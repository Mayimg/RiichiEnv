"""Offline behavior cloning trainer on mjai log data."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from riichienv_ml.config import import_class
from riichienv_ml.trainers.bc_logs import _create_evaluator
from riichienv_ml.utils import AverageMeter, load_model_weights


def _move_to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _state_dict_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _with_suffix(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


class BCPolicyTrainer:
    def __init__(
        self,
        data_glob: str,
        val_data_glob: str = "",
        load_model: str | None = None,
        device_str: str = "cuda",
        batch_size: int = 32,
        lr: float = 1e-4,
        lr_min: float = 1e-6,
        limit: int = 1000000,
        num_epochs: int = 1,
        num_workers: int = 8,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        max_grad_norm: float = 10.0,
        model_config: dict | None = None,
        model_class: str = "riichienv_ml.models.transformer.TransformerPolicyNetwork",
        dataset_class: str = "riichienv_ml.datasets.mjai_logs.BehaviorCloningDataset",
        encoder_class: str = "riichienv_ml.features.sequence_features.SequenceFeaturePackedEncoder",
        n_players: int = 4,
        replay_rule: str = "tenhou",
        tile_dim: int = 34,
        evaluator_config=None,
    ):
        self.data_glob = data_glob
        self.val_data_glob = val_data_glob
        self.load_model = load_model
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.batch_size = batch_size
        self.lr = lr
        self.lr_min = lr_min
        self.limit = int(limit)
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.max_grad_norm = max_grad_norm
        self.model_config = model_config or {}
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.encoder_class = encoder_class
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.tile_dim = tile_dim

        if evaluator_config is None:
            from riichienv_ml.config import EvaluatorConfig

            evaluator_config = EvaluatorConfig()
        self.evaluator_config = evaluator_config

        self.tp_evaluator = _create_evaluator(
            cfg_kwargs=dict(
                model_path=evaluator_config.model_path,
                evaluator_name=evaluator_config.evaluator_name,
                eval_device=evaluator_config.eval_device,
                opponents=[o.model_dump() for o in evaluator_config.opponents],
                model_class=model_class,
                encoder_class=encoder_class,
                tile_dim=tile_dim,
                device_str=device_str,
                n_players=n_players,
            ),
            model_config=self.model_config,
        )

    def _create_dataloader(self, dataset, *, is_train: bool) -> DataLoader:
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = is_train
            kwargs["prefetch_factor"] = 4
        return DataLoader(**kwargs)

    @staticmethod
    def _mask_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        legal = masks.bool() if masks.dtype != torch.bool else masks
        return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)

    def _forward_logits(self, model: torch.nn.Module, features):
        outputs = model(features)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _run_eval(self, model: torch.nn.Module, step: int):
        if self.tp_evaluator is None:
            return {}
        try:
            hw = _state_dict_for_save(model)
            model.eval()
            metrics = self.tp_evaluator.evaluate(hw, num_episodes=self.evaluator_config.eval_episodes)
            logline = self.tp_evaluator.metrics_to_logline(metrics)
            logger.info(f"Eval @ step {step}: {logline}")
            return metrics
        except Exception as e:
            logger.error(f"Mortal evaluation failed at step {step}: {e}")
            return {}
        finally:
            model.train()

    def train(self, output_path: str) -> None:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        train_files = glob.glob(self.data_glob, recursive=True)
        if not train_files:
            raise ValueError(f"No data found at {self.data_glob}")
        logger.info(f"Found {len(train_files)} training files")

        val_files = glob.glob(self.val_data_glob, recursive=True) if self.val_data_glob else []
        if self.val_data_glob:
            logger.info(f"Found {len(val_files)} validation files")

        EncoderClass = import_class(self.encoder_class)
        encoder = EncoderClass(tile_dim=self.tile_dim)

        DatasetClass = import_class(self.dataset_class)
        train_dataset = DatasetClass(
            train_files,
            is_train=True,
            n_players=self.n_players,
            replay_rule=self.replay_rule,
            encoder=encoder,
        )
        train_loader = self._create_dataloader(train_dataset, is_train=True)

        val_loader = None
        if val_files:
            val_dataset = DatasetClass(
                val_files,
                is_train=False,
                n_players=self.n_players,
                replay_rule=self.replay_rule,
                encoder=encoder,
            )
            val_loader = self._create_dataloader(val_dataset, is_train=False)

        ModelClass = import_class(self.model_class)
        model = ModelClass(**self.model_config).to(self.device)
        if self.load_model:
            load_model_weights(model, self.load_model, map_location=self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(self.limit, 1), eta_min=self.lr_min)
        model.train()

        step = 0
        best_val_loss = float("inf")
        latest_path = _with_suffix(output_path, "latest")

        try:
            for epoch in range(self.num_epochs):
                train_metrics, step = self._train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=epoch,
                )

                metrics = {
                    "epoch": epoch,
                    **train_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }

                if val_loader is not None:
                    val_metrics = self._eval_epoch(model=model, dataloader=val_loader, epoch=epoch)
                    metrics.update(val_metrics)

                if self.tp_evaluator is not None:
                    metrics.update(self._run_eval(model, step))

                state_dict = _state_dict_for_save(model)
                torch.save(state_dict, latest_path)
                logger.info(f"Saved latest checkpoint to {latest_path}")

                should_save_best = val_loader is None or metrics["val/loss"] < best_val_loss
                if should_save_best:
                    if val_loader is not None:
                        best_val_loss = metrics["val/loss"]
                    torch.save(state_dict, output_path)
                    logger.info(f"Saved model to {output_path}")

                wandb.log(metrics, step=step)

                if step >= self.limit:
                    break
        finally:
            wandb.finish()

    def _train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        step: int,
        epoch: int,
    ) -> tuple[dict[str, float], int]:
        loss_meter = AverageMeter("loss", ":.4f")
        acc_meter = AverageMeter("acc", ":.4f")

        for batch_idx, (features, actions, masks) in enumerate(dataloader):
            features = _move_to_device(features, self.device)
            actions = actions.long().to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            logits = self._forward_logits(model, features)
            logits = self._mask_logits(logits, masks)
            loss = F.cross_entropy(logits, actions, label_smoothing=self.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            predictions = logits.argmax(dim=1)
            acc = (predictions == actions).float().mean().item()

            batch_size = actions.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            if step % 100 == 0:
                logger.info(
                    "Epoch {} Step {} Batch {}: train/loss={:.4f} train/acc={:.4f}",
                    epoch,
                    step,
                    batch_idx,
                    loss_meter.avg,
                    acc_meter.avg,
                )

            step += 1
            if step >= self.limit:
                break

        metrics = {
            "train/loss": loss_meter.avg,
            "train/acc": acc_meter.avg,
        }
        logger.info(
            "Epoch {} train complete: loss={:.4f} acc={:.4f}",
            epoch,
            metrics["train/loss"],
            metrics["train/acc"],
        )
        return metrics, step

    @torch.inference_mode()
    def _eval_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        was_training = model.training
        model.eval()

        loss_meter = AverageMeter("loss", ":.4f")
        acc_meter = AverageMeter("acc", ":.4f")

        for features, actions, masks in dataloader:
            features = _move_to_device(features, self.device)
            actions = actions.long().to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self._forward_logits(model, features)
            logits = self._mask_logits(logits, masks)

            loss = F.cross_entropy(logits, actions, label_smoothing=self.label_smoothing)
            predictions = logits.argmax(dim=1)
            acc = (predictions == actions).float().mean().item()

            batch_size = actions.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

        metrics = {
            "val/loss": loss_meter.avg,
            "val/acc": acc_meter.avg,
        }
        logger.info(
            "Epoch {} validation complete: loss={:.4f} acc={:.4f}",
            epoch,
            metrics["val/loss"],
            metrics["val/acc"],
        )

        if was_training:
            model.train()
        return metrics
