"""Trainer for the joint hidden-allocation belief sampler."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import torch
import wandb
from loguru import logger
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from riichienv_ml.config import import_class
from riichienv_ml.features.belief_features import collate_belief_features
from riichienv_ml.trainers.bc_policy import _move_to_device, _state_dict_for_save, _with_suffix
from riichienv_ml.utils import AverageMeter, build_encoder, load_model_weights


def _belief_collate_fn(batch):
    features, targets = zip(*batch, strict=True)
    return collate_belief_features(list(features)), torch.stack(targets).long()


class BeliefAllocationTrainer:
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
        max_grad_norm: float = 10.0,
        model_config: dict | None = None,
        model_class: str = "riichienv_ml.models.belief_allocation.JointHiddenAllocationSampler",
        dataset_class: str = "riichienv_ml.datasets.belief_allocation.BeliefAllocationDataset",
        encoder_class: str = "riichienv_ml.features.belief_features.BeliefFeatureEncoder",
        n_players: int = 4,
        replay_rule: str = "tenhou",
        tile_dim: int = 34,
        skip_single_action: bool = True,
        shuffle_buffer_files: int = 1,
        sample_keep_prob: float = 1.0,
        eval_num_samples: int = 4,
        eval_sample_batches: int = 2,
    ):
        if n_players != 4:
            raise ValueError("BeliefAllocationTrainer currently supports 4-player mahjong only")
        self.data_glob = data_glob
        self.val_data_glob = val_data_glob
        self.load_model = load_model
        self.device = torch.device(device_str)
        self.batch_size = batch_size
        self.lr = lr
        self.lr_min = lr_min
        self.limit = int(limit)
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.model_config = model_config or {}
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.encoder_class = encoder_class
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.tile_dim = tile_dim
        self.skip_single_action = skip_single_action
        self.shuffle_buffer_files = int(shuffle_buffer_files)
        self.sample_keep_prob = float(sample_keep_prob)
        self.eval_num_samples = int(eval_num_samples)
        self.eval_sample_batches = int(eval_sample_batches)

    def _create_dataloader(self, dataset, *, is_train: bool) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
            "collate_fn": _belief_collate_fn,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = is_train
            kwargs["prefetch_factor"] = 4
        return DataLoader(**kwargs)

    def _build_dataset(self, files: list[str], *, is_train: bool):
        encoder_cls = import_class(self.encoder_class)
        encoder = build_encoder(encoder_cls, tile_dim=self.tile_dim, model_config=self.model_config)
        dataset_cls = import_class(self.dataset_class)
        return dataset_cls(
            files,
            is_train=is_train,
            n_players=self.n_players,
            replay_rule=self.replay_rule,
            encoder=encoder,
            skip_single_action=self.skip_single_action,
            shuffle_buffer_files=self.shuffle_buffer_files,
            sample_keep_prob=self.sample_keep_prob,
        )

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

        train_loader = self._create_dataloader(self._build_dataset(train_files, is_train=True), is_train=True)
        val_loader = None
        if val_files:
            val_loader = self._create_dataloader(self._build_dataset(val_files, is_train=False), is_train=False)

        model_cls = import_class(self.model_class)
        model = model_cls(**self.model_config).to(self.device)
        if self.load_model:
            load_model_weights(model, self.load_model, map_location=self.device)

        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(self.limit, 1), eta_min=self.lr_min)

        step = 0
        best_val_loss = float("inf")
        latest_path = _with_suffix(output_path, "latest")
        model.train()
        invalid_sum = 0.0
        invalid_count = 0

        try:
            for epoch in range(self.num_epochs):
                train_metrics, step, epoch_invalid_sum, epoch_invalid_count = self._train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    step,
                    epoch,
                )
                invalid_sum += epoch_invalid_sum
                invalid_count += epoch_invalid_count
                metrics = {"epoch": epoch, **train_metrics, "lr": optimizer.param_groups[0]["lr"]}

                if val_loader is not None:
                    metrics.update(self._eval_epoch(model, val_loader, epoch))

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

            if invalid_count > 0:
                logger.info("Training complete: train/invalid_target_rate={:.6f}", invalid_sum / invalid_count)
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
    ) -> tuple[dict[str, float], int, float, int]:
        loss_meter = AverageMeter("loss", ":.4f")
        acc_meter = AverageMeter("acc", ":.4f")
        invalid_meter = AverageMeter("invalid", ":.4f")
        window_loss_meter = AverageMeter("window_loss", ":.4f")
        window_acc_meter = AverageMeter("window_acc", ":.4f")
        log_interval = 100

        for batch_idx, (batch_features, batch_targets) in enumerate(dataloader):
            features = _move_to_device(batch_features, self.device)
            targets = batch_targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            out = model(features, target_counts=targets)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            batch_size = targets.shape[0]
            loss_meter.update(float(loss.item()), batch_size)
            acc_meter.update(float(out["acc"].item()), batch_size)
            invalid_meter.update(float(out.get("invalid_target_rate", torch.tensor(0.0)).item()), batch_size)
            window_loss_meter.update(float(loss.item()), batch_size)
            window_acc_meter.update(float(out["acc"].item()), batch_size)

            if step % log_interval == 0:
                logger.info(
                    "Epoch {} Step {} Batch {}: train/loss={:.4f} train/tile_acc={:.4f} "
                    "train/window100_loss={:.4f} train/window100_tile_acc={:.4f}",
                    epoch,
                    step,
                    batch_idx,
                    loss_meter.avg,
                    acc_meter.avg,
                    window_loss_meter.avg,
                    window_acc_meter.avg,
                )
                window_loss_meter.reset()
                window_acc_meter.reset()

            step += 1
            if step >= self.limit:
                break

        metrics = {
            "train/loss": loss_meter.avg,
            "train/tile_acc": acc_meter.avg,
        }
        logger.info(
            "Epoch {} train complete: loss={:.4f} tile_acc={:.4f}",
            epoch,
            metrics["train/loss"],
            metrics["train/tile_acc"],
        )
        return metrics, step, float(invalid_meter.sum), int(invalid_meter.count)

    @torch.inference_mode()
    def _eval_epoch(self, model: torch.nn.Module, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        was_training = model.training
        model.eval()

        loss_meter = AverageMeter("loss", ":.4f")
        acc_meter = AverageMeter("acc", ":.4f")
        invalid_meter = AverageMeter("invalid", ":.4f")
        sample_meters = {
            "allocation_legal_rate": AverageMeter("allocation_legal_rate", ":.4f"),
            "opponent_hand_size_exact_rate": AverageMeter("opponent_hand_size_exact_rate", ":.4f"),
            "unique_sample_rate": AverageMeter("unique_sample_rate", ":.4f"),
            "pairwise_l1_distance": AverageMeter("pairwise_l1_distance", ":.4f"),
        }
        for batch_idx, (batch_features, batch_targets) in enumerate(dataloader):
            features = _move_to_device(batch_features, self.device)
            targets = batch_targets.to(self.device, non_blocking=True)
            out = model(features, target_counts=targets)
            batch_size = targets.shape[0]
            loss_meter.update(float(out["loss"].item()), batch_size)
            acc_meter.update(float(out["acc"].item()), batch_size)
            invalid_meter.update(float(out.get("invalid_target_rate", torch.tensor(0.0)).item()), batch_size)
            if (
                self.eval_num_samples > 0
                and self.eval_sample_batches > 0
                and batch_idx < self.eval_sample_batches
                and hasattr(model, "allocation_diagnostics")
            ):
                samples = model.sample_allocations(features, num_samples=self.eval_num_samples)
                diagnostics = model.allocation_diagnostics(features, samples)
                for name, meter in sample_meters.items():
                    meter.update(float(diagnostics[name].item()), batch_size)

        if was_training:
            model.train()

        metrics = {
            "val/loss": loss_meter.avg,
            "val/tile_acc": acc_meter.avg,
            "val/invalid_target_rate": invalid_meter.avg,
        }
        if sample_meters["allocation_legal_rate"].count > 0:
            metrics.update(
                {
                    "val/sample_allocation_legal_rate": sample_meters["allocation_legal_rate"].avg,
                    "val/sample_opponent_hand_size_exact_rate": sample_meters[
                        "opponent_hand_size_exact_rate"
                    ].avg,
                    "val/sample_unique_rate": sample_meters["unique_sample_rate"].avg,
                    "val/sample_pairwise_l1_distance": sample_meters["pairwise_l1_distance"].avg,
                }
            )
        logger.info(
            "Epoch {} validation complete: loss={:.4f} tile_acc={:.4f} invalid_target_rate={:.6f}",
            epoch,
            metrics["val/loss"],
            metrics["val/tile_acc"],
            metrics["val/invalid_target_rate"],
        )
        return metrics
