"""Train the joint hidden-allocation belief sampler."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.multiprocessing
from riichienv_ml.config import load_config
from riichienv_ml.trainers.belief_allocation import BeliefAllocationTrainer
from riichienv_ml.utils import configure_matmul_precision, init_wandb, setup_logging

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv():
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hidden-allocation belief sampler")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data_glob", type=str, default=None)
    parser.add_argument("--val_data_glob", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--shuffle_buffer_files", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config).belief_sampler

    overrides = {}
    for field in (
        "data_glob",
        "val_data_glob",
        "load_model",
        "output",
        "device",
        "batch_size",
        "lr",
        "lr_min",
        "num_epochs",
        "num_workers",
        "shuffle_buffer_files",
        "limit",
        "matmul_precision",
    ):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    setup_logging(str(Path(cfg.output).parent), "train_belief")
    configure_matmul_precision(cfg.matmul_precision)
    init_wandb(cfg, config_path=args.config)

    game = cfg.game
    trainer = BeliefAllocationTrainer(
        data_glob=cfg.data_glob,
        val_data_glob=cfg.val_data_glob,
        load_model=cfg.load_model,
        device_str=cfg.device,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        lr_min=cfg.lr_min,
        limit=cfg.limit,
        num_epochs=cfg.num_epochs,
        num_workers=cfg.num_workers,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        model_config=cfg.model.model_dump(),
        model_class=cfg.model_class,
        dataset_class=cfg.dataset_class,
        encoder_class=cfg.encoder_class,
        n_players=game.n_players,
        replay_rule=game.replay_rule,
        tile_dim=game.tile_dim,
        skip_single_action=cfg.skip_single_action,
        shuffle_buffer_files=cfg.shuffle_buffer_files,
    )
    trainer.train(cfg.output)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
