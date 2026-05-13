"""Annotate MJAI logs with hidden-hand samples from a belief sampler."""

from __future__ import annotations

import argparse

from riichienv_ml.belief_log_sampling import BeliefLogSampler
from riichienv_ml.config import load_config

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv():
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample hidden hands and append MJAI metadata")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--input_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_logs", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include_single_action", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress_output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config).belief_log_sampling

    overrides = {}
    for field in (
        "input_glob",
        "output_dir",
        "summary_path",
        "model_path",
        "device",
        "num_logs",
        "num_samples",
        "batch_size",
        "temperature",
        "seed",
        "compress_output",
        "matmul_precision",
    ):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if args.include_single_action:
        overrides["skip_single_action"] = False
    if args.overwrite:
        overrides["overwrite"] = True
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    BeliefLogSampler(cfg).run()


if __name__ == "__main__":
    main()
