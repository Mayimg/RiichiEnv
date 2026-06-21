"""Benchmark fixed-observation belief sampling throughput."""

from __future__ import annotations

import argparse

from riichienv_ml.belief_sampling_benchmark import BeliefSamplingBenchmark
from riichienv_ml.config import load_config

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv():
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark belief sampler throughput on fixed observations")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--input_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--decisions_csv_path", type=str, default=None)
    parser.add_argument("--prog_len_csv_path", type=str, default=None)
    parser.add_argument("--prog_len_bucket_csv_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_logs", type=int, default=None)
    parser.add_argument("--max_decisions", type=int, default=None)
    parser.add_argument("--decision_stride", type=int, default=None)
    parser.add_argument("--samples_per_call", type=int, default=None)
    parser.add_argument("--target_duration_ms", type=float, default=None)
    parser.add_argument("--warmup_calls", type=int, default=None)
    parser.add_argument("--warmup_samples_per_call", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--decode_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=None)
    parser.add_argument("--include_single_action", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include_decisions_in_summary", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--inference_dtype", choices=["fp32", "bf16"], default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config).belief_sampling_benchmark

    overrides = {}
    for field in (
        "input_glob",
        "output_dir",
        "summary_path",
        "decisions_csv_path",
        "prog_len_csv_path",
        "prog_len_bucket_csv_path",
        "model_path",
        "device",
        "num_logs",
        "max_decisions",
        "decision_stride",
        "samples_per_call",
        "target_duration_ms",
        "warmup_calls",
        "warmup_samples_per_call",
        "temperature",
        "decode_steps",
        "seed",
        "progress_interval",
        "include_decisions_in_summary",
        "matmul_precision",
        "inference_dtype",
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

    BeliefSamplingBenchmark(cfg).run()


if __name__ == "__main__":
    main()
