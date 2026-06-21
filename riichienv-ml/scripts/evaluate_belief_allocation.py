"""Evaluate hidden-allocation belief sampler quality."""

from __future__ import annotations

import argparse

from riichienv_ml.belief_allocation_evaluation import BeliefAllocationEvaluator
from riichienv_ml.config import load_config

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv():
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate belief allocation sampler quality")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--input_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--decisions_csv_path", type=str, default=None)
    parser.add_argument("--opponents_csv_path", type=str, default=None)
    parser.add_argument("--opponent_state_csv_path", type=str, default=None)
    parser.add_argument("--opponent_discard_count_csv_path", type=str, default=None)
    parser.add_argument("--opponent_state_discard_count_csv_path", type=str, default=None)
    parser.add_argument("--shanten_distribution_csv_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_logs", type=int, default=None)
    parser.add_argument("--max_decisions", type=int, default=None)
    parser.add_argument("--decision_stride", type=int, default=None)
    parser.add_argument("--sample_keep_prob", type=float, default=None)
    parser.add_argument("--samples_per_decision", "--num_samples", dest="samples_per_decision", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--warmup_batches", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--decode_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=None)
    parser.add_argument("--dora_weight", type=float, default=None)
    parser.add_argument("--red_weight", type=float, default=None)
    parser.add_argument("--include_single_action", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include_decisions_in_summary", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include_opponents_in_summary", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--semantic_features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--inference_dtype", choices=["fp32", "bf16"], default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config).belief_allocation_evaluation

    overrides = {}
    for field in (
        "input_glob",
        "output_dir",
        "report_path",
        "summary_path",
        "decisions_csv_path",
        "opponents_csv_path",
        "opponent_state_csv_path",
        "opponent_discard_count_csv_path",
        "opponent_state_discard_count_csv_path",
        "shanten_distribution_csv_path",
        "model_path",
        "device",
        "num_logs",
        "max_decisions",
        "decision_stride",
        "sample_keep_prob",
        "samples_per_decision",
        "batch_size",
        "warmup_batches",
        "temperature",
        "decode_steps",
        "seed",
        "progress_interval",
        "dora_weight",
        "red_weight",
        "include_decisions_in_summary",
        "include_opponents_in_summary",
        "semantic_features",
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

    BeliefAllocationEvaluator(cfg).run()


if __name__ == "__main__":
    main()
