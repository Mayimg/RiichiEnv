"""Profile fixed-observation belief sampler execution with PyTorch profiler."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from riichienv_ml.belief_sampling_benchmark import BeliefSamplingBenchmark, _feature_metadata
from riichienv_ml.config import load_config
from riichienv_ml.models.belief_allocation import belief_profile_ranges
from torch.profiler import ProfilerActivity, profile, record_function

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv():
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile belief sampler throughput on one fixed observation")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--decision_index", type=int, default=0, help="Global selected decision index to profile")
    parser.add_argument("--samples_per_call", type=int, default=None)
    parser.add_argument("--profile_calls", type=int, default=1)
    parser.add_argument("--warmup_calls", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--decode_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--trace_name", type=str, default=None)
    parser.add_argument("--input_glob", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_logs", type=int, default=None)
    parser.add_argument("--decision_stride", type=int, default=None)
    parser.add_argument("--max_decisions", type=int, default=None)
    parser.add_argument("--include_single_action", action="store_true")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--activities", choices=["auto", "cpu", "cuda", "both"], default="auto")
    parser.add_argument("--sort_by", type=str, default=None)
    parser.add_argument("--row_limit", type=int, default=80)
    parser.add_argument("--record_shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with_stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--with_flops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile_context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group_by_input_shape", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _apply_overrides(cfg, args: argparse.Namespace):
    overrides = {}
    for field in (
        "input_glob",
        "model_path",
        "device",
        "num_logs",
        "decision_stride",
        "max_decisions",
        "matmul_precision",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if args.include_single_action:
        overrides["skip_single_action"] = False
    if overrides:
        return cfg.model_copy(update=overrides)
    return cfg


def _profiler_activities(activity_name: str, device: torch.device) -> list[ProfilerActivity]:
    if activity_name == "cpu":
        return [ProfilerActivity.CPU]
    if activity_name == "cuda":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("--activities=cuda requires a CUDA device")
        return [ProfilerActivity.CUDA]
    if activity_name == "both":
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return activities
    if device.type == "cuda" and torch.cuda.is_available():
        return [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    return [ProfilerActivity.CPU]


def _select_decision(benchmark: BeliefSamplingBenchmark, decision_index: int) -> dict[str, Any]:
    if decision_index < 0:
        raise ValueError("decision_index must be non-negative")

    selected_index = 0
    input_files = benchmark._input_files()
    if not input_files:
        raise ValueError(f"No input logs found at {benchmark.cfg.input_glob}")

    for log_index, input_path in enumerate(input_files):
        records, log_summary = benchmark._collect_decisions(input_path)
        for decision_index_in_log, record in enumerate(records):
            if decision_index_in_log % benchmark.cfg.decision_stride != 0:
                continue
            if benchmark.cfg.max_decisions is not None and selected_index >= benchmark.cfg.max_decisions:
                break
            if selected_index == decision_index:
                return {
                    "record": record,
                    "input_path": input_path,
                    "log_index": log_index,
                    "decision_index_in_log": decision_index_in_log,
                    "global_decision_index": selected_index,
                    "log_summary": log_summary,
                }
            selected_index += 1

    raise ValueError(f"decision_index {decision_index} is out of range; collected {selected_index} selected decisions")


def _output_paths(cfg, args: argparse.Namespace, effective_decode_steps: int) -> dict[str, Path]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir) / "profiler"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.trace_name or (
        f"decision_{args.decision_index}_samples_{args.samples_per_call or cfg.samples_per_call}"
        f"_steps_{effective_decode_steps}_{timestamp}"
    )
    return {
        "trace": output_dir / f"{name}.trace.json",
        "table": output_dir / f"{name}.table.txt",
        "metadata": output_dir / f"{name}.metadata.json",
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.profile_calls <= 0:
        raise ValueError("profile_calls must be positive")
    if args.warmup_calls < 0:
        raise ValueError("warmup_calls must be non-negative")
    if args.samples_per_call is not None and args.samples_per_call <= 0:
        raise ValueError("samples_per_call must be positive")


def _seed_everything(cfg) -> None:
    if cfg.seed is None:
        return
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))


def _run_warmup(
    benchmark: BeliefSamplingBenchmark,
    features: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    samples_per_call: int,
    temperature: float,
    decode_steps: int | None,
) -> None:
    with torch.inference_mode(), record_function("belief/profile_warmup"):
        warmup_context = benchmark.model.prepare_allocation_sampling_context(features)
        for _ in range(args.warmup_calls):
            samples = benchmark.model.sample_allocations_from_context(
                warmup_context,
                num_samples=samples_per_call,
                temperature=temperature,
                decode_steps=decode_steps,
            )
            benchmark._sync()
            del samples


def _run_profile(
    benchmark: BeliefSamplingBenchmark,
    features: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    activities: list[ProfilerActivity],
    samples_per_call: int,
    temperature: float,
    decode_steps: int | None,
):
    benchmark._reset_peak_memory()
    benchmark._sync()
    context = None
    if not args.profile_context:
        context = benchmark.model.prepare_allocation_sampling_context(features)
        benchmark._sync()

    started_at = time.perf_counter()
    with torch.inference_mode(), profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
    ) as prof, belief_profile_ranges():
        with record_function("belief/profile"):
            if args.profile_context:
                context = benchmark.model.prepare_allocation_sampling_context(features)
                benchmark._sync()
            if context is None:
                raise RuntimeError("sampling context was not prepared")
            for call_idx in range(args.profile_calls):
                with record_function(f"belief/profile_sample_call_{call_idx}"):
                    samples = benchmark.model.sample_allocations_from_context(
                        context,
                        num_samples=samples_per_call,
                        temperature=temperature,
                        decode_steps=decode_steps,
                    )
                    benchmark._sync()
                    del samples
    return prof, (time.perf_counter() - started_at) * 1000.0


def _build_metadata(
    benchmark: BeliefSamplingBenchmark,
    selected: dict[str, Any],
    args: argparse.Namespace,
    cfg,
    *,
    activities: list[ProfilerActivity],
    samples_per_call: int,
    temperature: float,
    decode_steps: int | None,
    effective_decode_steps: int,
    feature_prepare_ms: float,
    elapsed_ms: float,
    sort_by: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    record = selected["record"]
    return {
        "config_path": args.config,
        "model_path": cfg.model_path,
        "input_path": str(selected["input_path"]),
        "log_index": selected["log_index"],
        "decision_index_in_log": selected["decision_index_in_log"],
        "global_decision_index": selected["global_decision_index"],
        "feature_prepare_ms": feature_prepare_ms,
        "decision": {
            "observer": record.observer,
            "event_index": record.event_index,
            "response_source_index": record.response_source_index,
            "response_decision": record.response_decision,
            **_feature_metadata(record.feature),
        },
        "profile": {
            "device": str(benchmark.device),
            "activities": [activity.name for activity in activities],
            "samples_per_call": samples_per_call,
            "profile_calls": args.profile_calls,
            "warmup_calls": args.warmup_calls,
            "temperature": temperature,
            "decode_steps": decode_steps,
            "effective_decode_steps": effective_decode_steps,
            "profile_context": args.profile_context,
            "record_shapes": args.record_shapes,
            "profile_memory": args.profile_memory,
            "with_stack": args.with_stack,
            "with_flops": args.with_flops,
            "sort_by": sort_by,
            "elapsed_ms": elapsed_ms,
            "samples_per_second": (samples_per_call * args.profile_calls) / (elapsed_ms / 1000.0)
            if elapsed_ms > 0.0
            else None,
            **benchmark._memory_stats(),
        },
        "output": {key: str(path) for key, path in paths.items()},
        "log_summary": selected["log_summary"],
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    _validate_args(args)
    cfg = _apply_overrides(load_config(args.config).belief_sampling_benchmark, args)
    _seed_everything(cfg)

    benchmark = BeliefSamplingBenchmark(cfg)
    selected = _select_decision(benchmark, args.decision_index)
    record = selected["record"]
    features, feature_prepare_ms = benchmark._prepare_feature_batch(record.feature)
    samples_per_call = int(args.samples_per_call or cfg.samples_per_call)
    temperature = float(args.temperature if args.temperature is not None else cfg.temperature)
    decode_steps = args.decode_steps if args.decode_steps is not None else cfg.decode_steps
    effective_decode_steps = int(decode_steps or getattr(benchmark.model, "decode_steps", 0))
    paths = _output_paths(cfg, args, effective_decode_steps)
    activities = _profiler_activities(args.activities, benchmark.device)
    sort_by = args.sort_by or ("cuda_time_total" if ProfilerActivity.CUDA in activities else "cpu_time_total")

    _run_warmup(
        benchmark,
        features,
        args,
        samples_per_call=samples_per_call,
        temperature=temperature,
        decode_steps=decode_steps,
    )
    prof, elapsed_ms = _run_profile(
        benchmark,
        features,
        args,
        activities=activities,
        samples_per_call=samples_per_call,
        temperature=temperature,
        decode_steps=decode_steps,
    )

    table = prof.key_averages(group_by_input_shape=args.group_by_input_shape).table(
        sort_by=sort_by,
        row_limit=args.row_limit,
    )
    paths["table"].write_text(table + "\n", encoding="utf-8")
    prof.export_chrome_trace(str(paths["trace"]))

    metadata = _build_metadata(
        benchmark,
        selected,
        args,
        cfg,
        activities=activities,
        samples_per_call=samples_per_call,
        temperature=temperature,
        decode_steps=decode_steps,
        effective_decode_steps=effective_decode_steps,
        feature_prepare_ms=feature_prepare_ms,
        elapsed_ms=elapsed_ms,
        sort_by=sort_by,
        paths=paths,
    )
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(table)
    print(f"Saved profiler table to {paths['table']}")
    print(f"Saved Chrome trace to {paths['trace']}")
    print(f"Saved profiler metadata to {paths['metadata']}")


if __name__ == "__main__":
    main()
