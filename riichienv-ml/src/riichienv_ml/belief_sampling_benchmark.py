"""Benchmark belief-sampler throughput on fixed decision observations."""

from __future__ import annotations

import csv
import glob
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from riichienv import MjaiReplay
from riichienv_ml.belief_log_sampling import (
    _PASS_TYPE,
    _RESPONSE_SOURCE_TYPES,
    _action_to_mjai,
    _find_next_event,
    _read_jsonl,
    _to_device,
)
from riichienv_ml.config import BeliefSamplingBenchmarkConfig, import_class
from riichienv_ml.features.belief_features import BeliefFeatureEncoder, collate_belief_features
from riichienv_ml.utils import (
    build_encoder,
    configure_inference_dtype,
    configure_matmul_precision,
    inference_autocast,
    load_model_weights,
)

_PROG_LEN_BUCKETS = (
    (0, 4),
    (5, 9),
    (10, 19),
    (20, 39),
    (40, None),
)
_PHASE_NAMES = {
    BeliefFeatureEncoder.PHASE_ACT: "act",
    BeliefFeatureEncoder.PHASE_RESPONSE: "response",
    BeliefFeatureEncoder.PHASE_UNKNOWN: "unknown",
}


@dataclass
class _BenchmarkDecisionRecord:
    feature: dict[str, torch.Tensor]
    observer: int
    event_index: int | None
    response_source_index: int | None
    response_decision: bool


def _tensor_int(value: torch.Tensor | int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _masked_len(feature: dict[str, torch.Tensor], mask_key: str, value_key: str) -> int:
    mask = feature.get(mask_key)
    if isinstance(mask, torch.Tensor):
        return int(mask.detach().cpu().long().sum().item())
    value = feature.get(value_key)
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    return 0


def _feature_metadata(feature: dict[str, torch.Tensor]) -> dict[str, Any]:
    phase_id = _tensor_int(feature.get("belief_phase"), BeliefFeatureEncoder.PHASE_UNKNOWN)
    hand_sizes = feature["belief_hand_sizes"][:, 1].detach().cpu().long().tolist()
    visible_total = int(feature["visible_tile_counts"][:, 1].detach().cpu().long().sum().item())
    return {
        "phase": _PHASE_NAMES.get(phase_id, "unknown"),
        "phase_id": phase_id,
        "current_actor": _tensor_int(feature.get("belief_current_actor"), 4),
        "prog_len": _masked_len(feature, "prog_mask", "progression"),
        "sparse_meld_len": _masked_len(feature, "sparse_meld_mask", "sparse_melds"),
        "observer_hand_size": int(hand_sizes[0]),
        "opponent_slots": [int(value) for value in hand_sizes[1:4]],
        "opponent_slot_total": int(sum(hand_sizes[1:4])),
        "visible_tile_total": visible_total,
        "unseen_tile_total": int(136 - visible_total),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _value_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    total = sum(float(value) for value in values)
    return {
        "count": len(values),
        "mean": total / len(values),
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_seconds = sum(float(row["elapsed_ms"]) for row in rows) / 1000.0
    total_samples = sum(int(row["total_samples"]) for row in rows)
    total_calls = sum(int(row["num_calls"]) for row in rows)
    return {
        "num_decisions": len(rows),
        "total_samples": total_samples,
        "total_calls": total_calls,
        "sampling_elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_samples / elapsed_seconds if elapsed_seconds > 0.0 else None,
        "decision_samples_per_second": _value_stats([float(row["samples_per_second"]) for row in rows]),
        "elapsed_ms": _value_stats([float(row["elapsed_ms"]) for row in rows]),
        "encoder_cache_ms": _value_stats([float(row["encoder_cache_ms"]) for row in rows]),
        "decoder_elapsed_ms": _value_stats([float(row["decoder_elapsed_ms"]) for row in rows]),
        "call_ms": _value_stats([float(row["mean_call_ms"]) for row in rows]),
        "num_calls": _value_stats([float(row["num_calls"]) for row in rows]),
    }


def _prog_len_bucket_label(value: int) -> str:
    for lo, hi in _PROG_LEN_BUCKETS:
        if hi is None:
            if value >= lo:
                return f"{lo}+"
        elif lo <= value <= hi:
            return f"{lo}-{hi}"
    return "other"


def _group_aggregates(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [
        {
            key: group_key,
            **_aggregate_rows(group_rows),
        }
        for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0])
    ]


def _prog_len_bucket_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_prog_len_bucket_label(int(row["prog_len"]))].append(row)
    order = {_prog_len_bucket_label(lo): idx for idx, (lo, _hi) in enumerate(_PROG_LEN_BUCKETS)}
    return [
        {
            "prog_len_bucket": label,
            **_aggregate_rows(group_rows),
        }
        for label, group_rows in sorted(groups.items(), key=lambda item: order.get(item[0], 999))
    ]


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row[key]) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))


def _jsonable_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable_csv_value(row.get(key)) for key in fieldnames})


class BeliefSamplingBenchmark:
    """Run fixed-observation throughput measurements for the belief sampler."""

    def __init__(self, cfg: BeliefSamplingBenchmarkConfig):
        if cfg.game.n_players != 4:
            raise ValueError("Belief sampling benchmark currently supports 4-player mahjong only")
        if cfg.num_logs <= 0:
            raise ValueError("num_logs must be positive")
        if cfg.max_decisions is not None and cfg.max_decisions <= 0:
            raise ValueError("max_decisions must be positive when set")
        if cfg.decision_stride <= 0:
            raise ValueError("decision_stride must be positive")
        if cfg.samples_per_call <= 0:
            raise ValueError("samples_per_call must be positive")
        if cfg.target_duration_ms < 0.0:
            raise ValueError("target_duration_ms must be non-negative")
        if cfg.warmup_calls < 0:
            raise ValueError("warmup_calls must be non-negative")
        if cfg.warmup_samples_per_call is not None and cfg.warmup_samples_per_call <= 0:
            raise ValueError("warmup_samples_per_call must be positive when set")
        if cfg.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.inference_dtype = configure_inference_dtype(cfg.inference_dtype, self.device)
        self.output_dir = Path(cfg.output_dir)
        self.summary_path = Path(cfg.summary_path) if cfg.summary_path else self.output_dir / "summary.json"
        self.decisions_csv_path = (
            Path(cfg.decisions_csv_path) if cfg.decisions_csv_path else self.output_dir / "decisions.csv"
        )
        self.prog_len_csv_path = (
            Path(cfg.prog_len_csv_path) if cfg.prog_len_csv_path else self.output_dir / "prog_len.csv"
        )
        self.prog_len_bucket_csv_path = (
            Path(cfg.prog_len_bucket_csv_path)
            if cfg.prog_len_bucket_csv_path
            else self.output_dir / "prog_len_buckets.csv"
        )
        self.encoder = self._build_encoder()
        self.model = self._build_model()

    def _build_encoder(self) -> BeliefFeatureEncoder:
        encoder_cls = import_class(self.cfg.encoder_class)
        return build_encoder(encoder_cls, tile_dim=self.cfg.game.tile_dim, model_config=self.cfg.model.model_dump())

    def _build_model(self) -> torch.nn.Module:
        configure_matmul_precision(self.cfg.matmul_precision)
        model_cls = import_class(self.cfg.model_class)
        model = model_cls(**self.cfg.model.model_dump()).to(self.device)
        load_model_weights(model, self.cfg.model_path, map_location=self.device)
        model.eval()
        return model

    def _input_files(self) -> list[Path]:
        files = [Path(path) for path in sorted(glob.glob(self.cfg.input_glob, recursive=True))]
        return files[: self.cfg.num_logs]

    def _prepare_output(self) -> None:
        output_paths = [
            self.summary_path,
            self.decisions_csv_path,
            self.prog_len_csv_path,
            self.prog_len_bucket_csv_path,
        ]
        if not self.cfg.overwrite:
            existing = [path for path in output_paths if path.exists()]
            if existing:
                raise FileExistsError(f"Benchmark output already exists: {existing[0]}. Set overwrite=true.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _collect_decisions(self, input_path: Path) -> tuple[list[_BenchmarkDecisionRecord], dict[str, Any]]:
        events = _read_jsonl(input_path)
        replay = MjaiReplay.from_jsonl(str(input_path), rule=self.cfg.game.replay_rule)
        records: list[_BenchmarkDecisionRecord] = []
        event_pos = 0
        last_response_source_idx: int | None = None
        skipped_unmatched = 0
        skipped_response_without_source = 0

        for kyoku in replay.take_kyokus():
            for observer, obs, action in kyoku.steps(
                seat=None,
                skip_single_action=self.cfg.skip_single_action,
            ):
                action_mjai = _action_to_mjai(action, observer)
                if action_mjai.get("type") == _PASS_TYPE:
                    if last_response_source_idx is None:
                        skipped_response_without_source += 1
                        logger.warning(
                            "Skipping response benchmark sample without source event: {} observer={}",
                            input_path,
                            observer,
                        )
                        continue
                    records.append(
                        _BenchmarkDecisionRecord(
                            feature=self.encoder.encode(obs),
                            observer=int(observer),
                            event_index=None,
                            response_source_index=last_response_source_idx,
                            response_decision=True,
                        )
                    )
                    continue

                event_idx = _find_next_event(events, action_mjai, event_pos)
                if event_idx is None:
                    skipped_unmatched += 1
                    logger.warning(
                        "Could not match replay action to MJAI event: {} observer={} action={}",
                        input_path,
                        observer,
                        action_mjai,
                    )
                    continue

                for idx in range(event_pos, event_idx + 1):
                    if events[idx].get("type") in _RESPONSE_SOURCE_TYPES:
                        last_response_source_idx = idx
                event_pos = event_idx + 1
                records.append(
                    _BenchmarkDecisionRecord(
                        feature=self.encoder.encode(obs),
                        observer=int(observer),
                        event_index=event_idx,
                        response_source_index=None,
                        response_decision=False,
                    )
                )

        return records, {
            "input_path": str(input_path),
            "num_events": len(events),
            "num_decisions": len(records),
            "skipped_unmatched": skipped_unmatched,
            "skipped_response_without_source": skipped_response_without_source,
        }

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _reset_peak_memory(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _memory_stats(self) -> dict[str, float | None]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {
                "peak_allocated_mb": None,
                "peak_reserved_mb": None,
            }
        mb = 1024.0 * 1024.0
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / mb,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / mb,
        }

    def _prepare_feature_batch(self, feature: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        started_at = time.perf_counter()
        features = collate_belief_features([feature])
        features = _to_device(features, self.device)
        self._sync()
        return features, (time.perf_counter() - started_at) * 1000.0

    @torch.inference_mode()
    def _warmup(self, record: _BenchmarkDecisionRecord) -> dict[str, Any]:
        if self.cfg.warmup_calls == 0:
            return {"warmup_calls": 0, "warmup_samples": 0, "warmup_elapsed_seconds": 0.0}
        samples_per_call = int(self.cfg.warmup_samples_per_call or self.cfg.samples_per_call)
        features, _feature_prepare_ms = self._prepare_feature_batch(record.feature)
        self._sync()
        started_at = time.perf_counter()
        with inference_autocast(self.device, self.inference_dtype):
            sampling_context = self.model.prepare_allocation_sampling_context(features)
        self._sync()
        for _ in range(self.cfg.warmup_calls):
            with inference_autocast(self.device, self.inference_dtype):
                samples = self.model.sample_allocations_from_context(
                    sampling_context,
                    num_samples=samples_per_call,
                    temperature=self.cfg.temperature,
                    decode_steps=self.cfg.decode_steps,
                )
            self._sync()
            del samples
        elapsed = time.perf_counter() - started_at
        return {
            "warmup_calls": self.cfg.warmup_calls,
            "warmup_samples_per_call": samples_per_call,
            "warmup_samples": self.cfg.warmup_calls * samples_per_call,
            "warmup_elapsed_seconds": elapsed,
        }

    @torch.inference_mode()
    def _measure_decision(
        self,
        record: _BenchmarkDecisionRecord,
        *,
        input_path: Path,
        log_index: int,
        decision_index_in_log: int,
        global_decision_index: int,
    ) -> dict[str, Any]:
        feature_meta = _feature_metadata(record.feature)
        features, feature_prepare_ms = self._prepare_feature_batch(record.feature)
        self._reset_peak_memory()
        self._sync()

        target_seconds = self.cfg.target_duration_ms / 1000.0
        call_durations_ms: list[float] = []
        total_samples = 0
        started_at = time.perf_counter()
        cache_started_at = time.perf_counter()
        with inference_autocast(self.device, self.inference_dtype):
            sampling_context = self.model.prepare_allocation_sampling_context(features)
        self._sync()
        encoder_cache_ms = (time.perf_counter() - cache_started_at) * 1000.0
        elapsed = time.perf_counter() - started_at
        while not call_durations_ms or elapsed < target_seconds:
            call_started_at = time.perf_counter()
            with inference_autocast(self.device, self.inference_dtype):
                samples = self.model.sample_allocations_from_context(
                    sampling_context,
                    num_samples=self.cfg.samples_per_call,
                    temperature=self.cfg.temperature,
                    decode_steps=self.cfg.decode_steps,
                )
            self._sync()
            call_elapsed_ms = (time.perf_counter() - call_started_at) * 1000.0
            total_samples += int(samples.shape[1])
            del samples
            call_durations_ms.append(call_elapsed_ms)
            elapsed = time.perf_counter() - started_at

        elapsed_ms = elapsed * 1000.0
        memory_stats = self._memory_stats()
        return {
            "global_decision_index": global_decision_index,
            "log_index": log_index,
            "decision_index_in_log": decision_index_in_log,
            "input_path": str(input_path),
            "observer": record.observer,
            "event_index": record.event_index,
            "response_source_index": record.response_source_index,
            "response_decision": record.response_decision,
            **feature_meta,
            "target_duration_ms": self.cfg.target_duration_ms,
            "actual_duration_ms": elapsed_ms,
            "elapsed_ms": elapsed_ms,
            "feature_prepare_ms": feature_prepare_ms,
            "encoder_cache_ms": encoder_cache_ms,
            "decoder_elapsed_ms": sum(call_durations_ms),
            "samples_per_call": self.cfg.samples_per_call,
            "num_calls": len(call_durations_ms),
            "total_samples": total_samples,
            "samples_per_second": total_samples / elapsed if elapsed > 0.0 else 0.0,
            "mean_call_ms": sum(call_durations_ms) / len(call_durations_ms),
            "min_call_ms": min(call_durations_ms),
            "max_call_ms": max(call_durations_ms),
            "first_call_ms": call_durations_ms[0],
            "call_durations_ms": call_durations_ms,
            **memory_stats,
        }

    def _device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device": str(self.device),
            "torch_version": torch.__version__,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "inference_dtype": self.inference_dtype,
            "cuda_available": torch.cuda.is_available(),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            info.update(
                {
                    "cuda_device_name": props.name,
                    "cuda_capability": f"{props.major}.{props.minor}",
                    "cuda_total_memory_mb": props.total_memory / (1024.0 * 1024.0),
                }
            )
        return info

    def _build_summary(
        self,
        *,
        input_files: list[Path],
        log_summaries: list[dict[str, Any]],
        decision_rows: list[dict[str, Any]],
        warmup: dict[str, Any],
        wall_elapsed_seconds: float,
    ) -> dict[str, Any]:
        prog_len_rows = _group_aggregates(decision_rows, "prog_len")
        prog_len_bucket_rows = _prog_len_bucket_aggregates(decision_rows)
        phase_rows = _group_aggregates(decision_rows, "phase")
        observer_rows = _group_aggregates(decision_rows, "observer")
        per_log_rows = _group_aggregates(decision_rows, "log_index")
        for row in per_log_rows:
            log_index = int(row["log_index"])
            if 0 <= log_index < len(input_files):
                row["input_path"] = str(input_files[log_index])

        summary = {
            "input_glob": self.cfg.input_glob,
            "input_files": [str(path) for path in input_files],
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "decisions_csv_path": str(self.decisions_csv_path),
            "prog_len_csv_path": str(self.prog_len_csv_path),
            "prog_len_bucket_csv_path": str(self.prog_len_bucket_csv_path),
            "model_path": self.cfg.model_path,
            "replay_rule": self.cfg.game.replay_rule,
            "num_logs": len(input_files),
            "num_decisions": len(decision_rows),
            "samples_per_call": self.cfg.samples_per_call,
            "target_duration_ms": self.cfg.target_duration_ms,
            "temperature": self.cfg.temperature,
            "decode_steps": self.cfg.decode_steps,
            "effective_decode_steps": int(self.cfg.decode_steps or getattr(self.model, "decode_steps", 0)),
            "skip_single_action": self.cfg.skip_single_action,
            "decision_stride": self.cfg.decision_stride,
            "max_decisions": self.cfg.max_decisions,
            "wall_elapsed_seconds": wall_elapsed_seconds,
            "warmup": warmup,
            "overall": _aggregate_rows(decision_rows),
            "speed_by_prog_len": prog_len_rows,
            "speed_by_prog_len_bucket": prog_len_bucket_rows,
            "speed_by_phase": phase_rows,
            "speed_by_observer": observer_rows,
            "speed_by_log": per_log_rows,
            "prog_len_distribution": _distribution(decision_rows, "prog_len"),
            "sparse_meld_len_distribution": _distribution(decision_rows, "sparse_meld_len"),
            "phase_distribution": _distribution(decision_rows, "phase"),
            "log_summaries": log_summaries,
            "device_info": self._device_info(),
            "config": self.cfg.model_dump(),
        }
        if self.cfg.include_decisions_in_summary:
            summary["decisions"] = decision_rows
        return summary

    def _write_outputs(self, summary: dict[str, Any], decision_rows: list[dict[str, Any]]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        _write_csv(self.decisions_csv_path, decision_rows)
        _write_csv(self.prog_len_csv_path, summary["speed_by_prog_len"])
        _write_csv(self.prog_len_bucket_csv_path, summary["speed_by_prog_len_bucket"])

    def run(self) -> dict[str, Any]:
        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.cfg.seed))

        input_files = self._input_files()
        if not input_files:
            raise ValueError(f"No input logs found at {self.cfg.input_glob}")
        self._prepare_output()

        started_at = time.perf_counter()
        all_records: list[tuple[Path, int, int, _BenchmarkDecisionRecord]] = []
        log_summaries = []
        for log_index, input_path in enumerate(input_files):
            records, log_summary = self._collect_decisions(input_path)
            selected_records = [
                (decision_index_in_log, record)
                for decision_index_in_log, record in enumerate(records)
                if decision_index_in_log % self.cfg.decision_stride == 0
            ]
            log_summary["selected_decisions"] = len(selected_records)
            log_summaries.append(log_summary)
            for decision_index_in_log, record in selected_records:
                all_records.append((input_path, log_index, decision_index_in_log, record))
            logger.info(
                "Collected benchmark decisions {}/{}: {} selected from {} decisions",
                log_index + 1,
                len(input_files),
                len(selected_records),
                len(records),
            )

        if self.cfg.max_decisions is not None:
            all_records = all_records[: self.cfg.max_decisions]
        if not all_records:
            raise ValueError("No benchmark decision observations were collected")

        warmup = self._warmup(all_records[0][3])
        decision_rows: list[dict[str, Any]] = []
        for global_decision_index, (input_path, log_index, decision_index_in_log, record) in enumerate(all_records):
            row = self._measure_decision(
                record,
                input_path=input_path,
                log_index=log_index,
                decision_index_in_log=decision_index_in_log,
                global_decision_index=global_decision_index,
            )
            decision_rows.append(row)
            if (global_decision_index + 1) % self.cfg.progress_interval == 0:
                logger.info(
                    "Benchmark progress: {}/{} decisions, latest {:.1f} samples/s (prog_len={})",
                    global_decision_index + 1,
                    len(all_records),
                    row["samples_per_second"],
                    row["prog_len"],
                )

        wall_elapsed_seconds = time.perf_counter() - started_at
        summary = self._build_summary(
            input_files=input_files,
            log_summaries=log_summaries,
            decision_rows=decision_rows,
            warmup=warmup,
            wall_elapsed_seconds=wall_elapsed_seconds,
        )
        self._write_outputs(summary, decision_rows)
        logger.info(
            "Saved belief sampling benchmark summary to {} ({:.1f} samples/s)",
            self.summary_path,
            summary["overall"]["samples_per_second"],
        )
        return summary
