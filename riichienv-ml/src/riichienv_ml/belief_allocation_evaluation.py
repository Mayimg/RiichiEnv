"""Evaluate hidden-allocation belief sampler quality."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from riichienv import MjaiReplay, calculate_shanten
from riichienv_ml.belief_log_sampling import (
    _PASS_TYPE,
    _RESPONSE_SOURCE_TYPES,
    _action_to_mjai,
    _find_next_event,
    _read_jsonl,
    _to_device,
)
from riichienv_ml.config import BeliefAllocationEvaluationConfig, import_class
from riichienv_ml.features.belief_features import (
    BUCKET_COUNT,
    TILE37_COUNT,
    TOTAL_TILE_COUNTS37,
    BeliefFeatureEncoder,
    collate_belief_features,
    make_hidden_allocation_target,
)
from riichienv_ml.utils import build_encoder, configure_matmul_precision, load_model_weights

_OPPONENT_COUNT = BUCKET_COUNT - 1
_OPPONENT_REL_SEATS = (1, 2, 3)
_RED_TILE37 = (0, 10, 20)
_SHANTEN_CACHE_MAX_SIZE = 100_000
_TILE37_TO_TILE34 = (
    4,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    13,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    22,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
)
_TERMINAL_HONOR_TILE34 = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
_DECISION_METRIC_KEYS = (
    "allocation_legal_rate",
    "bucket_count_exact_rate",
    "opponent_hand_size_exact_rate",
    "tile_count_exact_rate",
    "unique_sample_rate",
    "pairwise_l1_distance",
    "raw_tile37_opponent_es",
    "raw_tile34_opponent_es",
    "raw_tile37_dora_red_opponent_es",
    "mean_shanten_crps",
    "mean_semantic_es",
    "samples_per_second",
)
_OPPONENT_METRIC_KEYS = (
    "raw_tile37_es",
    "raw_tile34_es",
    "raw_tile37_dora_red_es",
    "shanten_crps",
    "semantic_es",
)


@dataclass(frozen=True)
class _OpponentMeta:
    relative_seat: int
    absolute_seat: int
    discard_count: int
    meld_count: int
    concealed_size: int
    riichi_active: bool
    state: str


@dataclass(frozen=True)
class _EvaluationDecisionRecord:
    feature: dict[str, torch.Tensor]
    target: torch.Tensor
    observer: int
    event_index: int | None
    response_source_index: int | None
    response_decision: bool
    decision_index_in_log: int
    opponent_meta: tuple[_OpponentMeta, _OpponentMeta, _OpponentMeta]
    dora_tile34_counts: tuple[int, ...]


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
    phase_names = {
        BeliefFeatureEncoder.PHASE_ACT: "act",
        BeliefFeatureEncoder.PHASE_RESPONSE: "response",
        BeliefFeatureEncoder.PHASE_UNKNOWN: "unknown",
    }
    hand_sizes = feature["belief_hand_sizes"][:, 1].detach().cpu().long().tolist()
    visible_total = int(feature["visible_tile_counts"][:, 1].detach().cpu().long().sum().item())
    return {
        "phase": phase_names.get(phase_id, "unknown"),
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
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _mean_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _counter_to_rate(counter: Counter[str], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {}
    return {key: value / denominator for key, value in sorted(counter.items(), key=lambda item: int(item[0]))}


def _sum_shanten_counts(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for row in rows:
        counts = row.get(key)
        if isinstance(counts, dict):
            out.update({str(k): int(v) for k, v in counts.items()})
    return out


def _aggregate_batch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_seconds = sum(float(row["sampling_elapsed_ms"]) for row in rows) / 1000.0
    total_samples = sum(int(row["total_samples"]) for row in rows)
    return {
        "num_batches": len(rows),
        "total_samples": total_samples,
        "sampling_elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_samples / elapsed_seconds if elapsed_seconds > 0.0 else None,
        "batch_samples_per_second": _value_stats([float(row["samples_per_second"]) for row in rows]),
        "batch_elapsed_ms": _value_stats([float(row["sampling_elapsed_ms"]) for row in rows]),
        "batch_size": _value_stats([float(row["num_decisions"]) for row in rows]),
    }


def _aggregate_decision_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_decisions": len(rows),
        "total_samples": sum(int(row["num_samples"]) for row in rows),
    }
    for key in _DECISION_METRIC_KEYS:
        out[key] = _mean_or_none(rows, key)
    return out


def _aggregate_opponent_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_counts = Counter(str(row["true_shanten"]) for row in rows)
    sample_counts = _sum_shanten_counts(rows, "sample_shanten_counts")
    out: dict[str, Any] = {
        "num_opponent_states": len(rows),
        "total_samples": sum(int(row["num_samples"]) for row in rows),
        "true_shanten_distribution": dict(sorted(true_counts.items(), key=lambda item: int(item[0]))),
        "sample_shanten_distribution": dict(sorted(sample_counts.items(), key=lambda item: int(item[0]))),
        "true_shanten_rate": _counter_to_rate(true_counts, len(rows)),
        "sample_shanten_rate": _counter_to_rate(sample_counts, sum(sample_counts.values())),
    }
    for key in _OPPONENT_METRIC_KEYS:
        out[key] = _mean_or_none(rows, key)
    return out


def _group_opponent_aggregates(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        row = {key: value for key, value in zip(keys, group_key, strict=True)}
        row.update(_aggregate_opponent_rows(group_rows))
        out.append(row)
    return out


def _shanten_distribution_rows(
    rows: list[dict[str, Any]],
    *,
    stratum_type: str,
    stratum_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not stratum_keys:
        groups = {("overall",): rows}
        key_names = ("stratum",)
    else:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in stratum_keys)].append(row)
        key_names = stratum_keys

    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        true_counts = Counter(str(row["true_shanten"]) for row in group_rows)
        sample_counts = _sum_shanten_counts(group_rows, "sample_shanten_counts")
        shanten_values = sorted(set(true_counts) | set(sample_counts), key=int)
        true_total = len(group_rows)
        sample_total = sum(sample_counts.values())
        for shanten in shanten_values:
            base = {"stratum_type": stratum_type}
            base.update({key: value for key, value in zip(key_names, group_key, strict=True)})
            true_count = int(true_counts.get(shanten, 0))
            sample_count = int(sample_counts.get(shanten, 0))
            out.append(
                {
                    **base,
                    "shanten": int(shanten),
                    "true_count": true_count,
                    "true_rate": true_count / true_total if true_total > 0 else None,
                    "sample_count": sample_count,
                    "sample_rate": sample_count / sample_total if sample_total > 0 else None,
                }
            )
    return out


def _format_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.{digits}f}%"


def _format_pp(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    number = float(value) * 100.0
    return f"{number:+.{digits}f} pp"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def _rate_at(row: dict[str, Any], key: str, shanten: int) -> float:
    values = row.get(key)
    if not isinstance(values, dict):
        return 0.0
    return float(values.get(str(shanten), 0.0) or 0.0)


def _rate_gte(row: dict[str, Any], key: str, shanten: int) -> float:
    values = row.get(key)
    if not isinstance(values, dict):
        return 0.0
    return sum(float(value or 0.0) for raw_key, value in values.items() if int(raw_key) >= shanten)


def _count_at(row: dict[str, Any], key: str, shanten: int) -> int:
    values = row.get(key)
    if not isinstance(values, dict):
        return 0
    return int(values.get(str(shanten), 0) or 0)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _sorted_state_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"closed": 0, "meld1": 1, "meld2": 2, "meld3plus": 3, "riichi": 4}
    return sorted(rows, key=lambda row: (order.get(str(row.get("opponent_state")), 99), str(row.get("opponent_state"))))


def _format_metric_row(name: str, value: Any, digits: int = 3) -> list[str]:
    return [name, _format_float(value, digits)]


def _shanten_rate_headers() -> list[str]:
    headers = []
    for label in ("0", "1", "2", "3", "4+"):
        headers.extend((f"True {label}", f"Sample {label}"))
    return headers


def _shanten_rate_cells(row: dict[str, Any]) -> list[str]:
    cells = []
    for shanten in range(4):
        cells.extend(
            (
                _format_percent(_rate_at(row, "true_shanten_rate", shanten)),
                _format_percent(_rate_at(row, "sample_shanten_rate", shanten)),
            )
        )
    cells.extend(
        (
            _format_percent(_rate_gte(row, "true_shanten_rate", 4)),
            _format_percent(_rate_gte(row, "sample_shanten_rate", 4)),
        )
    )
    return cells


def _render_markdown_report(summary: dict[str, Any]) -> str:
    input_files = summary.get("input_files") or []
    log_summaries = summary.get("log_summaries") or []
    overall_decisions = summary.get("overall_decisions") or {}
    overall_opponents = summary.get("overall_opponents") or {}
    sampling_speed = summary.get("sampling_speed") or {}
    valid_decisions = sum(int(row.get("num_decisions", 0) or 0) for row in log_summaries)
    selected_decisions = int(summary.get("num_decisions", 0) or 0)
    selection_rate = selected_decisions / valid_decisions if valid_decisions > 0 else None
    true_tenpai = _rate_at(overall_opponents, "true_shanten_rate", 0)
    sample_tenpai = _rate_at(overall_opponents, "sample_shanten_rate", 0)
    state_rows = list(summary.get("metrics_by_opponent_state") or [])
    riichi_row = next((row for row in state_rows if row.get("opponent_state") == "riichi"), None)
    riichi_true_tenpai = _rate_at(riichi_row or {}, "true_shanten_rate", 0) if riichi_row else None
    riichi_sample_tenpai = _rate_at(riichi_row or {}, "sample_shanten_rate", 0) if riichi_row else None

    lines = [
        "# Belief Allocation Evaluation Report",
        "",
        "## Quick Read",
        "",
        "- Sampler health: legal, bucket-exact, hand-size-exact, and tile-exact rates are "
        f"{_format_percent(overall_decisions.get('allocation_legal_rate'))}, "
        f"{_format_percent(overall_decisions.get('bucket_count_exact_rate'))}, "
        f"{_format_percent(overall_decisions.get('opponent_hand_size_exact_rate'))}, and "
        f"{_format_percent(overall_decisions.get('tile_count_exact_rate'))}.",
        f"- Overall tenpai rate: true {_format_percent(true_tenpai)}, sample {_format_percent(sample_tenpai)} "
        f"({_format_pp(sample_tenpai - true_tenpai)}).",
    ]
    if riichi_row is not None and riichi_true_tenpai is not None and riichi_sample_tenpai is not None:
        lines.append(
            f"- Riichi tenpai rate: true {_format_percent(riichi_true_tenpai)}, "
            f"sample {_format_percent(riichi_sample_tenpai)} "
            f"({_format_pp(riichi_sample_tenpai - riichi_true_tenpai)})."
        )
    lines.extend(
        [
            f"- Throughput: {_format_float(sampling_speed.get('samples_per_second'), 1)} samples/sec "
            f"over {_format_int(sampling_speed.get('total_samples'))} samples.",
            "",
            "## Run",
            "",
            _markdown_table(
                ["Item", "Value"],
                [
                    ["Input files", _format_int(len(input_files))],
                    ["First input", Path(input_files[0]).name if input_files else "n/a"],
                    ["Last input", Path(input_files[-1]).name if input_files else "n/a"],
                    ["Valid decisions scanned", _format_int(valid_decisions) if valid_decisions else "n/a"],
                    ["Selected decisions", _format_int(selected_decisions)],
                    ["Selection rate", _format_percent(selection_rate)],
                    ["Opponent states", _format_int(summary.get("num_opponent_states"))],
                    ["Samples per decision", _format_int(summary.get("samples_per_decision"))],
                    ["Batch size", _format_int(summary.get("batch_size"))],
                    ["Temperature", _format_float(summary.get("temperature"), 3)],
                    ["Decode steps", _format_int(summary.get("effective_decode_steps"))],
                    ["Sample keep prob", _format_float(summary.get("sample_keep_prob"), 6)],
                    ["Decision stride", _format_int(summary.get("decision_stride"))],
                    ["Seed", summary.get("seed", "n/a")],
                    ["Wall elapsed", _format_seconds(summary.get("wall_elapsed_seconds"))],
                    ["Sampling elapsed", _format_seconds(sampling_speed.get("sampling_elapsed_seconds"))],
                ],
            ),
            "",
            "## Sampler Health",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Allocation legal rate", _format_percent(overall_decisions.get("allocation_legal_rate"))],
                    ["Bucket count exact rate", _format_percent(overall_decisions.get("bucket_count_exact_rate"))],
                    [
                        "Opponent hand size exact rate",
                        _format_percent(overall_decisions.get("opponent_hand_size_exact_rate")),
                    ],
                    ["Tile count exact rate", _format_percent(overall_decisions.get("tile_count_exact_rate"))],
                    ["Unique sample rate", _format_percent(overall_decisions.get("unique_sample_rate"))],
                    ["Pairwise L1 distance", _format_float(overall_decisions.get("pairwise_l1_distance"), 3)],
                    ["Samples/sec", _format_float(overall_decisions.get("samples_per_second"), 1)],
                ],
            ),
            "",
            "## Quality Scores",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    _format_metric_row(
                        "Decision raw tile37 opponent ES",
                        overall_decisions.get("raw_tile37_opponent_es"),
                    ),
                    _format_metric_row(
                        "Decision raw tile34 opponent ES",
                        overall_decisions.get("raw_tile34_opponent_es"),
                    ),
                    _format_metric_row(
                        "Decision dora/red-weighted tile37 ES",
                        overall_decisions.get("raw_tile37_dora_red_opponent_es"),
                    ),
                    _format_metric_row("Opponent raw tile37 ES", overall_opponents.get("raw_tile37_es")),
                    _format_metric_row("Opponent raw tile34 ES", overall_opponents.get("raw_tile34_es")),
                    _format_metric_row(
                        "Opponent dora/red-weighted tile37 ES",
                        overall_opponents.get("raw_tile37_dora_red_es"),
                    ),
                    _format_metric_row("Shanten CRPS", overall_opponents.get("shanten_crps")),
                    _format_metric_row("Semantic ES", overall_opponents.get("semantic_es")),
                ],
            ),
            "",
            "## Shanten Distribution",
            "",
        ]
    )

    true_dist = overall_opponents.get("true_shanten_distribution") or {}
    sample_dist = overall_opponents.get("sample_shanten_distribution") or {}
    shanten_values = sorted({int(key) for key in true_dist} | {int(key) for key in sample_dist})
    lines.append(
        _markdown_table(
            ["Shanten", "True Count", "True Rate", "Sample Count", "Sample Rate", "Sample - True"],
            [
                [
                    shanten,
                    _format_int(_count_at(overall_opponents, "true_shanten_distribution", shanten)),
                    _format_percent(_rate_at(overall_opponents, "true_shanten_rate", shanten)),
                    _format_int(_count_at(overall_opponents, "sample_shanten_distribution", shanten)),
                    _format_percent(_rate_at(overall_opponents, "sample_shanten_rate", shanten)),
                    _format_pp(
                        _rate_at(overall_opponents, "sample_shanten_rate", shanten)
                        - _rate_at(overall_opponents, "true_shanten_rate", shanten)
                    ),
                ]
                for shanten in shanten_values
            ],
        )
    )

    lines.extend(["", "## By Opponent State", ""])
    lines.append(
        _markdown_table(
            ["State", "N", *_shanten_rate_headers(), "CRPS", "Raw37 ES", "Semantic ES"],
            [
                [
                    row.get("opponent_state", "n/a"),
                    _format_int(row.get("num_opponent_states")),
                    *_shanten_rate_cells(row),
                    _format_float(row.get("shanten_crps"), 3),
                    _format_float(row.get("raw_tile37_es"), 3),
                    _format_float(row.get("semantic_es"), 3),
                ]
                for row in _sorted_state_rows(state_rows)
            ],
        )
    )

    discard_rows = sorted(
        list(summary.get("metrics_by_opponent_discard_count") or []),
        key=lambda row: int(row.get("opponent_discard_count", 0)),
    )
    lines.extend(["", "## By Opponent Discard Count", ""])
    lines.append(
        _markdown_table(
            ["Discards", "N", *_shanten_rate_headers(), "CRPS", "Raw37 ES"],
            [
                [
                    row.get("opponent_discard_count", "n/a"),
                    _format_int(row.get("num_opponent_states")),
                    *_shanten_rate_cells(row),
                    _format_float(row.get("shanten_crps"), 3),
                    _format_float(row.get("raw_tile37_es"), 3),
                ]
                for row in discard_rows
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Lower Energy Score, shanten CRPS, and semantic ES are better.",
            "- State-level raw ES is affected by concealed hand size; compare it mainly across runs with the "
            "same stratum.",
            "- Detailed state-by-discard metrics and per-decision rows are kept in the CSV files.",
            "",
            "## Output Files",
            "",
            _markdown_table(
                ["File", "Path"],
                [
                    ["Report Markdown", summary.get("report_path", "n/a")],
                    ["Summary JSON", summary.get("summary_path", "n/a")],
                    ["Decision CSV", summary.get("decisions_csv_path", "n/a")],
                    ["Opponent CSV", summary.get("opponents_csv_path", "n/a")],
                    ["State CSV", summary.get("opponent_state_csv_path", "n/a")],
                    ["Discard CSV", summary.get("opponent_discard_count_csv_path", "n/a")],
                    ["State x Discard CSV", summary.get("opponent_state_discard_count_csv_path", "n/a")],
                    ["Shanten Distribution CSV", summary.get("shanten_distribution_csv_path", "n/a")],
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _next_dora_tile34(indicator_tile: int) -> int:
    tile34 = int(indicator_tile) // 4
    if tile34 < 27:
        suit = tile34 // 9
        number = tile34 % 9
        return suit * 9 + ((number + 1) % 9)
    if 27 <= tile34 <= 30:
        return 27 + ((tile34 - 27 + 1) % 4)
    if 31 <= tile34 <= 33:
        return 31 + ((tile34 - 31 + 1) % 3)
    return tile34


def _dora_tile34_counts(dora_indicators: list[int]) -> tuple[int, ...]:
    counts = [0] * 34
    for indicator in dora_indicators:
        counts[_next_dora_tile34(int(indicator))] += 1
    return tuple(counts)


def _opponent_state(meld_count: int, riichi_active: bool) -> str:
    if riichi_active:
        return "riichi"
    if meld_count == 0:
        return "closed"
    if meld_count == 1:
        return "meld1"
    if meld_count == 2:
        return "meld2"
    return "meld3plus"


def _opponent_metadata(obs, hidden_hands: list[list[int]]) -> tuple[_OpponentMeta, _OpponentMeta, _OpponentMeta]:
    observer = int(obs.player_id)
    metas = []
    for rel in _OPPONENT_REL_SEATS:
        abs_seat = (observer + rel) % 4
        meld_count = len(obs.melds[abs_seat])
        riichi_stage = bool(obs.riichi_stage[abs_seat]) if hasattr(obs, "riichi_stage") else False
        riichi_active = bool(obs.riichi_declared[abs_seat]) or riichi_stage
        metas.append(
            _OpponentMeta(
                relative_seat=rel,
                absolute_seat=abs_seat,
                discard_count=len(obs.discards[abs_seat]),
                meld_count=meld_count,
                concealed_size=len(hidden_hands[abs_seat]),
                riichi_active=riichi_active,
                state=_opponent_state(meld_count, riichi_active),
            )
        )
    return tuple(metas)  # type: ignore[return-value]


def _stable_keep_decision(seed: int, input_path: Path, decision_index: int, sample_keep_prob: float) -> bool:
    if sample_keep_prob >= 1.0:
        return True
    payload = f"{seed}:{input_path}:{decision_index}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    return value < sample_keep_prob


def _collapse_tile34(counts: torch.Tensor) -> torch.Tensor:
    tile34 = torch.tensor(_TILE37_TO_TILE34, dtype=torch.long, device=counts.device)
    out = torch.zeros(*counts.shape[:-1], 34, dtype=counts.dtype, device=counts.device)
    index_shape = (1,) * (counts.dim() - 1) + (TILE37_COUNT,)
    index = tile34.view(index_shape).expand_as(counts)
    return out.scatter_add_(-1, index, counts)


def _energy_score_weighted_l1(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    bucket_weights: torch.Tensor,
    tile_weights: torch.Tensor,
) -> torch.Tensor:
    if samples.dim() != 4:
        raise ValueError(f"samples must have shape (B, S, K, T), got {tuple(samples.shape)}")
    batch_size, num_samples, bucket_count, tile_count = samples.shape
    if target.shape != (batch_size, bucket_count, tile_count):
        raise ValueError(f"target must have shape {(batch_size, bucket_count, tile_count)}, got {tuple(target.shape)}")
    weights = bucket_weights.float().view(batch_size, 1, bucket_count, 1) * tile_weights.float().view(
        batch_size,
        1,
        1,
        tile_count,
    )
    sample_values = samples.float() * weights
    target_values = target.float().unsqueeze(1) * weights
    to_truth = (sample_values - target_values).abs().flatten(2).sum(dim=2).mean(dim=1)
    if num_samples < 2:
        return to_truth
    distances = torch.cdist(sample_values.flatten(2), sample_values.flatten(2), p=1)
    tri = torch.triu(torch.ones(num_samples, num_samples, dtype=torch.bool, device=samples.device), diagonal=1)
    pair_term = distances[:, tri].sum(dim=1) / float(num_samples * (num_samples - 1))
    return to_truth - pair_term


def _allocation_diagnostics_per_decision(
    features: dict[str, torch.Tensor],
    allocations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    visible = features["visible_tile_counts"][:, :, 1].long()
    total_counts = torch.tensor(TOTAL_TILE_COUNTS37, dtype=torch.long, device=allocations.device).unsqueeze(0)
    unseen_counts = total_counts - visible
    hand_sizes = features["belief_hand_sizes"][:, :, 1].long()
    opponent_caps = hand_sizes[:, 1:4]
    wall = unseen_counts.sum(dim=1, keepdim=True) - opponent_caps.sum(dim=1, keepdim=True)
    rem = torch.cat([opponent_caps, wall.clamp(min=0)], dim=1)

    alloc = allocations.long()
    tile_exact = (alloc.sum(dim=2) == unseen_counts.unsqueeze(1)).all(dim=-1)
    bucket_exact = (alloc.sum(dim=3) == rem.unsqueeze(1)).all(dim=-1)
    opponent_exact = (alloc[:, :, :_OPPONENT_COUNT].sum(dim=3) == rem[:, :_OPPONENT_COUNT].unsqueeze(1)).all(dim=-1)
    nonnegative = (alloc >= 0).flatten(2).all(dim=-1)
    legal = tile_exact & bucket_exact & nonnegative

    flat_cpu = alloc.detach().reshape(alloc.shape[0], alloc.shape[1], -1).cpu()
    unique_rates = []
    for sample_set in flat_cpu:
        unique_rates.append(float(torch.unique(sample_set, dim=0).shape[0]) / float(sample_set.shape[0]))
    unique_rate = allocations.new_tensor(unique_rates, dtype=torch.float32)

    num_samples = alloc.shape[1]
    if num_samples < 2:
        pairwise_l1 = allocations.new_zeros(alloc.shape[0], dtype=torch.float32)
    else:
        flat = alloc.float().reshape(alloc.shape[0], num_samples, -1)
        distances = torch.cdist(flat, flat, p=1)
        tri = torch.triu(torch.ones(num_samples, num_samples, dtype=torch.bool, device=allocations.device), diagonal=1)
        pairwise_l1 = distances[:, tri].mean(dim=1)

    return {
        "allocation_legal_rate": legal.float().mean(dim=1),
        "bucket_count_exact_rate": bucket_exact.float().mean(dim=1),
        "opponent_hand_size_exact_rate": opponent_exact.float().mean(dim=1),
        "tile_count_exact_rate": tile_exact.float().mean(dim=1),
        "unique_sample_rate": unique_rate,
        "pairwise_l1_distance": pairwise_l1,
    }


def _fair_crps_1d(samples: list[int], target: int) -> float:
    if not samples:
        return 0.0
    values = torch.tensor(samples, dtype=torch.float32)
    to_truth = (values - float(target)).abs().mean().item()
    if len(samples) < 2:
        return to_truth
    distances = torch.cdist(values.view(1, -1, 1), values.view(1, -1, 1), p=1)[0]
    tri = torch.triu(torch.ones(len(samples), len(samples), dtype=torch.bool), diagonal=1)
    pair_term = distances[tri].sum().item() / float(len(samples) * (len(samples) - 1))
    return to_truth - pair_term


def _feature_energy_score(sample_features: list[list[float]], target_feature: list[float]) -> float | None:
    if not sample_features:
        return None
    samples = torch.tensor(sample_features, dtype=torch.float32).unsqueeze(0)
    target = torch.tensor(target_feature, dtype=torch.float32).view(1, 1, -1)
    to_truth = (samples - target).abs().sum(dim=2).mean(dim=1)
    if samples.shape[1] < 2:
        return float(to_truth.item())
    distances = torch.cdist(samples, samples, p=1)[0]
    tri = torch.triu(torch.ones(samples.shape[1], samples.shape[1], dtype=torch.bool), diagonal=1)
    pair_term = distances[tri].sum() / float(samples.shape[1] * (samples.shape[1] - 1))
    return float((to_truth[0] - pair_term).item())


def _counts37_to_counts34_list(counts37: list[int]) -> list[int]:
    counts34 = [0] * 34
    for tile37, count in enumerate(counts37):
        counts34[_TILE37_TO_TILE34[tile37]] += int(count)
    return counts34


def _semantic_feature_vector(counts37: list[int], shanten: int, dora_tile34_counts: tuple[int, ...]) -> list[float]:
    counts34 = _counts37_to_counts34_list(counts37)
    dora_count = sum(counts34[tile34] * dora_tile34_counts[tile34] for tile34 in range(34))
    terminal_honor = sum(counts34[tile34] for tile34 in _TERMINAL_HONOR_TILE34)
    honor = sum(counts34[27:34])
    pair_count = sum(1 for count in counts34 if count >= 2)
    triplet_count = sum(1 for count in counts34 if count >= 3)
    return [
        float(shanten),
        1.0 if shanten < 0 else 0.0,
        1.0 if shanten == 0 else 0.0,
        1.0 if shanten <= 1 else 0.0,
        float(dora_count),
        float(sum(counts37[tile37] for tile37 in _RED_TILE37)),
        float(terminal_honor),
        float(honor),
        float(pair_count),
        float(triplet_count),
        float(sum(counts34[0:9])),
        float(sum(counts34[9:18])),
        float(sum(counts34[18:27])),
        float(honor),
    ]


class BeliefAllocationEvaluator:
    """Run fixed-subset hidden-allocation quality evaluation."""

    def __init__(self, cfg: BeliefAllocationEvaluationConfig):
        if cfg.game.n_players != 4:
            raise ValueError("Belief allocation evaluation currently supports 4-player mahjong only")
        if cfg.num_logs <= 0:
            raise ValueError("num_logs must be positive")
        if cfg.max_decisions is not None and cfg.max_decisions <= 0:
            raise ValueError("max_decisions must be positive when set")
        if cfg.decision_stride <= 0:
            raise ValueError("decision_stride must be positive")
        if not 0.0 < cfg.sample_keep_prob <= 1.0:
            raise ValueError("sample_keep_prob must be in (0, 1]")
        if cfg.samples_per_decision <= 0:
            raise ValueError("samples_per_decision must be positive")
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if cfg.warmup_batches < 0:
            raise ValueError("warmup_batches must be non-negative")
        if cfg.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.output_dir = Path(cfg.output_dir)
        self.report_path = Path(cfg.report_path) if cfg.report_path else self.output_dir / "report.md"
        self.summary_path = Path(cfg.summary_path) if cfg.summary_path else self.output_dir / "summary.json"
        self.decisions_csv_path = (
            Path(cfg.decisions_csv_path) if cfg.decisions_csv_path else self.output_dir / "decisions.csv"
        )
        self.opponents_csv_path = (
            Path(cfg.opponents_csv_path) if cfg.opponents_csv_path else self.output_dir / "opponents.csv"
        )
        self.opponent_state_csv_path = (
            Path(cfg.opponent_state_csv_path) if cfg.opponent_state_csv_path else self.output_dir / "opponent_state.csv"
        )
        self.opponent_discard_count_csv_path = (
            Path(cfg.opponent_discard_count_csv_path)
            if cfg.opponent_discard_count_csv_path
            else self.output_dir / "opponent_discard_count.csv"
        )
        self.opponent_state_discard_count_csv_path = (
            Path(cfg.opponent_state_discard_count_csv_path)
            if cfg.opponent_state_discard_count_csv_path
            else self.output_dir / "opponent_state_discard_count.csv"
        )
        self.shanten_distribution_csv_path = (
            Path(cfg.shanten_distribution_csv_path)
            if cfg.shanten_distribution_csv_path
            else self.output_dir / "shanten_distributions.csv"
        )
        self.encoder = self._build_encoder()
        self.model = self._build_model()
        self._rng = random.Random(cfg.seed)
        self._shanten_cache: dict[tuple[int, ...], int] = {}

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
        files = [Path(path) for path in glob.glob(self.cfg.input_glob, recursive=True)]
        files = sorted(files, key=lambda path: (path.name, str(path)))
        return files[: self.cfg.num_logs]

    def _prepare_output(self) -> None:
        output_paths = [
            self.report_path,
            self.summary_path,
            self.decisions_csv_path,
            self.opponents_csv_path,
            self.opponent_state_csv_path,
            self.opponent_discard_count_csv_path,
            self.opponent_state_discard_count_csv_path,
            self.shanten_distribution_csv_path,
        ]
        if not self.cfg.overwrite:
            existing = [path for path in output_paths if path.exists()]
            if existing:
                raise FileExistsError(f"Evaluation output already exists: {existing[0]}. Set overwrite=true.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _keep_decision(self, input_path: Path, decision_index_in_log: int) -> bool:
        if decision_index_in_log % self.cfg.decision_stride != 0:
            return False
        if self.cfg.seed is None:
            return self._rng.random() < self.cfg.sample_keep_prob
        return _stable_keep_decision(int(self.cfg.seed), input_path, decision_index_in_log, self.cfg.sample_keep_prob)

    def _make_record(
        self,
        *,
        input_path: Path,
        decision_index_in_log: int,
        observer: int,
        obs,
        hidden_hands: list[list[int]],
        event_index: int | None,
        response_source_index: int | None,
        response_decision: bool,
    ) -> _EvaluationDecisionRecord | None:
        if not self._keep_decision(input_path, decision_index_in_log):
            return None
        feature = self.encoder.encode(obs)
        target = make_hidden_allocation_target(obs, hidden_hands, feature)
        return _EvaluationDecisionRecord(
            feature=feature,
            target=target,
            observer=int(observer),
            event_index=event_index,
            response_source_index=response_source_index,
            response_decision=response_decision,
            decision_index_in_log=decision_index_in_log,
            opponent_meta=_opponent_metadata(obs, hidden_hands),
            dora_tile34_counts=_dora_tile34_counts([int(tile) for tile in obs.dora_indicators]),
        )

    def _collect_decisions(
        self,
        input_path: Path,
        *,
        max_selected: int | None,
    ) -> tuple[list[_EvaluationDecisionRecord], dict[str, Any]]:
        try:
            events = _read_jsonl(input_path)
            replay = MjaiReplay.from_jsonl(str(input_path), rule=self.cfg.game.replay_rule)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Skipping unparseable replay: {}: {}", input_path, e)
            return [], {
                "input_path": str(input_path),
                "num_events": 0,
                "num_decisions": 0,
                "selected_decisions": 0,
                "skipped_unmatched": 0,
                "skipped_response_without_source": 0,
                "skipped_replay_error": 1,
            }

        records: list[_EvaluationDecisionRecord] = []
        event_pos = 0
        last_response_source_idx: int | None = None
        skipped_unmatched = 0
        skipped_response_without_source = 0
        valid_decisions = 0

        try:
            for kyoku in replay.take_kyokus():
                for observer, obs, action, hidden_hands in kyoku.steps(
                    seat=None,
                    skip_single_action=self.cfg.skip_single_action,
                    include_hidden=True,
                ):
                    action_mjai = _action_to_mjai(action, observer)
                    if action_mjai.get("type") == _PASS_TYPE:
                        if last_response_source_idx is None:
                            skipped_response_without_source += 1
                            logger.warning(
                                "Skipping response evaluation sample without source event: {} observer={}",
                                input_path,
                                observer,
                            )
                            continue
                        record = self._make_record(
                            input_path=input_path,
                            decision_index_in_log=valid_decisions,
                            observer=int(observer),
                            obs=obs,
                            hidden_hands=hidden_hands,
                            event_index=None,
                            response_source_index=last_response_source_idx,
                            response_decision=True,
                        )
                        valid_decisions += 1
                        if record is not None:
                            records.append(record)
                            if max_selected is not None and len(records) >= max_selected:
                                break
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
                    record = self._make_record(
                        input_path=input_path,
                        decision_index_in_log=valid_decisions,
                        observer=int(observer),
                        obs=obs,
                        hidden_hands=hidden_hands,
                        event_index=event_idx,
                        response_source_index=None,
                        response_decision=False,
                    )
                    valid_decisions += 1
                    if record is not None:
                        records.append(record)
                        if max_selected is not None and len(records) >= max_selected:
                            break
                if max_selected is not None and len(records) >= max_selected:
                    break
        except (RuntimeError, ValueError) as e:
            logger.warning("Skipping remaining replay due to error: {}: {}", input_path, e)

        return records, {
            "input_path": str(input_path),
            "num_events": len(events),
            "num_decisions": valid_decisions,
            "selected_decisions": len(records),
            "skipped_unmatched": skipped_unmatched,
            "skipped_response_without_source": skipped_response_without_source,
            "skipped_replay_error": 0,
        }

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _counts37_shanten(self, counts: torch.Tensor | list[int]) -> int:
        if isinstance(counts, torch.Tensor):
            key = tuple(int(value) for value in counts.detach().cpu().long().tolist())
        else:
            key = tuple(int(value) for value in counts)
        cached = self._shanten_cache.get(key)
        if cached is not None:
            return cached
        tiles = []
        for tile37, count in enumerate(key):
            tile34 = _TILE37_TO_TILE34[tile37]
            tiles.extend([tile34 * 4] * int(count))
        shanten = int(calculate_shanten(tiles))
        if len(self._shanten_cache) < _SHANTEN_CACHE_MAX_SIZE:
            self._shanten_cache[key] = shanten
        return shanten

    def _evaluate_opponent_semantics(
        self,
        sample_counts: torch.Tensor,
        target_counts: torch.Tensor,
        dora_tile34_counts: tuple[int, ...],
    ) -> dict[str, Any]:
        target_list = [int(value) for value in target_counts.detach().cpu().long().tolist()]
        sample_lists = [
            [int(value) for value in row.detach().cpu().long().tolist()]
            for row in sample_counts.detach().cpu().long()
        ]
        true_shanten = self._counts37_shanten(target_list)
        sample_shanten = [self._counts37_shanten(sample) for sample in sample_lists]
        sample_shanten_counts = Counter(str(value) for value in sample_shanten)

        semantic_es = None
        if self.cfg.semantic_features:
            target_feature = _semantic_feature_vector(target_list, true_shanten, dora_tile34_counts)
            sample_features = [
                _semantic_feature_vector(sample, shanten, dora_tile34_counts)
                for sample, shanten in zip(sample_lists, sample_shanten, strict=True)
            ]
            semantic_es = _feature_energy_score(sample_features, target_feature)

        return {
            "true_shanten": true_shanten,
            "sample_shanten_counts": dict(sorted(sample_shanten_counts.items(), key=lambda item: int(item[0]))),
            "sample_shanten_mean": sum(sample_shanten) / len(sample_shanten) if sample_shanten else None,
            "shanten_crps": _fair_crps_1d(sample_shanten, true_shanten),
            "semantic_es": semantic_es,
        }

    def _tile37_weights(self, records: list[_EvaluationDecisionRecord]) -> torch.Tensor:
        weights = []
        for record in records:
            row = []
            for tile37, tile34 in enumerate(_TILE37_TO_TILE34):
                weight = 1.0
                if tile37 in _RED_TILE37:
                    weight += float(self.cfg.red_weight)
                weight += float(self.cfg.dora_weight) * float(record.dora_tile34_counts[tile34])
                row.append(weight)
            weights.append(row)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def _warmup(self, records: list[_EvaluationDecisionRecord]) -> dict[str, Any]:
        if self.cfg.warmup_batches == 0:
            return {"warmup_batches": 0, "warmup_samples": 0, "warmup_elapsed_seconds": 0.0}
        batch_records = records[: self.cfg.batch_size]
        if not batch_records:
            return {"warmup_batches": 0, "warmup_samples": 0, "warmup_elapsed_seconds": 0.0}
        features = collate_belief_features([record.feature for record in batch_records])
        features = _to_device(features, self.device)
        self._sync()
        started_at = time.perf_counter()
        for _ in range(self.cfg.warmup_batches):
            samples = self.model.sample_allocations(
                features,
                num_samples=self.cfg.samples_per_decision,
                temperature=self.cfg.temperature,
                decode_steps=self.cfg.decode_steps,
            )
            self._sync()
            del samples
        elapsed = time.perf_counter() - started_at
        return {
            "warmup_batches": self.cfg.warmup_batches,
            "warmup_samples": self.cfg.warmup_batches * len(batch_records) * self.cfg.samples_per_decision,
            "warmup_elapsed_seconds": elapsed,
        }

    @torch.inference_mode()
    def _evaluate_batch(
        self,
        batch_records: list[_EvaluationDecisionRecord],
        *,
        input_paths: list[Path],
        log_indices: list[int],
        batch_index: int,
        global_start_index: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        features = collate_belief_features([record.feature for record in batch_records])
        targets = torch.stack([record.target for record in batch_records])
        features = _to_device(features, self.device)
        targets = targets.to(self.device, non_blocking=True)

        self._sync()
        started_at = time.perf_counter()
        samples = self.model.sample_allocations(
            features,
            num_samples=self.cfg.samples_per_decision,
            temperature=self.cfg.temperature,
            decode_steps=self.cfg.decode_steps,
        )
        self._sync()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        batch_size = len(batch_records)
        total_samples = batch_size * self.cfg.samples_per_decision
        samples_per_second = total_samples / (elapsed_ms / 1000.0) if elapsed_ms > 0.0 else 0.0

        diagnostics = _allocation_diagnostics_per_decision(features, samples)
        bucket_weights_opponent = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0]] * batch_size,
            dtype=torch.float32,
            device=self.device,
        )
        tile_weights37 = torch.ones(batch_size, TILE37_COUNT, dtype=torch.float32, device=self.device)
        tile_weights34 = torch.ones(batch_size, 34, dtype=torch.float32, device=self.device)
        dora_red_weights37 = self._tile37_weights(batch_records)

        raw_tile37_opponent_es = _energy_score_weighted_l1(
            samples,
            targets,
            bucket_weights=bucket_weights_opponent,
            tile_weights=tile_weights37,
        )
        samples34 = _collapse_tile34(samples)
        targets34 = _collapse_tile34(targets)
        raw_tile34_opponent_es = _energy_score_weighted_l1(
            samples34,
            targets34,
            bucket_weights=bucket_weights_opponent,
            tile_weights=tile_weights34,
        )
        raw_tile37_dora_red_opponent_es = _energy_score_weighted_l1(
            samples,
            targets,
            bucket_weights=bucket_weights_opponent,
            tile_weights=dora_red_weights37,
        )

        per_opponent_tile37_es = []
        per_opponent_tile34_es = []
        per_opponent_dora_red_es = []
        for opponent_idx in range(_OPPONENT_COUNT):
            weights = torch.zeros(batch_size, BUCKET_COUNT, dtype=torch.float32, device=self.device)
            weights[:, opponent_idx] = 1.0
            per_opponent_tile37_es.append(
                _energy_score_weighted_l1(samples, targets, bucket_weights=weights, tile_weights=tile_weights37)
            )
            per_opponent_tile34_es.append(
                _energy_score_weighted_l1(samples34, targets34, bucket_weights=weights, tile_weights=tile_weights34)
            )
            per_opponent_dora_red_es.append(
                _energy_score_weighted_l1(samples, targets, bucket_weights=weights, tile_weights=dora_red_weights37)
            )
        per_opponent_tile37_es_t = torch.stack(per_opponent_tile37_es, dim=1)
        per_opponent_tile34_es_t = torch.stack(per_opponent_tile34_es, dim=1)
        per_opponent_dora_red_es_t = torch.stack(per_opponent_dora_red_es, dim=1)

        samples_cpu = samples.detach().cpu().long()
        targets_cpu = targets.detach().cpu().long()
        decision_rows: list[dict[str, Any]] = []
        opponent_rows: list[dict[str, Any]] = []

        for row_idx, record in enumerate(batch_records):
            global_decision_index = global_start_index + row_idx
            opponent_semantic_rows = []
            for opponent_idx, meta in enumerate(record.opponent_meta):
                semantic = self._evaluate_opponent_semantics(
                    samples_cpu[row_idx, :, opponent_idx],
                    targets_cpu[row_idx, opponent_idx],
                    record.dora_tile34_counts,
                )
                opponent_row = {
                    "global_decision_index": global_decision_index,
                    "log_index": log_indices[row_idx],
                    "decision_index_in_log": record.decision_index_in_log,
                    "input_path": str(input_paths[row_idx]),
                    "observer": record.observer,
                    "event_index": record.event_index,
                    "response_source_index": record.response_source_index,
                    "response_decision": record.response_decision,
                    "opponent_bucket": opponent_idx,
                    "opponent_relative_seat": meta.relative_seat,
                    "opponent_absolute_seat": meta.absolute_seat,
                    "opponent_discard_count": meta.discard_count,
                    "opponent_meld_count": meta.meld_count,
                    "opponent_concealed_size": meta.concealed_size,
                    "opponent_riichi_active": meta.riichi_active,
                    "opponent_state": meta.state,
                    "num_samples": self.cfg.samples_per_decision,
                    "raw_tile37_es": float(per_opponent_tile37_es_t[row_idx, opponent_idx].item()),
                    "raw_tile34_es": float(per_opponent_tile34_es_t[row_idx, opponent_idx].item()),
                    "raw_tile37_dora_red_es": float(per_opponent_dora_red_es_t[row_idx, opponent_idx].item()),
                    **semantic,
                }
                opponent_rows.append(opponent_row)
                opponent_semantic_rows.append(opponent_row)

            feature_meta = _feature_metadata(record.feature)
            decision_row = {
                "global_decision_index": global_decision_index,
                "log_index": log_indices[row_idx],
                "decision_index_in_log": record.decision_index_in_log,
                "input_path": str(input_paths[row_idx]),
                "observer": record.observer,
                "event_index": record.event_index,
                "response_source_index": record.response_source_index,
                "response_decision": record.response_decision,
                **feature_meta,
                "num_samples": self.cfg.samples_per_decision,
                "batch_index": batch_index,
                "batch_size": batch_size,
                "sampling_elapsed_ms": elapsed_ms,
                "samples_per_second": samples_per_second,
                "allocation_legal_rate": float(diagnostics["allocation_legal_rate"][row_idx].item()),
                "bucket_count_exact_rate": float(diagnostics["bucket_count_exact_rate"][row_idx].item()),
                "opponent_hand_size_exact_rate": float(
                    diagnostics["opponent_hand_size_exact_rate"][row_idx].item()
                ),
                "tile_count_exact_rate": float(diagnostics["tile_count_exact_rate"][row_idx].item()),
                "unique_sample_rate": float(diagnostics["unique_sample_rate"][row_idx].item()),
                "pairwise_l1_distance": float(diagnostics["pairwise_l1_distance"][row_idx].item()),
                "raw_tile37_opponent_es": float(raw_tile37_opponent_es[row_idx].item()),
                "raw_tile34_opponent_es": float(raw_tile34_opponent_es[row_idx].item()),
                "raw_tile37_dora_red_opponent_es": float(raw_tile37_dora_red_opponent_es[row_idx].item()),
                "mean_shanten_crps": _mean_or_none(opponent_semantic_rows, "shanten_crps"),
                "mean_semantic_es": _mean_or_none(opponent_semantic_rows, "semantic_es"),
            }
            decision_rows.append(decision_row)

        batch_row = {
            "batch_index": batch_index,
            "num_decisions": batch_size,
            "samples_per_decision": self.cfg.samples_per_decision,
            "total_samples": total_samples,
            "sampling_elapsed_ms": elapsed_ms,
            "samples_per_second": samples_per_second,
        }
        return decision_rows, opponent_rows, batch_row

    def _build_summary(
        self,
        *,
        input_files: list[Path],
        log_summaries: list[dict[str, Any]],
        decision_rows: list[dict[str, Any]],
        opponent_rows: list[dict[str, Any]],
        batch_rows: list[dict[str, Any]],
        warmup: dict[str, Any],
        wall_elapsed_seconds: float,
    ) -> dict[str, Any]:
        by_state = _group_opponent_aggregates(opponent_rows, ("opponent_state",))
        by_discard_count = _group_opponent_aggregates(opponent_rows, ("opponent_discard_count",))
        by_state_discard_count = _group_opponent_aggregates(
            opponent_rows,
            ("opponent_state", "opponent_discard_count"),
        )
        shanten_distribution_rows = []
        shanten_distribution_rows.extend(
            _shanten_distribution_rows(opponent_rows, stratum_type="overall", stratum_keys=())
        )
        shanten_distribution_rows.extend(
            _shanten_distribution_rows(opponent_rows, stratum_type="opponent_state", stratum_keys=("opponent_state",))
        )
        shanten_distribution_rows.extend(
            _shanten_distribution_rows(
                opponent_rows,
                stratum_type="opponent_discard_count",
                stratum_keys=("opponent_discard_count",),
            )
        )
        shanten_distribution_rows.extend(
            _shanten_distribution_rows(
                opponent_rows,
                stratum_type="opponent_state_discard_count",
                stratum_keys=("opponent_state", "opponent_discard_count"),
            )
        )

        summary = {
            "input_glob": self.cfg.input_glob,
            "input_files": [str(path) for path in input_files],
            "output_dir": str(self.output_dir),
            "report_path": str(self.report_path),
            "summary_path": str(self.summary_path),
            "decisions_csv_path": str(self.decisions_csv_path),
            "opponents_csv_path": str(self.opponents_csv_path),
            "opponent_state_csv_path": str(self.opponent_state_csv_path),
            "opponent_discard_count_csv_path": str(self.opponent_discard_count_csv_path),
            "opponent_state_discard_count_csv_path": str(self.opponent_state_discard_count_csv_path),
            "shanten_distribution_csv_path": str(self.shanten_distribution_csv_path),
            "model_path": self.cfg.model_path,
            "replay_rule": self.cfg.game.replay_rule,
            "num_logs": len(input_files),
            "num_decisions": len(decision_rows),
            "num_opponent_states": len(opponent_rows),
            "sample_keep_prob": self.cfg.sample_keep_prob,
            "decision_stride": self.cfg.decision_stride,
            "samples_per_decision": self.cfg.samples_per_decision,
            "batch_size": self.cfg.batch_size,
            "temperature": self.cfg.temperature,
            "decode_steps": self.cfg.decode_steps,
            "effective_decode_steps": int(self.cfg.decode_steps or getattr(self.model, "decode_steps", 0)),
            "skip_single_action": self.cfg.skip_single_action,
            "seed": self.cfg.seed,
            "semantic_features": self.cfg.semantic_features,
            "shanten_cache_max_size": _SHANTEN_CACHE_MAX_SIZE,
            "wall_elapsed_seconds": wall_elapsed_seconds,
            "warmup": warmup,
            "sampling_speed": _aggregate_batch_rows(batch_rows),
            "overall_decisions": _aggregate_decision_rows(decision_rows),
            "overall_opponents": _aggregate_opponent_rows(opponent_rows),
            "metrics_by_opponent_state": by_state,
            "metrics_by_opponent_discard_count": by_discard_count,
            "metrics_by_opponent_state_discard_count": by_state_discard_count,
            "shanten_distributions": shanten_distribution_rows,
            "log_summaries": log_summaries,
            "config": self.cfg.model_dump(),
        }
        if self.cfg.include_decisions_in_summary:
            summary["decisions"] = decision_rows
        if self.cfg.include_opponents_in_summary:
            summary["opponents"] = opponent_rows
        return summary

    def _write_outputs(
        self,
        summary: dict[str, Any],
        decision_rows: list[dict[str, Any]],
        opponent_rows: list[dict[str, Any]],
    ) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.report_path.write_text(_render_markdown_report(summary), encoding="utf-8")
        _write_csv(self.decisions_csv_path, decision_rows)
        _write_csv(self.opponents_csv_path, opponent_rows)
        _write_csv(self.opponent_state_csv_path, summary["metrics_by_opponent_state"])
        _write_csv(self.opponent_discard_count_csv_path, summary["metrics_by_opponent_discard_count"])
        _write_csv(self.opponent_state_discard_count_csv_path, summary["metrics_by_opponent_state_discard_count"])
        _write_csv(self.shanten_distribution_csv_path, summary["shanten_distributions"])

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
        all_records: list[tuple[Path, int, _EvaluationDecisionRecord]] = []
        log_summaries = []
        processed_input_files = []
        for log_index, input_path in enumerate(input_files):
            remaining = None
            if self.cfg.max_decisions is not None:
                remaining = max(0, self.cfg.max_decisions - len(all_records))
                if remaining == 0:
                    break
            processed_input_files.append(input_path)
            records, log_summary = self._collect_decisions(input_path, max_selected=remaining)
            log_summaries.append(log_summary)
            for record in records:
                all_records.append((input_path, log_index, record))
            logger.info(
                "Collected evaluation decisions {}/{}: {} selected from {} valid decisions",
                log_index + 1,
                len(input_files),
                len(records),
                log_summary["num_decisions"],
            )

        if not all_records:
            raise ValueError("No evaluation decision observations were collected")

        warmup = self._warmup([record for _path, _log_index, record in all_records[: self.cfg.batch_size]])
        decision_rows: list[dict[str, Any]] = []
        opponent_rows: list[dict[str, Any]] = []
        batch_rows: list[dict[str, Any]] = []

        for batch_index, start in enumerate(range(0, len(all_records), self.cfg.batch_size)):
            batch_items = all_records[start : start + self.cfg.batch_size]
            input_paths = [item[0] for item in batch_items]
            log_indices = [item[1] for item in batch_items]
            batch_records = [item[2] for item in batch_items]
            batch_decision_rows, batch_opponent_rows, batch_row = self._evaluate_batch(
                batch_records,
                input_paths=input_paths,
                log_indices=log_indices,
                batch_index=batch_index,
                global_start_index=start,
            )
            decision_rows.extend(batch_decision_rows)
            opponent_rows.extend(batch_opponent_rows)
            batch_rows.append(batch_row)
            if len(decision_rows) % self.cfg.progress_interval == 0 or len(decision_rows) == len(all_records):
                logger.info(
                    "Evaluation progress: {}/{} decisions, latest {:.1f} samples/s",
                    len(decision_rows),
                    len(all_records),
                    batch_row["samples_per_second"],
                )

        wall_elapsed_seconds = time.perf_counter() - started_at
        summary = self._build_summary(
            input_files=processed_input_files,
            log_summaries=log_summaries,
            decision_rows=decision_rows,
            opponent_rows=opponent_rows,
            batch_rows=batch_rows,
            warmup=warmup,
            wall_elapsed_seconds=wall_elapsed_seconds,
        )
        self._write_outputs(summary, decision_rows, opponent_rows)
        logger.info(
            "Saved belief allocation evaluation summary to {} ({:.1f} samples/s)",
            self.summary_path,
            summary["sampling_speed"]["samples_per_second"] or 0.0,
        )
        return summary
