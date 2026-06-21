"""Annotate MJAI logs with hidden-hand samples from the belief sampler."""

from __future__ import annotations

import glob
import gzip
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from riichienv import MjaiReplay
from riichienv_ml.config import BeliefLogSamplingConfig, import_class
from riichienv_ml.features.belief_features import (
    BUCKET_COUNT,
    TILE37_COUNT,
    BeliefFeatureEncoder,
    collate_belief_features,
)
from riichienv_ml.utils import (
    build_encoder,
    configure_inference_dtype,
    configure_matmul_precision,
    inference_autocast,
    load_model_weights,
)

_RESPONSE_SOURCE_TYPES = {"dahai", "kakan", "ankan"}
_PASS_TYPE = "none"
_BUCKET_ORDER = ["shimocha", "toimen", "kamicha"]
_TILE37_MJAI = (
    "5mr",
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "5pr",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "5sr",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
)


@dataclass
class _DecisionRecord:
    feature: dict[str, torch.Tensor]
    observer: int
    event_index: int | None
    response_source_index: int | None
    response_decision: bool


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, events: list[dict[str, Any]], *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    with opener(path, "wt", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _output_path(input_path: Path, output_dir: Path, *, compress: bool) -> Path:
    name = input_path.name
    if compress and not name.endswith(".gz"):
        name = f"{name}.gz"
    if not compress and name.endswith(".gz"):
        name = name[:-3]
    return output_dir / name


def _ensure_meta(event: dict[str, Any]) -> dict[str, Any]:
    meta = event.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        event["meta"] = meta
    return meta


def _action_to_mjai(action, observer: int) -> dict[str, Any]:
    try:
        payload = json.loads(action.to_mjai())
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    payload.setdefault("actor", int(observer))
    return payload


def _normalize_mjai_tile(tile: Any) -> Any:
    if not isinstance(tile, str):
        return tile
    if tile in {"5mr", "5pr", "5sr"}:
        return tile[:2]
    return tile


def _same_mjai_tile(a: Any, b: Any) -> bool:
    return _normalize_mjai_tile(a) == _normalize_mjai_tile(b)


def _same_consumed_tiles(event_consumed: Any, action_consumed: Any) -> bool:
    if not isinstance(event_consumed, list) or not isinstance(action_consumed, list):
        return event_consumed == action_consumed
    event_counts = Counter(_normalize_mjai_tile(tile) for tile in event_consumed)
    action_counts = Counter(_normalize_mjai_tile(tile) for tile in action_consumed)
    return event_counts == action_counts


def _event_matches_action(event: dict[str, Any], action_mjai: dict[str, Any]) -> bool:
    if event.get("type") != action_mjai.get("type"):
        return False
    if "actor" in action_mjai and event.get("actor") != action_mjai["actor"]:
        return False
    if "pai" in action_mjai and "pai" in event and not _same_mjai_tile(event.get("pai"), action_mjai["pai"]):
        return False
    if "consumed" in action_mjai and event.get("consumed") != action_mjai["consumed"]:
        return _same_consumed_tiles(event.get("consumed"), action_mjai["consumed"])
    return True


def _find_next_event(events: list[dict[str, Any]], action_mjai: dict[str, Any], start: int) -> int | None:
    for idx in range(max(0, start), len(events)):
        if _event_matches_action(events[idx], action_mjai):
            return idx
    return None


def _tile37_to_mjai(tile37: int) -> str:
    if tile37 < 0:
        raise ValueError(f"tile37 out of range: {tile37}")
    try:
        return _TILE37_MJAI[tile37]
    except IndexError as exc:
        raise ValueError(f"tile37 out of range: {tile37}") from exc


def _counts37_to_mjai_hand(counts: torch.Tensor) -> list[str]:
    if tuple(counts.shape) != (TILE37_COUNT,):
        raise ValueError(f"counts must have shape {(TILE37_COUNT,)}, got {tuple(counts.shape)}")
    hand: list[str] = []
    for tile37, count in enumerate(counts.detach().cpu().long().tolist()):
        hand.extend([_tile37_to_mjai(tile37)] * int(count))
    return hand


def _allocation_to_meta(
    allocation: torch.Tensor,
    *,
    observer: int,
    num_samples: int,
    temperature: float,
) -> dict[str, Any]:
    if tuple(allocation.shape) != (num_samples, BUCKET_COUNT, TILE37_COUNT):
        raise ValueError(
            "allocation must have shape "
            f"{(num_samples, BUCKET_COUNT, TILE37_COUNT)}, got {tuple(allocation.shape)}"
        )
    opponent_seats = [(int(observer) + rel) % 4 for rel in (1, 2, 3)]
    samples = []
    for sample_idx in range(num_samples):
        samples.append(
            [
                _counts37_to_mjai_hand(allocation[sample_idx, bucket_idx])
                for bucket_idx in range(3)
            ]
        )
    return {
        "model": "belief_allocation",
        "observer": int(observer),
        "opponent_seats": opponent_seats,
        "bucket_order": _BUCKET_ORDER,
        "tile_format": "mjai",
        "tile_space": "tile37",
        "sample_count": int(num_samples),
        "temperature": float(temperature),
        "samples": samples,
    }


def _to_device(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in features.items()}


class BeliefLogSampler:
    """Run the belief sampler on MJAI logs and write annotated copies."""

    def __init__(self, cfg: BeliefLogSamplingConfig):
        if cfg.game.n_players != 4:
            raise ValueError("Belief log sampling currently supports 4-player mahjong only")
        if cfg.num_logs <= 0:
            raise ValueError("num_logs must be positive")
        if cfg.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.inference_dtype = configure_inference_dtype(cfg.inference_dtype, self.device)
        self.output_dir = Path(cfg.output_dir)
        self.summary_path = Path(cfg.summary_path) if cfg.summary_path else self.output_dir / "summary.json"
        self.encoder = self._build_encoder()
        self.model = self._build_model()
        self._diagnostic_sums: dict[str, float] = {}
        self._diagnostic_count = 0

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

    def _prepare_output(self, input_files: list[Path]) -> list[Path]:
        if not input_files:
            raise ValueError(f"No input logs found at {self.cfg.input_glob}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [
            _output_path(path, self.output_dir, compress=self.cfg.compress_output)
            for path in input_files
        ]
        for input_path, output_path in zip(input_files, output_paths, strict=True):
            if input_path.resolve() == output_path.resolve():
                raise ValueError(
                    f"Output path would overwrite the input log: {input_path}. Choose a different output_dir."
                )
        if not self.cfg.overwrite:
            existing = [path for path in output_paths if path.exists()]
            if existing:
                raise FileExistsError(
                    f"Output log already exists: {existing[0]}. Set overwrite=true or choose another output_dir."
                )
            if self.summary_path.exists():
                raise FileExistsError(
                    f"Summary already exists: {self.summary_path}. Set overwrite=true or choose another output_dir."
                )
        return output_paths

    def _collect_decisions(self, input_path: Path, events: list[dict[str, Any]]) -> list[_DecisionRecord]:
        replay = MjaiReplay.from_jsonl(str(input_path), rule=self.cfg.game.replay_rule)
        records: list[_DecisionRecord] = []
        event_pos = 0
        last_response_source_idx: int | None = None

        for kyoku in replay.take_kyokus():
            for observer, obs, action in kyoku.steps(
                seat=None,
                skip_single_action=self.cfg.skip_single_action,
            ):
                action_mjai = _action_to_mjai(action, observer)
                if action_mjai.get("type") == _PASS_TYPE:
                    if last_response_source_idx is None:
                        logger.warning(
                            "Skipping response belief sample without source event: {} observer={}",
                            input_path,
                            observer,
                        )
                        continue
                    records.append(
                        _DecisionRecord(
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
                    _DecisionRecord(
                        feature=self.encoder.encode(obs),
                        observer=int(observer),
                        event_index=event_idx,
                        response_source_index=None,
                        response_decision=False,
                    )
                )

        return records

    @torch.inference_mode()
    def _annotate_records(self, events: list[dict[str, Any]], records: list[_DecisionRecord]) -> None:
        for start in range(0, len(records), self.cfg.batch_size):
            batch_records = records[start : start + self.cfg.batch_size]
            features = collate_belief_features([record.feature for record in batch_records])
            features = _to_device(features, self.device)
            with inference_autocast(self.device, self.inference_dtype):
                allocations_device = self.model.sample_allocations(
                    features,
                    num_samples=self.cfg.num_samples,
                    temperature=self.cfg.temperature,
                )
            if hasattr(self.model, "allocation_diagnostics"):
                diagnostics = self.model.allocation_diagnostics(features, allocations_device)
                batch_size = len(batch_records)
                for key, value in diagnostics.items():
                    self._diagnostic_sums[key] = self._diagnostic_sums.get(key, 0.0) + float(value.item()) * batch_size
                self._diagnostic_count += batch_size
            allocations = allocations_device.cpu()
            for record, allocation in zip(batch_records, allocations, strict=True):
                payload = _allocation_to_meta(
                    allocation,
                    observer=record.observer,
                    num_samples=self.cfg.num_samples,
                    temperature=self.cfg.temperature,
                )
                payload["response_decision"] = record.response_decision
                if record.event_index is not None:
                    _ensure_meta(events[record.event_index])[self.cfg.metadata_key] = payload
                elif record.response_source_index is not None:
                    meta = _ensure_meta(events[record.response_source_index])
                    meta.setdefault(self.cfg.response_metadata_key, []).append(payload)

    def _annotate_one(self, input_path: Path, output_path: Path) -> dict[str, Any]:
        started_at = time.time()
        events = _read_jsonl(input_path)
        records = self._collect_decisions(input_path, events)
        self._annotate_records(events, records)
        _write_jsonl(output_path, events, compress=self.cfg.compress_output)
        MjaiReplay.from_jsonl(str(output_path), rule=self.cfg.game.replay_rule)
        elapsed = time.time() - started_at
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "num_events": len(events),
            "num_decisions": len(records),
            "elapsed_seconds": elapsed,
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def run(self) -> dict[str, Any]:
        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.cfg.seed))

        input_files = self._input_files()
        output_paths = self._prepare_output(input_files)
        started_at = time.time()
        logs = []
        for idx, (input_path, output_path) in enumerate(zip(input_files, output_paths, strict=True)):
            result = self._annotate_one(input_path, output_path)
            logs.append(result)
            logger.info(
                "Belief sampling {}/{} complete: {} decisions -> {}",
                idx + 1,
                len(input_files),
                result["num_decisions"],
                output_path,
            )

        elapsed = time.time() - started_at
        summary = {
            "num_logs": len(logs),
            "input_glob": self.cfg.input_glob,
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "model_path": self.cfg.model_path,
            "replay_rule": self.cfg.game.replay_rule,
            "num_samples": self.cfg.num_samples,
            "skip_single_action": self.cfg.skip_single_action,
            "compress_output": self.cfg.compress_output,
            "inference_dtype": self.inference_dtype,
            "metadata_key": self.cfg.metadata_key,
            "response_metadata_key": self.cfg.response_metadata_key,
            "elapsed_seconds": elapsed,
            "logs": logs,
        }
        if self._diagnostic_count > 0:
            summary["sample_diagnostics"] = {
                key: value / self._diagnostic_count
                for key, value in sorted(self._diagnostic_sums.items())
            }
        self._write_summary(summary)
        logger.info("Saved belief sampling summary to {}", self.summary_path)
        return summary
