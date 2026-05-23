import csv
from pathlib import Path

import pytest
import torch
from riichienv_ml import belief_sampling_benchmark as benchmark_mod
from riichienv_ml.belief_sampling_benchmark import BeliefSamplingBenchmark, _BenchmarkDecisionRecord
from riichienv_ml.config import BeliefSamplingBenchmarkConfig, GameConfig, ModelConfig
from riichienv_ml.datasets.belief_allocation import BeliefAllocationDataset
from riichienv_ml.features.belief_features import BUCKET_COUNT, TILE37_COUNT, BeliefFeatureEncoder
from riichienv_ml.models.belief_allocation import JointHiddenAllocationSampler

DATA_PATH = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"


def test_belief_sampling_benchmark_writes_summary_and_csv(tmp_path):
    model_path = tmp_path / "belief_model.pth"
    model_config = {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 128,
        "denoise_num_layers": 1,
        "denoise_dim_feedforward": 128,
        "decode_steps": 2,
        "dropout": 0.0,
    }
    model = JointHiddenAllocationSampler(**model_config)
    torch.save(model.state_dict(), model_path)

    cfg = BeliefSamplingBenchmarkConfig(
        game=GameConfig(n_players=4, replay_rule="tenhou"),
        input_glob=str(DATA_PATH),
        output_dir=str(tmp_path),
        summary_path=str(tmp_path / "summary.json"),
        model_path=str(model_path),
        device="cpu",
        num_logs=1,
        max_decisions=1,
        samples_per_call=2,
        target_duration_ms=0.0,
        warmup_calls=0,
        overwrite=False,
        model=ModelConfig(**model_config),
    )

    summary = BeliefSamplingBenchmark(cfg).run()

    assert summary["num_logs"] == 1
    assert summary["num_decisions"] == 1
    assert summary["overall"]["total_samples"] == 2
    assert summary["overall"]["total_calls"] == 1
    assert summary["decisions"][0]["samples_per_call"] == 2
    assert summary["decisions"][0]["num_calls"] == 1
    assert "encoder_cache_ms" in summary["decisions"][0]
    assert "decoder_elapsed_ms" in summary["decisions"][0]
    assert "encoder_cache_ms" in summary["overall"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "decisions.csv").exists()
    assert (tmp_path / "prog_len.csv").exists()
    assert (tmp_path / "prog_len_buckets.csv").exists()

    with (tmp_path / "decisions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["total_samples"] == "2"
    assert "encoder_cache_ms" in rows[0]


def test_belief_sampling_benchmark_reuses_encoder_cache_within_decision(monkeypatch):
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    feature, _target = next(iter(dataset))

    class CountingSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prepare_calls = 0
            self.sample_calls = 0

        def prepare_allocation_sampling_context(self, features):
            self.prepare_calls += 1
            return {"batch_size": features["visible_tile_counts"].shape[0]}

        def sample_allocations_from_context(self, sampling_context, *, num_samples, temperature, decode_steps):
            del temperature, decode_steps
            self.sample_calls += 1
            return torch.zeros(
                sampling_context["batch_size"],
                num_samples,
                BUCKET_COUNT,
                TILE37_COUNT,
                dtype=torch.long,
            )

    current_time = -0.001

    def fake_perf_counter():
        nonlocal current_time
        current_time += 0.001
        return current_time

    monkeypatch.setattr(benchmark_mod.time, "perf_counter", fake_perf_counter)

    benchmark = object.__new__(BeliefSamplingBenchmark)
    benchmark.cfg = BeliefSamplingBenchmarkConfig(
        device="cpu",
        samples_per_call=2,
        target_duration_ms=7.0,
    )
    benchmark.device = torch.device("cpu")
    benchmark.model = CountingSampler()
    record = _BenchmarkDecisionRecord(
        feature=feature,
        observer=0,
        event_index=0,
        response_source_index=None,
        response_decision=False,
    )

    row = benchmark._measure_decision(
        record,
        input_path=DATA_PATH,
        log_index=0,
        decision_index_in_log=0,
        global_decision_index=0,
    )

    assert benchmark.model.prepare_calls == 1
    assert benchmark.model.sample_calls == 2
    assert row["num_calls"] == 2
    assert row["total_samples"] == 4
    assert row["encoder_cache_ms"] == pytest.approx(1.0)
    assert row["decoder_elapsed_ms"] == pytest.approx(2.0)
    assert row["elapsed_ms"] == pytest.approx(9.0)
