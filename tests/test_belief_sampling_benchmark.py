import csv
from pathlib import Path

import torch
from riichienv_ml.belief_sampling_benchmark import BeliefSamplingBenchmark
from riichienv_ml.config import BeliefSamplingBenchmarkConfig, GameConfig, ModelConfig
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
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "decisions.csv").exists()
    assert (tmp_path / "prog_len.csv").exists()
    assert (tmp_path / "prog_len_buckets.csv").exists()

    with (tmp_path / "decisions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["total_samples"] == "2"
