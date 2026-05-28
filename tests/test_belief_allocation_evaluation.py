import csv
from pathlib import Path

import torch
from riichienv_ml.belief_allocation_evaluation import (
    BeliefAllocationEvaluator,
    _collapse_tile34,
    _energy_score_weighted_l1,
)
from riichienv_ml.config import BeliefAllocationEvaluationConfig, GameConfig, ModelConfig
from riichienv_ml.models.belief_allocation import JointHiddenAllocationSampler

DATA_PATH = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"


def test_energy_score_weighted_l1_uses_fair_pair_term():
    samples = torch.tensor([[[[1, 0]], [[0, 1]]]], dtype=torch.long)
    target = torch.tensor([[[1, 0]]], dtype=torch.long)
    bucket_weights = torch.ones(1, 1)
    tile_weights = torch.ones(1, 2)

    score = _energy_score_weighted_l1(
        samples,
        target,
        bucket_weights=bucket_weights,
        tile_weights=tile_weights,
    )

    assert score.shape == (1,)
    assert score.item() == 0.0


def test_collapse_tile34_merges_red_and_non_red_fives():
    counts = torch.zeros(1, 1, 1, 37, dtype=torch.long)
    counts[0, 0, 0, 0] = 1
    counts[0, 0, 0, 5] = 2

    collapsed = _collapse_tile34(counts)

    assert collapsed.shape == (1, 1, 1, 34)
    assert collapsed[0, 0, 0, 4].item() == 3
    assert collapsed.sum().item() == 3


def test_belief_allocation_evaluator_writes_summary_and_csv(tmp_path):
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

    cfg = BeliefAllocationEvaluationConfig(
        game=GameConfig(n_players=4, replay_rule="tenhou"),
        input_glob=str(DATA_PATH),
        output_dir=str(tmp_path),
        summary_path=str(tmp_path / "summary.json"),
        model_path=str(model_path),
        device="cpu",
        num_logs=1,
        max_decisions=1,
        sample_keep_prob=1.0,
        samples_per_decision=2,
        batch_size=1,
        warmup_batches=0,
        overwrite=False,
        seed=7,
        model=ModelConfig(**model_config),
    )

    summary = BeliefAllocationEvaluator(cfg).run()

    assert summary["num_logs"] == 1
    assert summary["num_decisions"] == 1
    assert summary["num_opponent_states"] == 3
    assert summary["samples_per_decision"] == 2
    assert summary["report_path"] == str(tmp_path / "report.md")
    assert summary["overall_decisions"]["allocation_legal_rate"] == 1.0
    assert "raw_tile37_opponent_es" in summary["overall_decisions"]
    assert "shanten_crps" in summary["overall_opponents"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "decisions.csv").exists()
    assert (tmp_path / "opponents.csv").exists()
    assert (tmp_path / "opponent_state.csv").exists()
    assert (tmp_path / "opponent_discard_count.csv").exists()
    assert (tmp_path / "opponent_state_discard_count.csv").exists()
    assert (tmp_path / "shanten_distributions.csv").exists()

    with (tmp_path / "opponents.csv").open(encoding="utf-8", newline="") as f:
        opponent_rows = list(csv.DictReader(f))
    assert len(opponent_rows) == 3
    assert all(row["num_samples"] == "2" for row in opponent_rows)
    assert all(row["opponent_state"] in {"closed", "meld1", "meld2", "meld3plus", "riichi"} for row in opponent_rows)

    with (tmp_path / "shanten_distributions.csv").open(encoding="utf-8", newline="") as f:
        shanten_rows = list(csv.DictReader(f))
    assert shanten_rows
    assert "overall" in {row["stratum_type"] for row in shanten_rows}

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# Belief Allocation Evaluation Report" in report
    assert "## Quick Read" in report
    assert "## By Opponent State" in report
    assert "True Tenpai" not in report
    assert "True 0" in report
    assert "Sample 4+" in report
