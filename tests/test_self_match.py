import json
from pathlib import Path

import torch
import yaml

from riichienv import MjaiReplay
from riichienv_ml.config import load_config
from riichienv_ml.models.transformer import TransformerPolicyNetwork
from riichienv_ml.self_match import SelfMatchRunner


def test_self_match_runner_writes_parseable_mjai_logs(tmp_path):
    train_cfg_path = tmp_path / "bc_config.yml"
    model_path = tmp_path / "model.pth"
    output_dir = tmp_path / "self_match_logs"
    self_match_cfg_path = tmp_path / "self_match.yml"

    train_cfg = {
        "bc": {
            "game": {
                "n_players": 4,
                "replay_rule": "tenhou",
            },
            "model_class": "riichienv_ml.models.transformer.TransformerPolicyNetwork",
            "encoder_class": "riichienv_ml.features.sequence_features.SequenceFeaturePackedEncoder",
            "model": {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "num_actions": 82,
                "max_prog_len": 256,
                "max_cand_len": 32,
            },
        }
    }
    with train_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f)

    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_actions=82,
        max_prog_len=256,
        max_cand_len=32,
    )
    torch.save(model.state_dict(), model_path)

    self_match_cfg = {
        "self_match": {
            "game": {
                "n_players": 4,
                "replay_rule": "tenhou",
            },
            "agents": [
                {
                    "config_path": str(train_cfg_path),
                    "model_path": str(model_path),
                    "device": "cpu",
                    "name": "test-agent",
                }
            ],
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "num_games": 1,
            "base_seed": 7,
            "seed_stride": 1,
            "progress_interval": 1,
            "overwrite": False,
            "compress_logs": False,
            "validate_saved_logs": True,
        }
    }
    with self_match_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(self_match_cfg, f)

    cfg = load_config(str(self_match_cfg_path)).self_match
    summary = SelfMatchRunner(cfg).run()

    log_files = sorted(output_dir.glob("game_*.jsonl"))
    assert len(log_files) == 1

    replay = MjaiReplay.from_jsonl(str(log_files[0]), rule="tenhou")
    assert replay.num_rounds() > 0

    summary_data = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_games"] == 1
    assert summary_data["num_games"] == 1
    assert Path(summary_data["games"][0]["log_path"]) == log_files[0]
