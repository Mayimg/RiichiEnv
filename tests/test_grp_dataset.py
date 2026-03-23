import json

import numpy as np
import torch

from riichienv_ml.datasets.grp_dataset import GrpReplayDataset
from riichienv_ml.models.grp_model import RankPredictor, RewardPredictor


def test_grp_replay_dataset_uses_kyoku_start_state_features(tmp_path):
    data = [
        {"type": "start_game", "names": ["A", "B", "C"], "id": "grp_dataset_3p"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [35000, 35000, 35000],
            "dora_marker": "1p",
            "tehais": [
                ["1p", "1p", "2p", "2p", "2p", "3p", "3p", "3p", "4p", "4p", "4p", "5z", "5z"],
                ["1s", "1s", "1s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "4s", "6z"],
                ["1z", "1z", "2z", "2z", "3z", "3z", "4z", "4z", "5z", "5z", "6z", "6z", "7z"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "1p"},
        {"type": "hora", "actor": 0, "target": 0, "pai": "1p"},
        {"type": "end_kyoku"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 2,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 1,
            "scores": [39000, 33000, 33000],
            "dora_marker": "2p",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5p", "6p", "7p", "1s", "2s", "3s", "E", "S", "W"],
                ["2m", "3m", "4m", "5m", "6p", "7p", "8p", "2s", "3s", "4s", "S", "W", "N"],
                ["3m", "4m", "5m", "6m", "7p", "8p", "9p", "3s", "4s", "5s", "W", "N", "P"],
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "2p"},
        {"type": "dahai", "actor": 1, "pai": "2p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    file_path = tmp_path / "grp_dataset_3p.jsonl"
    with open(file_path, "w") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")

    dataset = GrpReplayDataset(
        data_glob=str(file_path),
        n_players=3,
        replay_rule="mjsoul",
        is_train=False,
    )
    samples = list(dataset)

    assert len(samples) == 6

    first_x, first_y = samples[0]
    np.testing.assert_allclose(
        first_x.numpy(),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(first_y.numpy(), np.array([1.0, 0.0, 0.0], dtype=np.float32))

    fourth_x, fourth_y = samples[3]
    np.testing.assert_allclose(
        fourth_x.numpy(),
        np.array(
            [
                39000.0 / 35000.0,
                33000.0 / 35000.0,
                33000.0 / 35000.0,
                4000.0 / 12000.0,
                -2000.0 / 12000.0,
                -2000.0 / 12000.0,
                0.0,
                1.0 / 3.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(fourth_y.numpy(), np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_reward_predictor_autodetects_new_grp_input_shape(tmp_path):
    model_path = tmp_path / "grp_model_new_shape.pth"
    model = RankPredictor(input_dim=13, n_players=3)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    torch.save(model.state_dict(), model_path)

    predictor = RewardPredictor(
        str(model_path),
        pts_weight=[4.0, 2.0, 0.0],
        n_players=3,
        device="cpu",
    )

    rewards = predictor.calc_all_player_rewards(
        {
            "p0_start_score": 39000,
            "p1_start_score": 33000,
            "p2_start_score": 33000,
            "p0_delta_score": 4000,
            "p1_delta_score": -2000,
            "p2_delta_score": -2000,
            "chang": 0,
            "ju": 1,
            "ben": 0,
            "liqibang": 0,
        }
    )

    assert rewards == [0.0, 0.0, 0.0]
