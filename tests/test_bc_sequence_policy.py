import json

import torch

from riichienv import MjaiReplay
from riichienv_ml.datasets.mjai_logs import BehaviorCloningDataset
from riichienv_ml.features.sequence_features import SequenceFeaturePackedEncoder
from riichienv_ml.models.transformer import TransformerActorCritic, TransformerPolicyNetwork


class DummyEncoder:
    def encode(self, obs):
        return torch.tensor([len(obs.legal_actions())], dtype=torch.float32)


def _write_simple_4p_log(path):
    data = [
        {"type": "start_game", "names": ["A", "B", "C", "D"], "id": "bc_seq_test"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
                ["1p", "1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "9p", "9p"],
                ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m", "9m"],
                ["E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F", "C"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    with open(path, "w") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")


def test_behavior_cloning_dataset_yields_legal_action_labels(tmp_path):
    file_path = tmp_path / "bc_policy_sample.jsonl"
    _write_simple_4p_log(file_path)

    dataset = BehaviorCloningDataset(
        [str(file_path)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=DummyEncoder(),
    )

    samples = list(dataset)

    assert samples
    for features, action_id, mask in samples:
        assert isinstance(features, torch.Tensor)
        assert 0 <= action_id < len(mask)
        assert mask[action_id] == 1


def test_transformer_policy_network_returns_logits_only():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        policy_head_type="cls",
        max_prog_len=8,
        max_cand_len=4,
    )
    packed_size = 25 + 12 + 8 * 5 + 4 * 4 + 25 + 8 + 4
    x = torch.zeros(2, packed_size)

    logits = model(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 82)


def test_transformer_actor_critic_keeps_value_head_output():
    model = TransformerActorCritic(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        policy_head_type="cls",
        max_prog_len=8,
        max_cand_len=4,
    )
    packed_size = 25 + 12 + 8 * 5 + 4 * 4 + 25 + 8 + 4
    x = torch.zeros(2, packed_size)

    logits, value = model(x)

    assert logits.shape == (2, 82)
    assert value.shape == (2,)


def test_sequence_feature_packed_encoder_matches_transformer_policy_input(tmp_path):
    file_path = tmp_path / "bc_sequence_sample.jsonl"
    _write_simple_4p_log(file_path)

    replay = MjaiReplay.from_jsonl(str(file_path), rule="tenhou")
    kyoku = list(replay.take_kyokus())[0]
    obs, _ = next(iter(kyoku.steps(0)))

    encoder = SequenceFeaturePackedEncoder(max_prog_len=256, max_cand_len=32)
    features = encoder.encode(obs).unsqueeze(0)

    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=256,
        max_cand_len=32,
    )

    logits = model(features)

    assert logits.shape == (1, 82)
