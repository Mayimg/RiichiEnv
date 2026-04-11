import json
import re

import torch
import torch.nn.functional as F
import pytest

from riichienv import MjaiReplay
from riichienv_ml.datasets.mjai_logs import BehaviorCloningDataset
from riichienv_ml.features.sequence_features import SequenceFeaturePackedEncoder
from riichienv_ml.models.transformer import TransformerActorCritic, TransformerPolicyNetwork
import riichienv_ml.trainers.bc_policy as bc_policy_module


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


def test_bc_policy_trainer_logs_recent_100_batch_metrics(monkeypatch):
    class DummyLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            if args:
                message = message.format(*args)
            self.messages.append(message)

    trainer = object.__new__(bc_policy_module.BCPolicyTrainer)
    trainer.device = torch.device("cpu")
    trainer.label_smoothing = 0.0
    trainer.max_grad_norm = 10.0
    trainer.limit = 101

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    easy_batch = (
        torch.tensor([[5.0, 0.0]], dtype=torch.float32),
        torch.tensor([0]),
        torch.ones(1, 2, dtype=torch.float32),
    )
    hard_batch = (
        torch.tensor([[0.0, 5.0]], dtype=torch.float32),
        torch.tensor([0]),
        torch.ones(1, 2, dtype=torch.float32),
    )
    dataloader = [easy_batch, *([hard_batch] * 100)]

    dummy_logger = DummyLogger()
    monkeypatch.setattr(bc_policy_module, "logger", dummy_logger)

    metrics, step = trainer._train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        epoch=0,
    )

    assert step == 101
    assert metrics["train/acc"] == pytest.approx(1 / 101, abs=1e-6)

    step_100_log = next(msg for msg in dummy_logger.messages if "Step 100" in msg)
    match = re.search(
        r"train/loss=(?P<loss>\d+\.\d+) train/acc=(?P<acc>\d+\.\d+) "
        r"train/window100_loss=(?P<window_loss>\d+\.\d+) train/window100_acc=(?P<window_acc>\d+\.\d+)",
        step_100_log,
    )
    assert match is not None

    loss_first = F.cross_entropy(torch.tensor([[5.0, 0.0]]), torch.tensor([0])).item()
    loss_recent = F.cross_entropy(torch.tensor([[0.0, 5.0]]), torch.tensor([0])).item()
    loss_cumulative = (loss_first + 100 * loss_recent) / 101

    assert float(match["loss"]) == pytest.approx(loss_cumulative, abs=1e-4)
    assert float(match["acc"]) == pytest.approx(1 / 101, abs=1e-4)
    assert float(match["window_loss"]) == pytest.approx(loss_recent, abs=1e-4)
    assert float(match["window_acc"]) == pytest.approx(0.0, abs=1e-4)
