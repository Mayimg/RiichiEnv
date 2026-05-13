from pathlib import Path

import torch
from riichienv_ml.datasets.belief_allocation import BeliefAllocationDataset
from riichienv_ml.features.belief_features import (
    TOTAL_TILE_COUNTS37,
    BeliefFeatureEncoder,
    collate_belief_features,
)
from riichienv_ml.models.belief_allocation import JointHiddenAllocationSampler

from riichienv import MjaiReplay

DATA_PATH = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"


def test_mjai_steps_can_return_teacher_hidden_hands():
    replay = MjaiReplay.from_jsonl(str(DATA_PATH), rule="tenhou")
    kyoku = next(iter(replay.take_kyokus()))
    pid, obs, _action, hidden_hands = next(iter(kyoku.steps(skip_single_action=True, include_hidden=True)))

    assert pid == obs.player_id
    assert len(hidden_hands) == 4
    assert [len(hand) for hand in hidden_hands] == [14, 13, 13, 13]
    assert len(obs.hands[pid]) == len(hidden_hands[pid])


def test_belief_dataset_target_matches_unseen_counts():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    features, target = next(iter(dataset))

    visible = features["visible_tile_counts"][:, 1].numpy()
    unseen = torch.tensor(TOTAL_TILE_COUNTS37 - visible, dtype=torch.long)
    assert target.shape == (4, 37)
    assert torch.equal(target.sum(dim=0), unseen)
    assert target[3].sum().item() > 0


def test_belief_dataset_public_capacities_match_targets():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
        skip_single_action=False,
    )

    steps = 0
    for features, target in dataset:
        capacities = features["belief_hand_sizes"][1:4, 1]
        target_sizes = target[:3].sum(dim=1)
        assert torch.equal(capacities, target_sizes)
        steps += 1
    assert steps > 0


def test_belief_model_trains_and_samples_legal_allocations():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    items = [next(iter(dataset)) for _ in range(2)]
    features = collate_belief_features([item[0] for item in items])
    targets = torch.stack([item[1] for item in items])

    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        decoder_hidden_dim=64,
        dropout=0.0,
    )
    out = model(features, target_counts=targets)
    assert out["loss"].isfinite()
    assert out["allocation"].shape == (2, 4, 37)

    samples = model.sample_allocations(features, num_samples=2)
    visible = features["visible_tile_counts"][:, :, 1].long()
    unseen = torch.tensor(TOTAL_TILE_COUNTS37, dtype=torch.long).unsqueeze(0) - visible
    assert samples.shape == (2, 2, 4, 37)
    assert torch.equal(samples.sum(dim=2), unseen.unsqueeze(1).expand(-1, 2, -1))


def test_belief_model_skips_invalid_targets_without_huge_loss():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    features, target = next(iter(dataset))
    batch = collate_belief_features([features])
    targets = target.unsqueeze(0)
    batch["belief_hand_sizes"] = batch["belief_hand_sizes"].clone()
    batch["belief_hand_sizes"][0, 1, 1] += 1

    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        decoder_hidden_dim=64,
        dropout=0.0,
    )
    out = model(batch, target_counts=targets)

    assert out["loss"].isfinite()
    assert out["loss"].item() == 0.0
    assert out["invalid_target_rate"].item() == 1.0
