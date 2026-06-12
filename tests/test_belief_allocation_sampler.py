import json
import math
from pathlib import Path

import pytest
import torch
from riichienv_ml.belief_log_sampling import BeliefLogSampler, _event_matches_action
from riichienv_ml.config import BeliefLogSamplingConfig, GameConfig, ModelConfig
from riichienv_ml.datasets.belief_allocation import BeliefAllocationDataset
from riichienv_ml.features.belief_features import (
    BUCKET_COUNT,
    TILE37_COUNT,
    TOTAL_TILE_COUNTS37,
    BeliefFeatureEncoder,
    collate_belief_features,
)
from riichienv_ml.models.belief_allocation import AllocationDecodeState, JointHiddenAllocationSampler

from riichienv import MjaiReplay

DATA_PATH = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"


def _decode_unmask_counts(decode_steps: int) -> list[int]:
    token_count = TILE37_COUNT * (BUCKET_COUNT - 1)
    remaining_count = token_count
    counts = []
    for step in range(decode_steps):
        n_to_unmask = JointHiddenAllocationSampler._scheduled_unmask_count(step, decode_steps, remaining_count)
        counts.append(n_to_unmask)
        remaining_count -= n_to_unmask
        if remaining_count == 0:
            break
    return counts


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


def test_belief_dataset_buffers_configured_number_of_files_before_yield(monkeypatch):
    monkeypatch.setattr("riichienv_ml.datasets.belief_allocation.random.shuffle", lambda values: None)

    class DummyBeliefDataset(BeliefAllocationDataset):
        def __init__(self):
            super().__init__(
                ["a", "b", "c"],
                is_train=True,
                n_players=4,
                replay_rule="tenhou",
                encoder=None,
                shuffle_buffer_files=2,
            )
            self.loaded_files = []

        def _load_file_samples(self, file_path: str):
            self.loaded_files.append(file_path)
            return [(file_path, torch.zeros(4, 37, dtype=torch.long))]

    dataset = DummyBeliefDataset()
    next(iter(dataset))

    assert dataset.loaded_files == ["a", "b"]


def test_belief_dataset_applies_sample_keep_prob_after_buffering(monkeypatch):
    monkeypatch.setattr("riichienv_ml.datasets.belief_allocation.random.shuffle", lambda values: None)
    random_values = iter([0.0, 0.9, 0.2, 0.8])
    monkeypatch.setattr("riichienv_ml.datasets.belief_allocation.random.random", lambda: next(random_values))

    class DummyBeliefDataset(BeliefAllocationDataset):
        def __init__(self):
            super().__init__(
                ["a", "b", "c", "d"],
                is_train=True,
                n_players=4,
                replay_rule="tenhou",
                encoder=None,
                shuffle_buffer_files=2,
                sample_keep_prob=0.5,
            )

        def _load_file_samples(self, file_path: str):
            return [(file_path, torch.zeros(4, 37, dtype=torch.long))]

    assert [sample_id for sample_id, _target in DummyBeliefDataset()] == ["a", "c"]


def test_belief_dataset_applies_per_sample_keep_prob_after_buffering(monkeypatch):
    monkeypatch.setattr("riichienv_ml.datasets.belief_allocation.random.shuffle", lambda values: None)
    random_values = iter([0.3, 0.02, 0.1])
    monkeypatch.setattr("riichienv_ml.datasets.belief_allocation.random.random", lambda: next(random_values))

    class DummyBeliefDataset(BeliefAllocationDataset):
        def __init__(self):
            super().__init__(
                ["a"],
                is_train=True,
                n_players=4,
                replay_rule="tenhou",
                encoder=None,
                sample_keep_prob=0.2,
                stratified_sample_keep_prob={"enabled": True},
            )

        def _load_file_samples(self, file_path: str):
            del file_path
            target = torch.zeros(4, 37, dtype=torch.long)
            return [
                ("riichi", target, 0.4),
                ("early_closed", target, 0.01),
                ("legacy", target),
            ]

    assert {sample_id for sample_id, _target in DummyBeliefDataset()} == {"riichi", "legacy"}


def test_belief_dataset_stratified_keep_prob_uses_max_opponent_value():
    class DummyObs:
        player_id = 0
        riichi_declared = [False, True, False, False]
        riichi_stage = [False, False, False, False]
        melds = [[], [], [1, 2, 3], []]
        discards = [[], [], list(range(3)), list(range(20))]

    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=True,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
        stratified_sample_keep_prob={
            "riichi": 0.4,
            "meld1": 0.1,
            "meld2": 0.2,
            "meld3plus": 0.5,
            "closed_start": 0.01,
            "closed_end": 0.3,
            "closed_start_discard_count": 0,
            "closed_end_discard_count": 20,
        },
    )

    assert dataset._decision_sample_keep_prob(DummyObs()) == 0.5


def test_belief_dataset_rejects_invalid_shuffle_buffer_files():
    with pytest.raises(ValueError, match="shuffle_buffer_files"):
        BeliefAllocationDataset(
            [str(DATA_PATH)],
            is_train=True,
            n_players=4,
            replay_rule="tenhou",
            encoder=BeliefFeatureEncoder(),
            shuffle_buffer_files=0,
        )


@pytest.mark.parametrize("sample_keep_prob", [0.0, 1.1])
def test_belief_dataset_rejects_invalid_sample_keep_prob(sample_keep_prob):
    with pytest.raises(ValueError, match="sample_keep_prob"):
        BeliefAllocationDataset(
            [str(DATA_PATH)],
            is_train=True,
            n_players=4,
            replay_rule="tenhou",
            encoder=BeliefFeatureEncoder(),
            sample_keep_prob=sample_keep_prob,
        )


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
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
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
    diagnostics = model.allocation_diagnostics(features, samples)
    assert diagnostics["allocation_legal_rate"].item() == 1.0
    assert diagnostics["opponent_hand_size_exact_rate"].item() == 1.0

    longer_step_samples = model.sample_allocations(features, num_samples=1, decode_steps=8)
    assert longer_step_samples.shape == (2, 1, 4, 37)
    assert torch.equal(longer_step_samples.sum(dim=2), unseen.unsqueeze(1))


def test_belief_decode_steps_equal_allocation_tokens_unmasks_one_cell_per_step():
    token_count = TILE37_COUNT * (BUCKET_COUNT - 1)

    counts = _decode_unmask_counts(token_count)

    assert counts == [1] * token_count


def test_belief_decode_steps_other_than_allocation_tokens_keep_cosine_schedule():
    counts = _decode_unmask_counts(12)

    assert counts == [1, 3, 4, 7, 8, 10, 10, 12, 14, 13, 15, 14]


def test_belief_training_mask_decode_steps_equal_allocation_tokens_samples_uniform_count(monkeypatch):
    token_count = TILE37_COUNT * (BUCKET_COUNT - 1)
    expected_counts = torch.tensor([1, token_count // 2, token_count], dtype=torch.long)
    calls = []

    def fake_randint(low, high, size, *, device=None):
        calls.append((low, high, size, device))
        target_device = device if device is not None else torch.device("cpu")
        return expected_counts.to(target_device)

    monkeypatch.setattr(torch, "randint", fake_randint)

    mask = JointHiddenAllocationSampler._sample_random_mask(
        expected_counts.shape[0],
        torch.device("cpu"),
        decode_steps=token_count,
    )

    assert calls == [(1, token_count + 1, (expected_counts.shape[0],), torch.device("cpu"))]
    assert mask.flatten(1).sum(dim=1).tolist() == expected_counts.tolist()


def test_belief_training_forward_passes_decode_steps_to_random_mask(monkeypatch):
    d_model = 64
    token_count = TILE37_COUNT * (BUCKET_COUNT - 1)
    batch_size = 2
    model = JointHiddenAllocationSampler(
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=token_count,
        dropout=0.0,
    )
    captured = {}

    def fake_sample_random_mask(batch_size_arg, device, *, decode_steps=None):
        captured["batch_size"] = batch_size_arg
        captured["device"] = device
        captured["decode_steps"] = decode_steps
        return torch.ones(batch_size_arg, TILE37_COUNT, BUCKET_COUNT - 1, dtype=torch.bool, device=device)

    def fake_denoise_neural_logits(*args, **kwargs):
        del args, kwargs
        return torch.zeros(batch_size, TILE37_COUNT, BUCKET_COUNT - 1, 5)

    def fake_apply_prior_and_legality(neural_logits, *args, **kwargs):
        del args, kwargs
        return neural_logits, torch.ones_like(neural_logits, dtype=torch.bool)

    monkeypatch.setattr(model, "_sample_random_mask", fake_sample_random_mask)
    monkeypatch.setattr(model, "_denoise_neural_logits", fake_denoise_neural_logits)
    monkeypatch.setattr(model, "_apply_prior_and_legality", fake_apply_prior_and_legality)

    unseen_counts = torch.ones(batch_size, TILE37_COUNT, dtype=torch.long)
    rem = torch.tensor([[1, 1, 1, TILE37_COUNT - 3]] * batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, BUCKET_COUNT, TILE37_COUNT, dtype=torch.long)
    targets[:, 0, 0] = 1
    targets[:, 1, 1] = 1
    targets[:, 2, 2] = 1
    targets[:, 3] = unseen_counts
    targets[:, 3, 0] -= 1
    targets[:, 3, 1] -= 1
    targets[:, 3, 2] -= 1

    out = model._training_forward(
        torch.zeros(batch_size, d_model),
        torch.zeros(batch_size, 1, d_model),
        torch.zeros(batch_size, 1, dtype=torch.bool),
        unseen_counts,
        torch.zeros(batch_size, TILE37_COUNT, d_model),
        torch.zeros(BUCKET_COUNT - 1, d_model),
        rem,
        targets,
    )

    assert captured == {
        "batch_size": batch_size,
        "device": torch.device("cpu"),
        "decode_steps": token_count,
    }
    assert out["loss"].isfinite()


def test_belief_confidence_legal_normalized_entropy_scores_by_legal_candidate_count():
    probs = torch.tensor(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.90, 0.033, 0.033, 0.034, 0.00],
            [0.00, 1.00, 0.00, 0.00, 0.00],
            [0.20, 0.20, 0.20, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    feasible = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
        ],
    )

    confidence = JointHiddenAllocationSampler._confidence(probs, "legal_normalized_entropy", feasible)

    expected_two = 1.0 - (-(0.90 * math.log(0.90) + 0.10 * math.log(0.10))) / math.log(2)
    expected_four = 1.0 - (
        -(0.90 * math.log(0.90) + 0.033 * math.log(0.033) + 0.033 * math.log(0.033) + 0.034 * math.log(0.034))
    ) / math.log(4)

    assert torch.isclose(confidence[0], torch.tensor(expected_two, dtype=confidence.dtype))
    assert torch.isclose(confidence[1], torch.tensor(expected_four, dtype=confidence.dtype))
    assert confidence[1] > confidence[0]
    assert confidence[2].item() == 2.0
    assert confidence[3].item() == torch.finfo(confidence.dtype).min


def test_belief_model_accepts_legal_normalized_entropy_confidence_method():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
        confidence_method="legal_normalized_entropy",
    )

    assert model.confidence_method == "legal_normalized_entropy"


def test_belief_decode_steps_equal_allocation_tokens_uses_two_stage_cell_decoding(monkeypatch):
    d_model = 64
    model = JointHiddenAllocationSampler(
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    token_count = TILE37_COUNT * (BUCKET_COUNT - 1)
    calls = []

    def fake_denoise_neural_logits(*args, **kwargs):
        del args, kwargs
        return torch.zeros(1, TILE37_COUNT, BUCKET_COUNT - 1, 5)

    def fake_apply_prior_and_legality(neural_logits, *args, **kwargs):
        del args, kwargs
        return neural_logits, torch.ones_like(neural_logits, dtype=torch.bool)

    def fake_cell_step(
        neural_logits_cells,
        state,
        token,
        tile37,
        opponent,
        allocation_offset,
        batch_offsets,
        *,
        sample,
        temp,
        row_indices=None,
    ):
        del neural_logits_cells, sample, temp, row_indices
        chosen = torch.zeros(1, dtype=torch.long)
        state_index = batch_offsets.state + token
        state.apply_cell_update(batch_offsets, state_index, tile37, opponent, allocation_offset, chosen)
        calls.append(int(token.item()))
        return tile37, opponent, chosen

    monkeypatch.setattr(model, "_denoise_neural_logits", fake_denoise_neural_logits)
    monkeypatch.setattr(model, "_apply_prior_and_legality", fake_apply_prior_and_legality)
    monkeypatch.setattr(model, "_sample_and_apply_cell_step", fake_cell_step)
    monkeypatch.setattr(model, "_apply_forced_decode_cells", lambda *args, **kwargs: 0)

    allocation = model._iterative_decode(
        torch.zeros(1, d_model),
        torch.zeros(1, 1, d_model),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, TILE37_COUNT, dtype=torch.long),
        torch.zeros(1, TILE37_COUNT, d_model),
        torch.zeros(BUCKET_COUNT - 1, d_model),
        torch.zeros(1, BUCKET_COUNT, dtype=torch.long),
        sample=True,
        temperature=1.0,
        decode_steps=token_count,
    )

    assert len(calls) == token_count
    assert sorted(calls) == list(range(token_count))
    assert allocation.shape == (1, BUCKET_COUNT, TILE37_COUNT)


def test_belief_model_decoder_uses_denoise_transformer_and_mask_state():
    d_model = 64
    model = JointHiddenAllocationSampler(
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=2,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )

    assert not hasattr(model, "alloc_tile_embed")
    assert not hasattr(model, "alloc_bucket_embed")
    assert not hasattr(model, "alloc_mask_embed")
    assert model.alloc_tile_proj[0].in_features == model.encoder.tile_embed.proj.weight.shape[0]
    assert model.alloc_tile_proj[0].out_features == d_model
    assert model.alloc_tile_remaining_embed.num_embeddings == 5
    assert model.alloc_seat_remaining_embed.num_embeddings == BeliefFeatureEncoder.HAND_SIZE_DIMS
    assert model.alloc_state_embed.num_embeddings == 6
    assert model.mask_state_id == 5
    assert not hasattr(model, "joint_assignment_confidence_beta")
    assert not hasattr(model, "joint_assignment_score_gamma")
    assert not hasattr(model, "alloc_time_embed")
    assert not hasattr(model, "max_decode_steps")
    assert len(model.denoise_decoder.layers) == 2
    assert model.denoise_decoder.head.out_features == 5
    assert model.opponent_subset_masks.shape == (7, BUCKET_COUNT - 1)
    assert model.opponent_subset_masks.dtype == torch.bool
    assert torch.equal(model.opponent_subset_masks_long, model.opponent_subset_masks.long())
    assert "opponent_subset_masks" not in model.state_dict()
    assert "opponent_subset_masks_long" not in model.state_dict()
    assert model.logcomb_table.shape == (int(sum(TOTAL_TILE_COUNTS37)) + 1, int(sum(TOTAL_TILE_COUNTS37)) + 1)
    assert model.logcomb_table.dtype == torch.float32
    assert "logcomb_table" not in model.state_dict()
    assert model.tile37_to_tile34[0] == model.tile37_to_tile34[5] == 4
    assert model.tile37_to_tile34[10] == model.tile37_to_tile34[15] == 13
    assert model.tile37_to_tile34[20] == model.tile37_to_tile34[25] == 22


def test_belief_model_ignores_legacy_joint_assignment_config_keys():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        joint_assignment_confidence_beta=2.0,
        joint_assignment_score_gamma=1.5,
        dropout=0.0,
    )

    assert not hasattr(model, "joint_assignment_confidence_beta")
    assert not hasattr(model, "joint_assignment_score_gamma")


def test_belief_model_logcomb_lookup_matches_clamped_lgamma():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    n = torch.tensor([[4, 4, 3, -1, 2]])
    k = torch.tensor([[0, 2, 5, 1, -3]])

    n_f = n.float().clamp_min(0.0)
    k_f = k.float().clamp_min(0.0)
    k_f = torch.minimum(k_f, n_f)
    expected = torch.lgamma(n_f + 1.0) - torch.lgamma(k_f + 1.0) - torch.lgamma(n_f - k_f + 1.0)

    assert torch.allclose(model._safe_logcomb(n, k), expected)


def _reference_cell_hall_feasible(
    model: JointHiddenAllocationSampler,
    tile_remaining: torch.Tensor,
    seat_remaining: torch.Tensor,
    is_masked: torch.Tensor,
) -> torch.Tensor:
    opponent_count = BUCKET_COUNT - 1
    candidates = model.count_candidates.to(tile_remaining.device).view(1, 1, -1)
    tile_value = tile_remaining.unsqueeze(2).unsqueeze(-1)
    seat_value = seat_remaining.unsqueeze(1).unsqueeze(-1)
    feasible = (
        is_masked.unsqueeze(-1)
        & (candidates.view(1, 1, 1, -1) <= tile_value)
        & (candidates.view(1, 1, 1, -1) <= seat_value)
    )
    subset_masks = model.opponent_subset_masks.to(tile_remaining.device)
    subset_masks_long = model.opponent_subset_masks_long.to(tile_remaining.device)

    for subset_idx in range(subset_masks.shape[0]):
        subset = subset_masks[subset_idx]
        subset_long = subset_masks_long[subset_idx]
        base_demand = (seat_remaining * subset_long).sum(dim=1)
        subset_edges = is_masked & subset.view(1, 1, opponent_count)
        any_before = subset_edges.any(dim=2)
        edge_count = subset_edges.long().sum(dim=2)
        base_cap = (tile_remaining * any_before.long()).sum(dim=1)

        for seat in range(opponent_count):
            if bool(subset[seat].item()):
                demand_after = base_demand.view(-1, 1, 1) - candidates
                any_after = edge_count > 1
            else:
                demand_after = base_demand.view(-1, 1, 1)
                any_after = any_before
            cap_after = (
                base_cap.view(-1, 1, 1)
                - tile_remaining.unsqueeze(-1) * any_before.long().unsqueeze(-1)
                + (tile_remaining.unsqueeze(-1) - candidates) * any_after.long().unsqueeze(-1)
            )
            feasible[:, :, seat, :] &= demand_after <= cap_after

    return feasible


def test_belief_model_full_hall_legality_matches_reference_loop():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    generator = torch.Generator().manual_seed(23)
    batch_size = 4
    tile_remaining = torch.randint(0, 5, (batch_size, TILE37_COUNT), generator=generator)
    seat_remaining = torch.randint(0, 14, (batch_size, BUCKET_COUNT - 1), generator=generator)
    is_masked = torch.rand(batch_size, TILE37_COUNT, BUCKET_COUNT - 1, generator=generator) > 0.4

    actual = model._cell_hall_feasible(tile_remaining, seat_remaining, is_masked)
    expected = _reference_cell_hall_feasible(model, tile_remaining, seat_remaining, is_masked)

    assert torch.equal(actual, expected)


def test_belief_model_selected_cell_legality_matches_full_legality():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    generator = torch.Generator().manual_seed(11)
    batch_size = 5
    tile_remaining = torch.randint(0, 5, (batch_size, TILE37_COUNT), generator=generator)
    seat_remaining = torch.randint(0, 14, (batch_size, BUCKET_COUNT - 1), generator=generator)
    is_masked = torch.rand(batch_size, TILE37_COUNT, BUCKET_COUNT - 1, generator=generator) > 0.35
    tile37 = torch.tensor([0, 5, 10, 20, 36])
    opponent = torch.tensor([0, 1, 2, 0, 2])
    neural_logits = torch.randn(batch_size, TILE37_COUNT, BUCKET_COUNT - 1, 5, generator=generator)
    row_ids = torch.arange(batch_size)

    full_logits, full_feasible = model._apply_prior_and_legality(
        neural_logits,
        tile_remaining,
        seat_remaining,
        is_masked,
    )
    cell_logits, cell_feasible = model._apply_prior_and_legality_for_cells(
        neural_logits[row_ids, tile37, opponent],
        tile_remaining,
        seat_remaining,
        is_masked,
        tile37,
        opponent,
    )

    expected_logits = full_logits[row_ids, tile37, opponent]
    expected_feasible = full_feasible[row_ids, tile37, opponent]
    finite = torch.isfinite(expected_logits)
    assert torch.equal(cell_feasible, expected_feasible)
    assert torch.equal(torch.isneginf(cell_logits), torch.isneginf(expected_logits))
    assert torch.allclose(cell_logits[finite], expected_logits[finite])


def test_belief_model_detects_forced_cells_from_unique_legal_candidate():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    tile_remaining = torch.zeros(1, TILE37_COUNT, dtype=torch.long)
    tile_remaining[0, 7] = 3
    seat_remaining = torch.tensor([[3, 0, 0]], dtype=torch.long)
    is_masked = torch.ones(1, TILE37_COUNT, BUCKET_COUNT - 1, dtype=torch.bool)

    forced, values = model._forced_cells(tile_remaining, seat_remaining, is_masked)

    assert forced[0, 7, 0]
    assert values[0, 7, 0].item() == 3
    assert forced[0, :, 1].all()
    assert forced[0, :, 2].all()
    assert values[0, :, 1].sum().item() == 0
    assert values[0, :, 2].sum().item() == 0
    assert forced[0, 0, 0]
    assert values[0, 0, 0].item() == 0


def test_belief_model_initial_decode_resolves_fully_forced_state_without_neural_logits(monkeypatch):
    d_model = 64
    model = JointHiddenAllocationSampler(
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )

    def fail_denoise_neural_logits(*args, **kwargs):
        del args, kwargs
        raise AssertionError("forced decode should finish before neural logits are requested")

    monkeypatch.setattr(model, "_denoise_neural_logits", fail_denoise_neural_logits)
    unseen_counts = torch.zeros(1, TILE37_COUNT, dtype=torch.long)
    unseen_counts[0, 7] = 3
    rem = torch.tensor([[3, 0, 0, 0]], dtype=torch.long)

    allocation = model._iterative_decode(
        torch.zeros(1, d_model),
        torch.zeros(1, 1, d_model),
        torch.zeros(1, 1, dtype=torch.bool),
        unseen_counts,
        torch.zeros(1, TILE37_COUNT, d_model),
        torch.zeros(BUCKET_COUNT - 1, d_model),
        rem,
        sample=True,
        temperature=1.0,
        decode_steps=4,
    )

    assert allocation[0, 0, 7].item() == 3
    assert allocation[0, :3].sum().item() == 3
    assert allocation[0, 3].sum().item() == 0
    assert torch.equal(allocation.sum(dim=1), unseen_counts)


def test_belief_training_mask_resolves_forced_cells_and_retries_empty_rows(monkeypatch):
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    first = torch.zeros(1, TILE37_COUNT, BUCKET_COUNT - 1, dtype=torch.bool)
    first[0, 0, 1] = True
    second = first.clone()
    second[0, 0, 0] = True
    second[0, 1, 0] = True
    sampled_masks = [first, second]
    calls = []

    def fake_sample_random_mask(batch_size, device, *, decode_steps=None):
        calls.append((batch_size, device, decode_steps))
        return sampled_masks.pop(0).to(device)

    monkeypatch.setattr(model, "_sample_random_mask", fake_sample_random_mask)
    target_cells = torch.zeros(1, TILE37_COUNT, BUCKET_COUNT - 1, dtype=torch.long)
    target_cells[0, 0, 0] = 1
    target_ids = target_cells.clone()
    target_match = torch.ones_like(target_cells, dtype=torch.bool)
    unseen_counts = torch.zeros(1, TILE37_COUNT, dtype=torch.long)
    unseen_counts[0, 0] = 1
    unseen_counts[0, 1] = 1
    seat_caps = torch.tensor([[1, 0, 0]], dtype=torch.long)
    target_sample_valid = torch.ones(1, dtype=torch.bool)

    mask = model._sample_training_mask(
        target_cells,
        target_ids,
        target_match,
        unseen_counts,
        seat_caps,
        target_sample_valid,
        decode_steps=model.decode_steps,
    )

    assert calls == [(1, torch.device("cpu"), 4), (1, torch.device("cpu"), 4)]
    assert mask[0, 0, 0]
    assert mask[0, 1, 0]
    assert not mask[0, 0, 1]
    assert mask.sum().item() == 2


def test_belief_model_decode_token_lookup_buffers():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    tokens = torch.arange(TILE37_COUNT * (BUCKET_COUNT - 1))
    expected_tile37 = tokens // (BUCKET_COUNT - 1)
    expected_opponent = tokens % (BUCKET_COUNT - 1)
    expected_allocation_offset = expected_opponent * TILE37_COUNT + expected_tile37
    lookup = model._decode_token_lookup(torch.device("cpu"))

    assert torch.equal(lookup.tile37, expected_tile37)
    assert torch.equal(lookup.opponent, expected_opponent)
    assert torch.equal(lookup.allocation_offset, expected_allocation_offset)


def test_belief_model_cell_step_updates_decode_state():
    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    generator = torch.Generator().manual_seed(31)
    batch_size = 3
    batch_indices = torch.arange(batch_size)
    token = torch.tensor([0, 5 * (BUCKET_COUNT - 1) + 1, 10 * (BUCKET_COUNT - 1) + 2])
    tile37 = torch.div(token, BUCKET_COUNT - 1, rounding_mode="floor")
    opponent = token % (BUCKET_COUNT - 1)
    neural_logits = torch.randn(batch_size, TILE37_COUNT, BUCKET_COUNT - 1, 5, generator=generator)
    neural_logits_cells = neural_logits.reshape(batch_size * TILE37_COUNT * (BUCKET_COUNT - 1), 5)
    batch_offsets = model._decode_batch_offsets(batch_indices)
    token_lookup = model._decode_token_lookup(torch.device("cpu"))
    allocation_offset = token_lookup.allocation_offset.index_select(0, token)
    state = AllocationDecodeState(
        allocation=torch.zeros(batch_size, BUCKET_COUNT, TILE37_COUNT, dtype=torch.long),
        state_ids=torch.full((batch_size, TILE37_COUNT, BUCKET_COUNT - 1), model.mask_state_id, dtype=torch.long),
        tile_remaining=torch.full((batch_size, TILE37_COUNT), 4, dtype=torch.long),
        seat_remaining=torch.full((batch_size, BUCKET_COUNT - 1), 13, dtype=torch.long),
        is_masked=torch.ones(batch_size, TILE37_COUNT, BUCKET_COUNT - 1, dtype=torch.bool),
    )

    cell_logits, _cell_feasible = model._apply_prior_and_legality_for_cells(
        neural_logits[batch_indices, tile37, opponent],
        state.tile_remaining,
        state.seat_remaining,
        state.is_masked,
        tile37,
        opponent,
    )
    expected_chosen = cell_logits.argmax(dim=1)
    expected_allocation = state.allocation.clone()
    expected_state_ids = state.state_ids.clone()
    expected_tile_remaining = state.tile_remaining.clone()
    expected_seat_remaining = state.seat_remaining.clone()
    expected_is_masked = state.is_masked.clone()
    expected_allocation[batch_indices, opponent, tile37] = expected_chosen
    expected_state_ids[batch_indices, tile37, opponent] = expected_chosen
    expected_tile_remaining[batch_indices, tile37] -= expected_chosen
    expected_seat_remaining[batch_indices, opponent] -= expected_chosen
    expected_is_masked[batch_indices, tile37, opponent] = False

    actual_tile37, actual_opponent, actual_chosen = model._sample_and_apply_cell_step(
        neural_logits_cells,
        state,
        token,
        tile37,
        opponent,
        allocation_offset,
        batch_offsets,
        sample=False,
        temp=1.0,
    )

    assert torch.equal(actual_tile37, tile37)
    assert torch.equal(actual_opponent, opponent)
    assert torch.equal(actual_chosen, expected_chosen)
    assert torch.equal(state.allocation, expected_allocation)
    assert torch.equal(state.state_ids, expected_state_ids)
    assert torch.equal(state.tile_remaining, expected_tile_remaining)
    assert torch.equal(state.seat_remaining, expected_seat_remaining)
    assert torch.equal(state.is_masked, expected_is_masked)


def test_belief_encoder_returns_public_cross_attention_memory():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    items = [next(iter(dataset)) for _ in range(2)]
    features = collate_belief_features([item[0] for item in items])

    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )

    context, memory, memory_padding_mask, tile37_table = model.encoder.forward_context_and_memory(
        features,
        return_tile37_table=True,
    )
    prog_len = features["progression"].shape[1]
    sparse_meld_len = features["sparse_meld_mask"].shape[1]
    static_memory_len = 4 + sparse_meld_len + TILE37_COUNT

    assert context.shape == (2, 64)
    assert tile37_table.shape == (2, TILE37_COUNT, model.encoder.tile_embed.proj.weight.shape[0])
    assert memory.shape == (2, static_memory_len + prog_len, 64)
    assert memory_padding_mask.shape == (2, static_memory_len + prog_len)
    assert not memory_padding_mask[:, :4].any()
    assert torch.equal(memory_padding_mask[:, 4 : 4 + sparse_meld_len], ~features["sparse_meld_mask"])
    assert not memory_padding_mask[:, 4 + sparse_meld_len : static_memory_len].any()
    assert torch.equal(memory_padding_mask[:, static_memory_len:], ~features["prog_mask"])
    tile_emb = model._shared_alloc_tile_embeddings(tile37_table)
    assert tile_emb.shape == (2, TILE37_COUNT, 64)
    seat_emb = model._shared_alloc_seat_embeddings(context.device)
    assert seat_emb.shape == (BUCKET_COUNT - 1, 64)
    unseen_counts = model._unseen_counts(features)
    rem = model._initial_rem(features, unseen_counts)
    state_ids = torch.full((2, TILE37_COUNT, BUCKET_COUNT - 1), model.mask_state_id)
    tokens = model._token_embeddings(tile_emb, seat_emb, unseen_counts, rem[:, :3], state_ids)
    assert tokens.shape == (2, TILE37_COUNT * (BUCKET_COUNT - 1), 64)
    logits = model._denoise_neural_logits(
        context,
        memory,
        memory_padding_mask,
        tile_emb,
        seat_emb,
        unseen_counts,
        rem[:, :3],
        state_ids,
    )
    assert logits.shape == (2, TILE37_COUNT, BUCKET_COUNT - 1, 5)


def test_belief_model_samples_reuse_single_encoder_context():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    items = [next(iter(dataset)) for _ in range(2)]
    features = collate_belief_features([item[0] for item in items])

    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    original_forward_context_and_memory = model.encoder.forward_context_and_memory
    encoder_batch_sizes = []

    def wrapped_forward_context_and_memory(batch, **kwargs):
        encoder_batch_sizes.append(batch["visible_tile_counts"].shape[0])
        return original_forward_context_and_memory(batch, **kwargs)

    model.encoder.forward_context_and_memory = wrapped_forward_context_and_memory

    samples = model.sample_allocations(features, num_samples=3)

    assert encoder_batch_sizes == [2]
    assert samples.shape == (2, 3, 4, 37)


def test_belief_model_samples_from_prepared_context_without_rerunning_encoder():
    dataset = BeliefAllocationDataset(
        [str(DATA_PATH)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=BeliefFeatureEncoder(),
    )
    items = [next(iter(dataset)) for _ in range(2)]
    features = collate_belief_features([item[0] for item in items])

    model = JointHiddenAllocationSampler(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    original_forward_context_and_memory = model.encoder.forward_context_and_memory
    encoder_batch_sizes = []

    def wrapped_forward_context_and_memory(batch, **kwargs):
        encoder_batch_sizes.append(batch["visible_tile_counts"].shape[0])
        return original_forward_context_and_memory(batch, **kwargs)

    model.encoder.forward_context_and_memory = wrapped_forward_context_and_memory

    sampling_context = model.prepare_allocation_sampling_context(features)
    first = model.sample_allocations_from_context(sampling_context, num_samples=2)
    second = model.sample_allocations_from_context(sampling_context, num_samples=2)

    assert encoder_batch_sizes == [2]
    assert first.shape == (2, 2, 4, 37)
    assert second.shape == (2, 2, 4, 37)


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
        denoise_num_layers=1,
        denoise_dim_feedforward=128,
        decode_steps=4,
        dropout=0.0,
    )
    out = model(batch, target_counts=targets)

    assert out["loss"].isfinite()
    assert out["loss"].item() == 0.0
    assert out["invalid_target_rate"].item() == 1.0


def test_belief_log_sampler_writes_mjai_hand_metadata(tmp_path):
    model_path = tmp_path / "belief_model.pth"
    output_dir = tmp_path / "annotated"
    original_text = DATA_PATH.read_text(encoding="utf-8")
    model_config = {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 128,
        "denoise_num_layers": 1,
        "denoise_dim_feedforward": 128,
        "decode_steps": 4,
        "dropout": 0.0,
    }
    model = JointHiddenAllocationSampler(**model_config)
    torch.save(model.state_dict(), model_path)

    cfg = BeliefLogSamplingConfig(
        game=GameConfig(n_players=4, replay_rule="tenhou"),
        input_glob=str(DATA_PATH),
        output_dir=str(output_dir),
        summary_path=str(output_dir / "summary.json"),
        model_path=str(model_path),
        device="cpu",
        num_logs=1,
        num_samples=2,
        batch_size=8,
        skip_single_action=True,
        overwrite=False,
        compress_output=False,
        seed=7,
        model=ModelConfig(**model_config),
    )

    summary = BeliefLogSampler(cfg).run()

    assert DATA_PATH.read_text(encoding="utf-8") == original_text
    output_path = output_dir / DATA_PATH.name
    assert summary["num_logs"] == 1
    assert summary["sample_diagnostics"]["allocation_legal_rate"] == 1.0
    assert summary["sample_diagnostics"]["opponent_hand_size_exact_rate"] == 1.0
    assert output_path.exists()
    replay = MjaiReplay.from_jsonl(str(output_path), rule="tenhou")
    assert replay.num_rounds() > 0

    events = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    belief_events = [
        event for event in events if isinstance(event.get("meta"), dict) and "belief_allocation" in event["meta"]
    ]
    assert belief_events
    meta = belief_events[0]["meta"]["belief_allocation"]
    assert meta["tile_format"] == "mjai"
    assert meta["sample_count"] == 2
    assert len(meta["opponent_seats"]) == 3
    assert len(meta["samples"]) == 2
    assert all(len(sample) == 3 for sample in meta["samples"])
    assert all(isinstance(tile, str) for sample in meta["samples"] for hand in sample for tile in hand)


def test_belief_log_sampler_matches_kan_and_red_call_events():
    assert _event_matches_action(
        {"type": "ankan", "actor": 2, "consumed": ["8s", "8s", "8s", "8s"]},
        {"type": "ankan", "actor": 2, "pai": "8s", "consumed": ["8s", "8s", "8s", "8s"]},
    )
    assert _event_matches_action(
        {"type": "pon", "actor": 0, "target": 1, "pai": "5m", "consumed": ["5m", "5mr"]},
        {"type": "pon", "actor": 0, "pai": "5m", "consumed": ["5mr", "5m"]},
    )
    assert _event_matches_action(
        {"type": "daiminkan", "actor": 0, "target": 1, "pai": "5m", "consumed": ["5m", "5m", "5mr"]},
        {"type": "daiminkan", "actor": 0, "pai": "5m", "consumed": ["5mr", "5m", "5m"]},
    )
