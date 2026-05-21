import json
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

    context_bucket_count = BUCKET_COUNT - 1
    assert model.alloc_bucket_embed.num_embeddings == context_bucket_count
    assert model.alloc_state_embed.num_embeddings == 36
    assert model.mask_state_id == 35
    assert model.alloc_mask_embed.num_embeddings == 2
    assert model.alloc_time_embed.num_embeddings == 4
    assert len(model.denoise_decoder.layers) == 2
    assert model.denoise_decoder.head.out_features == 35
    assert model.tile37_to_tile34[0] == model.tile37_to_tile34[5] == 4
    assert model.tile37_to_tile34[10] == model.tile37_to_tile34[15] == 13
    assert model.tile37_to_tile34[20] == model.tile37_to_tile34[25] == 22


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

    context, memory, memory_padding_mask = model.encoder.forward_context_and_memory(features)
    prog_len = features["progression"].shape[1]
    sparse_meld_len = features["sparse_meld_mask"].shape[1]
    static_memory_len = 4 + sparse_meld_len + TILE37_COUNT

    assert context.shape == (2, 64)
    assert memory.shape == (2, static_memory_len + prog_len, 64)
    assert memory_padding_mask.shape == (2, static_memory_len + prog_len)
    assert not memory_padding_mask[:, :4].any()
    assert torch.equal(memory_padding_mask[:, 4 : 4 + sparse_meld_len], ~features["sparse_meld_mask"])
    assert not memory_padding_mask[:, 4 + sparse_meld_len : static_memory_len].any()
    assert torch.equal(memory_padding_mask[:, static_memory_len:], ~features["prog_mask"])
    tile_bucket_context = model._tile_bucket_context(model._unseen_counts(features), memory, memory_padding_mask)
    assert tile_bucket_context.shape == (2, TILE37_COUNT, BUCKET_COUNT - 1, 64)


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

    def wrapped_forward_context_and_memory(batch):
        encoder_batch_sizes.append(batch["visible_tile_counts"].shape[0])
        return original_forward_context_and_memory(batch)

    model.encoder.forward_context_and_memory = wrapped_forward_context_and_memory

    samples = model.sample_allocations(features, num_samples=3)

    assert encoder_batch_sizes == [2]
    assert samples.shape == (2, 3, 4, 37)


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
        event
        for event in events
        if isinstance(event.get("meta"), dict) and "belief_allocation" in event["meta"]
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
