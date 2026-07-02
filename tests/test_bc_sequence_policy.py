import json
import re

import pytest
import riichienv_ml.trainers.bc_policy as bc_policy_module
import torch
import torch.nn.functional as F
from riichienv_ml.datasets.mjai_logs import BehaviorCloningDataset, BehaviorCloningRankDataset
from riichienv_ml.features.sequence_features import (
    SequenceFeatureEncoder,
    SequenceFeaturePackedEncoder,
    collate_sequence_features,
    pack_sequence_features,
    packed_candidate_mask,
)
from riichienv_ml.models.transformer import (
    _ACTION_KIND_DISCARD,
    _ACTION_KIND_DORA,
    _ACTION_KIND_PAD,
    _MELD_KIND_CHI,
    _MELD_KIND_PAD,
    _MELD_KIND_PON,
    _MELD_ROLE_CALLED,
    _MELD_ROLE_CONSUMED,
    _RED_FLAG_PAD,
    _RED_FLAG_RED,
    _SEAT_ROLE_AGARI_WINNER,
    _SEAT_ROLE_MELD_OWNER,
    _SEAT_ROLE_PLAYER_INFO,
    _SEAT_ROLE_PROG_ACTOR,
    _SPARSE_DORA_OFFSET,
    _TILE34_PAD,
    SplitLinearLayerNorm,
    TransformerActorCritic,
    TransformerPolicyNetwork,
)
from riichienv_ml.utils import build_encoder

from riichienv import Action, ActionType, Meld, MeldType, MjaiReplay, Observation


class DummyEncoder:
    def encode(self, obs):
        return torch.tensor([len(obs.legal_actions())], dtype=torch.float32)

    def encode_kyoku_start(self, **kwargs):
        return torch.tensor([kwargs["player_id"]], dtype=torch.float32)


def test_split_linear_layer_norm_matches_concat_linear():
    torch.manual_seed(0)
    proj = SplitLinearLayerNorm([3, 5, 2], 7)
    parts = [
        torch.randn(4, 6, 3),
        torch.randn(4, 6, 5),
        torch.randn(4, 6, 2),
    ]

    expected = F.layer_norm(
        F.linear(torch.cat(parts, dim=-1), proj.weight, proj.bias),
        (proj.weight.shape[0],),
        proj.ln.weight,
        proj.ln.bias,
        proj.ln.eps,
    )

    torch.testing.assert_close(proj(parts), expected)


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


def _write_two_kyoku_4p_log(path):
    tehais = [
        ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
        ["1p", "1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "9p", "9p"],
        ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m", "9m"],
        ["E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F", "C"],
    ]
    data = [
        {"type": "start_game", "names": ["A", "B", "C", "D"], "id": "bc_rank_seq_test"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": tehais,
        },
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 2,
            "honba": 1,
            "kyoutaku": 0,
            "oya": 1,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "2m",
            "tehais": tehais,
        },
        {"type": "tsumo", "actor": 1, "pai": "3m"},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    with open(path, "w") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")


def _dummy_sequence_batch(batch_size: int = 2, prog_len: int = 2, cand_len: int = 3) -> dict[str, torch.Tensor]:
    player_stats = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    player_rank_stats = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        dtype=torch.long,
    )
    visible_tile_counts = torch.stack(
        [
            torch.arange(SequenceFeatureEncoder.VISIBLE_TILE_COUNT_TOKENS, dtype=torch.long),
            torch.zeros(SequenceFeatureEncoder.VISIBLE_TILE_COUNT_TOKENS, dtype=torch.long),
        ],
        dim=1,
    )
    batch = {
        "sparse": torch.full((batch_size, SequenceFeatureEncoder.MAX_SPARSE_LEN), SequenceFeatureEncoder.SPARSE_PAD),
        "dealer": torch.zeros(batch_size, dtype=torch.long),
        "player_stats": player_stats.unsqueeze(0).expand(batch_size, -1, -1).clone(),
        "player_rank_stats": player_rank_stats.unsqueeze(0).expand(batch_size, -1, -1).clone(),
        "sparse_melds": torch.tensor(SequenceFeatureEncoder.MELD_PAD).repeat(
            batch_size,
            SequenceFeatureEncoder.MAX_SPARSE_MELDS,
            1,
        ),
        "sparse_meld_owners": torch.full(
            (batch_size, SequenceFeatureEncoder.MAX_SPARSE_MELDS),
            SequenceFeatureEncoder.SPARSE_MELD_OWNER_PAD,
            dtype=torch.long,
        ),
        "hand": torch.tensor(SequenceFeatureEncoder.HAND_PAD).repeat(
            batch_size,
            SequenceFeatureEncoder.MAX_HAND_LEN,
            1,
        ),
        "visible_tile_counts": visible_tile_counts.unsqueeze(0).expand(batch_size, -1, -1).clone(),
        "numeric": torch.zeros(batch_size, SequenceFeatureEncoder.NUM_NUMERIC),
        "agari_overtakes": torch.zeros(batch_size, SequenceFeatureEncoder.AGARI_OVERTAKE_DIM),
        "progression": torch.tensor(SequenceFeatureEncoder.PROG_PAD).repeat(batch_size, prog_len, 1),
        "prog_melds": torch.tensor(SequenceFeatureEncoder.MELD_PAD).repeat(batch_size, prog_len, 1),
        "candidates": torch.tensor(SequenceFeatureEncoder.CAND_PAD).repeat(batch_size, cand_len, 1),
        "cand_melds": torch.tensor(SequenceFeatureEncoder.MELD_PAD).repeat(batch_size, cand_len, 1),
        "sparse_mask": torch.zeros(batch_size, SequenceFeatureEncoder.MAX_SPARSE_LEN, dtype=torch.bool),
        "sparse_meld_mask": torch.zeros(batch_size, SequenceFeatureEncoder.MAX_SPARSE_MELDS, dtype=torch.bool),
        "hand_mask": torch.zeros(batch_size, SequenceFeatureEncoder.MAX_HAND_LEN, dtype=torch.bool),
        "prog_mask": torch.ones(batch_size, prog_len, dtype=torch.bool),
        "cand_mask": torch.ones(batch_size, cand_len, dtype=torch.bool),
    }
    batch["sparse"][:, :4] = torch.tensor(
        [
            1,
            2,
            SequenceFeatureEncoder.SPARSE_KYOKU_OFFSET,
            SequenceFeatureEncoder.SPARSE_TILES_REMAINING_OFFSET + 69,
        ]
    )
    batch["sparse_mask"][:, :4] = True
    return batch


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


def test_behavior_cloning_rank_dataset_adds_non_initial_kyoku_start_samples(tmp_path):
    file_path = tmp_path / "bc_rank_policy_sample.jsonl"
    _write_two_kyoku_4p_log(file_path)

    dataset = BehaviorCloningRankDataset(
        [str(file_path)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=DummyEncoder(),
    )

    samples = list(dataset)
    rank_only = [sample for sample in samples if not sample[4]]
    policy_samples = [sample for sample in samples if sample[4]]

    assert policy_samples
    assert len(rank_only) == 4
    for features, action_id, mask, rank, has_policy in rank_only:
        assert isinstance(features, torch.Tensor)
        assert action_id == 0
        assert len(mask) == 0
        assert 0 <= rank < 4
        assert has_policy is False


def test_behavior_cloning_dataset_rejects_invalid_shuffle_buffer_files(tmp_path):
    file_path = tmp_path / "bc_policy_sample.jsonl"
    _write_simple_4p_log(file_path)

    with pytest.raises(ValueError, match="shuffle_buffer_files"):
        BehaviorCloningDataset(
            [str(file_path)],
            is_train=True,
            n_players=4,
            replay_rule="tenhou",
            encoder=DummyEncoder(),
            shuffle_buffer_files=0,
        )


def test_behavior_cloning_dataset_buffers_configured_number_of_files_before_yield(monkeypatch):
    monkeypatch.setattr("riichienv_ml.datasets.mjai_logs.random.shuffle", lambda values: None)

    class DummyBehaviorCloningDataset(BehaviorCloningDataset):
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
            return [(file_path, 0, torch.ones(1, dtype=torch.uint8))]

    dataset = DummyBehaviorCloningDataset()
    next(iter(dataset))

    assert dataset.loaded_files == ["a", "b"]


def test_behavior_cloning_dataset_flushes_remaining_files(monkeypatch):
    monkeypatch.setattr("riichienv_ml.datasets.mjai_logs.random.shuffle", lambda values: None)

    class DummyBehaviorCloningDataset(BehaviorCloningDataset):
        def __init__(self):
            super().__init__(
                ["a", "b"],
                is_train=True,
                n_players=4,
                replay_rule="tenhou",
                encoder=None,
                shuffle_buffer_files=3,
            )

        def _load_file_samples(self, file_path: str):
            return [(file_path, 0, torch.ones(1, dtype=torch.uint8))]

    samples = list(DummyBehaviorCloningDataset())

    assert [sample_id for sample_id, _action, _mask in samples] == ["a", "b"]


def test_behavior_cloning_dataset_validation_keeps_file_order(monkeypatch):
    def fail_shuffle(_values):
        raise AssertionError("validation should not shuffle files or samples")

    monkeypatch.setattr("riichienv_ml.datasets.mjai_logs.random.shuffle", fail_shuffle)

    class DummyBehaviorCloningDataset(BehaviorCloningDataset):
        def __init__(self):
            super().__init__(
                ["a", "b", "c"],
                is_train=False,
                n_players=4,
                replay_rule="tenhou",
                encoder=None,
                shuffle_buffer_files=2,
            )

        def _load_file_samples(self, file_path: str):
            return [(file_path, 0, torch.ones(1, dtype=torch.uint8))]

    samples = list(DummyBehaviorCloningDataset())

    assert [sample_id for sample_id, _action, _mask in samples] == ["a", "b", "c"]


def test_bc_policy_trainer_passes_shuffle_buffer_files_to_compatible_dataset():
    class CompatibleDataset:
        def __init__(self, files, *, is_train, n_players, replay_rule, encoder, shuffle_buffer_files=1):
            self.files = files
            self.is_train = is_train
            self.n_players = n_players
            self.replay_rule = replay_rule
            self.encoder = encoder
            self.shuffle_buffer_files = shuffle_buffer_files

    trainer = object.__new__(bc_policy_module.BCPolicyTrainer)
    trainer.n_players = 4
    trainer.replay_rule = "tenhou"
    trainer.shuffle_buffer_files = 4
    trainer.dataset_class = "tests.CompatibleDataset"

    dataset = trainer._create_dataset(CompatibleDataset, ["a"], is_train=True, encoder="encoder")

    assert dataset.files == ["a"]
    assert dataset.is_train is True
    assert dataset.n_players == 4
    assert dataset.replay_rule == "tenhou"
    assert dataset.encoder == "encoder"
    assert dataset.shuffle_buffer_files == 4


def test_replay_sequence_progression_starts_with_initial_dora(tmp_path):
    file_path = tmp_path / "bc_policy_sample.jsonl"
    _write_simple_4p_log(file_path)

    replay = MjaiReplay.from_jsonl(str(file_path), rule="tenhou")
    kyoku = next(iter(replay.take_kyokus()))
    obs, _action = next(iter(kyoku.steps(0)))

    features = SequenceFeatureEncoder().encode(obs)
    progression = features["progression"][features["prog_mask"]]

    assert progression[:2].tolist() == [
        [4, 0, 2, 2, 4],
        [4, 44, 2, 2, 4],
    ]


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
    x = _dummy_sequence_batch(batch_size=2, prog_len=8, cand_len=4)

    logits = model(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 82)


def test_transformer_policy_network_can_emit_rank_logits():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        policy_head_type="cls",
        emit_rank=True,
        max_prog_len=8,
        max_cand_len=4,
    )
    x = _dummy_sequence_batch(batch_size=2, prog_len=8, cand_len=4)

    logits, rank_logits = model(x)

    assert logits.shape == (2, 82)
    assert rank_logits.shape == (2, 4)


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
    x = _dummy_sequence_batch(batch_size=2, prog_len=8, cand_len=4)

    logits, value = model(x)

    assert logits.shape == (2, 82)
    assert value.shape == (2,)


def test_sequence_feature_packed_encoder_matches_transformer_policy_input(tmp_path):
    file_path = tmp_path / "bc_sequence_sample.jsonl"
    _write_simple_4p_log(file_path)

    replay = MjaiReplay.from_jsonl(str(file_path), rule="tenhou")
    kyoku = list(replay.take_kyokus())[0]
    obs, _ = next(iter(kyoku.steps(0)))

    encoder = SequenceFeaturePackedEncoder()
    sample = encoder.encode(obs)
    dict_features = collate_sequence_features([sample])
    features = pack_sequence_features([sample])

    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_prog_len=256,
        max_cand_len=32,
    )
    model.eval()

    logits = model(features)
    dict_logits = model(dict_features)

    assert logits.shape == (1, sample["candidates"].shape[0])
    assert packed_candidate_mask(features).shape == (1, sample["candidates"].shape[0])
    assert torch.allclose(logits, dict_logits, atol=1e-6)


def test_transformer_policy_training_step_with_sequence_features(tmp_path):
    file_path = tmp_path / "bc_sequence_train_sample.jsonl"
    _write_simple_4p_log(file_path)

    encoder = SequenceFeaturePackedEncoder()
    dataset = BehaviorCloningDataset(
        [str(file_path)],
        is_train=False,
        n_players=4,
        replay_rule="tenhou",
        encoder=encoder,
    )
    features, action_id, mask = next(iter(dataset))

    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=256,
        max_cand_len=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_features = pack_sequence_features([features])
    logits = model(batch_features)
    legal = packed_candidate_mask(batch_features)
    assert torch.equal(legal.cpu(), torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0))
    loss = F.cross_entropy(logits.masked_fill(~legal, torch.finfo(logits.dtype).min), torch.tensor([action_id]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_sequence_feature_encoder_counts_hand_tiles_and_adds_drawn_tile_token():
    obs = Observation(
        0,
        [[0, 1, 4], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [],
        ['{"type":"tsumo","actor":0,"pai":"1m"}'],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)
    hand = features["hand"]
    hand_mask = features["hand_mask"]

    assert hand.shape == (SequenceFeatureEncoder.MAX_HAND_LEN, len(SequenceFeatureEncoder.HAND_DIMS))
    assert hand[: SequenceFeatureEncoder.HAND_COUNT_TOKENS, 0].tolist() == list(
        range(SequenceFeatureEncoder.HAND_COUNT_TOKENS)
    )
    assert hand[1].tolist() == [1, 2]  # two 1m copies
    assert hand[2].tolist() == [2, 1]  # one 2m copy
    assert hand[SequenceFeatureEncoder.HAND_COUNT_TOKENS].tolist() == [
        1,
        SequenceFeatureEncoder.HAND_DRAWN_TOKEN_AUX,
    ]
    assert hand_mask[: SequenceFeatureEncoder.HAND_COUNT_TOKENS].all().item()
    assert hand_mask[SequenceFeatureEncoder.HAND_COUNT_TOKENS].item()


def test_sequence_feature_encoder_masks_absent_drawn_tile_token():
    obs = Observation(
        0,
        [[0, 4, 8], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [],
        [],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)
    hand = features["hand"]
    hand_mask = features["hand_mask"]

    assert hand[SequenceFeatureEncoder.HAND_COUNT_TOKENS].tolist() == list(SequenceFeatureEncoder.HAND_PAD)
    assert hand_mask[: SequenceFeatureEncoder.HAND_COUNT_TOKENS].all().item()
    assert not hand_mask[SequenceFeatureEncoder.HAND_COUNT_TOKENS].item()


def test_sequence_numeric_features_use_current_normalized_scores_only():
    obs = Observation(
        2,
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [35000, 15000, 25000, 42000],
        [False, False, False, False],
        [],
        ['{"type":"start_kyoku","honba":9,"kyotaku":8,"scores":[1000,2000,3000,4000]}'],
        3,
        2,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)

    assert features["numeric"].shape == (6,)
    assert torch.allclose(features["numeric"], torch.tensor([3.0, 2.0, 0.0, 1.7, 1.0, -1.0]))


def test_sequence_feature_encoder_includes_pairwise_agari_overtake_flags():
    obs = Observation(
        2,
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [25000, 24000, 24000, 24000],
        [False, False, False, False],
        [],
        [],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        None,
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)
    overtakes = features["agari_overtakes"].reshape(4, 96, 4)

    assert features["agari_overtakes"].shape == (SequenceFeatureEncoder.AGARI_OVERTAKE_DIM,)
    assert overtakes.shape == SequenceFeatureEncoder.AGARI_OVERTAKE_DIMS
    assert overtakes[1, 24, 2].item() == 1.0
    assert overtakes[1, 24, 1].item() == 0.0


def test_sequence_feature_encoder_separates_dealer_from_sparse_vocab():
    obs = Observation(
        2,
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [],
        [],
        0,
        0,
        1,
        1,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        None,
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)
    valid_sparse = features["sparse"][features["sparse_mask"]]

    assert features["dealer"].item() == 3
    assert features["sparse"].shape == (SequenceFeatureEncoder.MAX_SPARSE_LEN,)
    assert valid_sparse.tolist() == [
        1,
        3,
        SequenceFeatureEncoder.SPARSE_KYOKU_OFFSET,
        SequenceFeatureEncoder.SPARSE_TILES_REMAINING_OFFSET + 69,
    ]


def test_sequence_feature_encoder_encodes_kyoku_start_without_hand_dora_or_candidates():
    features = SequenceFeatureEncoder().encode_kyoku_start(
        scores=[30000, 25000, 25000, 20000],
        chang=1,
        ju=2,
        ben=3,
        liqibang=1,
        player_id=3,
    )
    valid_sparse = features["sparse"][features["sparse_mask"]]

    assert valid_sparse.tolist() == [
        1,
        3,
        SequenceFeatureEncoder.SPARSE_KYOKU_OFFSET + 2,
        SequenceFeatureEncoder.SPARSE_TILES_REMAINING_OFFSET + 69,
    ]
    assert features["dealer"].item() == 3
    assert features["player_rank_stats"].tolist() == [
        [0, 3, 1],
        [1, 0, 2],
        [2, 1, 3],
        [3, 2, 0],
    ]
    assert not features["hand_mask"].any()
    assert not features["sparse_meld_mask"].any()
    assert features["progression"].shape == (0, len(SequenceFeatureEncoder.PROG_DIMS))
    assert features["candidates"].shape == (0, SequenceFeatureEncoder.CAND_WIDTH)
    assert features["visible_tile_counts"][:, 1].sum().item() == 0
    torch.testing.assert_close(
        features["numeric"],
        torch.tensor([3.0, 1.0, -0.5, 0.5, 0.0, 0.0], dtype=torch.float32),
    )


def test_sequence_feature_encoder_includes_player_stats():
    events = []
    events.extend({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False} for _ in range(2))
    events.append({"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True})
    events.extend({"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": False} for _ in range(2))
    events.extend({"type": "dahai", "actor": 3, "pai": "4m", "tsumogiri": False} for _ in range(21))

    obs = Observation(
        2,
        [[], [], [], []],
        [
            [Meld(MeldType.Pon, [108, 109, 110], True, 1, 108)],
            [],
            [],
            [
                Meld(MeldType.Pon, [112, 113, 114], True, 2, 112),
                Meld(MeldType.Kakan, [116, 117, 118, 119], True, 1, 116),
            ],
        ],
        [[0, 4, 8], [12, 16], [], list(range(21))],
        [],
        [25000, 25000, 25000, 25000],
        [False, True, False, False],
        [],
        [json.dumps(event) for event in events],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        None,
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)

    assert features["player_stats"].shape == (
        SequenceFeatureEncoder.PLAYER_INFO_TOKENS,
        SequenceFeatureEncoder.PLAYER_INFO_WIDTH,
    )
    assert features["player_stats"].tolist() == [
        [0, 0, 0, 0, 0],
        [1, 0, 2, 20, 20],
        [2, 0, 1, 3, 2],
        [3, 1, 0, 2, 2],
    ]
    assert features["player_rank_stats"].shape == (
        SequenceFeatureEncoder.PLAYER_RANK_STATS_TOKENS,
        SequenceFeatureEncoder.PLAYER_RANK_STATS_WIDTH,
    )
    assert features["player_rank_stats"].tolist() == [
        [0, 2, 2],
        [1, 3, 3],
        [2, 0, 0],
        [3, 1, 1],
    ]


def test_sequence_feature_encoder_includes_visible_tile_counts_by_tile37():
    obs = Observation(
        0,
        [[16, 17], [], [], []],
        [[], [Meld(MeldType.Pon, [0, 1, 2], True, 0, 0)], [], []],
        [[], [], [0], []],
        [4],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [],
        [],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        None,
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)
    counts = features["visible_tile_counts"]

    assert counts.shape == (
        SequenceFeatureEncoder.VISIBLE_TILE_COUNT_TOKENS,
        SequenceFeatureEncoder.VISIBLE_TILE_COUNT_WIDTH,
    )
    assert counts[:, 0].tolist() == list(range(SequenceFeatureEncoder.VISIBLE_TILE_COUNT_TOKENS))
    assert counts[0].tolist() == [0, 1]  # red 5m in self hand
    assert counts[1].tolist() == [1, 3]  # discard + two consumed pon tiles; called tile skipped in meld
    assert counts[2].tolist() == [2, 1]  # visible dora indicator 2m
    assert counts[5].tolist() == [5, 1]  # normal 5m in self hand
    assert counts[:, 1].sum().item() == 6


def test_sequence_progression_includes_dora_reveals():
    obs = Observation(
        0,
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [],
        [
            '{"type":"start_kyoku","dora_marker":"5m"}',
            '{"type":"dora","dora_marker":"E"}',
        ],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        None,
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)
    progression = features["progression"][features["prog_mask"]]

    assert progression.tolist() == [
        [4, 0, 2, 2, 4],
        [4, 48, 2, 2, 4],
        [4, 73, 2, 2, 4],
    ]


def test_sequence_candidates_distinguish_red_and_moqie_discards():
    obs = Observation(
        0,
        [[16, 17, 18, 19], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [
            Action(ActionType.DISCARD, 16, []),
            Action(ActionType.DISCARD, 17, []),
            Action(ActionType.RIICHI, None, []),
        ],
        ['{"type":"tsumo","actor":0,"pai":"5m"}'],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        17,
    )

    candidates = obs.candidate_actions()
    assert len(candidates) == 3
    assert [a.tile for a in candidates if a.action_type == ActionType.DISCARD] == [16, 17]
    assert obs.find_candidate_action(obs.find_candidate_index(Action(ActionType.DISCARD, 16, []))).tile == 16
    assert obs.find_candidate_action(obs.find_candidate_index(Action(ActionType.DISCARD, 17, []))).tile == 17

    features = SequenceFeatureEncoder().encode(obs)
    valid = features["candidates"][features["cand_mask"]]
    assert valid.tolist() == [[0, 0, 0], [5, 1, 0], [37, 2, 0]]


def test_sequence_candidates_distinguish_red_consumed_melds():
    obs = Observation(
        1,
        [[], [16, 17, 18], [], []],
        [[], [], [], []],
        [[19], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [
            Action(ActionType.PON, 19, [17, 18]),
            Action(ActionType.PON, 19, [16, 17]),
        ],
        ['{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}'],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
    )

    candidates = obs.candidate_actions()
    assert len(candidates) == 2
    assert [a.action_type for a in candidates] == [ActionType.PON, ActionType.PON]
    assert [a.consume_tiles for a in candidates] == [[17, 18], [16, 17]]

    features = SequenceFeatureEncoder().encode(obs)
    valid = features["candidates"][features["cand_mask"]]
    assert valid.tolist() == [[44, 2, 3], [44, 2, 3]]
    valid_melds = features["cand_melds"][features["cand_mask"]]
    assert valid_melds.tolist() == [
        [_MELD_KIND_PON, 5, _MELD_ROLE_CALLED, 5, _MELD_ROLE_CONSUMED, 5, _MELD_ROLE_CONSUMED, 37, 3],
        [_MELD_KIND_PON, 5, _MELD_ROLE_CALLED, 0, _MELD_ROLE_CONSUMED, 5, _MELD_ROLE_CONSUMED, 37, 3],
    ]


def test_sequence_response_candidates_use_last_discard_actor_without_events():
    obs = Observation(
        1,
        [[], [16, 17, 18], [], []],
        [[], [], [], []],
        [[19], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [
            Action(ActionType.PON, 19, [17, 18]),
            Action(ActionType.RON, 19, []),
            Action(ActionType.PASS, None, []),
        ],
        [],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        19,
        last_discard_actor=0,
    )

    candidates = obs.candidate_actions()

    assert [a.action_type for a in candidates] == [ActionType.PON, ActionType.RON, ActionType.PASS]
    features = SequenceFeatureEncoder().encode(obs)
    valid = features["candidates"][features["cand_mask"]]
    assert valid.tolist() == [[44, 2, 3], [46, 2, 3], [42, 2, 0]]


def test_mjai_replay_claim_labels_remain_strict_candidate_actions(tmp_path):
    file_path = tmp_path / "claim_sample.jsonl"
    data = [
        {"type": "start_game", "names": ["A", "B", "C", "D"], "id": "claim_sample"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyoutaku": 0,
            "oya": 1,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["E", "E", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "9p", "9p"],
                ["E", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m", "9m"],
                ["1s", "1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "9s", "9s"],
                ["P", "P", "F", "F", "C", "C", "S", "S", "W", "W", "N", "N", "1p"],
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
        {"type": "pon", "actor": 0, "target": 1, "pai": "E", "consumed": ["E", "E"]},
        {"type": "ryukyoku", "reason": "test"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    with file_path.open("w", encoding="utf-8") as f:
        for event in data:
            f.write(json.dumps(event) + "\n")

    replay = MjaiReplay.from_jsonl(str(file_path), rule="tenhou")
    kyoku = next(iter(replay.take_kyokus()))
    pon_steps = [(obs, action) for obs, action in kyoku.steps(0) if action.action_type == ActionType.PON]

    assert pon_steps
    obs, action = pon_steps[0]
    assert len(action.consume_tiles) == 2
    assert obs.find_candidate_index(action) is not None


def test_transformer_decodes_current_dora_tiles_from_sparse_tokens():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )
    sparse = torch.tensor(
        [[_SPARSE_DORA_OFFSET + 4, _SPARSE_DORA_OFFSET + 36, SequenceFeatureEncoder.SPARSE_PAD]],
        dtype=torch.long,
    )

    dora_tile34 = model._decode_current_dora_tiles(sparse)

    assert dora_tile34.tolist() == [[4, 31, _TILE34_PAD]]


def test_transformer_tile_only_action_type_lookups_ignore_meld_patterns():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )

    assert model.prog_type_action_kind[1 + 5].item() == _ACTION_KIND_DISCARD
    assert model.prog_type_action_kind[38].item() == _ACTION_KIND_PAD
    assert model.prog_type_action_kind[39].item() == _ACTION_KIND_PAD
    assert model.prog_type_action_kind[40].item() == _ACTION_KIND_PAD
    assert model.prog_type_action_kind[41].item() == _ACTION_KIND_PAD
    assert model.prog_type_action_kind[42].item() == _ACTION_KIND_PAD
    assert model.prog_type_action_kind[43].item() == _ACTION_KIND_DORA
    assert model.prog_type_tile37[43].item() == 0
    assert model.prog_type_action_kind[79].item() == _ACTION_KIND_DORA
    assert model.prog_type_tile37[79].item() == 36

    assert model.cand_type_action_kind[0].item() == _ACTION_KIND_DISCARD
    assert model.cand_type_tile37[0].item() == 0
    assert model.cand_type_action_kind[5].item() == _ACTION_KIND_DISCARD
    assert model.cand_type_tile37[5].item() == 5
    assert model.cand_type_tile34[5].item() == _TILE34_PAD
    assert model.cand_type_action_kind[36].item() == _ACTION_KIND_DISCARD
    assert model.cand_type_action_kind[37].item() == _ACTION_KIND_PAD
    assert model.cand_type_action_kind[38].item() == _ACTION_KIND_PAD
    assert model.cand_type_action_kind[42].item() == _ACTION_KIND_PAD
    assert model.cand_type_action_kind[43].item() == _ACTION_KIND_PAD
    assert model.cand_type_action_kind[44].item() == _ACTION_KIND_PAD


def test_transformer_progression_discard_clock_uses_exclusive_prefix():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )
    progression = torch.tensor(
        [
            [
                [0, 1, 0, 0, 4],  # discard s
                [1, 38, 2, 2, 0],  # non-discard a
                [4, 43, 2, 2, 4],  # non-discard b
                [2, 39, 2, 2, 1],  # non-discard c
                [3, 2, 0, 0, 4],  # discard t
            ]
        ],
        dtype=torch.long,
    )
    prog_mask = torch.ones(1, 5, dtype=torch.bool)

    clock, total, player_clock, player_total = model._progression_discard_clock(progression, prog_mask)
    dist = clock[:, None, :] - clock[:, :, None]

    assert clock.tolist() == [[0, 1, 1, 1, 1]]
    assert total.tolist() == [2]
    assert player_clock.tolist() == [
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ]
    assert player_total.tolist() == [[1, 0, 0, 1]]
    assert dist[0, 1, 0].item() == -1  # a -> s
    assert dist[0, 0, 1].item() == 1  # s -> a
    assert dist[0, 1, 4].item() == 0  # a -> t
    assert dist[0, 4, 1].item() == 0  # t -> a
    assert torch.equal(torch.diagonal(dist[0]), torch.zeros(5, dtype=torch.long))


def test_transformer_progression_player_discard_clock_uses_actor_exclusive_prefix():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )
    progression = torch.tensor(
        [
            [
                [0, 1, 0, 0, 4],  # actor 0 discard
                [1, 38, 2, 2, 0],  # actor 1 call
                [1, 2, 0, 0, 4],  # actor 1 discard
                [2, 3, 0, 0, 4],  # actor 2 discard
                [1, 4, 0, 0, 4],  # actor 1 later discard
            ]
        ],
        dtype=torch.long,
    )
    prog_mask = torch.ones(1, 5, dtype=torch.bool)

    clock, total, player_clock, player_total = model._progression_discard_clock(progression, prog_mask)

    assert clock.tolist() == [[0, 1, 1, 2, 3]]
    assert total.tolist() == [4]
    assert player_clock.tolist() == [
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ]
    ]
    assert player_total.tolist() == [[1, 2, 1, 0]]
    actor1_clock = player_clock[:, :, 1]
    actor1_dist = actor1_clock[:, None, :] - actor1_clock[:, :, None]
    assert actor1_dist[0, 1, 2].item() == 0  # call -> its following discard
    assert actor1_dist[0, 1, 4].item() == 1  # call -> actor 1 later discard


def test_transformer_progression_player_discard_clock_handles_batched_actor_timelines():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )
    progression = torch.tensor(
        [
            [
                [0, 1, 0, 0, 4],
                [1, 2, 0, 0, 4],
                [1, 3, 0, 0, 4],
            ],
            [
                [2, 1, 0, 0, 4],
                [2, 2, 0, 0, 4],
                [0, 3, 0, 0, 4],
            ],
        ],
        dtype=torch.long,
    )
    prog_mask = torch.ones(2, 3, dtype=torch.bool)

    clock, total, player_clock, player_total = model._progression_discard_clock(progression, prog_mask)

    assert clock.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert total.tolist() == [3, 3]
    assert player_total.tolist() == [[1, 2, 0, 0], [1, 0, 2, 0]]
    assert player_clock.tolist() == [
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 0],
        ],
    ]


def test_transformer_progression_attention_bias_targets_progression_keys_only():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        progression_relative_position_cap=3,
        max_prog_len=8,
        max_cand_len=4,
    )
    with torch.no_grad():
        model.progression_relative_bias.zero_()
        model.progression_recency_bias.zero_()
        model.progression_player_relative_bias.zero_()
        model.progression_player_recency_bias.zero_()
        model.progression_relative_bias[0] = torch.arange(7, dtype=torch.float32)
        model.progression_recency_bias[0] = 100 + torch.arange(4, dtype=torch.float32)

    progression = torch.tensor(
        [
            [
                [0, 1, 0, 0, 4],
                [1, 38, 2, 2, 0],
                [4, 43, 2, 2, 4],
                [2, 39, 2, 2, 1],
                [3, 2, 0, 0, 4],
            ]
        ],
        dtype=torch.long,
    )
    prog_mask = torch.ones(1, 5, dtype=torch.bool)

    bias = model._progression_attention_bias(
        progression,
        prog_mask,
        total_len=8,
        prog_offset=2,
        dtype=torch.float32,
    )
    assert bias is not None
    head0 = bias[0]

    assert head0[3, 2].item() == 2  # progression query a -> progression key s, bucket -1
    assert head0[2, 3].item() == 4  # progression query s -> progression key a, bucket +1
    assert head0[3, 6].item() == 3  # progression query a -> progression key t, bucket 0
    assert head0[0, 3].item() == 102  # non-progression query -> progression key a, recency bucket -1
    assert head0[2, 0].item() == 0  # progression query -> non-progression key has no bias
    assert head0[0, 1].item() == 0  # non-progression pairs have no bias


def test_transformer_progression_attention_bias_adds_key_actor_player_clock():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        progression_relative_position_cap=3,
        progression_player_relative_position_cap=2,
        max_prog_len=8,
        max_cand_len=4,
    )
    with torch.no_grad():
        model.progression_relative_bias.zero_()
        model.progression_recency_bias.zero_()
        model.progression_player_relative_bias.zero_()
        model.progression_player_recency_bias.zero_()
        model.progression_player_relative_bias[0] = 10 + torch.arange(5, dtype=torch.float32)
        model.progression_player_recency_bias[0] = 100 + torch.arange(3, dtype=torch.float32)

    progression = torch.tensor(
        [
            [
                [0, 1, 0, 0, 4],  # actor 0 discard
                [1, 38, 2, 2, 0],  # actor 1 call
                [1, 2, 0, 0, 4],  # actor 1 discard
                [2, 3, 0, 0, 4],  # actor 2 discard
                [1, 4, 0, 0, 4],  # actor 1 later discard
                [4, 43, 2, 2, 4],  # actorless dora reveal
            ]
        ],
        dtype=torch.long,
    )
    prog_mask = torch.ones(1, 6, dtype=torch.bool)

    bias = model._progression_attention_bias(
        progression,
        prog_mask,
        total_len=8,
        prog_offset=2,
        dtype=torch.float32,
    )
    assert bias is not None
    head0 = bias[0]

    assert head0[3, 4].item() == 12  # actor 1 call -> following actor 1 discard, bucket 0
    assert head0[3, 6].item() == 13  # actor 1 call -> later actor 1 discard, bucket +1
    assert head0[6, 3].item() == 11  # later actor 1 discard -> actor 1 call, bucket -1
    assert head0[3, 2].item() == 11  # actor 1 call -> actor 0 discard, actor 0 clock bucket -1
    assert head0[3, 7].item() == 0  # actorless dora key receives no player-specific bias
    assert head0[0, 6].item() == 101  # non-progression query uses actor 1 recency bucket -1


def test_transformer_progression_attention_bias_zeroes_padded_keys():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        progression_relative_position_cap=3,
        max_prog_len=8,
        max_cand_len=4,
    )
    with torch.no_grad():
        model.progression_relative_bias.fill_(1.0)
        model.progression_recency_bias.fill_(1.0)
        model.progression_player_relative_bias.fill_(1.0)
        model.progression_player_recency_bias.fill_(1.0)

    progression = torch.tensor(
        [
            [
                [0, 1, 0, 0, 4],
                [1, 38, 2, 2, 0],
                [3, 2, 0, 0, 4],
            ]
        ],
        dtype=torch.long,
    )
    prog_mask = torch.tensor([[True, True, False]])

    bias = model._progression_attention_bias(
        progression,
        prog_mask,
        total_len=6,
        prog_offset=2,
        dtype=torch.float32,
    )
    assert bias is not None
    assert torch.count_nonzero(bias[:, :, 4]).item() == 0


def test_transformer_relative_position_forward_handles_padding_and_empty_progression():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        max_prog_len=8,
        max_cand_len=4,
    )
    padded = _dummy_sequence_batch(batch_size=2, prog_len=5, cand_len=3)
    padded["progression"][:, :, 1] = torch.tensor([1, 38, 2, 39, 2])
    padded["prog_mask"][:, 3:] = False

    logits = model(padded)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()

    empty = _dummy_sequence_batch(batch_size=2, prog_len=0, cand_len=3)
    empty_logits = model(empty)
    assert empty_logits.shape == (2, 3)
    assert torch.isfinite(empty_logits).all()


def test_transformer_progression_bias_disables_mha_fastpath_temporarily(monkeypatch):
    mha_backend = getattr(torch.backends, "mha", None)
    if (
        mha_backend is None
        or not hasattr(mha_backend, "get_fastpath_enabled")
        or not hasattr(mha_backend, "set_fastpath_enabled")
    ):
        pytest.skip("torch.backends.mha fastpath controls are unavailable")

    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        max_prog_len=8,
        max_cand_len=4,
    )
    features = _dummy_sequence_batch(batch_size=2, prog_len=5, cand_len=3)
    seen_fastpath_states = []
    original_forward = model.transformer.forward

    def wrapped_forward(*args, **kwargs):
        seen_fastpath_states.append(mha_backend.get_fastpath_enabled())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.transformer, "forward", wrapped_forward)

    previous = bool(mha_backend.get_fastpath_enabled())
    mha_backend.set_fastpath_enabled(True)
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(features)
    finally:
        restored = bool(mha_backend.get_fastpath_enabled())
        mha_backend.set_fastpath_enabled(previous)

    assert logits.shape == (2, 3)
    assert seen_fastpath_states == [False]
    assert restored is True


def test_transformer_embedding_padding_rows_remain_zero_after_custom_init():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )

    checked = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
            checked.append(name)
            padding_row = module.weight[module.padding_idx]
            assert torch.count_nonzero(padding_row).item() == 0, name

    assert checked


def test_transformer_relative_seat_embedding_shares_base_and_zeroes_special_values():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=8,
        max_cand_len=4,
    )
    seats = torch.tensor([[0, 1, 4]], dtype=torch.long)

    actor_emb = model.relative_seat_embed(seats, _SEAT_ROLE_PROG_ACTOR, out="other")
    owner_emb = model.relative_seat_embed(seats, _SEAT_ROLE_MELD_OWNER, out="model")
    player_emb = model.relative_seat_embed(seats, _SEAT_ROLE_PLAYER_INFO, out="other")

    assert actor_emb.shape == (1, 3, 8)
    assert owner_emb.shape == (1, 3, 64)
    assert player_emb.shape == (1, 3, 8)
    assert model.relative_seat_embed.base_embed.weight.shape[0] == 4
    assert torch.count_nonzero(actor_emb[:, :2]).item() > 0
    assert torch.count_nonzero(owner_emb[:, :2]).item() > 0
    assert torch.count_nonzero(player_emb[:, :2]).item() > 0
    assert torch.allclose(actor_emb[:, 2], torch.zeros_like(actor_emb[:, 2]))
    assert torch.allclose(owner_emb[:, 2], torch.zeros_like(owner_emb[:, 2]))
    assert torch.allclose(player_emb[:, 2], torch.zeros_like(player_emb[:, 2]))


def test_transformer_embeds_agari_overtakes_as_winner_seat_tokens():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=8,
        max_cand_len=4,
    )
    agari = torch.zeros(2, SequenceFeatureEncoder.AGARI_OVERTAKE_DIM, dtype=torch.float32)
    emb = model._embed_agari_overtakes(agari)

    assert emb.shape == (2, SequenceFeatureEncoder.AGARI_OVERTAKE_TOKENS, 64)
    assert model.agari_overtake_proj[0].in_features == SequenceFeatureEncoder.AGARI_OVERTAKE_TOKEN_DIM
    assert model.progression_relative_bias.shape == (4, 121)
    assert model.progression_recency_bias.shape == (4, 61)
    assert model.progression_player_relative_bias.shape == (4, 31)
    assert model.progression_player_recency_bias.shape == (4, 16)
    assert torch.count_nonzero(model.progression_relative_bias).item() == 0
    assert torch.count_nonzero(model.progression_recency_bias).item() == 0
    assert torch.count_nonzero(model.progression_player_relative_bias).item() == 0
    assert torch.count_nonzero(model.progression_player_recency_bias).item() == 0

    winner_seats = torch.arange(SequenceFeatureEncoder.AGARI_OVERTAKE_TOKENS).unsqueeze(0).expand(2, -1)
    seat_emb = model.relative_seat_embed(winner_seats, _SEAT_ROLE_AGARI_WINNER, out="model")
    torch.testing.assert_close(emb, seat_emb)


def test_transformer_factorized_meld_embedding_uses_slot_padding():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )
    sparse = torch.tensor([[SequenceFeatureEncoder.SPARSE_PAD]], dtype=torch.long)
    dora_tile34 = model._decode_current_dora_tiles(sparse)
    melds = torch.tensor(
        [
            [
                [_MELD_KIND_CHI, 1, _MELD_ROLE_CALLED, 2, _MELD_ROLE_CONSUMED, 3, _MELD_ROLE_CONSUMED, 37, 3],
                [_MELD_KIND_PAD, 37, 3, 37, 3, 37, 3, 37, 3],
            ]
        ],
        dtype=torch.long,
    )

    emb = model._embed_sparse_melds(melds, dora_tile34)

    assert emb.shape == (1, 2, 64)
    assert torch.count_nonzero(emb[:, 0]) > 0
    assert torch.allclose(emb[:, 1], torch.zeros_like(emb[:, 1]))


def test_sequence_sparse_melds_include_relative_owner_sidecar():
    obs = Observation(
        2,
        [[], [], [], []],
        [
            [Meld(MeldType.Pon, [108, 109, 110], True, 1, 108)],
            [],
            [Meld(MeldType.Chi, [36, 40, 44], True, 1, 36)],
            [Meld(MeldType.Pon, [112, 113, 114], True, 2, 112)],
        ],
        [[], [], [], []],
        [],
        [25000, 25000, 25000, 25000],
        [False, False, False, False],
        [],
        [],
        0,
        0,
        0,
        0,
        0,
        [],
        False,
        [None, None, None, None],
        [None, None, None, None],
        None,
        None,
        None,
    )

    features = SequenceFeatureEncoder().encode(obs)

    assert features["sparse_melds"].shape == (16, 9)
    assert features["sparse_meld_owners"].shape == (16,)
    assert features["sparse_meld_mask"][:3].tolist() == [True, True, True]
    assert features["sparse_meld_mask"][3:].any().item() is False
    assert features["sparse_meld_owners"][:4].tolist() == [0, 1, 2, 4]
    assert features["sparse_melds"][:3, 0].tolist() == [_MELD_KIND_CHI, _MELD_KIND_PON, _MELD_KIND_PON]


def test_transformer_shared_tile_embedding_marks_red_and_tile34_unknown_red():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
    )

    assert model.tile_embed.tile37_red_flag[0].item() == _RED_FLAG_RED
    assert model.tile_embed.tile34_red_flag[4].item() == _RED_FLAG_PAD


def test_transformer_tile_embedding_tables_match_direct_embedding():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=8,
        max_cand_len=4,
    )
    sparse = torch.tensor(
        [[_SPARSE_DORA_OFFSET + 4, _SPARSE_DORA_OFFSET + 36, SequenceFeatureEncoder.SPARSE_PAD]],
        dtype=torch.long,
    )
    dora_tile34 = model._decode_current_dora_tiles(sparse)
    dealer = torch.tensor([3], dtype=torch.long)
    round_wind = torch.tensor([1], dtype=torch.long)
    tile37_table, tile34_table = model.tile_embed.build_tables(
        dora_tile34,
        dealer,
        round_wind,
        model.relative_seat_embed,
    )

    tile37 = torch.tensor([[0, 5, 37]], dtype=torch.long)
    tile34 = torch.tensor([[4, 31, 34]], dtype=torch.long)

    torch.testing.assert_close(
        model.tile_embed.embed_tile37_from_table(tile37, tile37_table),
        model.tile_embed.embed_tile37(tile37, dora_tile34, dealer, round_wind, model.relative_seat_embed),
    )
    torch.testing.assert_close(
        model.tile_embed.embed_tile34_from_table(tile34, tile34_table),
        model.tile_embed.embed_tile34(tile34, dora_tile34, dealer, round_wind, model.relative_seat_embed),
    )


def test_transformer_visible_tile_count_embedding_uses_shared_tile_table():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=8,
        max_cand_len=4,
    )
    sparse = torch.tensor([[_SPARSE_DORA_OFFSET + 4, SequenceFeatureEncoder.SPARSE_PAD]], dtype=torch.long)
    dora_tile34 = model._decode_current_dora_tiles(sparse)
    dealer = torch.tensor([3], dtype=torch.long)
    round_wind = torch.tensor([1], dtype=torch.long)
    tile37_table, _ = model.tile_embed.build_tables(
        dora_tile34,
        dealer,
        round_wind,
        model.relative_seat_embed,
    )
    visible_counts = torch.tensor([[[0, 1], [5, 2], [36, 0]]], dtype=torch.long)

    table_emb = model._embed_visible_tile_counts(visible_counts, dora_tile34, tile37_table, dealer, round_wind)
    direct_emb = model._embed_visible_tile_counts(visible_counts, dora_tile34, None, dealer, round_wind)

    assert table_emb.shape == (1, 3, 64)
    torch.testing.assert_close(table_emb, direct_emb)


def test_transformer_hand_embedding_uses_shared_tile_table_and_drawn_marker():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=8,
        max_cand_len=4,
    )
    sparse = torch.tensor([[_SPARSE_DORA_OFFSET + 4, SequenceFeatureEncoder.SPARSE_PAD]], dtype=torch.long)
    dora_tile34 = model._decode_current_dora_tiles(sparse)
    dealer = torch.tensor([3], dtype=torch.long)
    round_wind = torch.tensor([1], dtype=torch.long)
    tile37_table, _ = model.tile_embed.build_tables(
        dora_tile34,
        dealer,
        round_wind,
        model.relative_seat_embed,
    )
    hand = torch.tensor(
        [[[1, 2], [5, 0], [1, SequenceFeatureEncoder.HAND_DRAWN_TOKEN_AUX], SequenceFeatureEncoder.HAND_PAD]],
        dtype=torch.long,
    )

    table_emb = model._embed_hand(hand, dora_tile34, tile37_table, dealer, round_wind)
    direct_emb = model._embed_hand(hand, dora_tile34, None, dealer, round_wind)

    assert table_emb.shape == (1, 4, 64)
    torch.testing.assert_close(table_emb, direct_emb)
    assert not torch.allclose(table_emb[:, 0], table_emb[:, 2])


def test_transformer_shared_tile_embedding_marks_wind_owner_and_round_wind():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=8,
        max_cand_len=4,
    )
    sparse = torch.tensor([[_SPARSE_DORA_OFFSET + 4, SequenceFeatureEncoder.SPARSE_PAD]], dtype=torch.long)
    dora_tile34 = model._decode_current_dora_tiles(sparse)
    round_wind = torch.tensor([1], dtype=torch.long)

    _, dealer_self_table = model.tile_embed.build_tables(
        dora_tile34,
        torch.tensor([0], dtype=torch.long),
        round_wind,
        model.relative_seat_embed,
    )
    _, dealer_shimo_table = model.tile_embed.build_tables(
        dora_tile34,
        torch.tensor([1], dtype=torch.long),
        round_wind,
        model.relative_seat_embed,
    )

    torch.testing.assert_close(dealer_self_table[:, 0], dealer_shimo_table[:, 0])
    assert not torch.allclose(dealer_self_table[:, 27:31], dealer_shimo_table[:, 27:31])

    _, east_round_table = model.tile_embed.build_tables(
        dora_tile34,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        model.relative_seat_embed,
    )
    _, south_round_table = model.tile_embed.build_tables(
        dora_tile34,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        model.relative_seat_embed,
    )

    torch.testing.assert_close(east_round_table[:, 0], south_round_table[:, 0])
    assert not torch.allclose(east_round_table[:, 27], south_round_table[:, 27])
    assert not torch.allclose(east_round_table[:, 28], south_round_table[:, 28])


def test_transformer_sparse_meld_action_embedding_matches_full_where():
    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        d_sub=8,
        max_prog_len=4,
        max_cand_len=4,
    )
    sparse = torch.tensor(
        [[_SPARSE_DORA_OFFSET + 4, SequenceFeatureEncoder.SPARSE_PAD]],
        dtype=torch.long,
    )
    dora_tile34 = model._decode_current_dora_tiles(sparse)
    tile37_table, _ = model.tile_embed.build_tables(dora_tile34)
    type_emb = torch.randn(1, 3, 8)
    melds = torch.tensor(
        [
            [
                [_MELD_KIND_PAD, 37, 3, 37, 3, 37, 3, 37, 3],
                [_MELD_KIND_CHI, 1, _MELD_ROLE_CALLED, 2, _MELD_ROLE_CONSUMED, 3, _MELD_ROLE_CONSUMED, 37, 3],
                [_MELD_KIND_PAD, 37, 3, 37, 3, 37, 3, 37, 3],
            ]
        ],
        dtype=torch.long,
    )

    actual = model._embed_meld_action_type(type_emb, melds, dora_tile34, tile37_table)
    full_meld_emb = model.meld_embed(melds, dora_tile34, model.tile_embed, tile37_table)
    expected = torch.where((melds[:, :, 0] != _MELD_KIND_PAD).unsqueeze(-1), full_meld_emb, type_emb)

    torch.testing.assert_close(actual, expected)


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


def test_build_encoder_ignores_removed_sequence_lengths_from_model_config():
    encoder = build_encoder(
        SequenceFeaturePackedEncoder,
        tile_dim=34,
        model_config={"max_prog_len": 11, "max_cand_len": 7, "unused": 123},
    )

    assert isinstance(encoder, SequenceFeaturePackedEncoder)
    assert not hasattr(encoder, "PACKED_SIZE")


def test_bc_policy_trainer_train_epoch_accepts_pointer_candidate_logits():
    trainer = object.__new__(bc_policy_module.BCPolicyTrainer)
    trainer.device = torch.device("cpu")
    trainer.label_smoothing = 0.0
    trainer.max_grad_norm = 10.0
    trainer.limit = 1

    max_prog_len = 2
    max_cand_len = 3
    features = _dummy_sequence_batch(batch_size=2, prog_len=max_prog_len, cand_len=max_cand_len)
    masks = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )
    features["cand_mask"] = masks
    actions = torch.tensor([0, 2])

    model = TransformerPolicyNetwork(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        d_sub=8,
        max_prog_len=max_prog_len,
        max_cand_len=max_cand_len,
        policy_head_type="pointer",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    metrics, step = trainer._train_epoch(
        model=model,
        dataloader=[(features, actions, masks)],
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        epoch=0,
    )

    assert step == 1
    assert metrics["train/loss"] > 0
    assert 0.0 <= metrics["train/acc"] <= 1.0


def test_bc_policy_trainer_train_epoch_accepts_mixed_rank_batches():
    trainer = object.__new__(bc_policy_module.BCPolicyTrainer)
    trainer.device = torch.device("cpu")
    trainer.label_smoothing = 0.0
    trainer.max_grad_norm = 10.0
    trainer.limit = 1
    trainer.value_coef = 1.0

    features = _dummy_sequence_batch(batch_size=3, prog_len=2, cand_len=2)
    masks = torch.tensor(
        [
            [1, 1],
            [0, 0],
            [1, 1],
        ],
        dtype=torch.bool,
    )
    features["cand_mask"] = masks
    actions = torch.tensor([0, 0, 1])
    ranks = torch.tensor([0, 2, 3])
    has_policy = torch.tensor([True, False, True])

    model = TransformerPolicyNetwork(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        d_sub=8,
        policy_head_type="pointer",
        emit_rank=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    metrics, step = trainer._train_epoch(
        model=model,
        dataloader=[(features, actions, masks, ranks, has_policy)],
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        epoch=0,
    )

    assert step == 1
    assert metrics["train/policy_loss"] > 0
    assert metrics["train/rank_loss"] > 0
    assert metrics["train/loss"] == pytest.approx(metrics["train/policy_loss"] + metrics["train/rank_loss"])


def test_bc_policy_trainer_train_epoch_accepts_all_rank_only_batches():
    trainer = object.__new__(bc_policy_module.BCPolicyTrainer)
    trainer.device = torch.device("cpu")
    trainer.label_smoothing = 0.0
    trainer.max_grad_norm = 10.0
    trainer.limit = 1
    trainer.value_coef = 1.0

    features = _dummy_sequence_batch(batch_size=2, prog_len=0, cand_len=0)
    masks = torch.zeros(2, 0, dtype=torch.bool)
    actions = torch.zeros(2, dtype=torch.long)
    ranks = torch.tensor([1, 3])
    has_policy = torch.zeros(2, dtype=torch.bool)

    model = TransformerPolicyNetwork(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        d_sub=8,
        policy_head_type="pointer",
        emit_rank=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    metrics, step = trainer._train_epoch(
        model=model,
        dataloader=[(features, actions, masks, ranks, has_policy)],
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        epoch=0,
    )

    assert step == 1
    assert metrics["train/policy_loss"] == 0
    assert metrics["train/rank_loss"] > 0
    assert metrics["train/loss"] == pytest.approx(metrics["train/rank_loss"])
