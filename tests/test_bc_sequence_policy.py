import json
import re

import pytest
import riichienv_ml.trainers.bc_policy as bc_policy_module
import torch
import torch.nn.functional as F
from riichienv_ml.datasets.mjai_logs import BehaviorCloningDataset
from riichienv_ml.features.sequence_features import SequenceFeatureEncoder, SequenceFeaturePackedEncoder
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
    _SEAT_ROLE_MELD_OWNER,
    _SEAT_ROLE_PROG_ACTOR,
    _SPARSE_DORA_OFFSET,
    _TILE34_PAD,
    TransformerActorCritic,
    TransformerPolicyNetwork,
)
from riichienv_ml.utils import build_encoder

from riichienv import Action, ActionType, Meld, MeldType, MjaiReplay, Observation


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
    packed_size = SequenceFeaturePackedEncoder(max_prog_len=8, max_cand_len=4).PACKED_SIZE
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
    packed_size = SequenceFeaturePackedEncoder(max_prog_len=8, max_cand_len=4).PACKED_SIZE
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

    assert logits.shape == (1, 32)


def test_sequence_feature_encoder_factorizes_hand_draw_state():
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
    valid_hand = hand[hand_mask]

    assert valid_hand.shape[1] == 2
    assert torch.count_nonzero(valid_hand[:, 1] == 1) == 1
    assert torch.count_nonzero(valid_hand[:, 1] == 0) == len(valid_hand) - 1


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
    assert valid_sparse.tolist() == [1, 3, 74]


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

    features = SequenceFeatureEncoder(max_cand_len=8).encode(obs)
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

    features = SequenceFeatureEncoder(max_cand_len=8).encode(obs)
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
    features = SequenceFeatureEncoder(max_cand_len=8).encode(obs)
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
    assert model.prog_type_action_kind[80].item() == _ACTION_KIND_PAD

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

    assert actor_emb.shape == (1, 3, 8)
    assert owner_emb.shape == (1, 3, 64)
    assert model.relative_seat_embed.base_embed.weight.shape[0] == 4
    assert torch.count_nonzero(actor_emb[:, :2]).item() > 0
    assert torch.count_nonzero(owner_emb[:, :2]).item() > 0
    assert torch.allclose(actor_emb[:, 2], torch.zeros_like(actor_emb[:, 2]))
    assert torch.allclose(owner_emb[:, 2], torch.zeros_like(owner_emb[:, 2]))


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


def test_build_encoder_passes_sequence_lengths_from_model_config():
    encoder = build_encoder(
        SequenceFeaturePackedEncoder,
        tile_dim=34,
        model_config={"max_prog_len": 11, "max_cand_len": 7, "unused": 123},
    )

    assert encoder._P == 11
    assert encoder._C == 7


def test_bc_policy_trainer_train_epoch_accepts_pointer_candidate_logits():
    trainer = object.__new__(bc_policy_module.BCPolicyTrainer)
    trainer.device = torch.device("cpu")
    trainer.label_smoothing = 0.0
    trainer.max_grad_norm = 10.0
    trainer.limit = 1

    max_prog_len = 2
    max_cand_len = 3
    encoder = SequenceFeaturePackedEncoder(max_prog_len=max_prog_len, max_cand_len=max_cand_len)
    features = torch.zeros(2, encoder.PACKED_SIZE, dtype=torch.float32)
    masks = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    features[:, -max_cand_len:] = masks
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
