from __future__ import annotations

import numpy as np

from riichienv import encode_grp_tenhou_4p

TENHOU_4P_AGARI_RANK_GAINS_INPUT_FORMAT = "kyoku_start_with_player_rank_and_tenhou_agari_rank_gains_v1"
TENHOU_4P_AGARI_RANK_GAINS_AND_OVERTAKES_INPUT_FORMAT = (
    "kyoku_start_with_player_rank_and_tenhou_agari_rank_gains_and_overtakes_v2"
)
TENHOU_4P_SELF_AGARI_DIM = 96
TENHOU_4P_OTHER_OVERTAKE_DIM = 288
TENHOU_4P_GRP_INPUT_DIM = 404
TENHOU_4P_AGARI_OFFSET = 20
TENHOU_4P_OVERTAKE_OFFSET = TENHOU_4P_AGARI_OFFSET + TENHOU_4P_SELF_AGARI_DIM


def supports_agari_rank_gains(n_players: int, replay_rule: str | None) -> bool:
    return n_players == 4 and replay_rule == "tenhou"


def get_agari_rank_gain_feature_dim(n_players: int, replay_rule: str | None) -> int:
    return TENHOU_4P_SELF_AGARI_DIM if supports_agari_rank_gains(n_players, replay_rule) else 0


def get_other_player_overtake_feature_dim(n_players: int, replay_rule: str | None) -> int:
    return TENHOU_4P_OTHER_OVERTAKE_DIM if supports_agari_rank_gains(n_players, replay_rule) else 0


def get_grp_input_dim(n_players: int, replay_rule: str | None) -> int:
    return TENHOU_4P_GRP_INPUT_DIM if supports_agari_rank_gains(n_players, replay_rule) else n_players * 4 + 4


def encode_tenhou_4p_all_player_grp_features(
    start_scores: list[int] | np.ndarray,
    delta_scores: list[int] | np.ndarray,
    chang: int,
    ju: int,
    ben: int,
    liqibang: int,
) -> np.ndarray:
    start_scores_list = np.asarray(start_scores, dtype=np.int32).tolist()
    delta_scores_list = np.asarray(delta_scores, dtype=np.int32).tolist()
    raw = encode_grp_tenhou_4p(
        start_scores_list,
        delta_scores_list,
        int(chang),
        int(ju),
        int(ben),
        int(liqibang),
    )
    return np.frombuffer(raw, dtype=np.float32).reshape(4, TENHOU_4P_GRP_INPUT_DIM).copy()


def encode_agari_rank_gains(
    start_scores: list[int] | np.ndarray,
    oya: int,
    honba: int,
    liqibang: int,
    player_idx: int,
    current_rank: int | None = None,
    n_players: int = 4,
    replay_rule: str | None = None,
) -> np.ndarray:
    if not supports_agari_rank_gains(n_players, replay_rule):
        return np.zeros(0, dtype=np.float32)

    features = encode_tenhou_4p_all_player_grp_features(
        start_scores,
        [0, 0, 0, 0],
        chang=0,
        ju=oya,
        ben=honba,
        liqibang=liqibang,
    )
    del current_rank
    return features[player_idx, TENHOU_4P_AGARI_OFFSET:TENHOU_4P_OVERTAKE_OFFSET]


def encode_other_player_overtake_flags(
    start_scores: list[int] | np.ndarray,
    oya: int,
    honba: int,
    liqibang: int,
    player_idx: int,
    current_rank: int | None = None,
    n_players: int = 4,
    replay_rule: str | None = None,
) -> np.ndarray:
    if not supports_agari_rank_gains(n_players, replay_rule):
        return np.zeros(0, dtype=np.float32)

    features = encode_tenhou_4p_all_player_grp_features(
        start_scores,
        [0, 0, 0, 0],
        chang=0,
        ju=oya,
        ben=honba,
        liqibang=liqibang,
    )
    del current_rank
    return features[player_idx, TENHOU_4P_OVERTAKE_OFFSET:]
