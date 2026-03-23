from __future__ import annotations

from functools import lru_cache

import numpy as np

from riichienv import calculate_score

TENHOU_4P_AGARI_RANK_GAINS_INPUT_FORMAT = "kyoku_start_with_player_rank_and_tenhou_agari_rank_gains_v1"
TENHOU_4P_AGARI_RANK_GAINS_AND_OVERTAKES_INPUT_FORMAT = (
    "kyoku_start_with_player_rank_and_tenhou_agari_rank_gains_and_overtakes_v2"
)
TENHOU_4P_MAX_YAKUMAN_MULTIPLIER = 4

# Reachable standard 4P riichi han/fu rows.
# This intentionally excludes impossible rows such as 20-fu 1-han.
_STANDARD_TSUMO_HAN_FU: tuple[tuple[int, int], ...] = (
    *((1, fu) for fu in (30, 40, 50, 60, 70, 80, 90, 100, 110)),
    *((2, fu) for fu in (20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110)),
    *((3, fu) for fu in (20, 25, 30, 40, 50, 60)),
    *((4, fu) for fu in (20, 25, 30)),
)
_STANDARD_RON_HAN_FU: tuple[tuple[int, int], ...] = (
    *((1, fu) for fu in (30, 40, 50, 60, 70, 80, 90, 100, 110)),
    *((2, fu) for fu in (25, 30, 40, 50, 60, 70, 80, 90, 100, 110)),
    *((3, fu) for fu in (25, 30, 40, 50, 60)),
    *((4, fu) for fu in (25, 30)),
)
_LIMIT_HAND_HANS: tuple[int, ...] = (5, 6, 8, 11)


def supports_agari_rank_gains(n_players: int, replay_rule: str | None) -> bool:
    return n_players == 4 and replay_rule == "tenhou"


def get_agari_rank_gain_feature_dim(n_players: int, replay_rule: str | None) -> int:
    if not supports_agari_rank_gains(n_players, replay_rule):
        return 0

    tsumo_patterns, ron_patterns = _tenhou_4p_base_agari_patterns(is_oya=False)
    return len(tsumo_patterns) + 3 * len(ron_patterns)


def get_grp_input_dim(n_players: int, replay_rule: str | None) -> int:
    return (
        n_players * 4
        + 4
        + get_agari_rank_gain_feature_dim(n_players, replay_rule)
        + get_other_player_overtake_feature_dim(n_players, replay_rule)
    )


def get_other_player_overtake_feature_dim(n_players: int, replay_rule: str | None) -> int:
    if not supports_agari_rank_gains(n_players, replay_rule):
        return 0

    return (n_players - 1) * get_agari_rank_gain_feature_dim(n_players, replay_rule)


@lru_cache(maxsize=2)
def _tenhou_4p_base_agari_patterns(is_oya: bool) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    tsumo_patterns: set[tuple[int, int]] = set()
    ron_patterns: set[int] = set()

    for han, fu in _STANDARD_TSUMO_HAN_FU:
        score = calculate_score(han, fu, is_oya, True, 0, 4)
        tsumo_patterns.add((int(score.pay_tsumo_oya), int(score.pay_tsumo_ko)))
    for han, fu in _STANDARD_RON_HAN_FU:
        score = calculate_score(han, fu, is_oya, False, 0, 4)
        ron_patterns.add(int(score.pay_ron))
    for han in _LIMIT_HAND_HANS:
        tsumo_score = calculate_score(han, 30, is_oya, True, 0, 4)
        ron_score = calculate_score(han, 30, is_oya, False, 0, 4)
        tsumo_patterns.add((int(tsumo_score.pay_tsumo_oya), int(tsumo_score.pay_tsumo_ko)))
        ron_patterns.add(int(ron_score.pay_ron))

    for yakuman_count in range(1, TENHOU_4P_MAX_YAKUMAN_MULTIPLIER + 1):
        han = 13 * yakuman_count
        tsumo_score = calculate_score(han, 0, is_oya, True, 0, 4)
        ron_score = calculate_score(han, 0, is_oya, False, 0, 4)
        tsumo_patterns.add((int(tsumo_score.pay_tsumo_oya), int(tsumo_score.pay_tsumo_ko)))
        ron_patterns.add(int(ron_score.pay_ron))

    return tuple(sorted(tsumo_patterns)), tuple(sorted(ron_patterns))


def _compute_stable_ranks(scores: list[float] | np.ndarray) -> list[int]:
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (-item[1], item[0]))
    ranks = [0] * len(indexed)
    for rank, (seat, _) in enumerate(indexed):
        ranks[seat] = rank
    return ranks


def _iter_agari_pattern_score_states(
    scores: np.ndarray,
    oya: int,
    honba: int,
    liqibang: int,
    winner_idx: int,
) -> tuple[np.ndarray, ...]:
    is_oya = winner_idx == oya
    tsumo_patterns, ron_patterns = _tenhou_4p_base_agari_patterns(is_oya=is_oya)
    deposit = int(liqibang) * 1000
    next_states: list[np.ndarray] = []

    for pay_oya, pay_ko in tsumo_patterns:
        next_scores = scores.copy()
        total_win = deposit
        for seat in range(4):
            if seat == winner_idx:
                continue
            pay = pay_ko if is_oya or seat != oya else pay_oya
            pay += int(honba) * 100
            next_scores[seat] -= pay
            total_win += pay
        next_scores[winner_idx] += total_win
        next_states.append(next_scores)

    for target in range(4):
        if target == winner_idx:
            continue
        for pay_ron in ron_patterns:
            next_scores = scores.copy()
            total_gain = pay_ron + int(honba) * 300
            next_scores[target] -= total_gain
            next_scores[winner_idx] += total_gain + deposit
            next_states.append(next_scores)

    return tuple(next_states)


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

    scores = np.asarray(start_scores, dtype=np.int32)
    if scores.shape != (4,):
        raise ValueError(f"Tenhou 4P agari rank gains require 4 scores, got shape={scores.shape}")

    if current_rank is None:
        current_rank = _compute_stable_ranks(scores)[player_idx]

    gains: list[float] = []

    for next_scores in _iter_agari_pattern_score_states(scores, oya, honba, liqibang, player_idx):
        next_rank = _compute_stable_ranks(next_scores)[player_idx]
        gains.append(max(current_rank - next_rank, 0) / 3.0)

    return np.asarray(gains, dtype=np.float32)


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

    scores = np.asarray(start_scores, dtype=np.int32)
    if scores.shape != (4,):
        raise ValueError(f"Tenhou 4P overtake flags require 4 scores, got shape={scores.shape}")

    current_ranks = _compute_stable_ranks(scores)
    if current_rank is None:
        current_rank = current_ranks[player_idx]

    overtakes: list[float] = []
    for other_idx in range(4):
        if other_idx == player_idx:
            continue
        started_below = current_ranks[other_idx] > current_rank
        for next_scores in _iter_agari_pattern_score_states(scores, oya, honba, liqibang, other_idx):
            if not started_below:
                overtakes.append(0.0)
                continue
            next_ranks = _compute_stable_ranks(next_scores)
            overtook = next_ranks[other_idx] < next_ranks[player_idx]
            overtakes.append(1.0 if overtook else 0.0)

    return np.asarray(overtakes, dtype=np.float32)
