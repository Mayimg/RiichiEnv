"""Feature helpers for hidden-allocation belief sampling."""

from __future__ import annotations

import numpy as np
import torch

from riichienv import ActionType
from riichienv_ml.features.sequence_features import SequenceFeatureEncoder, collate_sequence_features

TILE37_COUNT = 37
BUCKET_COUNT = 4
BUCKET_REL_SEATS = (1, 2, 3)
MAX_HIDDEN_HAND_SIZE = 14

TOTAL_TILE_COUNTS37 = np.full(TILE37_COUNT, 4, dtype=np.int64)
TOTAL_TILE_COUNTS37[[0, 10, 20]] = 1
TOTAL_TILE_COUNTS37[[5, 15, 25]] = 3

_RESPONSE_ACTION_TYPES = {
    int(ActionType.CHI),
    int(ActionType.PON),
    int(ActionType.DAIMINKAN),
    int(ActionType.RON),
    int(ActionType.PASS),
}
_RED_TILE37 = {
    16: 0,
    52: 10,
    88: 20,
}


def tile_id_to_tile37(tile_id: int) -> int:
    """Convert a 136-tile id to the existing sequence-feature tile37 id."""
    tile_id = int(tile_id)
    red_tile = _RED_TILE37.get(tile_id)
    if red_tile is not None:
        return red_tile

    tile34 = tile_id // 4
    if not 0 <= tile34 <= 33:
        raise ValueError(f"tile_id out of range: {tile_id}")
    if tile34 <= 8:
        return tile34 + 1
    if tile34 <= 17:
        return tile34 + 2
    return tile34 + 3


def counts37_from_tiles(tiles: list[int] | tuple[int, ...]) -> np.ndarray:
    counts = np.zeros(TILE37_COUNT, dtype=np.int64)
    for tile in tiles:
        counts[tile_id_to_tile37(int(tile))] += 1
    return counts


def _is_response_phase(obs) -> bool:
    return any(int(action.action_type) in _RESPONSE_ACTION_TYPES for action in obs.legal_actions())


def _relative_seat(observer: int, target: int) -> int:
    return (int(target) - int(observer)) % 4


def _meld_concealed_cost(obs, abs_seat: int) -> int:
    return 3 * len(obs.melds[abs_seat])


def public_hidden_hand_size_rows(obs) -> torch.Tensor:
    """Return four public hand-size rows: ``(relative_seat, concealed_count)``.

    Opponent counts are inferred from public state only.  For the observer we use
    the real hand length because it is part of the observation.
    """
    pid = int(obs.player_id)
    rows: list[list[int]] = []
    for rel in range(4):
        abs_seat = (pid + rel) % 4
        if abs_seat == pid:
            size = len(obs.hands[abs_seat])
        else:
            base_total = 13
            size = base_total - _meld_concealed_cost(obs, abs_seat)
        rows.append([rel, max(0, min(MAX_HIDDEN_HAND_SIZE, int(size)))])
    return torch.tensor(rows, dtype=torch.long)


class BeliefFeatureEncoder(SequenceFeatureEncoder):
    """Sequence encoder plus belief-sampler public state tokens.

    The base sequence features are kept compatible with the existing transformer
    encoder, while the model ignores candidate-action tokens.
    """

    PHASE_DIMS = 3
    PHASE_ACT = 0
    PHASE_RESPONSE = 1
    PHASE_UNKNOWN = 2
    CURRENT_ACTOR_DIMS = 5
    HAND_SIZE_DIMS = MAX_HIDDEN_HAND_SIZE + 1

    def encode(self, obs) -> dict[str, torch.Tensor]:
        features = super().encode(obs)

        response_phase = _is_response_phase(obs)
        if response_phase:
            phase = self.PHASE_RESPONSE
            if obs.last_discard_actor is None:
                current_actor = 4
            else:
                current_actor = _relative_seat(obs.player_id, obs.last_discard_actor)
        else:
            phase = self.PHASE_ACT
            current_actor = 0

        features["belief_phase"] = torch.tensor(phase, dtype=torch.long)
        features["belief_current_actor"] = torch.tensor(current_actor, dtype=torch.long)
        features["belief_hand_sizes"] = public_hidden_hand_size_rows(obs)
        return features


def make_hidden_allocation_target(
    obs,
    hidden_hands: list[list[int]],
    features: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build target counts with bucket order ``shimocha, toimen, kamicha, wall``."""
    pid = int(obs.player_id)
    visible_counts = features["visible_tile_counts"][:, 1].numpy().astype(np.int64, copy=True)
    unseen_counts = TOTAL_TILE_COUNTS37 - visible_counts

    target = np.zeros((BUCKET_COUNT, TILE37_COUNT), dtype=np.int64)
    for bucket_idx, rel in enumerate(BUCKET_REL_SEATS):
        abs_seat = (pid + rel) % 4
        target[bucket_idx] = counts37_from_tiles(hidden_hands[abs_seat])

    target[3] = unseen_counts - target[:3].sum(axis=0)
    if np.any(target < 0):
        raise ValueError("hidden allocation target has negative residual wall counts")
    if not np.array_equal(target.sum(axis=0), unseen_counts):
        raise ValueError("hidden allocation target does not match unseen tile counts")
    return torch.from_numpy(target.astype(np.int64, copy=False))


def collate_belief_features(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = collate_sequence_features(features)
    for key in ("belief_phase", "belief_current_actor", "belief_hand_sizes"):
        batch[key] = torch.stack([feature[key] for feature in features])
    return batch


def allocation_counts_to_tile37_lists(allocation: torch.Tensor) -> list[list[int]]:
    """Convert one ``(4, 37)`` allocation count tensor to tile37 id lists."""
    if allocation.shape != (BUCKET_COUNT, TILE37_COUNT):
        raise ValueError(f"allocation must have shape {(BUCKET_COUNT, TILE37_COUNT)}, got {tuple(allocation.shape)}")
    out: list[list[int]] = []
    counts = allocation.detach().cpu().long()
    for bucket in range(BUCKET_COUNT):
        tiles: list[int] = []
        for tile37, count in enumerate(counts[bucket].tolist()):
            tiles.extend([tile37] * int(count))
        out.append(tiles)
    return out
