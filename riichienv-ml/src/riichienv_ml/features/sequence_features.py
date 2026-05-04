"""Sequence feature encoder for transformer models.

Based on the Kanachan v3 encoding design. Wraps the Rust sequence feature
encoding methods on Observation. Progression and candidate groups are kept
at their observation-local lengths and padded only by the DataLoader collate
function.

See docs/SEQUENCE_FEATURE_ENCODING.md for the full specification.
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


class SequenceFeatureEncoder:
    """Sequence feature encoder for transformer models.

    Produces:
        sparse:      (MAX_SPARSE_LEN,)   int64   padded sparse embedding indices
        dealer:      ()                  int64   dealer relative seat
        player_stats:(PLAYER_INFO_TOKENS, PLAYER_INFO_WIDTH) int64 per-player public summaries
        sparse_melds:(MAX_SPARSE_MELDS, 9) int64 padded current visible meld rows
        sparse_meld_owners: (MAX_SPARSE_MELDS,) int64 padded current visible meld owner seats
        hand:        (MAX_HAND_LEN, 2)   int64   padded hand tuples
        current_shanten: (1,)            int64   current self minimum shanten
        numeric:     (NUM_NUMERIC,)      float32
        agari_overtakes: (AGARI_OVERTAKE_DIM,) float32 pairwise agari-rank-overtake flags,
                         reshapeable to (AGARI_OVERTAKE_TOKENS, AGARI_OVERTAKE_TOKEN_DIM)
        progression: (P, 5)              int64   action/dora-reveal history 5-tuples
        prog_melds:  (P, 9)              int64   progression meld rows
        candidates:  (C, 4)              int64   legal-action tuples
        cand_melds:  (C, 9)              int64   candidate meld rows
        sparse_mask: (MAX_SPARSE_LEN,)   bool    True for real tokens
        sparse_meld_mask: (MAX_SPARSE_MELDS,) bool True for real current visible melds
        hand_mask:   (MAX_HAND_LEN,)     bool    True for real hand entries
        prog_mask:   (P,)                bool    True for real entries
        cand_mask:   (C,)                bool    True for real entries
    """

    SPARSE_VOCAB_SIZE = 261
    SPARSE_PAD = 260
    MAX_SPARSE_LEN = 8

    DEALER_DIMS = 4

    PLAYER_INFO_DIMS = (4, 2, 5, 13, 13)
    PLAYER_INFO_TOKENS = 4
    PLAYER_INFO_WIDTH = len(PLAYER_INFO_DIMS)

    MELD_DIMS = (6, 38, 4, 38, 4, 38, 4, 38, 4)
    MELD_PAD = (5, 37, 3, 37, 3, 37, 3, 37, 3)
    MAX_SPARSE_MELDS = 16
    MELD_WIDTH = 9
    SPARSE_MELD_FEATURE_WIDTH = MELD_WIDTH + 1
    SPARSE_MELD_OWNER_DIMS = 5
    SPARSE_MELD_OWNER_PAD = 4

    HAND_DIMS = (38, 3)
    HAND_PAD = (37, 2)
    MAX_HAND_LEN = 14

    CURRENT_SHANTEN_WIDTH = 1
    SHANTEN_VALUE_DIMS = 9
    SHANTEN_VALUE_NA = 8
    SHANTEN_DIMS = (SHANTEN_VALUE_DIMS,) * CURRENT_SHANTEN_WIDTH

    PROG_DIMS = (5, 81, 3, 3, 5)
    PROG_PAD = (4, 80, 2, 2, 4)
    DEFAULT_PROG_POS_CAPACITY = 256

    SHANTEN_DELTA_DIMS = 4
    SHANTEN_DELTA_ADVANCE = 0
    SHANTEN_DELTA_SAME = 1
    SHANTEN_DELTA_REGRESS = 2
    SHANTEN_DELTA_NA = 3
    CAND_DIMS = (48, 3, 5, SHANTEN_DELTA_DIMS)
    CAND_PAD = (47, 2, 4, SHANTEN_DELTA_NA)
    CAND_WIDTH = len(CAND_DIMS)
    DEFAULT_CAND_POS_CAPACITY = 32

    NUM_NUMERIC = 6
    AGARI_OVERTAKE_DIMS = (4, 96, 4)
    AGARI_OVERTAKE_TOKENS = AGARI_OVERTAKE_DIMS[0]
    AGARI_OVERTAKE_TOKEN_DIM = AGARI_OVERTAKE_DIMS[1] * AGARI_OVERTAKE_DIMS[2]
    AGARI_OVERTAKE_DIM = AGARI_OVERTAKE_TOKENS * AGARI_OVERTAKE_TOKEN_DIM

    def __init__(
        self,
        n_players: int = 4,
        game_style: int = 1,
        max_prog_len: int | None = None,
        max_cand_len: int | None = None,
    ):
        self.n_players = n_players
        self.game_style = game_style  # 0=tonpuusen, 1=hanchan
        # Accepted for config reuse. Progression/candidate features are no
        # longer truncated here; padding is batch-local in sequence_feature_collate().
        self.max_prog_len = max_prog_len
        self.max_cand_len = max_cand_len

    def encode(self, obs) -> dict[str, torch.Tensor]:  # noqa: PLR0915
        """Encode observation into sequence features for transformer models.

        Args:
            obs: riichienv Observation object with encode_seq_* methods.

        Returns:
            Dict with padded sequence tensors and masks.
        """
        # Sparse
        raw = np.frombuffer(obs.encode_seq_sparse(self.game_style), dtype=np.uint16).copy()
        n_sparse = min(len(raw), self.MAX_SPARSE_LEN)
        sparse = np.full(self.MAX_SPARSE_LEN, self.SPARSE_PAD, dtype=np.int64)
        sparse[:n_sparse] = raw[:n_sparse]
        sparse_mask = np.zeros(self.MAX_SPARSE_LEN, dtype=np.bool_)
        sparse_mask[:n_sparse] = True

        # Dealer relative seat
        if hasattr(obs, "encode_seq_dealer"):
            dealer = np.int64(obs.encode_seq_dealer())
        else:
            dealer = np.int64((obs.oya - obs.player_id + self.n_players) % self.n_players)

        # Per-player public summary tokens
        player_stats_bytes = obs.encode_seq_player_stats()
        raw_player_stats = np.frombuffer(player_stats_bytes, dtype=np.uint16).reshape(-1, self.PLAYER_INFO_WIDTH)
        if raw_player_stats.shape[0] != self.PLAYER_INFO_TOKENS:
            raise ValueError(
                f"encode_seq_player_stats returned {raw_player_stats.shape[0]} rows; expected {self.PLAYER_INFO_TOKENS}"
            )
        player_stats = raw_player_stats.astype(np.int64, copy=True)

        # Current visible melds
        sparse_meld_feature_bytes = obs.encode_seq_sparse_meld_features()
        if len(sparse_meld_feature_bytes) > 0:
            raw_sparse_meld_features = np.frombuffer(sparse_meld_feature_bytes, dtype=np.uint16).reshape(
                -1, self.SPARSE_MELD_FEATURE_WIDTH
            )
            raw_sparse_melds = raw_sparse_meld_features[:, : self.MELD_WIDTH]
            raw_sparse_meld_owners = raw_sparse_meld_features[:, self.MELD_WIDTH]
            n_sparse_melds = min(len(raw_sparse_meld_features), self.MAX_SPARSE_MELDS)
        else:
            raw_sparse_melds = np.empty((0, self.MELD_WIDTH), dtype=np.uint16)
            raw_sparse_meld_owners = np.empty((0,), dtype=np.uint16)
            n_sparse_melds = 0
        sparse_melds = np.tile(np.array(self.MELD_PAD, dtype=np.int64), (self.MAX_SPARSE_MELDS, 1))
        if n_sparse_melds > 0:
            sparse_melds[:n_sparse_melds] = raw_sparse_melds[:n_sparse_melds]
        sparse_meld_owners = np.full(self.MAX_SPARSE_MELDS, self.SPARSE_MELD_OWNER_PAD, dtype=np.int64)
        if n_sparse_melds > 0:
            sparse_meld_owners[:n_sparse_melds] = raw_sparse_meld_owners[:n_sparse_melds]
        sparse_meld_mask = np.zeros(self.MAX_SPARSE_MELDS, dtype=np.bool_)
        sparse_meld_mask[:n_sparse_melds] = True

        # Hand
        hand_bytes = obs.encode_seq_hand()
        if len(hand_bytes) > 0:
            raw_hand = np.frombuffer(hand_bytes, dtype=np.uint16).reshape(-1, 2)
            n_hand = min(len(raw_hand), self.MAX_HAND_LEN)
        else:
            raw_hand = np.empty((0, 2), dtype=np.uint16)
            n_hand = 0
        hand = np.tile(np.array(self.HAND_PAD, dtype=np.int64), (self.MAX_HAND_LEN, 1))
        if n_hand > 0:
            hand[:n_hand] = raw_hand[:n_hand]
        hand_mask = np.zeros(self.MAX_HAND_LEN, dtype=np.bool_)
        hand_mask[:n_hand] = True

        # Current minimum shanten
        current_shanten_bytes = obs.encode_seq_current_shanten()
        current_shanten = np.frombuffer(current_shanten_bytes, dtype=np.uint16).astype(np.int64, copy=True)
        if current_shanten.shape[0] != self.CURRENT_SHANTEN_WIDTH:
            raise ValueError(
                f"encode_seq_current_shanten returned {current_shanten.shape[0]} values; "
                f"expected {self.CURRENT_SHANTEN_WIDTH}"
            )

        # Numeric
        numeric = np.frombuffer(obs.encode_seq_numeric(), dtype=np.float32).copy()

        # Pairwise agari-rank-overtake flags
        agari_overtakes = np.frombuffer(obs.encode_seq_agari_overtakes(), dtype=np.float32).copy()
        if agari_overtakes.shape[0] != self.AGARI_OVERTAKE_DIM:
            raise ValueError(
                f"encode_seq_agari_overtakes returned {agari_overtakes.shape[0]} floats; "
                f"expected {self.AGARI_OVERTAKE_DIM}"
            )

        # Progression
        prog_bytes = obs.encode_seq_progression()
        if len(prog_bytes) > 0:
            prog = np.frombuffer(prog_bytes, dtype=np.uint16).reshape(-1, 5).astype(np.int64, copy=True)
        else:
            prog = np.empty((0, 5), dtype=np.int64)
        n_prog = len(prog)
        prog_mask = np.ones(n_prog, dtype=np.bool_)

        prog_meld_bytes = obs.encode_seq_progression_melds()
        if len(prog_meld_bytes) > 0:
            raw_prog_melds = np.frombuffer(prog_meld_bytes, dtype=np.uint16).reshape(-1, self.MELD_WIDTH)
        else:
            raw_prog_melds = np.empty((0, self.MELD_WIDTH), dtype=np.uint16)
        prog_melds = np.tile(np.array(self.MELD_PAD, dtype=np.int64), (n_prog, 1))
        n_prog_sidecar = min(n_prog, len(raw_prog_melds))
        if n_prog_sidecar > 0:
            prog_melds[:n_prog_sidecar] = raw_prog_melds[:n_prog_sidecar]

        # Candidates
        raw_cand_features = None
        if hasattr(obs, "encode_seq_candidate_features"):
            cand_feature_bytes = obs.encode_seq_candidate_features()
            if len(cand_feature_bytes) > 0:
                raw_cand_features = np.frombuffer(cand_feature_bytes, dtype=np.uint16).reshape(
                    -1, self.CAND_WIDTH + self.MELD_WIDTH
                )

        if raw_cand_features is not None:
            raw_cand = raw_cand_features[:, : self.CAND_WIDTH]
            raw_cand_melds = raw_cand_features[:, self.CAND_WIDTH :]
        else:
            cand_bytes = obs.encode_seq_candidates()
            if len(cand_bytes) > 0:
                raw_cand = np.frombuffer(cand_bytes, dtype=np.uint16).reshape(-1, self.CAND_WIDTH)
            else:
                raw_cand = np.empty((0, self.CAND_WIDTH), dtype=np.uint16)

            cand_meld_bytes = obs.encode_seq_candidate_melds()
            if len(cand_meld_bytes) > 0:
                raw_cand_melds = np.frombuffer(cand_meld_bytes, dtype=np.uint16).reshape(-1, self.MELD_WIDTH)
            else:
                raw_cand_melds = np.empty((0, self.MELD_WIDTH), dtype=np.uint16)

        n_cand = len(raw_cand)
        cand = np.empty((n_cand, self.CAND_WIDTH), dtype=np.int64)
        if n_cand > 0:
            cand[:] = raw_cand
        cand_mask = np.ones(n_cand, dtype=np.bool_)

        cand_melds = np.tile(np.array(self.MELD_PAD, dtype=np.int64), (n_cand, 1))
        n_cand_sidecar = min(n_cand, len(raw_cand_melds))
        if n_cand_sidecar > 0:
            cand_melds[:n_cand_sidecar] = raw_cand_melds[:n_cand_sidecar]

        return {
            "sparse": torch.from_numpy(sparse),
            "dealer": torch.tensor(dealer, dtype=torch.long),
            "player_stats": torch.from_numpy(player_stats),
            "sparse_melds": torch.from_numpy(sparse_melds),
            "sparse_meld_owners": torch.from_numpy(sparse_meld_owners),
            "hand": torch.from_numpy(hand),
            "current_shanten": torch.from_numpy(current_shanten),
            "numeric": torch.from_numpy(numeric),
            "agari_overtakes": torch.from_numpy(agari_overtakes),
            "progression": torch.from_numpy(prog),
            "prog_melds": torch.from_numpy(prog_melds),
            "candidates": torch.from_numpy(cand),
            "cand_melds": torch.from_numpy(cand_melds),
            "sparse_mask": torch.from_numpy(sparse_mask),
            "sparse_meld_mask": torch.from_numpy(sparse_meld_mask),
            "hand_mask": torch.from_numpy(hand_mask),
            "prog_mask": torch.from_numpy(prog_mask),
            "cand_mask": torch.from_numpy(cand_mask),
        }


class SequenceFeaturePackedEncoder:
    """Compatibility-named dynamic sequence encoder.

    Existing configs reference this class name. It now returns the same dict as
    ``SequenceFeatureEncoder`` instead of a fixed-size packed tensor.
    """

    def __init__(
        self,
        tile_dim: int = 34,
        n_players: int = 4,
        game_style: int = 1,
        max_prog_len: int | None = None,
        max_cand_len: int | None = None,
    ):
        # tile_dim accepted for API compatibility with CNN encoders
        if tile_dim == 27:
            n_players = 3
        self.inner = SequenceFeatureEncoder(
            n_players=n_players, game_style=game_style, max_prog_len=max_prog_len, max_cand_len=max_cand_len
        )

    def encode(self, obs) -> dict[str, torch.Tensor]:
        """Encode observation into a dynamic sequence-feature dict."""
        return self.inner.encode(obs)


def _pad_sequence_tensors(values: list[torch.Tensor], pad_value: int | bool | tuple[int, ...]) -> torch.Tensor:
    max_len = max((int(v.shape[0]) for v in values), default=0)
    sample = values[0]
    out_shape = (len(values), max_len, *sample.shape[1:])
    out = sample.new_empty(out_shape)

    if isinstance(pad_value, tuple):
        pad = torch.tensor(pad_value, dtype=sample.dtype, device=sample.device)
        out[:] = pad.view(1, 1, *pad.shape)
    else:
        out.fill_(pad_value)

    for idx, value in enumerate(values):
        length = int(value.shape[0])
        if length > 0:
            out[idx, :length] = value
    return out


def _collate_feature_values(key: str, values: list[torch.Tensor]) -> torch.Tensor:
    if key == "progression":
        return _pad_sequence_tensors(values, SequenceFeatureEncoder.PROG_PAD)
    if key == "prog_melds":
        return _pad_sequence_tensors(values, SequenceFeatureEncoder.MELD_PAD)
    if key == "candidates":
        return _pad_sequence_tensors(values, SequenceFeatureEncoder.CAND_PAD)
    if key == "cand_melds":
        return _pad_sequence_tensors(values, SequenceFeatureEncoder.MELD_PAD)
    if key in {"prog_mask", "cand_mask"}:
        return _pad_sequence_tensors(values, False)
    return torch.stack(values)


def sequence_feature_collate(batch: list[Any]) -> Any:
    """Collate sequence-feature batches with batch-local dynamic padding."""
    elem = batch[0]

    if isinstance(elem, dict):
        collated = {key: _collate_feature_values(key, [sample[key] for sample in batch]) for key in elem}
    elif isinstance(elem, tuple):
        collated = tuple(sequence_feature_collate(list(items)) for items in zip(*batch, strict=True))
    elif isinstance(elem, torch.Tensor):
        if all(value.shape == elem.shape for value in batch):
            collated = torch.stack(batch)
        else:
            collated = _pad_sequence_tensors(batch, 0)
    elif isinstance(elem, np.ndarray):
        tensors = [torch.from_numpy(value) for value in batch]
        if all(value.shape == elem.shape for value in batch):
            collated = torch.stack(tensors)
        else:
            collated = _pad_sequence_tensors(tensors, 0)
    else:
        collated = default_collate(batch)

    return collated
