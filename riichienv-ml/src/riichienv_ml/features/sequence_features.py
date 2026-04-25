"""Sequence feature encoder for transformer models.

Based on the Kanachan v3 encoding design. Wraps the Rust sequence feature
encoding methods on Observation, producing padded tensors with masks
suitable for batched training.

See docs/SEQUENCE_FEATURE_ENCODING.md for the full specification.
"""

import numpy as np
import torch


class SequenceFeatureEncoder:
    """Sequence feature encoder for transformer models.

    Produces:
        sparse:      (MAX_SPARSE_LEN,)   int64   padded sparse embedding indices
        sparse_melds:(MAX_SPARSE_MELDS, 9) int64 padded current meld rows
        hand:        (MAX_HAND_LEN, 2)   int64   padded hand tuples
        numeric:     (NUM_NUMERIC,)      float32
        progression: (MAX_PROG_LEN, 5)   int64   padded action-history 5-tuples
        prog_melds:  (MAX_PROG_LEN, 9)   int64   padded progression meld rows
        candidates:  (MAX_CAND_LEN, 2)   int64   padded legal-action 2-tuples
        cand_melds:  (MAX_CAND_LEN, 9)   int64   padded candidate meld rows
        sparse_mask: (MAX_SPARSE_LEN,)   bool    True for real tokens
        sparse_meld_mask: (MAX_SPARSE_MELDS,) bool True for real current melds
        hand_mask:   (MAX_HAND_LEN,)     bool    True for real hand entries
        prog_mask:   (MAX_PROG_LEN,)     bool    True for real entries
        cand_mask:   (MAX_CAND_LEN,)     bool    True for real entries
    """

    SPARSE_VOCAB_SIZE = 269
    SPARSE_PAD = 268
    MAX_SPARSE_LEN = 10

    MELD_DIMS = (6, 38, 4, 38, 4, 38, 4, 38, 4)
    MELD_PAD = (5, 37, 3, 37, 3, 37, 3, 37, 3)
    MAX_SPARSE_MELDS = 4
    MELD_WIDTH = 9

    HAND_DIMS = (38, 3)
    HAND_PAD = (37, 2)
    MAX_HAND_LEN = 14

    PROG_DIMS = (5, 44, 3, 3, 5)
    PROG_PAD = (4, 43, 2, 2, 4)
    MAX_PROG_LEN = 256

    CAND_DIMS = (45, 4)
    CAND_PAD = (44, 3)
    MAX_CAND_LEN = 32

    NUM_NUMERIC = 12

    def __init__(self, n_players: int = 4, game_style: int = 1,
                 max_prog_len: int = 256, max_cand_len: int = 32):
        self.n_players = n_players
        self.game_style = game_style  # 0=tonpuusen, 1=hanchan
        self.MAX_PROG_LEN = max_prog_len
        self.MAX_CAND_LEN = max_cand_len

    def encode(self, obs) -> dict[str, torch.Tensor]:  # noqa: PLR0915
        """Encode observation into sequence features for transformer models.

        Args:
            obs: riichienv Observation object with encode_seq_* methods.

        Returns:
            Dict with padded sequence tensors and masks.
        """
        # Sparse
        raw = np.frombuffer(
            obs.encode_seq_sparse(self.game_style), dtype=np.uint16
        ).copy()
        n_sparse = min(len(raw), self.MAX_SPARSE_LEN)
        sparse = np.full(self.MAX_SPARSE_LEN, self.SPARSE_PAD, dtype=np.int64)
        sparse[:n_sparse] = raw[:n_sparse]
        sparse_mask = np.zeros(self.MAX_SPARSE_LEN, dtype=np.bool_)
        sparse_mask[:n_sparse] = True

        # Current melds
        sparse_meld_bytes = obs.encode_seq_sparse_melds()
        if len(sparse_meld_bytes) > 0:
            raw_sparse_melds = np.frombuffer(
                sparse_meld_bytes, dtype=np.uint16
            ).reshape(-1, self.MELD_WIDTH)
            n_sparse_melds = min(len(raw_sparse_melds), self.MAX_SPARSE_MELDS)
        else:
            raw_sparse_melds = np.empty((0, self.MELD_WIDTH), dtype=np.uint16)
            n_sparse_melds = 0
        sparse_melds = np.tile(
            np.array(self.MELD_PAD, dtype=np.int64), (self.MAX_SPARSE_MELDS, 1)
        )
        if n_sparse_melds > 0:
            sparse_melds[:n_sparse_melds] = raw_sparse_melds[:n_sparse_melds]
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
        hand = np.tile(
            np.array(self.HAND_PAD, dtype=np.int64), (self.MAX_HAND_LEN, 1)
        )
        if n_hand > 0:
            hand[:n_hand] = raw_hand[:n_hand]
        hand_mask = np.zeros(self.MAX_HAND_LEN, dtype=np.bool_)
        hand_mask[:n_hand] = True

        # Numeric
        numeric = np.frombuffer(
            obs.encode_seq_numeric(), dtype=np.float32
        ).copy()

        # Progression
        prog_bytes = obs.encode_seq_progression()
        if len(prog_bytes) > 0:
            raw_prog = np.frombuffer(prog_bytes, dtype=np.uint16).reshape(-1, 5)
            n_prog = min(len(raw_prog), self.MAX_PROG_LEN)
        else:
            raw_prog = np.empty((0, 5), dtype=np.uint16)
            n_prog = 0
        prog = np.tile(
            np.array(self.PROG_PAD, dtype=np.int64), (self.MAX_PROG_LEN, 1)
        )
        if n_prog > 0:
            prog[:n_prog] = raw_prog[:n_prog]
        prog_mask = np.zeros(self.MAX_PROG_LEN, dtype=np.bool_)
        prog_mask[:n_prog] = True

        prog_meld_bytes = obs.encode_seq_progression_melds()
        if len(prog_meld_bytes) > 0:
            raw_prog_melds = np.frombuffer(
                prog_meld_bytes, dtype=np.uint16
            ).reshape(-1, self.MELD_WIDTH)
            n_prog_melds = min(len(raw_prog_melds), self.MAX_PROG_LEN)
        else:
            raw_prog_melds = np.empty((0, self.MELD_WIDTH), dtype=np.uint16)
            n_prog_melds = 0
        prog_melds = np.tile(
            np.array(self.MELD_PAD, dtype=np.int64), (self.MAX_PROG_LEN, 1)
        )
        n_prog_sidecar = min(n_prog, n_prog_melds)
        if n_prog_sidecar > 0:
            prog_melds[:n_prog_sidecar] = raw_prog_melds[:n_prog_sidecar]

        # Candidates
        cand_bytes = obs.encode_seq_candidates()
        if len(cand_bytes) > 0:
            raw_cand = np.frombuffer(cand_bytes, dtype=np.uint16).reshape(-1, 2)
            n_cand = min(len(raw_cand), self.MAX_CAND_LEN)
        else:
            raw_cand = np.empty((0, 2), dtype=np.uint16)
            n_cand = 0
        cand = np.tile(
            np.array(self.CAND_PAD, dtype=np.int64), (self.MAX_CAND_LEN, 1)
        )
        if n_cand > 0:
            cand[:n_cand] = raw_cand[:n_cand]
        cand_mask = np.zeros(self.MAX_CAND_LEN, dtype=np.bool_)
        cand_mask[:n_cand] = True

        cand_meld_bytes = obs.encode_seq_candidate_melds()
        if len(cand_meld_bytes) > 0:
            raw_cand_melds = np.frombuffer(
                cand_meld_bytes, dtype=np.uint16
            ).reshape(-1, self.MELD_WIDTH)
            n_cand_melds = min(len(raw_cand_melds), self.MAX_CAND_LEN)
        else:
            raw_cand_melds = np.empty((0, self.MELD_WIDTH), dtype=np.uint16)
            n_cand_melds = 0
        cand_melds = np.tile(
            np.array(self.MELD_PAD, dtype=np.int64), (self.MAX_CAND_LEN, 1)
        )
        n_cand_sidecar = min(n_cand, n_cand_melds)
        if n_cand_sidecar > 0:
            cand_melds[:n_cand_sidecar] = raw_cand_melds[:n_cand_sidecar]

        return {
            "sparse": torch.from_numpy(sparse),
            "sparse_melds": torch.from_numpy(sparse_melds),
            "hand": torch.from_numpy(hand),
            "numeric": torch.from_numpy(numeric),
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
    """Packed single-tensor encoder for Ray worker compatibility.

    Packs all sequence features into a flat float32 tensor so the teacher
    worker (which expects ``encoder.encode(obs) -> Tensor``) can handle it
    transparently.  The ``TransformerActorCritic`` model unpacks this
    internally.

    Layout (all float32, P=max_prog_len, C=max_cand_len):
        sparse      (10)       int indices stored as float
        sparse_melds(4 * 9)    int meld rows stored as float
        hand        (14 * 2)   int tuples stored as float
        numeric     (12)       continuous values
        progression (P * 5)    int tuples stored as float
        prog_melds  (P * 9)    int meld rows stored as float
        candidates  (C * 2)    int tuples stored as float
        cand_melds  (C * 9)    int meld rows stored as float
        sparse_mask (10)       bool stored as float
        sparse_meld_mask (4)   bool stored as float
        hand_mask   (14)       bool stored as float
        prog_mask   (P)        bool stored as float
        cand_mask   (C)        bool stored as float
    """

    _S = SequenceFeatureEncoder.MAX_SPARSE_LEN
    _SM = SequenceFeatureEncoder.MAX_SPARSE_MELDS
    _MW = SequenceFeatureEncoder.MELD_WIDTH
    _H = SequenceFeatureEncoder.MAX_HAND_LEN     # 14
    _N = SequenceFeatureEncoder.NUM_NUMERIC       # 12

    def __init__(self, tile_dim: int = 34, n_players: int = 4,
                 game_style: int = 1,
                 max_prog_len: int = 256, max_cand_len: int = 32):
        # tile_dim accepted for API compatibility with CNN encoders
        if tile_dim == 27:
            n_players = 3
        self.inner = SequenceFeatureEncoder(
            n_players=n_players, game_style=game_style,
            max_prog_len=max_prog_len, max_cand_len=max_cand_len)
        self._P = max_prog_len
        self._C = max_cand_len
        self.PACKED_SIZE = (
            self._S + self._SM * self._MW + self._H * 2 + self._N
            + self._P * 5 + self._P * self._MW
            + self._C * 2 + self._C * self._MW
            + self._S + self._SM + self._H + self._P + self._C
        )

    def encode(self, obs) -> torch.Tensor:
        """Encode observation into a flat packed tensor (PACKED_SIZE,)."""
        d = self.inner.encode(obs)
        packed = torch.zeros(self.PACKED_SIZE, dtype=torch.float32)
        o = 0

        packed[o:o + self._S] = d["sparse"].float()
        o += self._S
        packed[o:o + self._SM * self._MW] = d["sparse_melds"].reshape(-1).float()
        o += self._SM * self._MW
        packed[o:o + self._H * 2] = d["hand"].reshape(-1).float()
        o += self._H * 2
        packed[o:o + self._N] = d["numeric"]
        o += self._N
        packed[o:o + self._P * 5] = d["progression"].reshape(-1).float()
        o += self._P * 5
        packed[o:o + self._P * self._MW] = d["prog_melds"].reshape(-1).float()
        o += self._P * self._MW
        packed[o:o + self._C * 2] = d["candidates"].reshape(-1).float()
        o += self._C * 2
        packed[o:o + self._C * self._MW] = d["cand_melds"].reshape(-1).float()
        o += self._C * self._MW
        packed[o:o + self._S] = d["sparse_mask"].float()
        o += self._S
        packed[o:o + self._SM] = d["sparse_meld_mask"].float()
        o += self._SM
        packed[o:o + self._H] = d["hand_mask"].float()
        o += self._H
        packed[o:o + self._P] = d["prog_mask"].float()
        o += self._P
        packed[o:o + self._C] = d["cand_mask"].float()

        return packed
