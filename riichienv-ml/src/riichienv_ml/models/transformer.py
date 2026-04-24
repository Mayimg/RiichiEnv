"""Transformer Actor-Critic for sequence feature encoding.

Accepts the packed flat tensor produced by SequenceFeaturePackedEncoder,
unpacks it into sparse / hand / numeric / progression / candidate groups,
embeds each group, and processes them through a TransformerEncoder.

Tile-only fields across hand / sparse dora / progression / candidates share
an attribute-based tile embedding module. Meld-pattern tokens (chi / pon and
their sparse meld counterparts) remain separate categorical embeddings.

Output: (logits, value) — same interface as ActorCriticNetwork.

NOTE: Sanma (3-player) is not supported. The sparse/progression/candidate
vocabularies and encoding logic assume 4-player mahjong.
"""

import math

import torch
from torch import nn

from riichienv_ml.features.sequence_features import SequenceFeatureEncoder

_TILE37_PAD = 37
_TILE34_PAD = 34

_SUIT_MAN = 0
_SUIT_PIN = 1
_SUIT_SOU = 2
_SUIT_HONOR = 3
_SUIT_PAD = 4

_RANK_NONE = 9
_RANK_PAD = 10

_HONOR_EAST = 0
_HONOR_SOUTH = 1
_HONOR_WEST = 2
_HONOR_NORTH = 3
_HONOR_WHITE = 4
_HONOR_GREEN = 5
_HONOR_RED = 6
_HONOR_NONE = 7
_HONOR_PAD = 8

_RED_FLAG_NORMAL = 0
_RED_FLAG_RED = 1
_RED_FLAG_PAD = 2

_TILE_CLASS_SIMPLE = 0
_TILE_CLASS_TERMINAL = 1
_TILE_CLASS_WIND = 2
_TILE_CLASS_DRAGON = 3
_TILE_CLASS_PAD = 4

_DORA_FLAG_DORA = 0
_DORA_FLAG_NONE = 1
_DORA_FLAG_PAD = 2

_ACTION_KIND_DISCARD = 0
_ACTION_KIND_DAIMINKAN = 1
_ACTION_KIND_ANKAN = 2
_ACTION_KIND_KAKAN = 3
_ACTION_KIND_PAD = 4

_SPARSE_DORA_OFFSET = 83
_SPARSE_DORA_SLOTS = 5
_DORA_SLOT_PAD = _SPARSE_DORA_SLOTS


def _tile37_to_tile34(tile37: int) -> int:
    if tile37 == 0:
        tile34 = 4
    elif 1 <= tile37 <= 9:
        tile34 = tile37 - 1
    elif tile37 == 10:
        tile34 = 13
    elif 11 <= tile37 <= 19:
        tile34 = tile37 - 2
    elif tile37 == 20:
        tile34 = 22
    elif 21 <= tile37 <= 29:
        tile34 = tile37 - 3
    elif 30 <= tile37 <= 36:
        tile34 = tile37 - 3
    else:
        tile34 = _TILE34_PAD
    return tile34


def _tile34_to_suit(tile34: int) -> int:
    if 0 <= tile34 <= 8:
        return _SUIT_MAN
    if 9 <= tile34 <= 17:
        return _SUIT_PIN
    if 18 <= tile34 <= 26:
        return _SUIT_SOU
    if 27 <= tile34 <= 33:
        return _SUIT_HONOR
    return _SUIT_PAD


def _tile34_to_rank(tile34: int) -> int:
    if 0 <= tile34 <= 26:
        return tile34 % 9
    if 27 <= tile34 <= 33:
        return _RANK_NONE
    return _RANK_PAD


def _tile34_to_honor_kind(tile34: int) -> int:
    if 27 <= tile34 <= 33:
        return tile34 - 27
    if 0 <= tile34 <= 26:
        return _HONOR_NONE
    return _HONOR_PAD


def _tile34_to_class(tile34: int) -> int:
    if 0 <= tile34 <= 26:
        rank = tile34 % 9
        if rank in (0, 8):
            return _TILE_CLASS_TERMINAL
        return _TILE_CLASS_SIMPLE
    if 27 <= tile34 <= 30:
        return _TILE_CLASS_WIND
    if 31 <= tile34 <= 33:
        return _TILE_CLASS_DRAGON
    return _TILE_CLASS_PAD


def _next_tile34(tile34: int) -> int:
    if 0 <= tile34 <= 8:
        return 0 if tile34 == 8 else tile34 + 1
    if 9 <= tile34 <= 17:
        return 9 if tile34 == 17 else tile34 + 1
    if 18 <= tile34 <= 26:
        return 18 if tile34 == 26 else tile34 + 1
    if 27 <= tile34 <= 30:
        return 27 if tile34 == 30 else tile34 + 1
    if 31 <= tile34 <= 33:
        return 31 if tile34 == 33 else tile34 + 1
    return _TILE34_PAD


def _build_tile37_lookups() -> dict[str, torch.Tensor]:
    tile34 = [_TILE34_PAD] * SequenceFeatureEncoder.HAND_DIMS[0]
    suit = [_SUIT_PAD] * SequenceFeatureEncoder.HAND_DIMS[0]
    rank = [_RANK_PAD] * SequenceFeatureEncoder.HAND_DIMS[0]
    honor_kind = [_HONOR_PAD] * SequenceFeatureEncoder.HAND_DIMS[0]
    red_flag = [_RED_FLAG_PAD] * SequenceFeatureEncoder.HAND_DIMS[0]
    tile_class = [_TILE_CLASS_PAD] * SequenceFeatureEncoder.HAND_DIMS[0]

    for tile37 in range(_TILE37_PAD):
        t34 = _tile37_to_tile34(tile37)
        tile34[tile37] = t34
        suit[tile37] = _tile34_to_suit(t34)
        rank[tile37] = _tile34_to_rank(t34)
        honor_kind[tile37] = _tile34_to_honor_kind(t34)
        red_flag[tile37] = _RED_FLAG_RED if tile37 in (0, 10, 20) else _RED_FLAG_NORMAL
        tile_class[tile37] = _tile34_to_class(t34)

    return {
        "tile34": torch.tensor(tile34, dtype=torch.long),
        "suit": torch.tensor(suit, dtype=torch.long),
        "rank": torch.tensor(rank, dtype=torch.long),
        "honor_kind": torch.tensor(honor_kind, dtype=torch.long),
        "red_flag": torch.tensor(red_flag, dtype=torch.long),
        "tile_class": torch.tensor(tile_class, dtype=torch.long),
    }


def _build_tile34_lookups() -> dict[str, torch.Tensor]:
    tile34 = [_TILE34_PAD] * (_TILE34_PAD + 1)
    suit = [_SUIT_PAD] * (_TILE34_PAD + 1)
    rank = [_RANK_PAD] * (_TILE34_PAD + 1)
    honor_kind = [_HONOR_PAD] * (_TILE34_PAD + 1)
    red_flag = [_RED_FLAG_PAD] * (_TILE34_PAD + 1)
    tile_class = [_TILE_CLASS_PAD] * (_TILE34_PAD + 1)

    for t34 in range(_TILE34_PAD):
        tile34[t34] = t34
        suit[t34] = _tile34_to_suit(t34)
        rank[t34] = _tile34_to_rank(t34)
        honor_kind[t34] = _tile34_to_honor_kind(t34)
        red_flag[t34] = _RED_FLAG_PAD
        tile_class[t34] = _tile34_to_class(t34)

    return {
        "tile34": torch.tensor(tile34, dtype=torch.long),
        "suit": torch.tensor(suit, dtype=torch.long),
        "rank": torch.tensor(rank, dtype=torch.long),
        "honor_kind": torch.tensor(honor_kind, dtype=torch.long),
        "red_flag": torch.tensor(red_flag, dtype=torch.long),
        "tile_class": torch.tensor(tile_class, dtype=torch.long),
    }


def _build_sparse_dora_lookups() -> dict[str, torch.Tensor]:
    vocab = SequenceFeatureEncoder.SPARSE_VOCAB_SIZE
    indicator_tile37 = [_TILE37_PAD] * vocab
    dora_slot = [_DORA_SLOT_PAD] * vocab
    dora_tile34 = [_TILE34_PAD] * vocab

    for slot in range(_SPARSE_DORA_SLOTS):
        base = _SPARSE_DORA_OFFSET + slot * _TILE37_PAD
        for tile37 in range(_TILE37_PAD):
            token = base + tile37
            indicator_tile37[token] = tile37
            dora_slot[token] = slot
            dora_tile34[token] = _next_tile34(_tile37_to_tile34(tile37))

    return {
        "indicator_tile37": torch.tensor(indicator_tile37, dtype=torch.long),
        "dora_slot": torch.tensor(dora_slot, dtype=torch.long),
        "dora_tile34": torch.tensor(dora_tile34, dtype=torch.long),
    }


def _build_prog_type_lookups(vocab_size: int) -> dict[str, torch.Tensor]:
    action_kind = [_ACTION_KIND_PAD] * vocab_size
    tile37 = [_TILE37_PAD] * vocab_size
    tile34 = [_TILE34_PAD] * vocab_size

    for k37 in range(_TILE37_PAD):
        for idx, kind in (
            (1 + k37, _ACTION_KIND_DISCARD),
            (168 + k37, _ACTION_KIND_DAIMINKAN),
            (239 + k37, _ACTION_KIND_KAKAN),
        ):
            if idx < vocab_size:
                action_kind[idx] = kind
                tile37[idx] = k37

    for t34 in range(_TILE34_PAD):
        idx = 205 + t34
        if idx < vocab_size:
            action_kind[idx] = _ACTION_KIND_ANKAN
            tile34[idx] = t34

    return {
        "action_kind": torch.tensor(action_kind, dtype=torch.long),
        "tile37": torch.tensor(tile37, dtype=torch.long),
        "tile34": torch.tensor(tile34, dtype=torch.long),
    }


def _build_cand_type_lookups(vocab_size: int) -> dict[str, torch.Tensor]:
    action_kind = [_ACTION_KIND_PAD] * vocab_size
    tile37 = [_TILE37_PAD] * vocab_size
    tile34 = [_TILE34_PAD] * vocab_size

    for k37 in range(_TILE37_PAD):
        for idx, kind in (
            (k37, _ACTION_KIND_DISCARD),
            (71 + k37, _ACTION_KIND_KAKAN),
            (241 + k37, _ACTION_KIND_DAIMINKAN),
        ):
            if idx < vocab_size:
                action_kind[idx] = kind
                tile37[idx] = k37

    for t34 in range(_TILE34_PAD):
        idx = 37 + t34
        if idx < vocab_size:
            action_kind[idx] = _ACTION_KIND_ANKAN
            tile34[idx] = t34

    return {
        "action_kind": torch.tensor(action_kind, dtype=torch.long),
        "tile37": torch.tensor(tile37, dtype=torch.long),
        "tile34": torch.tensor(tile34, dtype=torch.long),
    }


class SharedTileEmbedding(nn.Module):
    """Encode tiles via shared attribute embeddings."""

    def __init__(self, out_dim: int, attr_dim: int):
        super().__init__()
        self.tile34_embed = nn.Embedding(_TILE34_PAD + 1, attr_dim, padding_idx=_TILE34_PAD)
        self.suit_embed = nn.Embedding(_SUIT_PAD + 1, attr_dim, padding_idx=_SUIT_PAD)
        self.rank_embed = nn.Embedding(_RANK_PAD + 1, attr_dim, padding_idx=_RANK_PAD)
        self.honor_kind_embed = nn.Embedding(_HONOR_PAD + 1, attr_dim, padding_idx=_HONOR_PAD)
        self.red_flag_embed = nn.Embedding(_RED_FLAG_PAD + 1, attr_dim, padding_idx=_RED_FLAG_PAD)
        self.tile_class_embed = nn.Embedding(_TILE_CLASS_PAD + 1, attr_dim, padding_idx=_TILE_CLASS_PAD)
        self.dora_flag_embed = nn.Embedding(_DORA_FLAG_PAD + 1, attr_dim, padding_idx=_DORA_FLAG_PAD)
        self.proj = nn.Sequential(
            nn.Linear(attr_dim * 7, out_dim),
            nn.LayerNorm(out_dim),
        )

        for name, value in _build_tile37_lookups().items():
            self.register_buffer(f"tile37_{name}", value, persistent=False)
        for name, value in _build_tile34_lookups().items():
            self.register_buffer(f"tile34_{name}", value, persistent=False)

    def _compute_dora_flag(
        self,
        tile34: torch.Tensor,
        red_flag: torch.Tensor,
        pad_mask: torch.Tensor,
        dora_tile34: torch.Tensor,
    ) -> torch.Tensor:
        tile_dora = (tile34.unsqueeze(-1) == dora_tile34.unsqueeze(1)).any(dim=-1)
        is_red = red_flag == _RED_FLAG_RED
        dora_flag = torch.where(tile_dora | is_red, _DORA_FLAG_DORA, _DORA_FLAG_NONE)
        return torch.where(pad_mask, _DORA_FLAG_PAD, dora_flag)

    def _embed_attributes(
        self,
        tile34: torch.Tensor,
        suit: torch.Tensor,
        rank: torch.Tensor,
        honor_kind: torch.Tensor,
        red_flag: torch.Tensor,
        tile_class: torch.Tensor,
        pad_mask: torch.Tensor,
        dora_tile34: torch.Tensor,
    ) -> torch.Tensor:
        dora_flag = self._compute_dora_flag(tile34, red_flag, pad_mask, dora_tile34)
        parts = [
            self.tile34_embed(tile34),
            self.suit_embed(suit),
            self.rank_embed(rank),
            self.honor_kind_embed(honor_kind),
            self.red_flag_embed(red_flag),
            self.tile_class_embed(tile_class),
            self.dora_flag_embed(dora_flag),
        ]
        return self.proj(torch.cat(parts, dim=-1))

    def embed_tile37(self, tile37: torch.Tensor, dora_tile34: torch.Tensor) -> torch.Tensor:
        pad_mask = tile37 == _TILE37_PAD
        return self._embed_attributes(
            self.tile37_tile34[tile37],
            self.tile37_suit[tile37],
            self.tile37_rank[tile37],
            self.tile37_honor_kind[tile37],
            self.tile37_red_flag[tile37],
            self.tile37_tile_class[tile37],
            pad_mask,
            dora_tile34,
        )

    def embed_tile34(self, tile34: torch.Tensor, dora_tile34: torch.Tensor) -> torch.Tensor:
        pad_mask = tile34 == _TILE34_PAD
        return self._embed_attributes(
            self.tile34_tile34[tile34],
            self.tile34_suit[tile34],
            self.tile34_rank[tile34],
            self.tile34_honor_kind[tile34],
            self.tile34_red_flag[tile34],
            self.tile34_tile_class[tile34],
            pad_mask,
            dora_tile34,
        )


class TransformerActorCritic(nn.Module):
    """Transformer Actor-Critic over packed sequence features.

    Input:  (B, PACKED_SIZE)  float32 — from SequenceFeaturePackedEncoder
    Output: (logits, value) tuple — (B, num_actions), (B,)

    V2 defaults: d_model=384, max_prog_len=256, max_cand_len=32, d_type=96, d_other=32
    V1 compat:   pass d_sub=32, max_prog_len=512, max_cand_len=64
    """

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        num_actions: int = 82,
        # Policy head type: "cls" (V2) or "cross_attn" (V3)
        policy_head_type: str = "cross_attn",
        emit_value: bool = True,
        # Embedding sub-dimensions (asymmetric)
        d_sub: int | None = None,   # V1 compat: if set, d_type=d_other=d_sub
        d_type: int = 96,           # type field embedding dim
        d_other: int = 32,          # other field embedding dim
        # Sequence length (must match encoder)
        max_prog_len: int = 256,
        max_cand_len: int = 32,
        # Vocab sizes (from SequenceFeatureEncoder)
        sparse_vocab: int = SequenceFeatureEncoder.SPARSE_VOCAB_SIZE,   # 549
        sparse_pad: int = SequenceFeatureEncoder.SPARSE_PAD,            # 548
        hand_dims: tuple = SequenceFeatureEncoder.HAND_DIMS,            # (38,3)
        prog_dims: tuple = SequenceFeatureEncoder.PROG_DIMS,            # (5,277,3,3,5)
        cand_dims: tuple = SequenceFeatureEncoder.CAND_DIMS,            # (280,3,3,4)
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.policy_head_type = policy_head_type
        self.emit_value = emit_value

        # V1 backward compat: uniform d_sub overrides asymmetric dims
        if d_sub is not None:
            d_type = d_sub
            d_other = d_sub

        # Packed layout constants (must match SequenceFeaturePackedEncoder)
        self._S = SequenceFeatureEncoder.MAX_SPARSE_LEN   # 14
        self._H = SequenceFeatureEncoder.MAX_HAND_LEN     # 14
        self._N = SequenceFeatureEncoder.NUM_NUMERIC       # 12
        self._P = max_prog_len
        self._C = max_cand_len

        # --- Embedding layers ---
        self.sparse_embed = nn.Embedding(
            sparse_vocab, d_model, padding_idx=sparse_pad)
        self.tile_embed = SharedTileEmbedding(out_dim=d_type, attr_dim=d_other)
        self.tile_action_kind_embed = nn.Embedding(
            _ACTION_KIND_PAD + 1, d_other, padding_idx=_ACTION_KIND_PAD)
        self.tile_action_proj = nn.Sequential(
            nn.Linear(d_type + d_other, d_type),
            nn.LayerNorm(d_type),
        )

        # Hand: embed (tile37, draw_state) → concat → project
        hand_sub_dims = [d_type, d_other]
        self.hand_draw_state_embed = nn.Embedding(hand_dims[1], d_other)
        self.hand_proj = nn.Sequential(
            nn.Linear(sum(hand_sub_dims), d_model),
            nn.LayerNorm(d_model),
        )

        self.numeric_proj = nn.Sequential(
            nn.Linear(self._N, d_model),
            nn.LayerNorm(d_model),
        )

        # Progression: embed each of 5 fields → concat → project
        # field[1] is type (vocab=277) → d_type; others → d_other
        prog_sub_dims = [d_other if i != 1 else d_type for i in range(len(prog_dims))]
        self.prog_embeds = nn.ModuleList([
            nn.Embedding(dim, d_s) for dim, d_s in zip(prog_dims, prog_sub_dims)
        ])
        prog_cat_dim = sum(prog_sub_dims)
        self.prog_proj = nn.Sequential(
            nn.Linear(prog_cat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Candidates: embed each of 4 fields → concat → project
        # field[0] is type (vocab=280) → d_type; others → d_other
        cand_sub_dims = [d_other if i != 0 else d_type for i in range(len(cand_dims))]
        self.cand_embeds = nn.ModuleList([
            nn.Embedding(dim, d_s) for dim, d_s in zip(cand_dims, cand_sub_dims)
        ])
        cand_cat_dim = sum(cand_sub_dims)
        self.cand_proj = nn.Sequential(
            nn.Linear(cand_cat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.dora_slot_embed = nn.Embedding(
            _DORA_SLOT_PAD + 1, d_other, padding_idx=_DORA_SLOT_PAD)
        self.sparse_dora_proj = nn.Sequential(
            nn.Linear(d_type + d_other, d_model),
            nn.LayerNorm(d_model),
        )

        # --- CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- Segment embeddings (5 groups: sparse / hand / numeric / prog / cand) ---
        self.segment_embed = nn.Embedding(5, d_model)

        # --- Positional encoding (sinusoidal) ---
        max_seq = 1 + self._S + self._H + 1 + self._P + self._C
        self.register_buffer("pos_enc", self._sinusoidal_pe(max_seq, d_model))

        # --- Transformer encoder (pre-LN for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model)

        # --- Cross-attention for policy head (V3) ---
        if self.policy_head_type == "cross_attn":
            self.cand_cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn_norm = nn.LayerNorm(d_model)

        # --- Output heads ---
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_actions),
        )
        self.value_head = None
        if self.emit_value:
            self.value_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        for name, value in _build_sparse_dora_lookups().items():
            self.register_buffer(f"sparse_{name}", value, persistent=False)
        for name, value in _build_prog_type_lookups(prog_dims[1]).items():
            self.register_buffer(f"prog_type_{name}", value, persistent=False)
        for name, value in _build_cand_type_lookups(cand_dims[0]).items():
            self.register_buffer(f"cand_type_{name}", value, persistent=False)

        self._init_weights()

    # ------------------------------------------------------------------
    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    def _unpack(self, x: torch.Tensor):
        """Unpack flat (B, PACKED_SIZE) tensor into components."""
        o = 0
        sparse = x[:, o:o + self._S].long()
        o += self._S
        hand = x[:, o:o + self._H * 2].reshape(-1, self._H, 2).long()
        o += self._H * 2
        numeric = x[:, o:o + self._N]
        o += self._N
        prog = x[:, o:o + self._P * 5].reshape(-1, self._P, 5).long()
        o += self._P * 5
        cand = x[:, o:o + self._C * 4].reshape(-1, self._C, 4).long()
        o += self._C * 4
        sparse_mask = x[:, o:o + self._S].bool()
        o += self._S
        hand_mask = x[:, o:o + self._H].bool()
        o += self._H
        prog_mask = x[:, o:o + self._P].bool()
        o += self._P
        cand_mask = x[:, o:o + self._C].bool()
        return sparse, hand, numeric, prog, cand, sparse_mask, hand_mask, prog_mask, cand_mask

    # ------------------------------------------------------------------
    def _decode_current_dora_tiles(self, sparse: torch.Tensor) -> torch.Tensor:
        """Return current dora tiles as tile34 ids, one slot per sparse token."""
        return self.sparse_dora_tile34[sparse]

    def _embed_sparse(self, sparse: torch.Tensor, dora_tile34: torch.Tensor) -> torch.Tensor:
        sparse_emb = self.sparse_embed(sparse)
        dora_slot = self.sparse_dora_slot[sparse]
        dora_mask = dora_slot != _DORA_SLOT_PAD
        if not torch.any(dora_mask):
            return sparse_emb

        indicator_tile37 = self.sparse_indicator_tile37[sparse]
        tile_emb = self.tile_embed.embed_tile37(indicator_tile37, dora_tile34)
        slot_emb = self.dora_slot_embed(dora_slot)
        dora_emb = self.sparse_dora_proj(torch.cat([tile_emb, slot_emb], dim=-1))
        return torch.where(dora_mask.unsqueeze(-1), dora_emb, sparse_emb)

    def _embed_tile_only_action_type(
        self,
        type_ids: torch.Tensor,
        generic_embed: nn.Embedding,
        action_kind_lookup: torch.Tensor,
        tile37_lookup: torch.Tensor,
        tile34_lookup: torch.Tensor,
        dora_tile34: torch.Tensor,
    ) -> torch.Tensor:
        type_emb = generic_embed(type_ids)
        action_kind = action_kind_lookup[type_ids]
        target_mask = action_kind != _ACTION_KIND_PAD
        if not torch.any(target_mask):
            return type_emb

        tile37 = tile37_lookup[type_ids]
        tile34 = tile34_lookup[type_ids]
        tile37_emb = self.tile_embed.embed_tile37(tile37, dora_tile34)
        tile34_emb = self.tile_embed.embed_tile34(tile34, dora_tile34)
        tile_emb = torch.where((tile37 != _TILE37_PAD).unsqueeze(-1), tile37_emb, tile34_emb)
        action_emb = self.tile_action_kind_embed(action_kind)
        target_emb = self.tile_action_proj(torch.cat([tile_emb, action_emb], dim=-1))
        return torch.where(target_mask.unsqueeze(-1), target_emb, type_emb)

    def _embed_hand(self, hand: torch.Tensor, dora_tile34: torch.Tensor) -> torch.Tensor:
        tile_emb = self.tile_embed.embed_tile37(hand[:, :, 0], dora_tile34)
        draw_state_emb = self.hand_draw_state_embed(hand[:, :, 1])
        return self.hand_proj(torch.cat([tile_emb, draw_state_emb], dim=-1))

    def _embed_progression(self, prog: torch.Tensor, dora_tile34: torch.Tensor) -> torch.Tensor:
        prog_parts = []
        for i, emb in enumerate(self.prog_embeds):
            if i == 1:
                prog_parts.append(
                    self._embed_tile_only_action_type(
                        prog[:, :, i],
                        emb,
                        self.prog_type_action_kind,
                        self.prog_type_tile37,
                        self.prog_type_tile34,
                        dora_tile34,
                    )
                )
            else:
                prog_parts.append(emb(prog[:, :, i]))
        return self.prog_proj(torch.cat(prog_parts, dim=-1))

    def _embed_candidates(self, cand: torch.Tensor, dora_tile34: torch.Tensor) -> torch.Tensor:
        cand_parts = []
        for i, emb in enumerate(self.cand_embeds):
            if i == 0:
                cand_parts.append(
                    self._embed_tile_only_action_type(
                        cand[:, :, i],
                        emb,
                        self.cand_type_action_kind,
                        self.cand_type_tile37,
                        self.cand_type_tile34,
                        dora_tile34,
                    )
                )
            else:
                cand_parts.append(emb(cand[:, :, i]))
        return self.cand_proj(torch.cat(cand_parts, dim=-1))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        sparse, hand, numeric, prog, cand, sparse_mask, hand_mask, prog_mask, cand_mask = \
            self._unpack(x)
        dora_tile34 = self._decode_current_dora_tiles(sparse)

        # Embed sparse tokens: (B, S, d)
        sparse_emb = self._embed_sparse(sparse, dora_tile34)

        # Embed hand tuples: (B, H, d)
        hand_emb = self._embed_hand(hand, dora_tile34)

        # Project numeric: (B, 1, d)
        numeric_emb = self.numeric_proj(numeric).unsqueeze(1)

        # Embed progression 5-tuples: (B, P, d)
        prog_emb = self._embed_progression(prog, dora_tile34)

        # Embed candidate 4-tuples: (B, C, d)
        cand_emb = self._embed_candidates(cand, dora_tile34)

        # CLS token: (B, 1, d)
        cls = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate: [CLS, sparse(S), hand(H), numeric(1), prog(P), cand(C)]
        tokens = torch.cat([cls, sparse_emb, hand_emb, numeric_emb, prog_emb, cand_emb], dim=1)

        # Add segment embeddings
        seg_ids = torch.cat([
            torch.zeros(batch_size, 1 + self._S, dtype=torch.long, device=x.device),
            torch.ones(batch_size, self._H, dtype=torch.long, device=x.device),
            torch.full((batch_size, 1), 2, dtype=torch.long, device=x.device),
            torch.full((batch_size, self._P), 3, dtype=torch.long, device=x.device),
            torch.full((batch_size, self._C), 4, dtype=torch.long, device=x.device),
        ], dim=1)
        tokens = tokens + self.segment_embed(seg_ids)

        # Add positional encoding
        tokens = tokens + self.pos_enc[:, :tokens.shape[1]]

        # Build padding mask: True = ignore (PyTorch convention)
        cls_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        numeric_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([
            cls_valid,         # CLS is always valid
            ~sparse_mask,      # True where sparse is padding
            ~hand_mask,        # True where hand is padding
            numeric_valid,     # numeric is always valid
            ~prog_mask,        # True where prog is padding
            ~cand_mask,        # True where cand is padding
        ], dim=1)

        # Transformer
        output = self.transformer(tokens, src_key_padding_mask=pad_mask)
        output = self.final_norm(output)

        # CLS output is shared by policy and value heads.
        cls_out = output[:, 0]

        # Policy head
        if self.policy_head_type == "cross_attn":
            # Cross-attention: CLS queries candidate token outputs
            cand_offset = 1 + self._S + self._H + 1 + self._P
            cand_out = output[:, cand_offset:cand_offset + self._C]  # (B, C, d_model)
            cls_q = cls_out.unsqueeze(1)  # (B, 1, d_model)
            cand_attn_mask = ~cand_mask   # True = padding (PyTorch convention)
            attn_out, _ = self.cand_cross_attn(
                cls_q, cand_out, cand_out,
                key_padding_mask=cand_attn_mask)  # (B, 1, d_model)
            policy_input = self.cross_attn_norm(cls_out + attn_out.squeeze(1))
        else:
            policy_input = cls_out

        logits = self.policy_head(policy_input)
        if not self.emit_value:
            return logits

        value = self.value_head(cls_out)
        return logits, value.squeeze(-1)


class TransformerPolicyNetwork(TransformerActorCritic):
    """Policy-only transformer over packed sequence features."""

    def __init__(self, **kwargs):
        super().__init__(emit_value=False, **kwargs)
