"""Transformer Actor-Critic for sequence feature encoding.

Accepts the feature dictionaries produced by SequenceFeatureEncoder, embeds
each group, and processes them through a TransformerEncoder.  Progression and
candidate groups are padded only to the maximum length in the current batch.

Tile-only fields across hand / sparse dora / progression / candidates share
an attribute-based tile embedding module. Chi/pon/kan melds use a shared
factorized embedding over meld kind, slot tiles, and slot roles. Dealer,
player summary, progression actor/from, candidate from, and sparse meld owner
fields share an observer-relative seat base embedding with role-specific context.
Agari-overtake features are reshaped into four winner-relative-seat tokens
with a shared projection and the shared relative-seat embedding.

Visible tile-count tokens embed each tile37 id through the shared tile embedding
and combine it with a count embedding shared by all tile types.

Progression order is encoded with learned relative attention bias over a
discard-count clock. Absolute event-index positional embeddings are not used.

Output: (logits, value) — same interface as ActorCriticNetwork.

NOTE: Sanma (3-player) is not supported. The sparse/progression/candidate
vocabularies and encoding logic assume 4-player mahjong.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from riichienv_ml.features.sequence_features import SequenceFeatureEncoder, packed_sequence_size


def _transformer_without_mha_fastpath(
    transformer: nn.TransformerEncoder,
    tokens: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    src_key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    mha_backend = getattr(torch.backends, "mha", None)
    if (
        mask is None
        or mha_backend is None
        or not hasattr(mha_backend, "get_fastpath_enabled")
        or not hasattr(mha_backend, "set_fastpath_enabled")
    ):
        return transformer(tokens, mask=mask, src_key_padding_mask=src_key_padding_mask)

    previous = bool(mha_backend.get_fastpath_enabled())
    mha_backend.set_fastpath_enabled(False)
    try:
        return transformer(tokens, mask=mask, src_key_padding_mask=src_key_padding_mask)
    finally:
        mha_backend.set_fastpath_enabled(previous)


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
_ACTION_KIND_DORA = 1
_ACTION_KIND_DAIMINKAN = 2
_ACTION_KIND_ANKAN = 3
_ACTION_KIND_KAKAN = 4
_ACTION_KIND_PAD = 5

_MELD_KIND_CHI = 0
_MELD_KIND_PON = 1
_MELD_KIND_DAIMINKAN = 2
_MELD_KIND_ANKAN = 3
_MELD_KIND_KAKAN = 4
_MELD_KIND_PAD = 5

_MELD_ROLE_CALLED = 0
_MELD_ROLE_CONSUMED = 1
_MELD_ROLE_ADDED = 2
_MELD_ROLE_PAD = 3
_MELD_WIDTH = 9

_SPARSE_DORA_OFFSET = 75
_SPARSE_DORA_SLOTS = 5
_DORA_SLOT_PAD = _SPARSE_DORA_SLOTS

_SEAT_ROLE_DEALER = 0
_SEAT_ROLE_PROG_ACTOR = 1
_SEAT_ROLE_PROG_FROM = 2
_SEAT_ROLE_CAND_FROM = 3
_SEAT_ROLE_MELD_OWNER = 4
_SEAT_ROLE_TILE_WIND_OWNER = 5
_SEAT_ROLE_AGARI_WINNER = 6
_SEAT_ROLE_PLAYER_INFO = 7
_SEAT_NUM_ROLES = 8
_SEAT_PAD_OR_NA = 4
_PLAYER_COUNT = 4

_ROUND_WIND_FLAG_YES = 0
_ROUND_WIND_FLAG_NO = 1
_ROUND_WIND_FLAG_PAD = 2


class SplitLinearLayerNorm(nn.Module):
    """Linear(concat(parts)) + LayerNorm, evaluated as a sum over field slices."""

    def __init__(self, input_dims: tuple[int, ...] | list[int], out_dim: int):
        super().__init__()
        self.input_dims = tuple(int(dim) for dim in input_dims)
        if not self.input_dims:
            raise ValueError("input_dims must not be empty")
        self.offsets = tuple([0, *torch.tensor(self.input_dims).cumsum(0).tolist()])
        self.weight = nn.Parameter(torch.empty(out_dim, sum(self.input_dims)))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.ln = nn.LayerNorm(out_dim)
        nn.init.trunc_normal_(self.weight, std=0.02)

    def _weight_slice(self, part_idx: int) -> torch.Tensor:
        start = self.offsets[part_idx]
        end = self.offsets[part_idx + 1]
        return self.weight[:, start:end]

    def project_part(self, part_idx: int, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._weight_slice(part_idx), None)

    def project_embedding_weight(self, part_idx: int, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(weight, self._weight_slice(part_idx), None)

    def project_embedding(self, part_idx: int, embedding: nn.Embedding, indices: torch.Tensor) -> torch.Tensor:
        projected_weight = self.project_embedding_weight(part_idx, embedding.weight)
        return F.embedding(indices, projected_weight, padding_idx=embedding.padding_idx)

    def finish(self, raw: torch.Tensor) -> torch.Tensor:
        return self.ln(raw + self.bias)

    def forward(self, parts: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(parts) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} parts, got {len(parts)}")
        raw = self.project_part(0, parts[0])
        for part_idx, part in enumerate(parts[1:], start=1):
            raw = raw + self.project_part(part_idx, part)
        return self.finish(raw)


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
        discard_idx = 1 + k37
        if discard_idx < vocab_size:
            action_kind[discard_idx] = _ACTION_KIND_DISCARD
            tile37[discard_idx] = k37

        dora_idx = 43 + k37
        if dora_idx < vocab_size:
            action_kind[dora_idx] = _ACTION_KIND_DORA
            tile37[dora_idx] = k37

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
        idx = k37
        if idx < vocab_size:
            action_kind[idx] = _ACTION_KIND_DISCARD
            tile37[idx] = k37

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
        self.round_wind_flag_embed = nn.Embedding(
            _ROUND_WIND_FLAG_PAD + 1,
            attr_dim,
            padding_idx=_ROUND_WIND_FLAG_PAD,
        )
        self.proj = SplitLinearLayerNorm([attr_dim] * 9, out_dim)

        for name, value in _build_tile37_lookups().items():
            self.register_buffer(f"tile37_{name}", value, persistent=False)
        for name, value in _build_tile34_lookups().items():
            self.register_buffer(f"tile34_{name}", value, persistent=False)

    @staticmethod
    def _gather_table(table: torch.Tensor, tile_ids: torch.Tensor) -> torch.Tensor:
        batch_size = table.shape[0]
        flat_ids = tile_ids.reshape(batch_size, -1)
        gather_ids = flat_ids.unsqueeze(-1).expand(-1, -1, table.shape[-1])
        gathered = table.gather(1, gather_ids)
        return gathered.reshape(*tile_ids.shape, table.shape[-1])

    @staticmethod
    def _broadcast_context(context: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while context.ndim < target_ndim:
            context = context.unsqueeze(1)
        return context

    def _zero_attribute(self, tile34: torch.Tensor) -> torch.Tensor:
        return self.tile34_embed.weight.new_zeros(*tile34.shape, self.tile34_embed.embedding_dim)

    def _compute_dora_flag(
        self,
        tile34: torch.Tensor,
        red_flag: torch.Tensor,
        pad_mask: torch.Tensor,
        dora_tile34: torch.Tensor,
    ) -> torch.Tensor:
        dora = dora_tile34
        for _ in range(tile34.ndim - 1):
            dora = dora.unsqueeze(1)
        tile_dora = (tile34.unsqueeze(-1) == dora).any(dim=-1)
        is_red = red_flag == _RED_FLAG_RED
        dora_flag = torch.where(tile_dora | is_red, _DORA_FLAG_DORA, _DORA_FLAG_NONE)
        return torch.where(pad_mask, _DORA_FLAG_PAD, dora_flag)

    def _compute_wind_attributes(
        self,
        tile34: torch.Tensor,
        pad_mask: torch.Tensor,
        dealer: torch.Tensor | None,
        round_wind: torch.Tensor | None,
        relative_seat_embed: nn.Module | None,
        seat_other_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_wind = (tile34 >= 27) & (tile34 <= 30) & ~pad_mask
        wind_index = (tile34 - 27).clamp(min=0, max=3)

        if dealer is None or (relative_seat_embed is None and seat_other_table is None):
            wind_owner_emb = self._zero_attribute(tile34)
        else:
            dealer_ctx = self._broadcast_context(dealer.clamp(min=0, max=3), tile34.ndim)
            wind_owner = (dealer_ctx + wind_index) % 4
            wind_owner = torch.where(is_wind, wind_owner, torch.full_like(tile34, _SEAT_PAD_OR_NA))
            if seat_other_table is None:
                wind_owner_emb = relative_seat_embed(wind_owner, _SEAT_ROLE_TILE_WIND_OWNER, out="other")
            else:
                wind_owner_emb = RelativeSeatEmbedding.embed_from_table(
                    seat_other_table,
                    wind_owner,
                    _SEAT_ROLE_TILE_WIND_OWNER,
                )

        if round_wind is None:
            round_flag = torch.full_like(tile34, _ROUND_WIND_FLAG_PAD)
        else:
            round_wind_ctx = self._broadcast_context(round_wind.clamp(min=0, max=3), tile34.ndim)
            round_flag = torch.where(
                wind_index == round_wind_ctx,
                torch.full_like(tile34, _ROUND_WIND_FLAG_YES),
                torch.full_like(tile34, _ROUND_WIND_FLAG_NO),
            )
            round_flag = torch.where(is_wind, round_flag, torch.full_like(tile34, _ROUND_WIND_FLAG_PAD))

        return wind_owner_emb, self.round_wind_flag_embed(round_flag)

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
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        relative_seat_embed: nn.Module | None = None,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dora_flag = self._compute_dora_flag(tile34, red_flag, pad_mask, dora_tile34)
        wind_owner_emb, round_wind_flag_emb = self._compute_wind_attributes(
            tile34,
            pad_mask,
            dealer,
            round_wind,
            relative_seat_embed,
            seat_other_table,
        )
        raw = (
            self.proj.project_embedding(0, self.tile34_embed, tile34)
            + self.proj.project_embedding(1, self.suit_embed, suit)
            + self.proj.project_embedding(2, self.rank_embed, rank)
            + self.proj.project_embedding(3, self.honor_kind_embed, honor_kind)
            + self.proj.project_embedding(4, self.red_flag_embed, red_flag)
            + self.proj.project_embedding(5, self.tile_class_embed, tile_class)
            + self.proj.project_embedding(6, self.dora_flag_embed, dora_flag)
            + self.proj.project_part(7, wind_owner_emb)
            + self.proj.project_part(8, round_wind_flag_emb)
        )
        return self.proj.finish(raw)

    def embed_tile37(
        self,
        tile37: torch.Tensor,
        dora_tile34: torch.Tensor,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        relative_seat_embed: nn.Module | None = None,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            dealer,
            round_wind,
            relative_seat_embed,
            seat_other_table,
        )

    def embed_tile34(
        self,
        tile34: torch.Tensor,
        dora_tile34: torch.Tensor,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        relative_seat_embed: nn.Module | None = None,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            dealer,
            round_wind,
            relative_seat_embed,
            seat_other_table,
        )

    def build_tables(
        self,
        dora_tile34: torch.Tensor,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        relative_seat_embed: nn.Module | None = None,
        seat_other_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = dora_tile34.shape[0]
        tile37 = torch.arange(_TILE37_PAD + 1, device=dora_tile34.device).expand(batch_size, -1)
        tile34 = torch.arange(_TILE34_PAD + 1, device=dora_tile34.device).expand(batch_size, -1)
        return (
            self.embed_tile37(tile37, dora_tile34, dealer, round_wind, relative_seat_embed, seat_other_table),
            self.embed_tile34(tile34, dora_tile34, dealer, round_wind, relative_seat_embed, seat_other_table),
        )

    def embed_tile37_from_table(self, tile37: torch.Tensor, tile37_table: torch.Tensor) -> torch.Tensor:
        return self._gather_table(tile37_table, tile37)

    def embed_tile34_from_table(self, tile34: torch.Tensor, tile34_table: torch.Tensor) -> torch.Tensor:
        return self._gather_table(tile34_table, tile34)


class SharedMeldEmbedding(nn.Module):
    """Encode chi/pon/kan melds from tile slots, slot roles, and meld kind."""

    def __init__(self, out_dim: int, role_dim: int):
        super().__init__()
        self.kind_embed = nn.Embedding(_MELD_KIND_PAD + 1, role_dim, padding_idx=_MELD_KIND_PAD)
        self.role_embed = nn.Embedding(_MELD_ROLE_PAD + 1, role_dim, padding_idx=_MELD_ROLE_PAD)
        self.slot_proj = SplitLinearLayerNorm([out_dim, role_dim], out_dim)
        self.proj = SplitLinearLayerNorm([out_dim, out_dim, out_dim, out_dim, role_dim], out_dim)

    def forward(
        self,
        meld: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile_embed: SharedTileEmbedding,
        tile37_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        relative_seat_embed: nn.Module | None = None,
    ) -> torch.Tensor:
        kind = meld[..., 0]
        slot_tiles = torch.stack([meld[..., 1], meld[..., 3], meld[..., 5], meld[..., 7]], dim=-1)
        slot_roles = torch.stack([meld[..., 2], meld[..., 4], meld[..., 6], meld[..., 8]], dim=-1)
        slot_mask = slot_tiles != _TILE37_PAD

        if tile37_table is None:
            tile_emb = tile_embed.embed_tile37(slot_tiles, dora_tile34, dealer, round_wind, relative_seat_embed)
        else:
            tile_emb = tile_embed.embed_tile37_from_table(slot_tiles, tile37_table)
        slot_raw = self.slot_proj.project_part(0, tile_emb) + self.slot_proj.project_embedding(
            1,
            self.role_embed,
            slot_roles,
        )
        slot_emb = self.slot_proj.finish(slot_raw)
        slot_emb = torch.where(slot_mask.unsqueeze(-1), slot_emb, torch.zeros_like(slot_emb))

        raw = (
            self.proj.project_part(0, slot_emb[..., 0, :])
            + self.proj.project_part(1, slot_emb[..., 1, :])
            + self.proj.project_part(2, slot_emb[..., 2, :])
            + self.proj.project_part(3, slot_emb[..., 3, :])
            + self.proj.project_embedding(4, self.kind_embed, kind)
        )
        out = self.proj.finish(raw)
        meld_mask = kind != _MELD_KIND_PAD
        return torch.where(meld_mask.unsqueeze(-1), out, torch.zeros_like(out))


class RelativeSeatEmbedding(nn.Module):
    """Shared observer-relative seat embedding with role-specific context.

    Only real seats 0..3 share the base table. Value 4 is treated as
    padding/N/A/marker and returns zero; the corresponding action type or mask
    carries the special meaning.
    """

    def __init__(self, base_dim: int, d_model: int):
        super().__init__()
        self.base_embed = nn.Embedding(4, base_dim)
        self.role_embed = nn.Embedding(_SEAT_NUM_ROLES, base_dim)
        self.other_proj = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
        )
        self.model_proj = nn.Sequential(
            nn.Linear(base_dim, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def embed_from_table(table: torch.Tensor, seats: torch.Tensor, role: int) -> torch.Tensor:
        safe_seats = seats.clamp(min=0, max=_SEAT_PAD_OR_NA)
        return F.embedding(safe_seats, table[role])

    def build_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        seats = torch.arange(_SEAT_PAD_OR_NA + 1, dtype=torch.long, device=self.base_embed.weight.device)
        roles = torch.arange(_SEAT_NUM_ROLES, dtype=torch.long, device=self.base_embed.weight.device)
        valid = seats < _SEAT_PAD_OR_NA
        safe_seats = seats.clamp(min=0, max=3)
        emb = self.base_embed(safe_seats).unsqueeze(0) + self.role_embed(roles).unsqueeze(1)
        other = self.other_proj(emb)
        model = self.model_proj(emb)
        valid_mask = valid.reshape(1, -1, 1)
        return (
            torch.where(valid_mask, other, torch.zeros_like(other)),
            torch.where(valid_mask, model, torch.zeros_like(model)),
        )

    def forward(self, seats: torch.Tensor, role: int, *, out: str) -> torch.Tensor:
        table_other, table_model = self.build_tables()
        if out == "other":
            return self.embed_from_table(table_other, seats, role)
        elif out == "model":
            return self.embed_from_table(table_model, seats, role)
        else:
            raise ValueError(f"unsupported relative seat embedding output: {out}")


class TransformerActorCritic(nn.Module):
    """Transformer Actor-Critic over sequence feature dictionaries.

    Input:  dict[str, Tensor] — from SequenceFeatureEncoder and its collator
    Output: (logits, value) tuple. For policy_head_type="pointer",
    logits are (B, batch_max_cand_len); otherwise logits are (B, num_actions).
    """

    def __init__(  # noqa: PLR0915
        self,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        num_actions: int = 82,
        # Policy head type: "pointer" (candidate logits), "cls", or "cross_attn"
        policy_head_type: str = "pointer",
        emit_value: bool = True,
        # Embedding sub-dimensions (asymmetric)
        d_sub: int | None = None,  # V1 compat: if set, d_type=d_other=d_sub
        d_type: int = 96,  # type field embedding dim
        d_other: int = 32,  # other field embedding dim
        progression_relative_position_cap: int = 60,
        progression_player_relative_position_cap: int = 15,
        # Accepted for old configs; sequence lengths are now dynamic.
        max_prog_len: int | None = None,
        max_cand_len: int | None = None,
        # Vocab sizes (from SequenceFeatureEncoder)
        sparse_vocab: int = SequenceFeatureEncoder.SPARSE_VOCAB_SIZE,  # 261
        sparse_pad: int = SequenceFeatureEncoder.SPARSE_PAD,  # 260
        player_info_dims: tuple = SequenceFeatureEncoder.PLAYER_INFO_DIMS,  # (4,2,5,21,21)
        hand_dims: tuple = SequenceFeatureEncoder.HAND_DIMS,  # (38,3)
        visible_tile_count_dims: tuple = SequenceFeatureEncoder.VISIBLE_TILE_COUNT_DIMS,  # (37,5)
        prog_dims: tuple = SequenceFeatureEncoder.PROG_DIMS,  # (5,80,3,3,5)
        cand_dims: tuple = SequenceFeatureEncoder.CAND_DIMS,  # (47,3,5)
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = int(nhead)
        self.num_actions = num_actions
        self.policy_head_type = policy_head_type
        self.emit_value = emit_value
        self.progression_relative_position_cap = int(progression_relative_position_cap)
        if self.progression_relative_position_cap < 0:
            raise ValueError("progression_relative_position_cap must be non-negative")
        self._RPB = self.progression_relative_position_cap * 2 + 1
        self.progression_player_relative_position_cap = int(progression_player_relative_position_cap)
        if self.progression_player_relative_position_cap < 0:
            raise ValueError("progression_player_relative_position_cap must be non-negative")
        self._PLAYER_RPB = self.progression_player_relative_position_cap * 2 + 1

        # V1 backward compat: uniform d_sub overrides asymmetric dims
        if d_sub is not None:
            d_type = d_sub
            d_other = d_sub

        # Fixed feature layout constants.
        self._S = SequenceFeatureEncoder.MAX_SPARSE_LEN
        self._D = 1
        self._PI = SequenceFeatureEncoder.PLAYER_INFO_TOKENS
        self._PIW = SequenceFeatureEncoder.PLAYER_INFO_WIDTH
        self._SM = SequenceFeatureEncoder.MAX_SPARSE_MELDS
        self._MW = SequenceFeatureEncoder.MELD_WIDTH
        self._H = SequenceFeatureEncoder.MAX_HAND_LEN  # 14
        self._VC = SequenceFeatureEncoder.VISIBLE_TILE_COUNT_TOKENS
        self._VCW = SequenceFeatureEncoder.VISIBLE_TILE_COUNT_WIDTH
        self._N = SequenceFeatureEncoder.NUM_NUMERIC  # 6
        self._A = SequenceFeatureEncoder.AGARI_OVERTAKE_DIM
        self._AT = SequenceFeatureEncoder.AGARI_OVERTAKE_TOKENS
        self._AW = SequenceFeatureEncoder.AGARI_OVERTAKE_TOKEN_DIM
        self._PW = len(prog_dims)
        self._CW = len(cand_dims)
        if len(player_info_dims) != self._PIW:
            raise ValueError(f"player_info_dims must have {self._PIW} entries")
        self.player_info_dims = tuple(int(v) for v in player_info_dims)
        if len(visible_tile_count_dims) != self._VCW:
            raise ValueError(f"visible_tile_count_dims must have {self._VCW} entries")
        self.visible_tile_count_dims = tuple(int(v) for v in visible_tile_count_dims)

        # --- Embedding layers ---
        self.sparse_embed = nn.Embedding(sparse_vocab, d_model, padding_idx=sparse_pad)
        self.relative_seat_embed = RelativeSeatEmbedding(base_dim=d_other, d_model=d_model)
        self.player_riichi_embed = nn.Embedding(self.player_info_dims[1], d_other)
        self.player_meld_count_embed = nn.Embedding(self.player_info_dims[2], d_other)
        self.player_discard_count_embed = nn.Embedding(self.player_info_dims[3], d_other)
        self.player_tedashi_count_embed = nn.Embedding(self.player_info_dims[4], d_other)
        self.player_info_proj = SplitLinearLayerNorm([d_other] * self._PIW, d_model)
        self.tile_embed = SharedTileEmbedding(out_dim=d_type, attr_dim=d_other)
        self.tile_action_kind_embed = nn.Embedding(_ACTION_KIND_PAD + 1, d_other, padding_idx=_ACTION_KIND_PAD)
        self.tile_action_proj = SplitLinearLayerNorm([d_type, d_other], d_type)
        self.meld_embed = SharedMeldEmbedding(out_dim=d_type, role_dim=d_other)
        self.sparse_meld_proj = nn.Sequential(
            nn.Linear(d_type, d_model),
            nn.LayerNorm(d_model),
        )

        # Hand: embed (tile37, draw_state) with split Linear(concat(...)) equivalent.
        hand_sub_dims = [d_type, d_other]
        self.hand_draw_state_embed = nn.Embedding(hand_dims[1], d_other)
        self.hand_proj = SplitLinearLayerNorm(hand_sub_dims, d_model)

        self.visible_tile_count_embed = nn.Embedding(self.visible_tile_count_dims[1], d_other)
        self.visible_tile_count_proj = SplitLinearLayerNorm([d_type, d_other], d_model)

        self.numeric_proj = nn.Sequential(
            nn.Linear(self._N, d_model),
            nn.LayerNorm(d_model),
        )
        self.agari_overtake_proj = nn.Sequential(
            nn.Linear(self._AW, d_model),
            nn.LayerNorm(d_model),
        )

        # Progression: shared seat fields + type/moqie/liqi embeddings.
        # field[1] is type (vocab=80) -> d_type; others -> d_other
        prog_sub_dims = [d_other if i != 1 else d_type for i in range(len(prog_dims))]
        self.prog_type_embed = nn.Embedding(prog_dims[1], d_type)
        self.prog_moqie_embed = nn.Embedding(prog_dims[2], d_other)
        self.prog_liqi_embed = nn.Embedding(prog_dims[3], d_other)
        self.prog_proj = SplitLinearLayerNorm(prog_sub_dims, d_model)

        # Candidates: shared from-seat field + type/moqie embeddings.
        # field[0] is type (vocab=47) -> d_type; others -> d_other
        cand_sub_dims = [d_other if i != 0 else d_type for i in range(len(cand_dims))]
        self.cand_type_embed = nn.Embedding(cand_dims[0], d_type)
        self.cand_moqie_embed = nn.Embedding(cand_dims[1], d_other)
        self.cand_proj = SplitLinearLayerNorm(cand_sub_dims, d_model)
        self.dora_slot_embed = nn.Embedding(_DORA_SLOT_PAD + 1, d_other, padding_idx=_DORA_SLOT_PAD)
        self.sparse_dora_proj = SplitLinearLayerNorm([d_type, d_other], d_model)

        # --- CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- Segment embeddings (8 groups: sparse / player / hand / visible-count / numeric / agari / prog / cand) ---
        self.segment_embed = nn.Embedding(8, d_model)

        # Progression relative attention bias, indexed by signed discard-clock distance.
        self.progression_relative_bias = nn.Parameter(torch.zeros(self.nhead, self._RPB))
        # Non-progression queries use a recency-only table. Distances are always
        # non-positive because those queries are anchored at the current clock.
        self.progression_recency_bias = nn.Parameter(
            torch.zeros(self.nhead, self.progression_relative_position_cap + 1)
        )
        self.progression_player_relative_bias = nn.Parameter(torch.zeros(self.nhead, self._PLAYER_RPB))
        self.progression_player_recency_bias = nn.Parameter(
            torch.zeros(self.nhead, self.progression_player_relative_position_cap + 1)
        )

        # --- Transformer encoder (pre-LN for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model)

        # --- Cross-attention for fixed policy head (V3) ---
        if self.policy_head_type == "cross_attn":
            self.cand_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn_norm = nn.LayerNorm(d_model)

        # --- Output heads ---
        if self.policy_head_type == "pointer":
            self.policy_head = None
            self.candidate_scorer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.candidate_scorer = None
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
    def _progression_discard_clock(
        self,
        prog: torch.Tensor,
        prog_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exclusive discard-count clocks for progression rows.

        A discard token is in the same clock tick as non-discard tokens that
        precede it after the previous discard. This makes call/kan/dora rows
        positionally close to the following discard while same-position
        distances remain zero. Player clocks use the same rule, but advance
        only on discards by the corresponding actor.
        """
        batch_size = int(prog.shape[0])
        if prog.shape[1] == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=prog.device)
            empty_player = torch.empty(batch_size, 0, _PLAYER_COUNT, dtype=torch.long, device=prog.device)
            total = torch.zeros(batch_size, dtype=torch.long, device=prog.device)
            player_total = torch.zeros(batch_size, _PLAYER_COUNT, dtype=torch.long, device=prog.device)
            return empty, total, empty_player, player_total
        action_kind = self.prog_type_action_kind[prog[:, :, 1].clamp(min=0, max=self.prog_type_action_kind.numel() - 1)]
        is_discard = (action_kind == _ACTION_KIND_DISCARD) & prog_mask
        increments = is_discard.long()
        clock = increments.cumsum(dim=1) - increments
        total = increments.sum(dim=1)

        actor = prog[:, :, 0].clamp(min=0, max=_SEAT_PAD_OR_NA)
        player_ids = torch.arange(_PLAYER_COUNT, dtype=torch.long, device=prog.device).view(1, 1, _PLAYER_COUNT)
        player_increments = (is_discard[:, :, None] & (actor[:, :, None] == player_ids)).long()
        player_clock = player_increments.cumsum(dim=1) - player_increments
        player_total = player_increments.sum(dim=1)
        return clock, total, player_clock, player_total

    @staticmethod
    def _additive_padding_mask(pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        out = torch.zeros(pad_mask.shape, dtype=dtype, device=pad_mask.device)
        return out.masked_fill(pad_mask, torch.finfo(dtype).min)

    def _progression_attention_bias(
        self,
        prog: torch.Tensor,
        prog_mask: torch.Tensor,
        *,
        total_len: int,
        prog_offset: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        prog_len = int(prog.shape[1])
        if prog_len == 0:
            return None

        batch_size = int(prog.shape[0])
        device = prog.device
        prog_clock, total_clock, player_clock, player_total_clock = self._progression_discard_clock(prog, prog_mask)

        token_clock = total_clock[:, None].expand(batch_size, total_len).clone()
        token_clock[:, prog_offset : prog_offset + prog_len] = prog_clock
        dist = prog_clock[:, None, :] - token_clock[:, :, None]
        cap = self.progression_relative_position_cap
        buckets = dist.clamp(min=-cap, max=cap) + cap
        recency_buckets = dist.clamp(min=-cap, max=0) + cap

        prog_table = self.progression_relative_bias.transpose(0, 1).contiguous()
        recency_table = self.progression_recency_bias.transpose(0, 1).contiguous()
        prog_query_values = F.embedding(buckets, prog_table)
        recency_values = F.embedding(recency_buckets, recency_table)

        query_is_prog = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        query_is_prog[:, prog_offset : prog_offset + prog_len] = prog_mask
        values = torch.where(query_is_prog[:, :, None, None], prog_query_values, recency_values)

        key_actor = prog[:, :, 0].clamp(min=0, max=_SEAT_PAD_OR_NA)
        key_actor_is_player = (key_actor < _PLAYER_COUNT) & prog_mask
        key_actor_index = key_actor.clamp(max=_PLAYER_COUNT - 1)

        token_player_clock = player_total_clock[:, None, :].expand(batch_size, total_len, _PLAYER_COUNT).clone()
        token_player_clock[:, prog_offset : prog_offset + prog_len, :] = player_clock
        key_player_clock = player_clock.gather(2, key_actor_index[:, :, None]).squeeze(2)
        query_player_clock = token_player_clock.gather(
            2,
            key_actor_index[:, None, :].expand(batch_size, total_len, prog_len),
        )
        player_dist = key_player_clock[:, None, :] - query_player_clock

        player_cap = self.progression_player_relative_position_cap
        player_buckets = player_dist.clamp(min=-player_cap, max=player_cap) + player_cap
        player_recency_buckets = player_dist.clamp(min=-player_cap, max=0) + player_cap

        player_prog_table = self.progression_player_relative_bias.transpose(0, 1).contiguous()
        player_recency_table = self.progression_player_recency_bias.transpose(0, 1).contiguous()
        player_prog_values = F.embedding(player_buckets, player_prog_table)
        player_recency_values = F.embedding(player_recency_buckets, player_recency_table)
        player_values = torch.where(
            query_is_prog[:, :, None, None],
            player_prog_values,
            player_recency_values,
        )
        player_values = torch.where(
            key_actor_is_player[:, None, :, None],
            player_values,
            torch.zeros_like(player_values),
        )
        values = values + player_values
        values = torch.where(prog_mask[:, None, :, None], values, torch.zeros_like(values))

        bias = torch.zeros(
            batch_size,
            total_len,
            total_len,
            self.nhead,
            dtype=values.dtype,
            device=device,
        )
        bias[:, :, prog_offset : prog_offset + prog_len, :] = values
        return bias.permute(0, 3, 1, 2).reshape(batch_size * self.nhead, total_len, total_len).to(dtype=dtype)

    def _transformer_with_progression_bias(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        prog: torch.Tensor,
        prog_mask: torch.Tensor,
        *,
        prog_offset: int,
    ) -> torch.Tensor:
        attn_bias = self._progression_attention_bias(
            prog,
            prog_mask,
            total_len=int(tokens.shape[1]),
            prog_offset=prog_offset,
            dtype=tokens.dtype,
        )
        padding_mask = self._additive_padding_mask(pad_mask, tokens.dtype)
        return _transformer_without_mha_fastpath(
            self.transformer,
            tokens,
            mask=attn_bias,
            src_key_padding_mask=padding_mask,
        )

    def _segment_ids(self, prog_len: int, cand_len: int, device: torch.device) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(1 + self._S + self._D, dtype=torch.long, device=device),
                torch.full((self._PI,), 1, dtype=torch.long, device=device),
                torch.zeros(self._SM, dtype=torch.long, device=device),
                torch.full((self._H,), 2, dtype=torch.long, device=device),
                torch.full((self._VC,), 3, dtype=torch.long, device=device),
                torch.full((1,), 4, dtype=torch.long, device=device),
                torch.full((self._AT,), 5, dtype=torch.long, device=device),
                torch.full((prog_len,), 6, dtype=torch.long, device=device),
                torch.full((cand_len,), 7, dtype=torch.long, device=device),
            ],
            dim=0,
        ).unsqueeze(0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)

    # ------------------------------------------------------------------
    def _unpack_packed(self, x: torch.Tensor, prog_len: int, cand_len: int):
        """Unpack the DataLoader fast-path flattened sequence feature tensor."""
        if x.ndim != 2:
            raise ValueError(f"packed sequence features must be rank-2, got shape={tuple(x.shape)}")
        prog_len = int(prog_len)
        cand_len = int(cand_len)
        expected = packed_sequence_size(prog_len, cand_len)
        if x.shape[1] != expected:
            raise ValueError(f"packed sequence feature width {x.shape[1]} does not match expected {expected}")

        batch_size = x.shape[0]
        offset = 0

        def take(shape: tuple[int, ...], kind: str) -> torch.Tensor:
            nonlocal offset
            width = math.prod(shape)
            tensor = x[:, offset : offset + width].reshape(batch_size, *shape)
            offset += width
            if kind == "long":
                return tensor.long()
            if kind == "bool":
                return tensor.bool()
            return tensor.float()

        sparse = take((self._S,), "long")
        dealer = take((self._D,), "long").squeeze(1)
        player_stats = take((self._PI, self._PIW), "long")
        sparse_melds = take((self._SM, self._MW), "long")
        sparse_meld_owners = take((self._SM,), "long")
        hand = take((self._H, 2), "long")
        visible_tile_counts = take((self._VC, self._VCW), "long")
        numeric = take((self._N,), "float")
        agari_overtakes = take((self._A,), "float")
        sparse_mask = take((self._S,), "bool")
        sparse_meld_mask = take((self._SM,), "bool")
        hand_mask = take((self._H,), "bool")
        progression = take((prog_len, self._PW), "long")
        prog_melds = take((prog_len, self._MW), "long")
        candidates = take((cand_len, self._CW), "long")
        cand_melds = take((cand_len, self._MW), "long")
        prog_mask = take((prog_len,), "bool")
        cand_mask = take((cand_len,), "bool")
        if offset != x.shape[1]:
            raise RuntimeError(f"packed sequence unpack width mismatch: read {offset}, expected {x.shape[1]}")

        return (
            sparse,
            dealer,
            player_stats,
            sparse_melds,
            sparse_meld_owners,
            hand,
            visible_tile_counts,
            numeric,
            agari_overtakes,
            progression,
            prog_melds,
            candidates,
            cand_melds,
            sparse_mask,
            sparse_meld_mask,
            hand_mask,
            prog_mask,
            cand_mask,
        )

    def _feature_tensors(self, x: dict[str, torch.Tensor] | tuple[torch.Tensor, int, int]):
        """Normalize a batched feature dict into typed component tensors."""
        if isinstance(x, tuple):
            if len(x) != 3:
                raise TypeError("packed sequence features must be (tensor, prog_len, cand_len)")
            return self._unpack_packed(x[0], int(x[1]), int(x[2]))
        return (
            x["sparse"].long(),
            x["dealer"].long(),
            x["player_stats"].long(),
            x["sparse_melds"].long(),
            x["sparse_meld_owners"].long(),
            x["hand"].long(),
            x["visible_tile_counts"].long(),
            x["numeric"].float(),
            x["agari_overtakes"].float(),
            x["progression"].long(),
            x["prog_melds"].long(),
            x["candidates"].long(),
            x["cand_melds"].long(),
            x["sparse_mask"].bool(),
            x["sparse_meld_mask"].bool(),
            x["hand_mask"].bool(),
            x["prog_mask"].bool(),
            x["cand_mask"].bool(),
        )

    # ------------------------------------------------------------------
    def _decode_current_dora_tiles(self, sparse: torch.Tensor) -> torch.Tensor:
        """Return current dora tiles as tile34 ids, one slot per sparse token."""
        return self.sparse_dora_tile34[sparse]

    @staticmethod
    def _decode_round_wind(sparse: torch.Tensor) -> torch.Tensor:
        """Return current round wind as 0=E, 1=S, 2=W, 3=N."""
        return (sparse[:, 1] - 2).clamp(min=0, max=3)

    def _seat_other(
        self,
        seats: torch.Tensor,
        role: int,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seat_other_table is None:
            return self.relative_seat_embed(seats, role, out="other")
        return self.relative_seat_embed.embed_from_table(seat_other_table, seats, role)

    def _seat_model(
        self,
        seats: torch.Tensor,
        role: int,
        seat_model_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seat_model_table is None:
            return self.relative_seat_embed(seats, role, out="model")
        return self.relative_seat_embed.embed_from_table(seat_model_table, seats, role)

    def _embed_sparse(
        self,
        sparse: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor,
    ) -> torch.Tensor:
        sparse_emb = self.sparse_embed(sparse)
        dora_slot = self.sparse_dora_slot[sparse]
        dora_mask = dora_slot != _DORA_SLOT_PAD

        indicator_tile37 = self.sparse_indicator_tile37[sparse]
        tile_emb = self.tile_embed.embed_tile37_from_table(indicator_tile37, tile37_table)
        dora_raw = self.sparse_dora_proj.project_part(0, tile_emb) + self.sparse_dora_proj.project_embedding(
            1,
            self.dora_slot_embed,
            dora_slot,
        )
        dora_emb = self.sparse_dora_proj.finish(dora_raw)
        return torch.where(dora_mask.unsqueeze(-1), dora_emb, sparse_emb)

    def _embed_player_info(
        self,
        player_stats: torch.Tensor,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seats = player_stats[:, :, 0].clamp(min=0, max=3)
        riichi = player_stats[:, :, 1].clamp(min=0, max=self.player_info_dims[1] - 1)
        meld_count = player_stats[:, :, 2].clamp(min=0, max=self.player_info_dims[2] - 1)
        discard_count = player_stats[:, :, 3].clamp(min=0, max=self.player_info_dims[3] - 1)
        tedashi_count = player_stats[:, :, 4].clamp(min=0, max=self.player_info_dims[4] - 1)
        raw = (
            self.player_info_proj.project_part(
                0,
                self._seat_other(seats, _SEAT_ROLE_PLAYER_INFO, seat_other_table),
            )
            + self.player_info_proj.project_embedding(1, self.player_riichi_embed, riichi)
            + self.player_info_proj.project_embedding(2, self.player_meld_count_embed, meld_count)
            + self.player_info_proj.project_embedding(3, self.player_discard_count_embed, discard_count)
            + self.player_info_proj.project_embedding(4, self.player_tedashi_count_embed, tedashi_count)
        )
        return self.player_info_proj.finish(raw)

    def _embed_sparse_melds(
        self,
        melds: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        owners: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        meld_emb: torch.Tensor | None = None,
        seat_model_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if meld_emb is None:
            meld_emb = self.meld_embed(
                melds,
                dora_tile34,
                self.tile_embed,
                tile37_table,
                dealer,
                round_wind,
                self.relative_seat_embed,
            )
        out = self.sparse_meld_proj(meld_emb)
        if owners is not None:
            out = out + self._seat_model(owners, _SEAT_ROLE_MELD_OWNER, seat_model_table)
        return out

    def _embed_tile_only_action_type(
        self,
        type_ids: torch.Tensor,
        generic_embed: nn.Embedding,
        action_kind_lookup: torch.Tensor,
        tile37_lookup: torch.Tensor,
        tile34_lookup: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        tile34_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_emb = generic_embed(type_ids)
        action_kind = action_kind_lookup[type_ids]
        target_mask = action_kind != _ACTION_KIND_PAD

        tile37 = tile37_lookup[type_ids]
        tile34 = tile34_lookup[type_ids]
        if tile37_table is None:
            tile37_emb = self.tile_embed.embed_tile37(
                tile37,
                dora_tile34,
                dealer,
                round_wind,
                self.relative_seat_embed,
            )
        else:
            tile37_emb = self.tile_embed.embed_tile37_from_table(tile37, tile37_table)
        if tile34_table is None:
            tile34_emb = self.tile_embed.embed_tile34(
                tile34,
                dora_tile34,
                dealer,
                round_wind,
                self.relative_seat_embed,
            )
        else:
            tile34_emb = self.tile_embed.embed_tile34_from_table(tile34, tile34_table)
        tile_emb = torch.where((tile37 != _TILE37_PAD).unsqueeze(-1), tile37_emb, tile34_emb)
        target_raw = self.tile_action_proj.project_part(0, tile_emb) + self.tile_action_proj.project_embedding(
            1,
            self.tile_action_kind_embed,
            action_kind,
        )
        target_emb = self.tile_action_proj.finish(target_raw)
        return torch.where(target_mask.unsqueeze(-1), target_emb, type_emb)

    def _embed_meld_action_type(
        self,
        type_emb: torch.Tensor,
        melds: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        meld_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        meld_kind = melds[:, :, 0]
        meld_mask = meld_kind != _MELD_KIND_PAD
        if meld_emb is None:
            meld_emb = self.meld_embed(
                melds,
                dora_tile34,
                self.tile_embed,
                tile37_table,
                dealer,
                round_wind,
                self.relative_seat_embed,
            )
        return torch.where(meld_mask.unsqueeze(-1), meld_emb, type_emb)

    def _embed_hand(
        self,
        hand: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tile37_table is None:
            tile_emb = self.tile_embed.embed_tile37(
                hand[:, :, 0],
                dora_tile34,
                dealer,
                round_wind,
                self.relative_seat_embed,
            )
        else:
            tile_emb = self.tile_embed.embed_tile37_from_table(hand[:, :, 0], tile37_table)
        raw = self.hand_proj.project_part(0, tile_emb) + self.hand_proj.project_embedding(
            1,
            self.hand_draw_state_embed,
            hand[:, :, 1],
        )
        return self.hand_proj.finish(raw)

    def _embed_visible_tile_counts(
        self,
        visible_tile_counts: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tile37 = visible_tile_counts[:, :, 0]
        counts = visible_tile_counts[:, :, 1].clamp(
            min=0,
            max=self.visible_tile_count_dims[1] - 1,
        )
        if tile37_table is None:
            tile_emb = self.tile_embed.embed_tile37(
                tile37,
                dora_tile34,
                dealer,
                round_wind,
                self.relative_seat_embed,
            )
        else:
            tile_emb = self.tile_embed.embed_tile37_from_table(tile37, tile37_table)
        raw = self.visible_tile_count_proj.project_part(
            0,
            tile_emb,
        ) + self.visible_tile_count_proj.project_embedding(
            1,
            self.visible_tile_count_embed,
            counts,
        )
        return self.visible_tile_count_proj.finish(raw)

    def _embed_progression(
        self,
        prog: torch.Tensor,
        prog_melds: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        tile34_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        meld_emb: torch.Tensor | None = None,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_emb = self._embed_tile_only_action_type(
            prog[:, :, 1],
            self.prog_type_embed,
            self.prog_type_action_kind,
            self.prog_type_tile37,
            self.prog_type_tile34,
            dora_tile34,
            tile37_table,
            tile34_table,
            dealer,
            round_wind,
        )
        type_emb = self._embed_meld_action_type(
            type_emb,
            prog_melds,
            dora_tile34,
            tile37_table,
            dealer,
            round_wind,
            meld_emb,
        )
        raw = (
            self.prog_proj.project_part(
                0,
                self._seat_other(prog[:, :, 0], _SEAT_ROLE_PROG_ACTOR, seat_other_table),
            )
            + self.prog_proj.project_part(1, type_emb)
            + self.prog_proj.project_embedding(2, self.prog_moqie_embed, prog[:, :, 2])
            + self.prog_proj.project_embedding(3, self.prog_liqi_embed, prog[:, :, 3])
            + self.prog_proj.project_part(
                4,
                self._seat_other(prog[:, :, 4], _SEAT_ROLE_PROG_FROM, seat_other_table),
            )
        )
        return self.prog_proj.finish(raw)

    def _embed_candidates(
        self,
        cand: torch.Tensor,
        cand_melds: torch.Tensor,
        dora_tile34: torch.Tensor,
        tile37_table: torch.Tensor | None = None,
        tile34_table: torch.Tensor | None = None,
        dealer: torch.Tensor | None = None,
        round_wind: torch.Tensor | None = None,
        meld_emb: torch.Tensor | None = None,
        seat_other_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_emb = self._embed_tile_only_action_type(
            cand[:, :, 0],
            self.cand_type_embed,
            self.cand_type_action_kind,
            self.cand_type_tile37,
            self.cand_type_tile34,
            dora_tile34,
            tile37_table,
            tile34_table,
            dealer,
            round_wind,
        )
        type_emb = self._embed_meld_action_type(
            type_emb,
            cand_melds,
            dora_tile34,
            tile37_table,
            dealer,
            round_wind,
            meld_emb,
        )
        raw = (
            self.cand_proj.project_part(0, type_emb)
            + self.cand_proj.project_embedding(1, self.cand_moqie_embed, cand[:, :, 1])
            + self.cand_proj.project_part(
                2,
                self._seat_other(cand[:, :, 2], _SEAT_ROLE_CAND_FROM, seat_other_table),
            )
        )
        return self.cand_proj.finish(raw)

    def _embed_agari_overtakes(
        self,
        agari_overtakes: torch.Tensor,
        seat_model_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        agari_by_winner = agari_overtakes.reshape(-1, self._AT, self._AW)
        winner_seats = torch.arange(self._AT, dtype=torch.long, device=agari_overtakes.device)
        winner_seats = winner_seats.unsqueeze(0).expand(agari_by_winner.shape[0], -1)
        return self.agari_overtake_proj(agari_by_winner) + self._seat_model(
            winner_seats,
            _SEAT_ROLE_AGARI_WINNER,
            seat_model_table,
        )

    # ------------------------------------------------------------------
    def forward(  # noqa: PLR0915
        self,
        x: dict[str, torch.Tensor] | tuple[torch.Tensor, int, int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, (dict, tuple)):
            raise TypeError("TransformerActorCritic expects sequence feature dict or packed sequence tuple")
        (
            sparse,
            dealer,
            player_stats,
            sparse_melds,
            sparse_meld_owners,
            hand,
            visible_tile_counts,
            numeric,
            agari_overtakes,
            prog,
            prog_melds,
            cand,
            cand_melds,
            sparse_mask,
            sparse_meld_mask,
            hand_mask,
            prog_mask,
            cand_mask,
        ) = self._feature_tensors(x)
        batch_size = sparse.shape[0]
        device = sparse.device
        prog_len = prog.shape[1]
        cand_len = cand.shape[1]
        dora_tile34 = self._decode_current_dora_tiles(sparse)
        round_wind = self._decode_round_wind(sparse)
        seat_other_table, seat_model_table = self.relative_seat_embed.build_tables()
        tile37_table, tile34_table = self.tile_embed.build_tables(
            dora_tile34,
            dealer,
            round_wind,
            self.relative_seat_embed,
            seat_other_table,
        )
        all_meld_emb = self.meld_embed(
            torch.cat([sparse_melds, prog_melds, cand_melds], dim=1),
            dora_tile34,
            self.tile_embed,
            tile37_table,
            dealer,
            round_wind,
            self.relative_seat_embed,
        )
        sparse_meld_type, prog_meld_type, cand_meld_type = all_meld_emb.split(
            [self._SM, prog_len, cand_len],
            dim=1,
        )

        # Embed sparse tokens: (B, S, d)
        sparse_emb = self._embed_sparse(sparse, dora_tile34, tile37_table)

        # Embed dealer relative seat: (B, 1, d)
        dealer_emb = self._seat_model(dealer, _SEAT_ROLE_DEALER, seat_model_table).unsqueeze(1)

        # Embed per-player public summaries: (B, 4, d)
        player_info_emb = self._embed_player_info(player_stats, seat_other_table)

        # Embed current visible melds: (B, SM, d)
        sparse_meld_emb = self._embed_sparse_melds(
            sparse_melds,
            dora_tile34,
            tile37_table,
            sparse_meld_owners,
            dealer,
            round_wind,
            sparse_meld_type,
            seat_model_table,
        )

        # Embed hand tuples: (B, H, d)
        hand_emb = self._embed_hand(hand, dora_tile34, tile37_table, dealer, round_wind)

        # Embed visible tile-count tokens: (B, 37, d)
        visible_tile_count_emb = self._embed_visible_tile_counts(
            visible_tile_counts,
            dora_tile34,
            tile37_table,
            dealer,
            round_wind,
        )

        # Project numeric: (B, 1, d)
        numeric_emb = self.numeric_proj(numeric).unsqueeze(1)

        # Project pairwise agari-rank-overtake flags by winner seat: (B, 4, d)
        agari_overtake_emb = self._embed_agari_overtakes(agari_overtakes, seat_model_table)

        # Embed progression 5-tuples: (B, P, d)
        prog_emb = self._embed_progression(
            prog,
            prog_melds,
            dora_tile34,
            tile37_table,
            tile34_table,
            dealer,
            round_wind,
            prog_meld_type,
            seat_other_table,
        )

        # Embed candidate tuples: (B, C, d)
        cand_emb = self._embed_candidates(
            cand,
            cand_melds,
            dora_tile34,
            tile37_table,
            tile34_table,
            dealer,
            round_wind,
            cand_meld_type,
            seat_other_table,
        )

        # CLS token: (B, 1, d)
        cls = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate: [CLS, sparse(S), dealer(1), player_info(4), sparse_meld(SM), hand(H),
        # visible_count(37), numeric(1), agari_overtake(4), prog(P), cand(C)]
        tokens = torch.cat(
            [
                cls,
                sparse_emb,
                dealer_emb,
                player_info_emb,
                sparse_meld_emb,
                hand_emb,
                visible_tile_count_emb,
                numeric_emb,
                agari_overtake_emb,
                prog_emb,
                cand_emb,
            ],
            dim=1,
        )

        # Add segment embeddings
        seg_ids = self._segment_ids(prog_len, cand_len, device).expand(batch_size, -1)
        tokens = tokens + self.segment_embed(seg_ids)

        # Build padding mask: True = ignore (PyTorch convention)
        cls_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        dealer_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        player_info_valid = torch.zeros(batch_size, self._PI, dtype=torch.bool, device=device)
        numeric_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        visible_tile_count_valid = torch.zeros(batch_size, self._VC, dtype=torch.bool, device=device)
        agari_overtake_valid = torch.zeros(batch_size, self._AT, dtype=torch.bool, device=device)
        pad_mask = torch.cat(
            [
                cls_valid,  # CLS is always valid
                ~sparse_mask,  # True where sparse is padding
                dealer_valid,  # dealer is always valid
                player_info_valid,  # player summaries are always valid
                ~sparse_meld_mask,  # True where current visible meld is padding
                ~hand_mask,  # True where hand is padding
                visible_tile_count_valid,  # visible tile-count tokens are always valid
                numeric_valid,  # numeric is always valid
                agari_overtake_valid,  # agari-overtake token is always valid
                ~prog_mask,  # True where prog is padding
                ~cand_mask,  # True where cand is padding
            ],
            dim=1,
        )

        # Transformer
        prog_offset = 1 + self._S + self._D + self._PI + self._SM + self._H + self._VC + 1 + self._AT
        output = self._transformer_with_progression_bias(
            tokens,
            pad_mask,
            prog,
            prog_mask,
            prog_offset=prog_offset,
        )
        output = self.final_norm(output)

        # CLS output is shared by policy and value heads.
        cls_out = output[:, 0]

        cand_offset = 1 + self._S + self._D + self._PI + self._SM + self._H + self._VC + 1 + self._AT + prog_len
        cand_out = output[:, cand_offset : cand_offset + cand_len]  # (B, C, d_model)

        # Policy head
        if self.policy_head_type == "cross_attn":
            # Cross-attention: CLS queries candidate token outputs
            cls_q = cls_out.unsqueeze(1)  # (B, 1, d_model)
            cand_attn_mask = ~cand_mask  # True = padding (PyTorch convention)
            attn_out, _ = self.cand_cross_attn(
                cls_q, cand_out, cand_out, key_padding_mask=cand_attn_mask
            )  # (B, 1, d_model)
            policy_input = self.cross_attn_norm(cls_out + attn_out.squeeze(1))
            logits = self.policy_head(policy_input)
        elif self.policy_head_type == "pointer":
            cls_expanded = cls_out.unsqueeze(1).expand(-1, cand_len, -1)
            logits = self.candidate_scorer(torch.cat([cand_out, cls_expanded], dim=-1)).squeeze(-1)
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
