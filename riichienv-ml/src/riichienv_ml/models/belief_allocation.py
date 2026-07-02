"""Joint hidden-allocation belief sampler."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from riichienv_ml.features.belief_features import (
    BUCKET_COUNT,
    SHANTEN_CLASS_COUNT,
    TILE37_COUNT,
    TOTAL_TILE_COUNTS37,
    BeliefFeatureEncoder,
)
from riichienv_ml.models.transformer import (
    _SEAT_ROLE_DEALER,
    _SEAT_ROLE_PLAYER_INFO,
    _SEAT_ROLE_PROG_ACTOR,
    SplitLinearLayerNorm,
    TransformerActorCritic,
)

_COUNT_CANDIDATES = 5
_CONFIDENCE_METHODS = ("max_prob", "neg_entropy", "legal_normalized_entropy")
_TILE34_COUNT = 34
_OPPONENT_COUNT = BUCKET_COUNT - 1
_ALLOCATION_TOKEN_COUNT = TILE37_COUNT * _OPPONENT_COUNT
_SHANTEN_TOKEN_COUNT = _OPPONENT_COUNT
_MAX_LOGCOMB_COUNT = int(sum(TOTAL_TILE_COUNTS37))
_TRAINING_MASK_MAX_RETRIES = 16


def _build_tile37_to_tile34() -> torch.Tensor:
    tile34 = torch.empty(TILE37_COUNT, dtype=torch.long)
    tile34[0] = 4
    tile34[1:10] = torch.arange(9, dtype=torch.long)
    tile34[10] = 13
    tile34[11:20] = torch.arange(9, 18, dtype=torch.long)
    tile34[20] = 22
    tile34[21:30] = torch.arange(18, 27, dtype=torch.long)
    tile34[30:37] = torch.arange(27, 34, dtype=torch.long)
    return tile34


def _build_opponent_subset_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [(subset_mask >> seat) & 1 for seat in range(_OPPONENT_COUNT)]
            for subset_mask in range(1, 1 << _OPPONENT_COUNT)
        ],
        dtype=torch.bool,
    )


def _build_allocation_token_tile37() -> torch.Tensor:
    return torch.arange(_ALLOCATION_TOKEN_COUNT, dtype=torch.long) // _OPPONENT_COUNT


def _build_allocation_token_opponent() -> torch.Tensor:
    return torch.arange(_ALLOCATION_TOKEN_COUNT, dtype=torch.long) % _OPPONENT_COUNT


def _build_allocation_token_allocation_offset() -> torch.Tensor:
    tile37 = _build_allocation_token_tile37()
    opponent = _build_allocation_token_opponent()
    return opponent * TILE37_COUNT + tile37


def _build_logcomb_table() -> torch.Tensor:
    counts = torch.arange(_MAX_LOGCOMB_COUNT + 1, dtype=torch.float32)
    n = counts.view(-1, 1)
    k = torch.minimum(counts.view(1, -1), n)
    return torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)


_TILE37_TO_TILE34 = tuple(int(tile34) for tile34 in _build_tile37_to_tile34().tolist())
_OPPONENT_SUBSET_FLAGS = tuple(
    tuple(bool((subset_mask >> seat) & 1) for seat in range(_OPPONENT_COUNT))
    for subset_mask in range(1, 1 << _OPPONENT_COUNT)
)
_PROFILE_RANGES_ENABLED = False


@contextmanager
def belief_profile_ranges(enabled: bool = True) -> Iterator[None]:
    """Enable detailed profiler ranges for belief sampler inference in this context."""

    global _PROFILE_RANGES_ENABLED  # noqa: PLW0603
    previous = _PROFILE_RANGES_ENABLED
    _PROFILE_RANGES_ENABLED = bool(enabled)
    try:
        yield
    finally:
        _PROFILE_RANGES_ENABLED = previous


def _profile_range(name: str):
    if _PROFILE_RANGES_ENABLED:
        return record_function(name)
    return nullcontext()


@dataclass(frozen=True)
class AllocationSamplingContext:
    """Encoder-side tensors reused when sampling many allocations for one observation batch."""

    context: torch.Tensor
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    unseen_counts: torch.Tensor
    tile_emb: torch.Tensor
    seat_emb: torch.Tensor
    rem: torch.Tensor
    shanten_logits: torch.Tensor
    riichi_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])


@dataclass(frozen=True)
class AllocationDecodeBatchOffsets:
    """Batch-specific flat-index offsets used while decoding."""

    allocation: torch.Tensor
    state: torch.Tensor
    tile: torch.Tensor
    seat: torch.Tensor


@dataclass(frozen=True)
class AllocationDecodeTokenLookup:
    """Token-indexed lookup buffers for hidden-allocation cells."""

    tile37: torch.Tensor
    opponent: torch.Tensor
    allocation_offset: torch.Tensor


@dataclass
class AllocationDecodeState:
    """Mutable tensors updated while decoding hidden allocations."""

    allocation: torch.Tensor
    state_ids: torch.Tensor
    tile_remaining: torch.Tensor
    seat_remaining: torch.Tensor
    is_masked: torch.Tensor

    def apply_cell_update(
        self,
        batch_offsets: AllocationDecodeBatchOffsets,
        state_index: torch.Tensor,
        tile37: torch.Tensor,
        opponent: torch.Tensor,
        allocation_offset: torch.Tensor,
        chosen: torch.Tensor,
    ) -> None:
        allocation_index = batch_offsets.allocation + allocation_offset
        tile_index = batch_offsets.tile + tile37
        seat_index = batch_offsets.seat + opponent

        self.apply_flat_cell_updates(
            state_index,
            allocation_index,
            tile_index,
            seat_index,
            chosen,
        )

    def apply_flat_cell_updates(
        self,
        state_index: torch.Tensor,
        allocation_index: torch.Tensor,
        tile_index: torch.Tensor,
        seat_index: torch.Tensor,
        chosen: torch.Tensor,
    ) -> None:
        neg_chosen = -chosen

        self.allocation.view(-1).index_copy_(0, allocation_index, chosen)
        self.state_ids.view(-1).index_copy_(0, state_index, chosen)
        self.tile_remaining.view(-1).scatter_add_(0, tile_index, neg_chosen)
        self.seat_remaining.view(-1).scatter_add_(0, seat_index, neg_chosen)
        self.is_masked.view(-1).index_fill_(0, state_index, False)

    def apply_dense_cell_updates(
        self,
        update_mask: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        update_values = values.long() * update_mask.long()
        opponent_update_mask = update_mask.transpose(1, 2)

        self.allocation[:, :_OPPONENT_COUNT] = torch.where(
            opponent_update_mask,
            update_values.transpose(1, 2),
            self.allocation[:, :_OPPONENT_COUNT],
        )
        self.state_ids = torch.where(update_mask, values.long(), self.state_ids)
        self.tile_remaining = self.tile_remaining - update_values.sum(dim=2)
        self.seat_remaining = self.seat_remaining - update_values.sum(dim=1)
        self.is_masked = self.is_masked & ~update_mask


@dataclass(frozen=True)
class CrossAttentionKVCache:
    """Per-layer cross-attention K/V tensors for fixed decoder memory."""

    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None


@dataclass(frozen=True)
class ActiveAllocationDecodeInputs:
    """Batch tensors for one active-row decode step."""

    context: torch.Tensor
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    tile_emb: torch.Tensor
    state: AllocationDecodeState
    shanten_classes: torch.Tensor | None
    cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None
    batch_indices: torch.Tensor
    uses_full_batch: bool


class _DenoiseLayer(nn.Module):
    """Pre-LN bidirectional decoder layer with public-memory cross attention."""

    def __init__(
        self,
        d_dec: int,
        nhead: int,
        d_ff: int,
        dropout: float,
        *,
        d_memory: int,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_dec)
        self.self_attn = nn.MultiheadAttention(d_dec, nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_dec)
        self.cross_attn = nn.MultiheadAttention(
            d_dec,
            nhead,
            dropout=dropout,
            batch_first=True,
            kdim=d_memory,
            vdim=d_memory,
        )
        self.ffn_norm = nn.LayerNorm(d_dec)
        self.ffn = nn.Sequential(
            nn.Linear(d_dec, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_dec),
        )
        self.dropout = nn.Dropout(dropout)

    def _cross_projection_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.cross_attn.in_proj_weight is not None:
            q_weight, k_weight, v_weight = self.cross_attn.in_proj_weight.chunk(3, dim=0)
        else:
            q_weight = self.cross_attn.q_proj_weight
            k_weight = self.cross_attn.k_proj_weight
            v_weight = self.cross_attn.v_proj_weight
            if q_weight is None or k_weight is None or v_weight is None:
                raise RuntimeError("cross attention projection weights are not initialized")

        if self.cross_attn.in_proj_bias is None:
            return q_weight, k_weight, v_weight, None, None, None
        q_bias, k_bias, v_bias = self.cross_attn.in_proj_bias.chunk(3, dim=0)
        return q_weight, k_weight, v_weight, q_bias, k_bias, v_bias

    def _to_attention_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _dim = x.shape
        return x.view(batch_size, seq_len, self.cross_attn.num_heads, self.cross_attn.head_dim).transpose(1, 2)

    def build_cross_attention_cache(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> CrossAttentionKVCache:
        _q_weight, k_weight, v_weight, _q_bias, k_bias, v_bias = self._cross_projection_weights()
        key = self._to_attention_heads(F.linear(memory, k_weight, k_bias))
        value = self._to_attention_heads(F.linear(memory, v_weight, v_bias))
        attention_mask = None
        if memory_padding_mask is not None:
            attention_mask = ~memory_padding_mask[:, None, None, :]
        return CrossAttentionKVCache(key=key, value=value, attention_mask=attention_mask)

    def _cross_attention_from_cache(
        self,
        query: torch.Tensor,
        cache: CrossAttentionKVCache,
    ) -> torch.Tensor:
        q_weight, _k_weight, _v_weight, q_bias, _k_bias, _v_bias = self._cross_projection_weights()
        projected = self._to_attention_heads(F.linear(query, q_weight, q_bias))
        dropout_p = float(self.cross_attn.dropout) if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            projected,
            cache.key,
            cache.value,
            attn_mask=cache.attention_mask,
            dropout_p=dropout_p,
        )
        attended = attended.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.cross_attn.embed_dim)
        return self.cross_attn.out_proj(attended)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        memory_padding_mask: torch.Tensor | None,
        *,
        cross_attention_cache: CrossAttentionKVCache | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.self_norm(x)
        attended, _weights = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout(attended)

        residual = x
        cross_query = self.cross_norm(x)
        if cross_attention_cache is None:
            if memory is None:
                raise ValueError("memory is required when cross_attention_cache is not provided")
            attended, _weights = self.cross_attn(
                cross_query,
                memory,
                memory,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )
        else:
            attended = self._cross_attention_from_cache(cross_query, cross_attention_cache)
        x = residual + self.dropout(attended)

        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DenoiseTransformer(nn.Module):
    """Small bidirectional denoising transformer over allocation-cell tokens."""

    def __init__(
        self,
        d_dec: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        *,
        d_memory: int,
        output_dim: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DenoiseLayer(
                    d_dec,
                    nhead,
                    d_ff,
                    dropout,
                    d_memory=d_memory,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor | None,
        memory_padding_mask: torch.Tensor | None,
        *,
        cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None = None,
    ) -> torch.Tensor:
        if cross_attention_cache is not None and len(cross_attention_cache) != len(self.layers):
            raise ValueError("cross_attention_cache length must match decoder layer count")
        x = tokens
        for idx, layer in enumerate(self.layers):
            layer_cache = None if cross_attention_cache is None else cross_attention_cache[idx]
            x = layer(x, memory, memory_padding_mask, cross_attention_cache=layer_cache)
        return self.head(self.final_norm(x))

    def build_cross_attention_cache(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> tuple[CrossAttentionKVCache, ...]:
        return tuple(layer.build_cross_attention_cache(memory, memory_padding_mask) for layer in self.layers)


class BeliefObservationEncoder(TransformerActorCritic):
    """Observation transformer that emits a CLS context for belief decoding."""

    def __init__(self, **kwargs):
        kwargs.setdefault("emit_value", False)
        kwargs.setdefault("policy_head_type", "pointer")
        d_sub = kwargs.get("d_sub")
        d_other = int(d_sub if d_sub is not None else kwargs.get("d_other", 32))
        super().__init__(**kwargs)
        self.belief_phase_embed = nn.Embedding(BeliefFeatureEncoder.PHASE_DIMS, self.d_model)
        self.belief_hand_size_embed = nn.Embedding(BeliefFeatureEncoder.HAND_SIZE_DIMS, d_other)
        self.belief_hand_size_proj = SplitLinearLayerNorm([d_other, d_other], self.d_model)
        self.belief_shanten_query_embed = nn.Embedding(1, self.d_model)
        self.belief_shanten_head = nn.Linear(self.d_model, SHANTEN_CLASS_COUNT)
        self.belief_segment_embed = nn.Embedding(4, self.d_model)
        self._init_weights()

    def _embed_belief_hand_sizes(
        self,
        hand_sizes: torch.Tensor,
        seat_other_table: torch.Tensor,
    ) -> torch.Tensor:
        seats = hand_sizes[:, :, 0].clamp(min=0, max=3)
        counts = hand_sizes[:, :, 1].clamp(min=0, max=BeliefFeatureEncoder.HAND_SIZE_DIMS - 1)
        raw = self.belief_hand_size_proj.project_part(
            0,
            self._seat_other(seats, _SEAT_ROLE_PLAYER_INFO, seat_other_table),
        ) + self.belief_hand_size_proj.project_embedding(
            1,
            self.belief_hand_size_embed,
            counts,
        )
        return self.belief_hand_size_proj.finish(raw)

    def _embed_belief_shanten_queries(
        self,
        batch_size: int,
        device: torch.device,
        seat_model_table: torch.Tensor,
    ) -> torch.Tensor:
        seats = torch.arange(1, BUCKET_COUNT, dtype=torch.long, device=device)
        seat_emb = self._seat_model(seats, _SEAT_ROLE_PLAYER_INFO, seat_model_table)
        query_emb = self.belief_shanten_query_embed.weight[0].view(1, 1, self.d_model)
        return seat_emb.unsqueeze(0).expand(batch_size, -1, -1) + query_emb

    def forward_context(self, x: dict[str, torch.Tensor] | tuple[torch.Tensor, int, int]) -> torch.Tensor:
        context, _memory, _memory_padding_mask = self.forward_context_and_memory(x)
        return context

    def forward_context_and_memory(  # noqa: PLR0915
        self,
        x: dict[str, torch.Tensor] | tuple[torch.Tensor, int, int],
        *,
        return_tile37_table: bool = False,
        return_shanten_logits: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        (
            sparse,
            dealer,
            player_stats,
            player_rank_stats,
            sparse_melds,
            sparse_meld_owners,
            hand,
            visible_tile_counts,
            numeric,
            agari_overtakes,
            prog,
            prog_melds,
            _cand,
            _cand_melds,
            sparse_mask,
            sparse_meld_mask,
            hand_mask,
            prog_mask,
            _cand_mask,
        ) = self._feature_tensors(x)
        if not isinstance(x, dict):
            raise TypeError("BeliefObservationEncoder expects collated feature dictionaries")

        phase = x["belief_phase"].long()
        current_actor = x["belief_current_actor"].long()
        hand_sizes = x["belief_hand_sizes"].long()

        batch_size = sparse.shape[0]
        device = sparse.device
        prog_len = prog.shape[1]
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
            torch.cat([sparse_melds, prog_melds], dim=1),
            dora_tile34,
            self.tile_embed,
            tile37_table,
            dealer,
            round_wind,
            self.relative_seat_embed,
        )
        sparse_meld_type, prog_meld_type = all_meld_emb.split([self._SM, prog_len], dim=1)

        sparse_emb = self._embed_sparse(sparse, dora_tile34, tile37_table)
        dealer_emb = self._seat_model(dealer, _SEAT_ROLE_DEALER, seat_model_table).unsqueeze(1)
        player_info_emb = self._embed_player_info(player_stats, seat_other_table)
        player_rank_stats_emb = self._embed_player_rank_stats(player_rank_stats, seat_other_table)
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
        hand_emb = self._embed_hand(hand, dora_tile34, tile37_table, dealer, round_wind)
        visible_tile_count_emb = self._embed_visible_tile_counts(
            visible_tile_counts,
            dora_tile34,
            tile37_table,
            dealer,
            round_wind,
        )
        numeric_emb = self.numeric_proj(numeric).unsqueeze(1)
        agari_overtake_emb = self._embed_agari_overtakes(agari_overtakes, seat_model_table)
        phase_emb = self.belief_phase_embed(phase).unsqueeze(1)
        current_actor_emb = self._seat_model(
            current_actor,
            _SEAT_ROLE_PROG_ACTOR,
            seat_model_table,
        ).unsqueeze(1)
        belief_hand_size_emb = self._embed_belief_hand_sizes(hand_sizes, seat_other_table)
        belief_shanten_query_emb = self._embed_belief_shanten_queries(batch_size, device, seat_model_table)

        belief_extra = torch.cat([phase_emb, current_actor_emb, belief_hand_size_emb, belief_shanten_query_emb], dim=1)
        belief_extra_len = int(belief_extra.shape[1])
        belief_seg_ids = torch.tensor([0, 1, 2, 2, 2, 2, 3, 3, 3], dtype=torch.long, device=device).unsqueeze(0)
        belief_extra = belief_extra + self.belief_segment_embed(belief_seg_ids)

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

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat(
            [
                cls,
                sparse_emb,
                dealer_emb,
                player_info_emb,
                player_rank_stats_emb,
                sparse_meld_emb,
                hand_emb,
                visible_tile_count_emb,
                numeric_emb,
                agari_overtake_emb,
                belief_extra,
                prog_emb,
            ],
            dim=1,
        )

        seg_ids = self._segment_ids(prog_len, 0, device)
        public_prefix_len = 1 + self._S + self._D + self._PI + self._PRS + self._SM + self._H + self._VC + 1 + self._AT
        seg_ids = torch.cat(
            [
                seg_ids[:, :public_prefix_len],
                torch.full((1, belief_extra_len), 5, dtype=torch.long, device=device),
                seg_ids[:, public_prefix_len:],
            ],
            dim=1,
        )
        tokens = tokens + self.segment_embed(seg_ids.expand(batch_size, -1))

        cls_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        dealer_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        player_info_valid = torch.zeros(batch_size, self._PI, dtype=torch.bool, device=device)
        player_rank_stats_valid = torch.zeros(batch_size, self._PRS, dtype=torch.bool, device=device)
        numeric_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        visible_tile_count_valid = torch.zeros(batch_size, self._VC, dtype=torch.bool, device=device)
        agari_overtake_valid = torch.zeros(batch_size, self._AT, dtype=torch.bool, device=device)
        belief_valid = torch.zeros(batch_size, belief_extra_len, dtype=torch.bool, device=device)
        pad_mask = torch.cat(
            [
                cls_valid,
                ~sparse_mask,
                dealer_valid,
                player_info_valid,
                player_rank_stats_valid,
                ~sparse_meld_mask,
                ~hand_mask,
                visible_tile_count_valid,
                numeric_valid,
                agari_overtake_valid,
                belief_valid,
                ~prog_mask,
            ],
            dim=1,
        )

        prog_offset = (
            1 + self._S + self._D + self._PI + self._PRS + self._SM + self._H + self._VC + 1 + self._AT
            + belief_extra_len
        )
        output = self._transformer_with_progression_bias(
            tokens,
            pad_mask,
            prog,
            prog_mask,
            prog_offset=prog_offset,
        )
        output = self.final_norm(output)

        cls_out = output[:, 0]
        player_offset = 1 + self._S + self._D
        rank_stats_offset = player_offset + self._PI
        sparse_meld_offset = rank_stats_offset + self._PRS
        visible_offset = sparse_meld_offset + self._SM + self._H
        belief_offset = visible_offset + self._VC + 1 + self._AT
        shanten_offset = belief_offset + 6
        prog_offset = belief_offset + belief_extra_len
        shanten_logits = self.belief_shanten_head(output[:, shanten_offset : shanten_offset + _SHANTEN_TOKEN_COUNT])
        memory = torch.cat(
            [
                output[:, player_offset : player_offset + self._PI],
                output[:, sparse_meld_offset : sparse_meld_offset + self._SM],
                output[:, visible_offset : visible_offset + self._VC],
                output[:, prog_offset : prog_offset + prog_len],
            ],
            dim=1,
        )
        memory_padding_mask = torch.cat(
            [
                player_info_valid,
                ~sparse_meld_mask,
                visible_tile_count_valid,
                ~prog_mask,
            ],
            dim=1,
        )
        if return_tile37_table and return_shanten_logits:
            return cls_out, memory, memory_padding_mask, tile37_table[:, :TILE37_COUNT], shanten_logits
        if return_tile37_table:
            return cls_out, memory, memory_padding_mask, tile37_table[:, :TILE37_COUNT]
        if return_shanten_logits:
            return cls_out, memory, memory_padding_mask, shanten_logits
        return cls_out, memory, memory_padding_mask


class JointHiddenAllocationSampler(nn.Module):
    """MaskGIT-style legal sampler over opponent hands and residual wall counts."""

    def __init__(
        self,
        d_model: int = 384,
        decoder_hidden_dim: int | None = None,
        tile_order: list[int] | None = None,
        denoise_num_layers: int = 3,
        denoise_dim_feedforward: int | None = None,
        decode_steps: int = 12,
        max_decode_steps: int | None = None,
        confidence_method: str = "max_prob",
        shanten_aux_loss_weight: float = 0.2,
        active_row_compaction: bool = False,
        **encoder_kwargs: Any,
    ):
        super().__init__()
        del decoder_hidden_dim, max_decode_steps
        encoder_kwargs.pop("joint_assignment_confidence_beta", None)
        encoder_kwargs.pop("joint_assignment_score_gamma", None)
        encoder_kwargs["d_model"] = d_model
        self.encoder = BeliefObservationEncoder(**encoder_kwargs)
        self.d_model = d_model
        self.d_dec = d_model
        self.tile_order = tuple(tile_order or range(TILE37_COUNT))
        if sorted(self.tile_order) != list(range(TILE37_COUNT)):
            raise ValueError("tile_order must be a permutation of 0..36")
        self.decode_steps = int(decode_steps)
        if self.decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        if confidence_method not in _CONFIDENCE_METHODS:
            allowed = "', '".join(_CONFIDENCE_METHODS)
            raise ValueError(f"confidence_method must be one of: '{allowed}'")
        self.confidence_method = confidence_method
        self.active_row_compaction = bool(active_row_compaction)
        self.mask_state_id = _COUNT_CANDIDATES
        self.shanten_aux_loss_weight = float(shanten_aux_loss_weight)
        if self.shanten_aux_loss_weight < 0.0:
            raise ValueError("shanten_aux_loss_weight must be non-negative")

        cross_attn_heads = int(encoder_kwargs.get("nhead", 8))
        dropout = float(encoder_kwargs.get("dropout", 0.1))
        shared_tile_dim = int(self.encoder.tile_embed.proj.weight.shape[0])
        self.alloc_tile_proj = nn.Sequential(
            nn.Linear(shared_tile_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.alloc_tile_remaining_embed = nn.Embedding(5, d_model)
        self.alloc_seat_remaining_embed = nn.Embedding(BeliefFeatureEncoder.HAND_SIZE_DIMS, d_model)
        self.alloc_state_embed = nn.Embedding(_COUNT_CANDIDATES + 1, self.d_dec)
        self.decoder_shanten_class_embed = nn.Embedding(SHANTEN_CLASS_COUNT, self.d_dec)
        d_ff = int(denoise_dim_feedforward or max(self.d_dec * 2, 512))
        self.denoise_decoder = DenoiseTransformer(
            self.d_dec,
            cross_attn_heads,
            int(denoise_num_layers),
            d_ff,
            dropout,
            d_memory=d_model,
            output_dim=_COUNT_CANDIDATES,
        )

        self.register_buffer("count_candidates", torch.arange(_COUNT_CANDIDATES, dtype=torch.long), persistent=False)
        self.register_buffer(
            "total_counts37",
            torch.tensor(TOTAL_TILE_COUNTS37, dtype=torch.long),
            persistent=False,
        )
        subset_masks = _build_opponent_subset_masks()
        self.register_buffer("opponent_subset_masks", subset_masks, persistent=False)
        self.register_buffer("opponent_subset_masks_long", subset_masks.long(), persistent=False)
        self.register_buffer("allocation_token_tile37", _build_allocation_token_tile37(), persistent=False)
        self.register_buffer("allocation_token_opponent", _build_allocation_token_opponent(), persistent=False)
        self.register_buffer(
            "allocation_token_allocation_offset",
            _build_allocation_token_allocation_offset(),
            persistent=False,
        )
        self.register_buffer("logcomb_table", _build_logcomb_table(), persistent=False)
        self.tile37_to_tile34 = _TILE37_TO_TILE34

    def _shared_alloc_tile_embeddings(self, tile37_table: torch.Tensor) -> torch.Tensor:
        return self.alloc_tile_proj(tile37_table[:, :TILE37_COUNT])

    def _shared_alloc_seat_embeddings(self, device: torch.device) -> torch.Tensor:
        seats = torch.arange(1, BUCKET_COUNT, dtype=torch.long, device=device)
        return self.encoder._seat_model(seats, _SEAT_ROLE_PLAYER_INFO)

    def _decode_token_lookup(self, device: torch.device) -> AllocationDecodeTokenLookup:
        return AllocationDecodeTokenLookup(
            tile37=self.allocation_token_tile37.to(device),
            opponent=self.allocation_token_opponent.to(device),
            allocation_offset=self.allocation_token_allocation_offset.to(device),
        )

    @staticmethod
    def _decode_batch_offsets(batch_indices: torch.Tensor) -> AllocationDecodeBatchOffsets:
        return AllocationDecodeBatchOffsets(
            allocation=batch_indices * (BUCKET_COUNT * TILE37_COUNT),
            state=batch_indices * _ALLOCATION_TOKEN_COUNT,
            tile=batch_indices * TILE37_COUNT,
            seat=batch_indices * _OPPONENT_COUNT,
        )

    def _unseen_counts(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        visible = features["visible_tile_counts"][:, :, 1].long()
        return (self.total_counts37.to(visible.device).unsqueeze(0) - visible).clamp(min=0, max=4)

    @staticmethod
    def _opponent_riichi_mask(features: dict[str, torch.Tensor]) -> torch.Tensor:
        return features["player_stats"][:, 1:BUCKET_COUNT, 1].long().bool()

    @staticmethod
    def _apply_riichi_shanten_override(
        shanten_classes: torch.Tensor,
        riichi_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if riichi_mask is None:
            return shanten_classes
        return shanten_classes.masked_fill(riichi_mask.to(shanten_classes.device), 0)

    def _sample_shanten_classes(
        self,
        shanten_logits: torch.Tensor,
        riichi_mask: torch.Tensor | None,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if sample:
            probs = F.softmax(shanten_logits.float(), dim=-1)
            classes = torch.multinomial(probs.reshape(-1, SHANTEN_CLASS_COUNT), 1).view(
                shanten_logits.shape[0],
                _SHANTEN_TOKEN_COUNT,
            )
        else:
            classes = shanten_logits.argmax(dim=-1)
        return self._apply_riichi_shanten_override(classes, riichi_mask)

    def _initial_rem(self, features: dict[str, torch.Tensor], unseen_counts: torch.Tensor) -> torch.Tensor:
        hand_sizes = features["belief_hand_sizes"][:, :, 1].long()
        opp_caps = hand_sizes[:, 1:4]
        wall = unseen_counts.sum(dim=1, keepdim=True) - opp_caps.sum(dim=1, keepdim=True)
        return torch.cat([opp_caps, wall.clamp(min=0)], dim=1)

    @staticmethod
    def _target_validity(
        target_counts: torch.Tensor,
        rem: torch.Tensor,
        unseen_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonnegative = (target_counts >= 0).flatten(1).all(dim=1)
        tile_valid = (target_counts.sum(dim=1) == unseen_counts).all(dim=1)
        bucket_valid = (target_counts.sum(dim=2) == rem).all(dim=1)
        teacher_path_valid = nonnegative & tile_valid
        sample_valid = teacher_path_valid & bucket_valid
        return sample_valid, teacher_path_valid

    def _safe_logcomb(self, n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        table = self.logcomb_table
        if table.dtype != torch.float32:
            table = _build_logcomb_table().to(device=n.device)
            self.logcomb_table = table
        elif table.device != n.device:
            table = table.to(device=n.device)
        n_idx = n.clamp(min=0, max=_MAX_LOGCOMB_COUNT).long()
        k_idx = k.clamp(min=0, max=_MAX_LOGCOMB_COUNT).long()
        k_idx = torch.minimum(k_idx, n_idx)
        return table[n_idx, k_idx]

    def _decoder_memory(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        decoder_memory = torch.cat([context.unsqueeze(1), memory], dim=1)
        prefix_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=context.device)
        decoder_memory_padding_mask = torch.cat([prefix_mask, memory_padding_mask], dim=1)
        return decoder_memory, decoder_memory_padding_mask

    def _build_decoder_cross_attention_cache(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[CrossAttentionKVCache, ...]:
        with _profile_range("belief/build_cross_attention_cache"):
            decoder_memory, decoder_memory_padding_mask = self._decoder_memory(
                context,
                memory,
                memory_padding_mask,
            )
            return self.denoise_decoder.build_cross_attention_cache(decoder_memory, decoder_memory_padding_mask)

    @staticmethod
    def _index_cross_attention_cache(
        cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None,
        row_indices: torch.Tensor,
    ) -> tuple[CrossAttentionKVCache, ...] | None:
        if cross_attention_cache is None:
            return None
        return tuple(
            CrossAttentionKVCache(
                key=cache.key.index_select(0, row_indices),
                value=cache.value.index_select(0, row_indices),
                attention_mask=None
                if cache.attention_mask is None
                else cache.attention_mask.index_select(0, row_indices),
            )
            for cache in cross_attention_cache
        )

    @staticmethod
    def _index_decode_state(state: AllocationDecodeState, row_indices: torch.Tensor) -> AllocationDecodeState:
        return AllocationDecodeState(
            allocation=state.allocation.index_select(0, row_indices),
            state_ids=state.state_ids.index_select(0, row_indices),
            tile_remaining=state.tile_remaining.index_select(0, row_indices),
            seat_remaining=state.seat_remaining.index_select(0, row_indices),
            is_masked=state.is_masked.index_select(0, row_indices),
        )

    @staticmethod
    def _scatter_decode_state(
        state: AllocationDecodeState,
        row_indices: torch.Tensor,
        compact_state: AllocationDecodeState,
    ) -> None:
        state.allocation.index_copy_(0, row_indices, compact_state.allocation)
        state.state_ids.index_copy_(0, row_indices, compact_state.state_ids)
        state.tile_remaining.index_copy_(0, row_indices, compact_state.tile_remaining)
        state.seat_remaining.index_copy_(0, row_indices, compact_state.seat_remaining)
        state.is_masked.index_copy_(0, row_indices, compact_state.is_masked)

    def _active_decode_inputs(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        tile_emb: torch.Tensor,
        state: AllocationDecodeState,
        shanten_classes: torch.Tensor | None,
        cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None,
        active_indices: torch.Tensor,
        full_batch_indices: torch.Tensor,
    ) -> ActiveAllocationDecodeInputs:
        active_count = int(active_indices.shape[0])
        if active_count == context.shape[0]:
            return ActiveAllocationDecodeInputs(
                context=context,
                memory=memory,
                memory_padding_mask=memory_padding_mask,
                tile_emb=tile_emb,
                state=state,
                shanten_classes=shanten_classes,
                cross_attention_cache=cross_attention_cache,
                batch_indices=full_batch_indices,
                uses_full_batch=True,
            )

        with _profile_range("belief/compact_active_rows"):
            return ActiveAllocationDecodeInputs(
                context=context.index_select(0, active_indices),
                memory=memory.index_select(0, active_indices),
                memory_padding_mask=memory_padding_mask.index_select(0, active_indices),
                tile_emb=tile_emb.index_select(0, active_indices),
                state=self._index_decode_state(state, active_indices),
                shanten_classes=None if shanten_classes is None else shanten_classes.index_select(0, active_indices),
                cross_attention_cache=self._index_cross_attention_cache(cross_attention_cache, active_indices),
                batch_indices=torch.arange(active_count, device=context.device),
                uses_full_batch=False,
            )

    def _token_embeddings(
        self,
        tile_emb: torch.Tensor,
        seat_emb: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tile_emb.shape[0]
        tile_token = tile_emb.unsqueeze(2).expand(-1, -1, _OPPONENT_COUNT, -1)
        seat_token = seat_emb.view(1, 1, _OPPONENT_COUNT, self.d_dec).expand(batch_size, TILE37_COUNT, -1, -1)
        tile_rem_emb = self.alloc_tile_remaining_embed(tile_remaining.clamp(min=0, max=4).long()).unsqueeze(2)
        seat_rem_emb = self.alloc_seat_remaining_embed(
            seat_remaining.clamp(min=0, max=BeliefFeatureEncoder.HAND_SIZE_DIMS - 1).long()
        ).unsqueeze(1)
        tokens = (
            tile_token
            + seat_token
            + tile_rem_emb
            + seat_rem_emb
            + self.alloc_state_embed(state_ids.clamp(min=0, max=self.mask_state_id))
        )
        return tokens.reshape(batch_size, _ALLOCATION_TOKEN_COUNT, self.d_dec)

    def _shanten_condition_tokens(
        self,
        seat_emb: torch.Tensor,
        shanten_classes: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if shanten_classes is None:
            shanten_classes = torch.zeros(batch_size, _SHANTEN_TOKEN_COUNT, dtype=torch.long, device=device)
        shanten_classes = shanten_classes.clamp(min=0, max=SHANTEN_CLASS_COUNT - 1).long()
        seat_token = seat_emb.view(1, _SHANTEN_TOKEN_COUNT, self.d_dec).expand(batch_size, -1, -1)
        return seat_token + self.decoder_shanten_class_embed(shanten_classes)

    def _decoder_tokens(
        self,
        allocation_tokens: torch.Tensor,
        seat_emb: torch.Tensor,
        shanten_classes: torch.Tensor | None,
    ) -> torch.Tensor:
        shanten_tokens = self._shanten_condition_tokens(
            seat_emb,
            shanten_classes,
            int(allocation_tokens.shape[0]),
            allocation_tokens.device,
        )
        return torch.cat([shanten_tokens, allocation_tokens], dim=1)

    def _denoise_neural_logits(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        tile_emb: torch.Tensor,
        seat_emb: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        state_ids: torch.Tensor,
        shanten_classes: torch.Tensor | None = None,
        cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None = None,
    ) -> torch.Tensor:
        with _profile_range("belief/denoise_neural_logits"):
            with _profile_range("belief/token_embeddings"):
                allocation_tokens = self._token_embeddings(
                    tile_emb,
                    seat_emb,
                    tile_remaining,
                    seat_remaining,
                    state_ids,
                )
                tokens = self._decoder_tokens(allocation_tokens, seat_emb, shanten_classes)
            decoder_memory = None
            decoder_memory_padding_mask = None
            if cross_attention_cache is None:
                decoder_memory, decoder_memory_padding_mask = self._decoder_memory(
                    context,
                    memory,
                    memory_padding_mask,
                )
            with _profile_range("belief/denoise_decoder"):
                logits = self.denoise_decoder(
                    tokens,
                    decoder_memory,
                    decoder_memory_padding_mask,
                    cross_attention_cache=cross_attention_cache,
                )
        allocation_logits = logits[:, _SHANTEN_TOKEN_COUNT:]
        return allocation_logits.reshape(context.shape[0], TILE37_COUNT, _OPPONENT_COUNT, _COUNT_CANDIDATES)

    def _hypergeom_prior(
        self,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.count_candidates.to(tile_remaining.device).view(1, 1, 1, _COUNT_CANDIDATES)
        u_t = tile_remaining.unsqueeze(2).unsqueeze(-1)
        n_b = seat_remaining.unsqueeze(1).unsqueeze(-1)
        available_for_seat = (tile_remaining.unsqueeze(2) * is_masked.long()).sum(dim=1)
        u_total = available_for_seat.unsqueeze(1).unsqueeze(-1)
        return (
            self._safe_logcomb(u_t, candidates)
            + self._safe_logcomb(u_total - u_t, n_b - candidates)
            - self._safe_logcomb(u_total, n_b)
        )

    def _cell_hall_feasible(
        self,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.count_candidates.to(tile_remaining.device).view(1, 1, _COUNT_CANDIDATES)
        tile_value = tile_remaining.unsqueeze(2).unsqueeze(-1)
        seat_value = seat_remaining.unsqueeze(1).unsqueeze(-1)
        feasible = (
            is_masked.unsqueeze(-1)
            & (candidates.view(1, 1, 1, -1) <= tile_value)
            & (candidates.view(1, 1, 1, -1) <= seat_value)
        )
        subset_masks = self.opponent_subset_masks.to(tile_remaining.device)
        subset_masks_long = self.opponent_subset_masks_long.to(tile_remaining.device)
        subset_count = subset_masks.shape[0]

        base_demand = (seat_remaining.unsqueeze(1) * subset_masks_long.view(1, subset_count, _OPPONENT_COUNT)).sum(
            dim=2
        )
        subset_edges = is_masked.unsqueeze(1) & subset_masks.view(1, subset_count, 1, _OPPONENT_COUNT)
        any_before = subset_edges.any(dim=3)
        edge_count = subset_edges.long().sum(dim=3)
        any_before_long = any_before.long()
        base_cap = (tile_remaining.unsqueeze(1) * any_before_long).sum(dim=2)

        candidate_values = candidates.view(1, 1, 1, 1, _COUNT_CANDIDATES)
        tile_values = tile_remaining.view(tile_remaining.shape[0], 1, TILE37_COUNT, 1, 1)
        opponent_in_subset = subset_masks.view(1, subset_count, 1, _OPPONENT_COUNT)
        opponent_in_subset_long = subset_masks_long.view(1, subset_count, 1, _OPPONENT_COUNT, 1)
        any_after = torch.where(opponent_in_subset, edge_count.unsqueeze(3) > 1, any_before.unsqueeze(3))
        demand_after = base_demand.view(-1, subset_count, 1, 1, 1) - opponent_in_subset_long * candidate_values
        cap_after = (
            base_cap.view(-1, subset_count, 1, 1, 1)
            - tile_values * any_before_long.unsqueeze(3).unsqueeze(4)
            + (tile_values - candidate_values) * any_after.long().unsqueeze(4)
        )
        return feasible & (demand_after <= cap_after).all(dim=1)

    def _apply_prior_and_legality(
        self,
        neural_logits: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _profile_range("belief/apply_prior_legality"):
            feasible = self._cell_hall_feasible(tile_remaining, seat_remaining, is_masked)
            logits = self._hypergeom_prior(tile_remaining, seat_remaining, is_masked) + neural_logits.float()
            return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min), feasible

    def _forced_cells(
        self,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feasible = self._cell_hall_feasible(tile_remaining, seat_remaining, is_masked)
        forced = is_masked & (feasible.sum(dim=-1) == 1)
        values = feasible.long().argmax(dim=-1)
        return forced, values

    def _selected_cell_hall_feasible(
        self,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
        tile37: torch.Tensor,
        opponent: torch.Tensor,
        candidates: torch.Tensor,
        u_t: torch.Tensor,
    ) -> torch.Tensor:
        subset_masks = self.opponent_subset_masks.to(tile_remaining.device)
        subset_masks_long = self.opponent_subset_masks_long.to(tile_remaining.device)
        subset_count = subset_masks.shape[0]

        base_demand = (seat_remaining.unsqueeze(1) * subset_masks_long.view(1, subset_count, _OPPONENT_COUNT)).sum(
            dim=2
        )
        subset_edges = is_masked.unsqueeze(1) & subset_masks.view(1, subset_count, 1, _OPPONENT_COUNT)
        any_before = subset_edges.any(dim=3)
        edge_count = subset_edges.long().sum(dim=3)
        base_cap = (tile_remaining.unsqueeze(1) * any_before.long()).sum(dim=2)

        tile_index = tile37.view(-1, 1, 1).expand(-1, subset_count, 1)
        any_before_cell = any_before.gather(2, tile_index).squeeze(2)
        edge_count_cell = edge_count.gather(2, tile_index).squeeze(2)
        opponent_in_subset = subset_masks.gather(1, opponent.view(1, -1).expand(subset_count, -1)).transpose(0, 1)

        candidate_values = candidates.view(1, 1, _COUNT_CANDIDATES)
        opponent_in_subset_long = opponent_in_subset.long()
        any_before_cell_long = any_before_cell.long()
        any_after_cell_long = torch.where(opponent_in_subset, edge_count_cell > 1, any_before_cell).long()
        demand_after = base_demand.unsqueeze(2) - opponent_in_subset_long.unsqueeze(2) * candidate_values
        cap_after = (
            base_cap.unsqueeze(2)
            - u_t.unsqueeze(1) * any_before_cell_long.unsqueeze(2)
            + (u_t.unsqueeze(1) - candidate_values) * any_after_cell_long.unsqueeze(2)
        )
        return (demand_after <= cap_after).all(dim=1)

    def _apply_prior_and_legality_for_cells(
        self,
        neural_logits: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
        tile37: torch.Tensor,
        opponent: torch.Tensor,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if row_indices is not None:
            tile_remaining = tile_remaining.index_select(0, row_indices)
            seat_remaining = seat_remaining.index_select(0, row_indices)
            is_masked = is_masked.index_select(0, row_indices)

        row_ids = torch.arange(tile_remaining.shape[0], device=tile_remaining.device)
        candidates = self.count_candidates.to(tile_remaining.device).view(1, _COUNT_CANDIDATES)
        u_t = tile_remaining[row_ids, tile37].unsqueeze(1)
        n_b = seat_remaining[row_ids, opponent].unsqueeze(1)
        available_for_seat = (tile_remaining.unsqueeze(2) * is_masked.long()).sum(dim=1)
        u_total = available_for_seat[row_ids, opponent].unsqueeze(1)
        prior = (
            self._safe_logcomb(u_t, candidates)
            + self._safe_logcomb(u_total - u_t, n_b - candidates)
            - self._safe_logcomb(u_total, n_b)
        )
        feasible = is_masked[row_ids, tile37, opponent].unsqueeze(1) & (candidates <= u_t) & (candidates <= n_b)
        feasible &= self._selected_cell_hall_feasible(
            tile_remaining,
            seat_remaining,
            is_masked,
            tile37,
            opponent,
            candidates,
            u_t,
        )

        logits = prior + neural_logits.float()
        return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min), feasible

    @staticmethod
    def _sample_random_mask(
        batch_size: int,
        device: torch.device,
        *,
        decode_steps: int | None = None,
    ) -> torch.Tensor:
        if decode_steps == _ALLOCATION_TOKEN_COUNT:
            # Match a uniformly sampled state from the one-cell-per-step decode path.
            n_masked = torch.randint(1, _ALLOCATION_TOKEN_COUNT + 1, (batch_size,), device=device)
        else:
            t = torch.rand(batch_size, device=device)
            mask_ratio = torch.cos(t * math.pi / 2.0)
            n_masked = (
                torch.round(_ALLOCATION_TOKEN_COUNT * mask_ratio)
                .long()
                .clamp(
                    min=1,
                    max=_ALLOCATION_TOKEN_COUNT,
                )
            )
        order = torch.rand(batch_size, _ALLOCATION_TOKEN_COUNT, device=device).argsort(dim=1)
        ranks = torch.arange(_ALLOCATION_TOKEN_COUNT, device=device).unsqueeze(0)
        chosen = ranks < n_masked.unsqueeze(1)
        mask = torch.zeros(batch_size, _ALLOCATION_TOKEN_COUNT, dtype=torch.bool, device=device)
        mask.scatter_(1, order, chosen)
        return mask.reshape(batch_size, TILE37_COUNT, _OPPONENT_COUNT)

    def _resolve_forced_training_cells(
        self,
        mask: torch.Tensor,
        target_cells: torch.Tensor,
        target_ids: torch.Tensor,
        target_match: torch.Tensor,
        unseen_counts: torch.Tensor,
        seat_caps: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.clone()
        valid = valid_rows.view(-1, 1, 1)
        for _ in range(_ALLOCATION_TOKEN_COUNT):
            known_counts = target_cells * (~mask).long()
            tile_remaining = unseen_counts - known_counts.sum(dim=2)
            seat_remaining = seat_caps - known_counts.sum(dim=1)
            forced, values = self._forced_cells(tile_remaining, seat_remaining, mask)
            forced &= valid & target_match & (values == target_ids)
            if not forced.any():
                break
            mask = mask & ~forced
        return mask

    def _sample_training_mask(
        self,
        target_cells: torch.Tensor,
        target_ids: torch.Tensor,
        target_match: torch.Tensor,
        unseen_counts: torch.Tensor,
        seat_caps: torch.Tensor,
        target_sample_valid: torch.Tensor,
        *,
        decode_steps: int | None = None,
    ) -> torch.Tensor:
        mask = self._sample_random_mask(target_cells.shape[0], target_cells.device, decode_steps=decode_steps)
        for attempt in range(_TRAINING_MASK_MAX_RETRIES):
            mask = self._resolve_forced_training_cells(
                mask,
                target_cells,
                target_ids,
                target_match,
                unseen_counts,
                seat_caps,
                target_sample_valid,
            )
            empty_valid_rows = target_sample_valid & ~mask.flatten(1).any(dim=1)
            if not empty_valid_rows.any() or attempt == _TRAINING_MASK_MAX_RETRIES - 1:
                break
            retry_count = int(empty_valid_rows.long().sum().item())
            mask[empty_valid_rows] = self._sample_random_mask(
                retry_count,
                target_cells.device,
                decode_steps=decode_steps,
            )
        return mask

    @staticmethod
    def _cosine_target_unmasked(step: int, decode_steps: int) -> int:
        ratio_masked = math.cos(math.pi * float(step + 1) / (2.0 * float(decode_steps)))
        target = int(round(_ALLOCATION_TOKEN_COUNT * (1.0 - ratio_masked)))
        if step < decode_steps - 1 and target == 0:
            target = 1
        return max(0, min(_ALLOCATION_TOKEN_COUNT, target))

    @staticmethod
    def _scheduled_unmask_count(step: int, decode_steps: int, remaining_count: int) -> int:
        if decode_steps == _ALLOCATION_TOKEN_COUNT:
            return min(1, remaining_count)
        if step == decode_steps - 1:
            return remaining_count
        target_unmasked = JointHiddenAllocationSampler._cosine_target_unmasked(step, decode_steps)
        current_unmasked = _ALLOCATION_TOKEN_COUNT - remaining_count
        return max(0, min(target_unmasked - current_unmasked, remaining_count))

    @staticmethod
    def _scheduled_unmask_counts(step: int, decode_steps: int, remaining_counts: torch.Tensor) -> torch.Tensor:
        if decode_steps == _ALLOCATION_TOKEN_COUNT:
            return remaining_counts.clamp(max=1)
        if step == decode_steps - 1:
            return remaining_counts
        target_unmasked = JointHiddenAllocationSampler._cosine_target_unmasked(step, decode_steps)
        current_unmasked = _ALLOCATION_TOKEN_COUNT - remaining_counts
        return (target_unmasked - current_unmasked).clamp(min=0).minimum(remaining_counts)

    @staticmethod
    def _confidence(
        probs: torch.Tensor,
        method: str,
        feasible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if method == "neg_entropy":
            safe_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            return (safe_probs * safe_probs.log()).sum(dim=-1)
        if method == "legal_normalized_entropy":
            if feasible is None:
                raise ValueError("feasible mask is required for legal_normalized_entropy")
            legal_count = feasible.sum(dim=-1)
            legal_probs = probs.masked_fill(~feasible, 0.0)
            legal_mass = legal_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
            legal_probs = legal_probs / legal_mass
            safe_legal_probs = legal_probs.clamp_min(torch.finfo(probs.dtype).tiny)
            entropy = -(legal_probs * safe_legal_probs.log()).sum(dim=-1)
            denominator = legal_count.clamp_min(2).to(probs.dtype).log()
            score = (1.0 - entropy / denominator).clamp(min=0.0, max=1.0)
            score = torch.where(legal_count == 1, score.new_full(score.shape, 2.0), score)
            return torch.where(legal_count > 0, score, score.new_full(score.shape, torch.finfo(score.dtype).min))
        return probs.max(dim=-1).values

    @staticmethod
    def _mean_unique_sample_rate(allocations: torch.Tensor) -> torch.Tensor:
        if allocations.shape[1] == 0:
            return allocations.new_tensor(0.0, dtype=torch.float32)
        rates = []
        flat = allocations.detach().reshape(allocations.shape[0], allocations.shape[1], -1).cpu()
        for sample_set in flat:
            rates.append(float(torch.unique(sample_set, dim=0).shape[0]) / float(sample_set.shape[0]))
        return allocations.new_tensor(sum(rates) / max(len(rates), 1), dtype=torch.float32)

    @staticmethod
    def _mean_pairwise_l1(allocations: torch.Tensor) -> torch.Tensor:
        samples = allocations.shape[1]
        if samples < 2:
            return allocations.new_tensor(0.0, dtype=torch.float32)
        flat = allocations.float().reshape(allocations.shape[0], samples, -1)
        distances = torch.cdist(flat, flat, p=1)
        tri = torch.triu(torch.ones(samples, samples, dtype=torch.bool, device=allocations.device), diagonal=1)
        return distances[:, tri].mean()

    def allocation_diagnostics(
        self,
        features: dict[str, torch.Tensor],
        allocations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return legality and diversity diagnostics for sampled allocations."""
        if allocations.dim() == 3:
            allocations = allocations.unsqueeze(1)
        if allocations.dim() != 4 or allocations.shape[2:] != (BUCKET_COUNT, TILE37_COUNT):
            raise ValueError(
                "allocations must have shape "
                f"(B, S, {BUCKET_COUNT}, {TILE37_COUNT}) or (B, {BUCKET_COUNT}, {TILE37_COUNT})"
            )
        unseen_counts = self._unseen_counts(features)
        rem = self._initial_rem(features, unseen_counts)
        alloc = allocations.to(unseen_counts.device).long()
        tile_exact = (alloc.sum(dim=2) == unseen_counts.unsqueeze(1)).all(dim=-1)
        bucket_exact = (alloc.sum(dim=3) == rem.unsqueeze(1)).all(dim=-1)
        opponent_exact = (alloc[:, :, :3].sum(dim=3) == rem[:, :3].unsqueeze(1)).all(dim=-1)
        nonnegative = (alloc >= 0).flatten(2).all(dim=-1)
        legal = tile_exact & bucket_exact & nonnegative
        return {
            "allocation_legal_rate": legal.float().mean(),
            "opponent_hand_size_exact_rate": opponent_exact.float().mean(),
            "tile_count_exact_rate": tile_exact.float().mean(),
            "bucket_count_exact_rate": bucket_exact.float().mean(),
            "unique_sample_rate": self._mean_unique_sample_rate(alloc),
            "pairwise_l1_distance": self._mean_pairwise_l1(alloc),
        }

    def forward(  # noqa: PLR0915
        self,
        features: dict[str, torch.Tensor],
        target_counts: torch.Tensor | None = None,
        *,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        context, memory, memory_padding_mask, tile37_table, shanten_logits = self.encoder.forward_context_and_memory(
            features,
            return_tile37_table=True,
            return_shanten_logits=True,
        )
        unseen_counts = self._unseen_counts(features)
        tile_emb = self._shared_alloc_tile_embeddings(tile37_table)
        seat_emb = self._shared_alloc_seat_embeddings(context.device)
        rem = self._initial_rem(features, unseen_counts)
        riichi_mask = self._opponent_riichi_mask(features)
        if target_counts is not None:
            return self._training_forward(
                context,
                memory,
                memory_padding_mask,
                unseen_counts,
                tile_emb,
                seat_emb,
                rem,
                target_counts,
                shanten_logits=shanten_logits,
                target_shanten_classes=features.get("belief_shanten_labels"),
                riichi_mask=riichi_mask,
            )
        shanten_classes = self._sample_shanten_classes(
            shanten_logits,
            riichi_mask,
            sample=sample,
        )
        cross_attention_cache = self._build_decoder_cross_attention_cache(context, memory, memory_padding_mask)
        allocation = self._iterative_decode(
            context,
            memory,
            memory_padding_mask,
            unseen_counts,
            tile_emb,
            seat_emb,
            rem,
            sample=sample,
            temperature=temperature,
            decode_steps=self.decode_steps,
            shanten_classes=shanten_classes,
            cross_attention_cache=cross_attention_cache,
        )
        return {"allocation": allocation, "shanten_logits": shanten_logits, "shanten_classes": shanten_classes}

    def _training_forward(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        unseen_counts: torch.Tensor,
        tile_emb: torch.Tensor,
        seat_emb: torch.Tensor,
        rem: torch.Tensor,
        target_counts: torch.Tensor,
        *,
        shanten_logits: torch.Tensor | None = None,
        target_shanten_classes: torch.Tensor | None = None,
        riichi_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_counts = target_counts.long()
        has_shanten_targets = target_shanten_classes is not None
        if target_shanten_classes is not None:
            target_shanten_classes = target_shanten_classes.to(context.device).long()
            target_shanten_classes = self._apply_riichi_shanten_override(target_shanten_classes, riichi_mask)
        elif shanten_logits is not None:
            target_shanten_classes = self._sample_shanten_classes(
                shanten_logits.detach(),
                riichi_mask,
                sample=False,
            )
        target_sample_valid, target_teacher_path_valid = self._target_validity(
            target_counts,
            rem,
            unseen_counts,
        )
        target_rem = target_counts.sum(dim=2)
        use_target_rem = (~target_sample_valid & target_teacher_path_valid).unsqueeze(1)
        rem = torch.where(use_target_rem, target_rem, rem)

        target_cells = target_counts[:, :_OPPONENT_COUNT].permute(0, 2, 1).long()
        target_match = (target_cells >= 0) & (target_cells < _COUNT_CANDIDATES)
        target_ids = target_cells.clamp(min=0, max=_COUNT_CANDIDATES - 1)
        mask = self._sample_training_mask(
            target_cells,
            target_ids,
            target_match,
            unseen_counts,
            rem[:, :_OPPONENT_COUNT],
            target_sample_valid,
            decode_steps=self.decode_steps,
        )
        state_ids = target_ids.masked_fill(mask, self.mask_state_id)
        known_counts = target_cells * (~mask).long()
        tile_remaining = unseen_counts - known_counts.sum(dim=2)
        seat_remaining = rem[:, :_OPPONENT_COUNT] - known_counts.sum(dim=1)

        neural_logits = self._denoise_neural_logits(
            context,
            memory,
            memory_padding_mask,
            tile_emb,
            seat_emb,
            tile_remaining,
            seat_remaining,
            state_ids,
            shanten_classes=target_shanten_classes,
        )
        logits, feasible = self._apply_prior_and_legality(
            neural_logits,
            tile_remaining,
            seat_remaining,
            mask,
        )

        target_feasible = feasible.gather(3, target_ids.unsqueeze(3)).squeeze(3) & target_match
        valid = mask & target_sample_valid.view(-1, 1, 1) & target_feasible

        out = {
            "allocation": target_counts,
            "invalid_target_rate": (~target_sample_valid).float().mean(),
        }
        zero = context.sum() * 0.0
        if valid.any():
            allocation_loss = F.cross_entropy(logits[valid], target_cells[valid], reduction="mean")
            pred = logits[valid].argmax(dim=1)
            acc = (pred == target_cells[valid]).float().mean()
        else:
            allocation_loss = zero
            acc = zero.detach()

        loss = allocation_loss
        out.update({"allocation_loss": allocation_loss, "acc": acc})
        if has_shanten_targets and shanten_logits is not None and target_shanten_classes is not None:
            shanten_valid = target_sample_valid
            if shanten_valid.any():
                shanten_loss = F.cross_entropy(
                    shanten_logits[shanten_valid].reshape(-1, SHANTEN_CLASS_COUNT),
                    target_shanten_classes[shanten_valid].reshape(-1),
                    reduction="mean",
                )
                shanten_pred = shanten_logits[shanten_valid].argmax(dim=-1)
                shanten_acc = (shanten_pred == target_shanten_classes[shanten_valid]).float().mean()
                loss = loss + self.shanten_aux_loss_weight * shanten_loss
            else:
                shanten_loss = zero
                shanten_acc = zero.detach()
            out.update(
                {
                    "shanten_loss": shanten_loss,
                    "shanten_acc": shanten_acc,
                    "shanten_aux_loss_weight": loss.new_tensor(self.shanten_aux_loss_weight),
                }
            )
        out["loss"] = loss
        return out

    def _sample_and_apply_cell_step(
        self,
        neural_logits_cells: torch.Tensor,
        state: AllocationDecodeState,
        token: torch.Tensor,
        tile37: torch.Tensor,
        opponent: torch.Tensor,
        allocation_offset: torch.Tensor,
        batch_offsets: AllocationDecodeBatchOffsets,
        *,
        sample: bool,
        temp: float,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _PROFILE_RANGES_ENABLED:
            with _profile_range("belief/cell_lookup"):
                state_index = batch_offsets.state + token
                cell_neural_logits = neural_logits_cells.index_select(0, state_index)
            with _profile_range("belief/cell_legality"):
                cell_logits, _cell_feasible = self._apply_prior_and_legality_for_cells(
                    cell_neural_logits,
                    state.tile_remaining,
                    state.seat_remaining,
                    state.is_masked,
                    tile37,
                    opponent,
                    row_indices=row_indices,
                )
            with _profile_range("belief/cell_sample"):
                if sample:
                    cell_probs = F.softmax((cell_logits / temp).float(), dim=1)
                    chosen = torch.multinomial(cell_probs, 1).squeeze(1)
                else:
                    chosen = cell_logits.argmax(dim=1)
            with _profile_range("belief/cell_state_update"):
                state.apply_cell_update(batch_offsets, state_index, tile37, opponent, allocation_offset, chosen)
        else:
            state_index = batch_offsets.state + token
            cell_neural_logits = neural_logits_cells.index_select(0, state_index)
            cell_logits, _cell_feasible = self._apply_prior_and_legality_for_cells(
                cell_neural_logits,
                state.tile_remaining,
                state.seat_remaining,
                state.is_masked,
                tile37,
                opponent,
                row_indices=row_indices,
            )
            if sample:
                cell_probs = F.softmax((cell_logits / temp).float(), dim=1)
                chosen = torch.multinomial(cell_probs, 1).squeeze(1)
            else:
                chosen = cell_logits.argmax(dim=1)
            state.apply_cell_update(batch_offsets, state_index, tile37, opponent, allocation_offset, chosen)
        return tile37, opponent, chosen

    def _sample_and_apply_dense_cell_step(
        self,
        logits_cells: torch.Tensor,
        state: AllocationDecodeState,
        token: torch.Tensor,
        tile37: torch.Tensor,
        opponent: torch.Tensor,
        batch_indices: torch.Tensor,
        *,
        sample: bool,
        temp: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_offsets = self._decode_batch_offsets(batch_indices)
        state_index = batch_offsets.state + token
        with _profile_range("belief/cell_lookup"):
            cell_logits = logits_cells.index_select(0, state_index)
        with _profile_range("belief/cell_sample"):
            if sample:
                cell_probs = F.softmax((cell_logits / temp).float(), dim=1)
                chosen = torch.multinomial(cell_probs, 1).squeeze(1)
            else:
                chosen = cell_logits.argmax(dim=1)
        with _profile_range("belief/cell_state_update"):
            update_mask = F.one_hot(token, num_classes=_ALLOCATION_TOKEN_COUNT).view(
                token.shape[0],
                TILE37_COUNT,
                _OPPONENT_COUNT,
            )
            update_mask = update_mask.to(torch.bool) & state.is_masked
            update_values = chosen.view(-1, 1, 1).expand_as(state.state_ids)
            state.apply_dense_cell_updates(update_mask, update_values)
        return tile37, opponent, chosen

    def _apply_forced_decode_cells_dense_once(
        self,
        state: AllocationDecodeState,
    ) -> None:
        forced, values = self._forced_cells(state.tile_remaining, state.seat_remaining, state.is_masked)
        state.apply_dense_cell_updates(forced, values)

    def _apply_forced_decode_cells(
        self,
        state: AllocationDecodeState,
        token_lookup: AllocationDecodeTokenLookup,
    ) -> int:
        resolved = 0
        for _ in range(_ALLOCATION_TOKEN_COUNT):
            forced, values = self._forced_cells(state.tile_remaining, state.seat_remaining, state.is_masked)
            if not forced.any():
                break

            state_index = forced.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            chosen = values.reshape(-1).index_select(0, state_index)
            batch_index = torch.div(state_index, _ALLOCATION_TOKEN_COUNT, rounding_mode="floor")
            token = state_index.remainder(_ALLOCATION_TOKEN_COUNT)
            tile37 = token_lookup.tile37.index_select(0, token)
            opponent = token_lookup.opponent.index_select(0, token)
            allocation_offset = token_lookup.allocation_offset.index_select(0, token)
            allocation_index = batch_index * (BUCKET_COUNT * TILE37_COUNT) + allocation_offset
            tile_index = batch_index * TILE37_COUNT + tile37
            seat_index = batch_index * _OPPONENT_COUNT + opponent
            state.apply_flat_cell_updates(state_index, allocation_index, tile_index, seat_index, chosen)
            resolved += int(state_index.numel())
        return resolved

    def _iterative_decode_compacted_fast_path(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        tile_emb: torch.Tensor,
        seat_emb: torch.Tensor,
        state: AllocationDecodeState,
        *,
        sample: bool,
        temp: float,
        decode_steps: int,
        shanten_classes: torch.Tensor | None,
        cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None,
    ) -> None:
        batch_size = context.shape[0]
        device = context.device
        token_lookup = self._decode_token_lookup(device)

        with _profile_range("belief/forced_initial_decode_cells"):
            self._apply_forced_decode_cells_dense_once(state)
        remaining_counts = state.is_masked.flatten(1).long().sum(dim=1)
        active_indices = remaining_counts.nonzero(as_tuple=False).squeeze(1)
        full_batch_indices = torch.arange(batch_size, device=device)

        for _step in range(decode_steps):
            active_count = int(active_indices.shape[0])
            if active_count == 0:
                break
            active = self._active_decode_inputs(
                context,
                memory,
                memory_padding_mask,
                tile_emb,
                state,
                shanten_classes,
                cross_attention_cache,
                active_indices,
                full_batch_indices,
            )

            with _profile_range("belief/decode_step"):
                neural_logits = self._denoise_neural_logits(
                    active.context,
                    active.memory,
                    active.memory_padding_mask,
                    active.tile_emb,
                    seat_emb,
                    active.state.tile_remaining,
                    active.state.seat_remaining,
                    active.state.state_ids,
                    shanten_classes=active.shanten_classes,
                    cross_attention_cache=active.cross_attention_cache,
                )
                logits, feasible = self._apply_prior_and_legality(
                    neural_logits,
                    active.state.tile_remaining,
                    active.state.seat_remaining,
                    active.state.is_masked,
                )
                logits_cells = logits.reshape(active_count * _ALLOCATION_TOKEN_COUNT, _COUNT_CANDIDATES)
                with _profile_range("belief/confidence_order"):
                    probs = F.softmax((logits / temp).float(), dim=-1)
                    confidence = self._confidence(probs, self.confidence_method, feasible)
                    confidence = confidence.masked_fill(~active.state.is_masked, torch.finfo(confidence.dtype).min)
                    token = confidence.reshape(active_count, _ALLOCATION_TOKEN_COUNT).argmax(dim=1)

                with _profile_range("belief/cell_sampling_loop"):
                    tile37 = token_lookup.tile37.index_select(0, token)
                    opponent = token_lookup.opponent.index_select(0, token)
                    self._sample_and_apply_dense_cell_step(
                        logits_cells,
                        active.state,
                        token,
                        tile37,
                        opponent,
                        active.batch_indices,
                        sample=sample,
                        temp=temp,
                    )
                    with _profile_range("belief/forced_decode_cells"):
                        self._apply_forced_decode_cells_dense_once(active.state)
                    active_remaining_counts = active.state.is_masked.flatten(1).long().sum(dim=1)

            if not active.uses_full_batch:
                with _profile_range("belief/scatter_active_rows"):
                    self._scatter_decode_state(state, active_indices, active.state)
                    remaining_counts.index_copy_(0, active_indices, active_remaining_counts)
            else:
                remaining_counts = active_remaining_counts
            still_active = active_remaining_counts > 0
            active_indices = active_indices.index_select(0, still_active.nonzero(as_tuple=False).squeeze(1))

    def _iterative_decode(  # noqa: PLR0915
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        unseen_counts: torch.Tensor,
        tile_emb: torch.Tensor,
        seat_emb: torch.Tensor,
        rem: torch.Tensor,
        *,
        sample: bool,
        temperature: float,
        decode_steps: int,
        shanten_classes: torch.Tensor | None = None,
        cross_attention_cache: tuple[CrossAttentionKVCache, ...] | None = None,
    ) -> torch.Tensor:
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")

        batch_size = context.shape[0]
        device = context.device
        state = AllocationDecodeState(
            allocation=torch.zeros(batch_size, BUCKET_COUNT, TILE37_COUNT, dtype=torch.long, device=device),
            state_ids=torch.full(
                (batch_size, TILE37_COUNT, _OPPONENT_COUNT),
                self.mask_state_id,
                dtype=torch.long,
                device=device,
            ),
            tile_remaining=unseen_counts.clone(),
            seat_remaining=rem[:, :_OPPONENT_COUNT].clone(),
            is_masked=torch.ones(batch_size, TILE37_COUNT, _OPPONENT_COUNT, dtype=torch.bool, device=device),
        )
        temp = max(float(temperature), 1e-6)
        batch_indices = torch.arange(batch_size, device=device)
        token_lookup = self._decode_token_lookup(device)

        with _profile_range("belief/iterative_decode"):
            if decode_steps == _ALLOCATION_TOKEN_COUNT:
                if self.active_row_compaction:
                    self._iterative_decode_compacted_fast_path(
                        context,
                        memory,
                        memory_padding_mask,
                        tile_emb,
                        seat_emb,
                        state,
                        sample=sample,
                        temp=temp,
                        decode_steps=decode_steps,
                        shanten_classes=shanten_classes,
                        cross_attention_cache=cross_attention_cache,
                    )
                else:
                    with _profile_range("belief/forced_initial_decode_cells"):
                        self._apply_forced_decode_cells_dense_once(state)
                    remaining_counts = state.is_masked.flatten(1).long().sum(dim=1)
                    for _step in range(decode_steps):
                        if not remaining_counts.any():
                            break
                        with _profile_range("belief/decode_step"):
                            neural_logits = self._denoise_neural_logits(
                                context,
                                memory,
                                memory_padding_mask,
                                tile_emb,
                                seat_emb,
                                state.tile_remaining,
                                state.seat_remaining,
                                state.state_ids,
                                shanten_classes=shanten_classes,
                                cross_attention_cache=cross_attention_cache,
                            )
                            logits, feasible = self._apply_prior_and_legality(
                                neural_logits,
                                state.tile_remaining,
                                state.seat_remaining,
                                state.is_masked,
                            )
                            logits_cells = logits.reshape(batch_size * _ALLOCATION_TOKEN_COUNT, _COUNT_CANDIDATES)
                            with _profile_range("belief/confidence_order"):
                                probs = F.softmax((logits / temp).float(), dim=-1)
                                confidence = self._confidence(probs, self.confidence_method, feasible)
                                confidence = confidence.masked_fill(~state.is_masked, torch.finfo(confidence.dtype).min)
                                token = confidence.reshape(batch_size, _ALLOCATION_TOKEN_COUNT).argmax(dim=1)

                            with _profile_range("belief/cell_sampling_loop"):
                                tile37 = token_lookup.tile37.index_select(0, token)
                                opponent = token_lookup.opponent.index_select(0, token)
                                self._sample_and_apply_dense_cell_step(
                                    logits_cells,
                                    state,
                                    token,
                                    tile37,
                                    opponent,
                                    batch_indices,
                                    sample=sample,
                                    temp=temp,
                                )
                                with _profile_range("belief/forced_decode_cells"):
                                    self._apply_forced_decode_cells_dense_once(state)
                                remaining_counts = state.is_masked.flatten(1).long().sum(dim=1)
            else:
                with _profile_range("belief/forced_initial_decode_cells"):
                    self._apply_forced_decode_cells(state, token_lookup)
                remaining_counts = state.is_masked.flatten(1).long().sum(dim=1)
                for step in range(decode_steps):
                    if not remaining_counts.any():
                        break
                    with _profile_range("belief/decode_step"):
                        neural_logits = self._denoise_neural_logits(
                            context,
                            memory,
                            memory_padding_mask,
                            tile_emb,
                            seat_emb,
                            state.tile_remaining,
                            state.seat_remaining,
                            state.state_ids,
                            shanten_classes=shanten_classes,
                            cross_attention_cache=cross_attention_cache,
                        )
                        neural_logits_cells = neural_logits.reshape(
                            batch_size * _ALLOCATION_TOKEN_COUNT,
                            _COUNT_CANDIDATES,
                        )
                        logits, feasible = self._apply_prior_and_legality(
                            neural_logits,
                            state.tile_remaining,
                            state.seat_remaining,
                            state.is_masked,
                        )
                        with _profile_range("belief/confidence_order"):
                            probs = F.softmax((logits / temp).float(), dim=-1)
                            confidence = self._confidence(probs, self.confidence_method, feasible)
                            confidence = confidence.masked_fill(~state.is_masked, torch.finfo(confidence.dtype).min)

                            n_to_unmask = self._scheduled_unmask_counts(step, decode_steps, remaining_counts)
                            max_to_unmask = int(n_to_unmask.max().item())

                            if max_to_unmask == 0:
                                continue
                            token_order = confidence.reshape(batch_size, _ALLOCATION_TOKEN_COUNT).argsort(
                                dim=1,
                                descending=True,
                            )
                            selected_token_order = token_order[:, :max_to_unmask]
                        with _profile_range("belief/cell_sampling_loop"):
                            for rank in range(max_to_unmask):
                                active = n_to_unmask > rank
                                if not active.any():
                                    continue
                                active_indices = batch_indices[active]
                                token = selected_token_order[active, rank]
                                tile37 = token_lookup.tile37.index_select(0, token)
                                opponent = token_lookup.opponent.index_select(0, token)
                                still_masked = state.is_masked[active_indices, tile37, opponent]
                                if not still_masked.any():
                                    continue
                                active_indices = active_indices[still_masked]
                                token = token[still_masked]
                                tile37 = tile37[still_masked]
                                opponent = opponent[still_masked]
                                allocation_offset = token_lookup.allocation_offset.index_select(0, token)
                                batch_offsets = self._decode_batch_offsets(active_indices)
                                self._sample_and_apply_cell_step(
                                    neural_logits_cells,
                                    state,
                                    token,
                                    tile37,
                                    opponent,
                                    allocation_offset,
                                    batch_offsets,
                                    sample=sample,
                                    temp=temp,
                                    row_indices=active_indices,
                                )
                                with _profile_range("belief/forced_decode_cells"):
                                    self._apply_forced_decode_cells(state, token_lookup)
                                remaining_counts = state.is_masked.flatten(1).long().sum(dim=1)
                                if not remaining_counts.any():
                                    break

        state.allocation[:, BUCKET_COUNT - 1] = unseen_counts - state.allocation[:, :_OPPONENT_COUNT].sum(dim=1)
        return state.allocation

    @torch.inference_mode()
    def prepare_allocation_sampling_context(
        self,
        features: dict[str, torch.Tensor],
    ) -> AllocationSamplingContext:
        with _profile_range("belief/prepare_allocation_sampling_context"):
            (
                context,
                memory,
                memory_padding_mask,
                tile37_table,
                shanten_logits,
            ) = self.encoder.forward_context_and_memory(
                features,
                return_tile37_table=True,
                return_shanten_logits=True,
            )
            unseen_counts = self._unseen_counts(features)
            tile_emb = self._shared_alloc_tile_embeddings(tile37_table)
            seat_emb = self._shared_alloc_seat_embeddings(context.device)
            rem = self._initial_rem(features, unseen_counts)
            riichi_mask = self._opponent_riichi_mask(features)
        return AllocationSamplingContext(
            context=context,
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            unseen_counts=unseen_counts,
            tile_emb=tile_emb,
            seat_emb=seat_emb,
            rem=rem,
            shanten_logits=shanten_logits,
            riichi_mask=riichi_mask,
        )

    @torch.inference_mode()
    def sample_allocations_from_context(
        self,
        sampling_context: AllocationSamplingContext,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        decode_steps: int | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        with _profile_range("belief/sample_allocations_from_context"):
            with _profile_range("belief/repeat_sampling_context"):
                context = sampling_context.context.repeat_interleave(num_samples, dim=0)
                tile_emb = sampling_context.tile_emb.repeat_interleave(num_samples, dim=0)
                memory = sampling_context.memory.repeat_interleave(num_samples, dim=0)
                memory_padding_mask = sampling_context.memory_padding_mask.repeat_interleave(num_samples, dim=0)
                unseen_counts = sampling_context.unseen_counts.repeat_interleave(num_samples, dim=0)
                rem = sampling_context.rem.repeat_interleave(num_samples, dim=0)
                shanten_logits = sampling_context.shanten_logits.repeat_interleave(num_samples, dim=0)
                riichi_mask = sampling_context.riichi_mask.repeat_interleave(num_samples, dim=0)
                shanten_classes = self._sample_shanten_classes(
                    shanten_logits,
                    riichi_mask,
                    sample=True,
                )
            cross_attention_cache = self._build_decoder_cross_attention_cache(context, memory, memory_padding_mask)
            sampled = self._iterative_decode(
                context,
                memory,
                memory_padding_mask,
                unseen_counts,
                tile_emb,
                sampling_context.seat_emb,
                rem,
                sample=True,
                temperature=temperature,
                decode_steps=int(decode_steps or self.decode_steps),
                shanten_classes=shanten_classes,
                cross_attention_cache=cross_attention_cache,
            )
        return sampled.reshape(sampling_context.batch_size, num_samples, BUCKET_COUNT, TILE37_COUNT)

    @torch.inference_mode()
    def sample_allocations(
        self,
        features: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        decode_steps: int | None = None,
    ) -> torch.Tensor:
        sampling_context = self.prepare_allocation_sampling_context(features)
        return self.sample_allocations_from_context(
            sampling_context,
            num_samples=num_samples,
            temperature=temperature,
            decode_steps=decode_steps,
        )
