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
_TILE34_COUNT = 34
_OPPONENT_COUNT = BUCKET_COUNT - 1
_ALLOCATION_TOKEN_COUNT = TILE37_COUNT * _OPPONENT_COUNT


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


_TILE37_TO_TILE34 = tuple(int(tile34) for tile34 in _build_tile37_to_tile34().tolist())
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

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])


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

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.self_norm(x)
        attended, _weights = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout(attended)

        residual = x
        attended, _weights = self.cross_attn(
            self.cross_norm(x),
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
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
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = tokens
        for layer in self.layers:
            x = layer(x, memory, memory_padding_mask)
        return self.head(self.final_norm(x))


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
        self.belief_segment_embed = nn.Embedding(3, self.d_model)
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

    def forward_context(self, x: dict[str, torch.Tensor] | tuple[torch.Tensor, int, int]) -> torch.Tensor:
        context, _memory, _memory_padding_mask = self.forward_context_and_memory(x)
        return context

    def forward_context_and_memory(  # noqa: PLR0915
        self,
        x: dict[str, torch.Tensor] | tuple[torch.Tensor, int, int],
        *,
        return_tile37_table: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
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

        belief_extra = torch.cat([phase_emb, current_actor_emb, belief_hand_size_emb], dim=1)
        belief_seg_ids = torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.long, device=device).unsqueeze(0)
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
        prog_emb = prog_emb + self._progression_pe(prog_len, device, prog_emb.dtype)

        cls = self.cls_token.expand(batch_size, -1, -1)
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
                belief_extra,
                prog_emb,
            ],
            dim=1,
        )

        seg_ids = self._segment_ids(prog_len, 0, device)
        seg_ids = torch.cat(
            [
                seg_ids[:, : 1 + self._S + self._D + self._PI + self._SM + self._H + self._VC + 1 + self._AT],
                torch.full((1, 6), 5, dtype=torch.long, device=device),
                seg_ids[:, 1 + self._S + self._D + self._PI + self._SM + self._H + self._VC + 1 + self._AT :],
            ],
            dim=1,
        )
        tokens = tokens + self.segment_embed(seg_ids.expand(batch_size, -1))

        cls_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        dealer_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        player_info_valid = torch.zeros(batch_size, self._PI, dtype=torch.bool, device=device)
        numeric_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        visible_tile_count_valid = torch.zeros(batch_size, self._VC, dtype=torch.bool, device=device)
        agari_overtake_valid = torch.zeros(batch_size, self._AT, dtype=torch.bool, device=device)
        belief_valid = torch.zeros(batch_size, 6, dtype=torch.bool, device=device)
        pad_mask = torch.cat(
            [
                cls_valid,
                ~sparse_mask,
                dealer_valid,
                player_info_valid,
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

        output = self.transformer(tokens, src_key_padding_mask=pad_mask)
        output = self.final_norm(output)

        cls_out = output[:, 0]
        player_offset = 1 + self._S + self._D
        sparse_meld_offset = player_offset + self._PI
        visible_offset = player_offset + self._PI + self._SM + self._H
        prog_offset = visible_offset + self._VC + 1 + self._AT + 6
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
        if return_tile37_table:
            return cls_out, memory, memory_padding_mask, tile37_table[:, :TILE37_COUNT]
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
        **encoder_kwargs: Any,
    ):
        super().__init__()
        del decoder_hidden_dim
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
        self.max_decode_steps = int(max_decode_steps or self.decode_steps)
        if self.max_decode_steps < self.decode_steps:
            raise ValueError("max_decode_steps must be >= decode_steps")
        if confidence_method not in {"max_prob", "neg_entropy"}:
            raise ValueError("confidence_method must be 'max_prob' or 'neg_entropy'")
        self.confidence_method = confidence_method
        self.mask_state_id = _COUNT_CANDIDATES

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
        self.alloc_time_embed = nn.Embedding(self.max_decode_steps, self.d_dec)
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
        self.tile37_to_tile34 = _TILE37_TO_TILE34

    def _shared_alloc_tile_embeddings(self, tile37_table: torch.Tensor) -> torch.Tensor:
        return self.alloc_tile_proj(tile37_table[:, :TILE37_COUNT])

    def _shared_alloc_seat_embeddings(self, device: torch.device) -> torch.Tensor:
        seats = torch.arange(1, BUCKET_COUNT, dtype=torch.long, device=device)
        return self.encoder._seat_model(seats, _SEAT_ROLE_PLAYER_INFO)

    def _unseen_counts(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        visible = features["visible_tile_counts"][:, :, 1].long()
        return (self.total_counts37.to(visible.device).unsqueeze(0) - visible).clamp(min=0, max=4)

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

    @staticmethod
    def _safe_logcomb(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        n_f = n.float().clamp_min(0.0)
        k_f = k.float().clamp_min(0.0)
        k_f = torch.minimum(k_f, n_f)
        return torch.lgamma(n_f + 1.0) - torch.lgamma(k_f + 1.0) - torch.lgamma(n_f - k_f + 1.0)

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

    def _token_embeddings(
        self,
        tile_emb: torch.Tensor,
        seat_emb: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        state_ids: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tile_emb.shape[0]
        tile_token = tile_emb.unsqueeze(2).expand(-1, -1, _OPPONENT_COUNT, -1)
        seat_token = seat_emb.view(1, 1, _OPPONENT_COUNT, self.d_dec).expand(batch_size, TILE37_COUNT, -1, -1)
        tile_rem_emb = self.alloc_tile_remaining_embed(tile_remaining.clamp(min=0, max=4).long()).unsqueeze(2)
        seat_rem_emb = self.alloc_seat_remaining_embed(
            seat_remaining.clamp(min=0, max=BeliefFeatureEncoder.HAND_SIZE_DIMS - 1).long()
        ).unsqueeze(1)
        step_ids = step_ids.clamp(min=0, max=self.max_decode_steps - 1)
        tokens = (
            tile_token
            + seat_token
            + tile_rem_emb
            + seat_rem_emb
            + self.alloc_state_embed(state_ids.clamp(min=0, max=self.mask_state_id))
            + self.alloc_time_embed(step_ids).view(batch_size, 1, 1, self.d_dec)
        )
        return tokens.reshape(batch_size, _ALLOCATION_TOKEN_COUNT, self.d_dec)

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
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        with _profile_range("belief/denoise_neural_logits"):
            with _profile_range("belief/token_embeddings"):
                tokens = self._token_embeddings(
                    tile_emb,
                    seat_emb,
                    tile_remaining,
                    seat_remaining,
                    state_ids,
                    step_ids,
                )
            decoder_memory, decoder_memory_padding_mask = self._decoder_memory(
                context,
                memory,
                memory_padding_mask,
            )
            with _profile_range("belief/denoise_decoder"):
                logits = self.denoise_decoder(tokens, decoder_memory, decoder_memory_padding_mask)
        return logits.reshape(context.shape[0], TILE37_COUNT, _OPPONENT_COUNT, _COUNT_CANDIDATES)

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

        for subset_mask in range(1, 1 << _OPPONENT_COUNT):
            subset = torch.tensor(
                [(subset_mask >> seat) & 1 for seat in range(_OPPONENT_COUNT)],
                dtype=torch.bool,
                device=tile_remaining.device,
            )
            base_demand = (seat_remaining * subset.long()).sum(dim=1)
            subset_edges = is_masked & subset.view(1, 1, _OPPONENT_COUNT)
            any_before = subset_edges.any(dim=2)
            edge_count = subset_edges.long().sum(dim=2)
            base_cap = (tile_remaining * any_before.long()).sum(dim=1)

            for seat in range(_OPPONENT_COUNT):
                if subset[seat]:
                    demand_after = base_demand.view(-1, 1, 1) - candidates
                    any_after = edge_count > 1
                else:
                    demand_after = base_demand.view(-1, 1, 1)
                    any_after = any_before
                cap_after = (
                    base_cap.view(-1, 1, 1)
                    - tile_remaining.unsqueeze(-1) * any_before.long().unsqueeze(-1)
                    + (tile_remaining.unsqueeze(-1) - candidates) * any_after.long().unsqueeze(-1)
                )
                feasible[:, :, seat, :] &= demand_after <= cap_after

        return feasible

    def _apply_prior_and_legality(
        self,
        neural_logits: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _profile_range("belief/apply_prior_legality"):
            feasible = self._cell_hall_feasible(tile_remaining, seat_remaining, is_masked)
            logits = self._hypergeom_prior(tile_remaining, seat_remaining, is_masked) + neural_logits
            return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min), feasible

    def _apply_prior_and_legality_for_cells(
        self,
        neural_logits: torch.Tensor,
        tile_remaining: torch.Tensor,
        seat_remaining: torch.Tensor,
        is_masked: torch.Tensor,
        tile37: torch.Tensor,
        opponent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        for subset_mask in range(1, 1 << _OPPONENT_COUNT):
            subset = torch.tensor(
                [(subset_mask >> seat) & 1 for seat in range(_OPPONENT_COUNT)],
                dtype=torch.bool,
                device=tile_remaining.device,
            )
            base_demand = (seat_remaining * subset.long()).sum(dim=1)
            subset_edges = is_masked & subset.view(1, 1, _OPPONENT_COUNT)
            any_before = subset_edges.any(dim=2)
            edge_count = subset_edges.long().sum(dim=2)
            base_cap = (tile_remaining * any_before.long()).sum(dim=1)
            any_before_cell = any_before[row_ids, tile37]
            edge_count_cell = edge_count[row_ids, tile37]
            opponent_in_subset = subset[opponent]
            any_after_cell = torch.where(opponent_in_subset, edge_count_cell > 1, any_before_cell)
            demand_after = base_demand.unsqueeze(1) - torch.where(
                opponent_in_subset.unsqueeze(1),
                candidates,
                torch.zeros_like(candidates),
            )
            cap_after = (
                base_cap.unsqueeze(1)
                - u_t * any_before_cell.long().unsqueeze(1)
                + (u_t - candidates) * any_after_cell.long().unsqueeze(1)
            )
            feasible &= demand_after <= cap_after

        logits = prior + neural_logits
        return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min), feasible

    @staticmethod
    def _sample_random_mask(
        batch_size: int,
        device: torch.device,
        decode_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        step_ids = torch.floor(t * float(decode_steps)).long().clamp(max=decode_steps - 1)
        return mask.reshape(batch_size, TILE37_COUNT, _OPPONENT_COUNT), step_ids

    @staticmethod
    def _cosine_target_unmasked(step: int, decode_steps: int) -> int:
        ratio_masked = math.cos(math.pi * float(step + 1) / (2.0 * float(decode_steps)))
        target = int(round(_ALLOCATION_TOKEN_COUNT * (1.0 - ratio_masked)))
        if step < decode_steps - 1 and target == 0:
            target = 1
        return max(0, min(_ALLOCATION_TOKEN_COUNT, target))

    @staticmethod
    def _confidence(probs: torch.Tensor, method: str) -> torch.Tensor:
        if method == "neg_entropy":
            safe_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            return (safe_probs * safe_probs.log()).sum(dim=-1)
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
        context, memory, memory_padding_mask, tile37_table = self.encoder.forward_context_and_memory(
            features,
            return_tile37_table=True,
        )
        unseen_counts = self._unseen_counts(features)
        tile_emb = self._shared_alloc_tile_embeddings(tile37_table)
        seat_emb = self._shared_alloc_seat_embeddings(context.device)
        rem = self._initial_rem(features, unseen_counts)
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
            )
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
        )
        return {"allocation": allocation}

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
    ) -> dict[str, torch.Tensor]:
        target_counts = target_counts.long()
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
        mask, step_ids = self._sample_random_mask(context.shape[0], context.device, self.decode_steps)
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
            step_ids,
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
        if valid.any():
            loss = F.cross_entropy(logits[valid], target_cells[valid], reduction="mean")
            pred = logits[valid].argmax(dim=1)
            acc = (pred == target_cells[valid]).float().mean()
            out.update({"loss": loss, "acc": acc})
        else:
            zero = context.sum() * 0.0
            out.update({"loss": zero, "acc": zero.detach()})
        return out

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
    ) -> torch.Tensor:
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        if decode_steps > self.max_decode_steps:
            raise ValueError("decode_steps cannot exceed max_decode_steps")

        batch_size = context.shape[0]
        device = context.device
        state_ids = torch.full(
            (batch_size, TILE37_COUNT, _OPPONENT_COUNT),
            self.mask_state_id,
            dtype=torch.long,
            device=device,
        )
        is_masked = torch.ones(batch_size, TILE37_COUNT, _OPPONENT_COUNT, dtype=torch.bool, device=device)
        allocation = torch.zeros(batch_size, BUCKET_COUNT, TILE37_COUNT, dtype=torch.long, device=device)
        tile_remaining = unseen_counts.clone()
        seat_remaining = rem[:, :_OPPONENT_COUNT].clone()
        temp = max(float(temperature), 1e-6)
        batch_indices = torch.arange(batch_size, device=device)

        with _profile_range("belief/iterative_decode"):
            for step in range(decode_steps):
                with _profile_range("belief/decode_step"):
                    step_ids = torch.full((batch_size,), step, dtype=torch.long, device=device)
                    neural_logits = self._denoise_neural_logits(
                        context,
                        memory,
                        memory_padding_mask,
                        tile_emb,
                        seat_emb,
                        tile_remaining,
                        seat_remaining,
                        state_ids,
                        step_ids,
                    )
                    logits, _feasible = self._apply_prior_and_legality(
                        neural_logits,
                        tile_remaining,
                        seat_remaining,
                        is_masked,
                    )
                    with _profile_range("belief/confidence_order"):
                        probs = F.softmax(logits / temp, dim=-1)
                        confidence = self._confidence(probs, self.confidence_method)
                        confidence = confidence.masked_fill(~is_masked, torch.finfo(confidence.dtype).min)

                        remaining = is_masked.flatten(1).sum(dim=1)
                        if step == decode_steps - 1:
                            n_to_unmask = remaining
                        else:
                            target_unmasked = self._cosine_target_unmasked(step, decode_steps)
                            current_unmasked = _ALLOCATION_TOKEN_COUNT - remaining
                            n_to_unmask = (target_unmasked - current_unmasked).clamp(min=0)
                            n_to_unmask = torch.minimum(n_to_unmask, remaining)

                        max_to_unmask = int(n_to_unmask.max().item())
                        if max_to_unmask == 0:
                            continue
                        token_order = confidence.reshape(batch_size, _ALLOCATION_TOKEN_COUNT).argsort(
                            dim=1,
                            descending=True,
                        )
                    with _profile_range("belief/cell_sampling_loop"):
                        for rank in range(max_to_unmask):
                            active = rank < n_to_unmask
                            if not active.any():
                                continue
                            rows = batch_indices[active]
                            token = token_order[rows, rank]
                            tile37 = torch.div(token, _OPPONENT_COUNT, rounding_mode="floor")
                            opponent = token % _OPPONENT_COUNT
                            cell_neural_logits = neural_logits[rows, tile37, opponent]
                            cell_logits, _cell_feasible = self._apply_prior_and_legality_for_cells(
                                cell_neural_logits,
                                tile_remaining[rows],
                                seat_remaining[rows],
                                is_masked[rows],
                                tile37,
                                opponent,
                            )
                            if sample:
                                cell_probs = F.softmax(cell_logits / temp, dim=1)
                                chosen = torch.multinomial(cell_probs, 1).squeeze(1)
                            else:
                                chosen = cell_logits.argmax(dim=1)

                            allocation[rows, opponent, tile37] = chosen
                            state_ids[rows, tile37, opponent] = chosen
                            tile_remaining[rows, tile37] = tile_remaining[rows, tile37] - chosen
                            seat_remaining[rows, opponent] = seat_remaining[rows, opponent] - chosen
                            is_masked[rows, tile37, opponent] = False

        allocation[:, BUCKET_COUNT - 1] = unseen_counts - allocation[:, :_OPPONENT_COUNT].sum(dim=1)
        return allocation

    @torch.inference_mode()
    def prepare_allocation_sampling_context(
        self,
        features: dict[str, torch.Tensor],
    ) -> AllocationSamplingContext:
        with _profile_range("belief/prepare_allocation_sampling_context"):
            context, memory, memory_padding_mask, tile37_table = self.encoder.forward_context_and_memory(
                features,
                return_tile37_table=True,
            )
            unseen_counts = self._unseen_counts(features)
            tile_emb = self._shared_alloc_tile_embeddings(tile37_table)
            seat_emb = self._shared_alloc_seat_embeddings(context.device)
            rem = self._initial_rem(features, unseen_counts)
        return AllocationSamplingContext(
            context=context,
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            unseen_counts=unseen_counts,
            tile_emb=tile_emb,
            seat_emb=seat_emb,
            rem=rem,
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
