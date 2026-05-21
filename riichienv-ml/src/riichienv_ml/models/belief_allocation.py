"""Joint hidden-allocation belief sampler."""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

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

_MAX_CANDIDATES = 35
_TILE34_COUNT = 34
_CONTEXT_BUCKET_COUNT = BUCKET_COUNT - 1


def _build_candidate_tuples() -> tuple[torch.Tensor, torch.Tensor]:
    tuples = torch.zeros(5, _MAX_CANDIDATES, BUCKET_COUNT, dtype=torch.long)
    mask = torch.zeros(5, _MAX_CANDIDATES, dtype=torch.bool)
    for u in range(5):
        rows = [row for row in itertools.product(range(u + 1), repeat=BUCKET_COUNT) if sum(row) == u]
        for idx, row in enumerate(rows):
            tuples[u, idx] = torch.tensor(row, dtype=torch.long)
            mask[u, idx] = True
    return tuples, mask


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
    """Small bidirectional denoising transformer over the 37 allocation tokens."""

    def __init__(
        self,
        d_dec: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        *,
        d_memory: int,
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
        self.head = nn.Linear(d_dec, _MAX_CANDIDATES)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        decode_steps: int = 8,
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
        self.mask_state_id = _MAX_CANDIDATES

        cross_attn_heads = int(encoder_kwargs.get("nhead", 8))
        dropout = float(encoder_kwargs.get("dropout", 0.1))
        self.alloc_tile_embed = nn.Embedding(TILE37_COUNT, d_model)
        self.alloc_u_embed = nn.Embedding(5, d_model)
        self.alloc_bucket_embed = nn.Embedding(_CONTEXT_BUCKET_COUNT, d_model)
        self.alloc_cross_attn = nn.MultiheadAttention(
            d_model,
            cross_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.alloc_cross_attn_norm = nn.LayerNorm(d_model)
        self.alloc_state_embed = nn.Embedding(_MAX_CANDIDATES + 1, self.d_dec)
        self.alloc_mask_embed = nn.Embedding(2, self.d_dec)
        self.alloc_time_embed = nn.Embedding(self.max_decode_steps, self.d_dec)
        d_ff = int(denoise_dim_feedforward or max(self.d_dec * 2, 512))
        self.denoise_decoder = DenoiseTransformer(
            self.d_dec,
            cross_attn_heads,
            int(denoise_num_layers),
            d_ff,
            dropout,
            d_memory=d_model,
        )

        candidate_tuples, candidate_mask = _build_candidate_tuples()
        self.register_buffer("candidate_tuples", candidate_tuples, persistent=False)
        self.register_buffer("candidate_mask", candidate_mask, persistent=False)
        self.register_buffer(
            "total_counts37",
            torch.tensor(TOTAL_TILE_COUNTS37, dtype=torch.long),
            persistent=False,
        )
        self.tile37_to_tile34 = _TILE37_TO_TILE34

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
    def _base_logits(rem: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        rem_f = rem.float()
        for _ in range(candidates.dim() - rem.dim()):
            rem_f = rem_f.unsqueeze(1)
        cand_f = candidates.float()
        remaining = (rem_f - cand_f).clamp_min(0)
        return (torch.lgamma(rem_f + 1) - torch.lgamma(cand_f + 1) - torch.lgamma(remaining + 1)).sum(dim=-1)

    def _tile_bucket_context(
        self,
        unseen_counts: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = unseen_counts.shape[0]
        tile_ids = torch.arange(TILE37_COUNT, dtype=torch.long, device=unseen_counts.device)
        bucket_ids = torch.arange(_CONTEXT_BUCKET_COUNT, dtype=torch.long, device=unseen_counts.device)
        tile_query = self.alloc_tile_embed(tile_ids).unsqueeze(0) + self.alloc_u_embed(unseen_counts.long())
        bucket_query = self.alloc_bucket_embed(bucket_ids).view(1, 1, _CONTEXT_BUCKET_COUNT, self.d_model)
        query = (tile_query.unsqueeze(2) + bucket_query).reshape(
            batch_size,
            TILE37_COUNT * _CONTEXT_BUCKET_COUNT,
            self.d_model,
        )
        attended, _weights = self.alloc_cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        return self.alloc_cross_attn_norm(query + attended).reshape(
            batch_size,
            TILE37_COUNT,
            _CONTEXT_BUCKET_COUNT,
            self.d_model,
        )

    def _decoder_memory(
        self,
        context: torch.Tensor,
        tile_bucket_context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        bucket_memory = tile_bucket_context.reshape(
            batch_size,
            TILE37_COUNT * _CONTEXT_BUCKET_COUNT,
            self.d_model,
        )
        decoder_memory = torch.cat([context.unsqueeze(1), bucket_memory, memory], dim=1)
        prefix_mask = torch.zeros(
            batch_size,
            1 + TILE37_COUNT * _CONTEXT_BUCKET_COUNT,
            dtype=torch.bool,
            device=context.device,
        )
        decoder_memory_padding_mask = torch.cat([prefix_mask, memory_padding_mask], dim=1)
        return decoder_memory, decoder_memory_padding_mask

    def _token_embeddings(
        self,
        unseen_counts: torch.Tensor,
        state_ids: torch.Tensor,
        is_masked: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = unseen_counts.shape[0]
        tile_ids = torch.arange(TILE37_COUNT, dtype=torch.long, device=unseen_counts.device)
        tile_emb = self.alloc_tile_embed(tile_ids).unsqueeze(0).expand(batch_size, -1, -1)
        unseen_emb = self.alloc_u_embed(unseen_counts.long())
        step_ids = step_ids.clamp(min=0, max=self.max_decode_steps - 1)
        return (
            tile_emb
            + unseen_emb
            + self.alloc_state_embed(state_ids.clamp(min=0, max=self.mask_state_id))
            + self.alloc_mask_embed(is_masked.long())
            + self.alloc_time_embed(step_ids).unsqueeze(1)
        )

    def _denoise_neural_logits(
        self,
        context: torch.Tensor,
        tile_bucket_context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        unseen_counts: torch.Tensor,
        state_ids: torch.Tensor,
        is_masked: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._token_embeddings(unseen_counts, state_ids, is_masked, step_ids)
        decoder_memory, decoder_memory_padding_mask = self._decoder_memory(
            context,
            tile_bucket_context,
            memory,
            memory_padding_mask,
        )
        return self.denoise_decoder(tokens, decoder_memory, decoder_memory_padding_mask)

    def _apply_prior_and_legality(
        self,
        neural_logits: torch.Tensor,
        rem: torch.Tensor,
        candidates: torch.Tensor,
        cand_mask: torch.Tensor,
        future_unseen_total: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rem_expanded = rem
        for _ in range(candidates.dim() - rem.dim()):
            rem_expanded = rem_expanded.unsqueeze(1)
        future = future_unseen_total
        while future.dim() < candidates.dim():
            future = future.unsqueeze(-1)
        feasible = (
            cand_mask
            & (candidates <= rem_expanded).all(dim=-1)
            & ((rem_expanded - candidates) <= future).all(dim=-1)
        )
        logits = self._base_logits(rem, candidates) + neural_logits
        return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min), feasible

    def _target_indices(
        self,
        target_counts: torch.Tensor,
        unseen_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_tuples[unseen_counts.long()]
        cand_mask = self.candidate_mask[unseen_counts.long()]
        target_by_tile = target_counts.permute(0, 2, 1).long()
        matches = (candidates == target_by_tile.unsqueeze(2)).all(dim=-1) & cand_mask
        return matches.float().argmax(dim=-1).long(), matches.any(dim=-1)

    @staticmethod
    def _sample_random_mask(
        batch_size: int,
        device: torch.device,
        decode_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(batch_size, device=device)
        mask_ratio = torch.cos(t * math.pi / 2.0)
        n_masked = torch.round(TILE37_COUNT * mask_ratio).long().clamp(min=1, max=TILE37_COUNT)
        order = torch.rand(batch_size, TILE37_COUNT, device=device).argsort(dim=1)
        ranks = torch.arange(TILE37_COUNT, device=device).unsqueeze(0)
        chosen = ranks < n_masked.unsqueeze(1)
        mask = torch.zeros(batch_size, TILE37_COUNT, dtype=torch.bool, device=device)
        mask.scatter_(1, order, chosen)
        step_ids = torch.floor(t * float(decode_steps)).long().clamp(max=decode_steps - 1)
        return mask, step_ids

    @staticmethod
    def _cosine_target_unmasked(step: int, decode_steps: int) -> int:
        ratio_masked = math.cos(math.pi * float(step + 1) / (2.0 * float(decode_steps)))
        target = int(round(TILE37_COUNT * (1.0 - ratio_masked)))
        if step < decode_steps - 1 and target == 0:
            target = 1
        return max(0, min(TILE37_COUNT, target))

    @staticmethod
    def _future_unseen_total(unseen_counts: torch.Tensor, is_masked: torch.Tensor) -> torch.Tensor:
        masked_total = (unseen_counts.long() * is_masked.long()).sum(dim=1, keepdim=True)
        return (masked_total - unseen_counts.long()).clamp_min(0)

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
        context, memory, memory_padding_mask = self.encoder.forward_context_and_memory(features)
        unseen_counts = self._unseen_counts(features)
        tile_bucket_context = self._tile_bucket_context(unseen_counts, memory, memory_padding_mask)
        rem = self._initial_rem(features, unseen_counts)
        if target_counts is not None:
            return self._training_forward(
                context,
                tile_bucket_context,
                memory,
                memory_padding_mask,
                unseen_counts,
                rem,
                target_counts,
            )
        allocation = self._iterative_decode(
            context,
            tile_bucket_context,
            memory,
            memory_padding_mask,
            unseen_counts,
            rem,
            sample=sample,
            temperature=temperature,
            decode_steps=self.decode_steps,
        )
        return {"allocation": allocation}

    def _training_forward(
        self,
        context: torch.Tensor,
        tile_bucket_context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        unseen_counts: torch.Tensor,
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

        target_idx, target_match = self._target_indices(target_counts, unseen_counts)
        mask, step_ids = self._sample_random_mask(context.shape[0], context.device, self.decode_steps)
        state_ids = target_idx.masked_fill(mask, self.mask_state_id)
        target_by_tile = target_counts.permute(0, 2, 1)
        known_counts = target_by_tile * (~mask).unsqueeze(-1).long()
        rem_for_masked = rem - known_counts.sum(dim=1)
        future_unseen_total = self._future_unseen_total(unseen_counts, mask)

        neural_logits = self._denoise_neural_logits(
            context,
            tile_bucket_context,
            memory,
            memory_padding_mask,
            unseen_counts,
            state_ids,
            mask,
            step_ids,
        )
        candidates = self.candidate_tuples[unseen_counts.long()]
        cand_mask = self.candidate_mask[unseen_counts.long()]
        logits, feasible = self._apply_prior_and_legality(
            neural_logits,
            rem_for_masked,
            candidates,
            cand_mask,
            future_unseen_total,
        )

        target_feasible = feasible.gather(2, target_idx.unsqueeze(2)).squeeze(2) & target_match
        valid = mask & target_sample_valid.unsqueeze(1) & target_feasible

        out = {
            "allocation": target_counts,
            "invalid_target_rate": (~target_sample_valid).float().mean(),
        }
        if valid.any():
            loss = F.cross_entropy(logits[valid], target_idx[valid], reduction="mean")
            pred = logits[valid].argmax(dim=1)
            acc = (pred == target_idx[valid]).float().mean()
            out.update({"loss": loss, "acc": acc})
        else:
            zero = context.sum() * 0.0
            out.update({"loss": zero, "acc": zero.detach()})
        return out

    def _iterative_decode(  # noqa: PLR0915
        self,
        context: torch.Tensor,
        tile_bucket_context: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        unseen_counts: torch.Tensor,
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
            (batch_size, TILE37_COUNT),
            self.mask_state_id,
            dtype=torch.long,
            device=device,
        )
        is_masked = torch.ones(batch_size, TILE37_COUNT, dtype=torch.bool, device=device)
        allocation = torch.zeros(batch_size, BUCKET_COUNT, TILE37_COUNT, dtype=torch.long, device=device)
        temp = max(float(temperature), 1e-6)
        all_candidates = self.candidate_tuples[unseen_counts.long()]
        all_candidate_mask = self.candidate_mask[unseen_counts.long()]
        batch_indices = torch.arange(batch_size, device=device)

        for step in range(decode_steps):
            step_ids = torch.full((batch_size,), step, dtype=torch.long, device=device)
            neural_logits = self._denoise_neural_logits(
                context,
                tile_bucket_context,
                memory,
                memory_padding_mask,
                unseen_counts,
                state_ids,
                is_masked,
                step_ids,
            )
            future_unseen_total = self._future_unseen_total(unseen_counts, is_masked)
            logits, _feasible = self._apply_prior_and_legality(
                neural_logits,
                rem,
                all_candidates,
                all_candidate_mask,
                future_unseen_total,
            )
            probs = F.softmax(logits / temp, dim=-1)
            confidence = self._confidence(probs, self.confidence_method)
            confidence = confidence.masked_fill(~is_masked, torch.finfo(confidence.dtype).min)

            remaining = is_masked.sum(dim=1)
            if step == decode_steps - 1:
                n_to_unmask = remaining
            else:
                target_unmasked = self._cosine_target_unmasked(step, decode_steps)
                current_unmasked = TILE37_COUNT - remaining
                n_to_unmask = (target_unmasked - current_unmasked).clamp(min=0)
                n_to_unmask = torch.minimum(n_to_unmask, remaining)

            max_to_unmask = int(n_to_unmask.max().item())
            if max_to_unmask == 0:
                continue
            tile_order = confidence.argsort(dim=1, descending=True)
            for rank in range(max_to_unmask):
                active = rank < n_to_unmask
                if not active.any():
                    continue
                rows = batch_indices[active]
                tile37 = tile_order[rows, rank]
                unseen_for_tile = unseen_counts[rows, tile37].long()
                candidates = self.candidate_tuples[unseen_for_tile]
                cand_mask = self.candidate_mask[unseen_for_tile]
                masked_total = (unseen_counts[rows].long() * is_masked[rows].long()).sum(dim=1)
                future_total = (masked_total - unseen_for_tile).clamp_min(0)
                tile_neural_logits = neural_logits[rows, tile37]
                tile_logits, _tile_feasible = self._apply_prior_and_legality(
                    tile_neural_logits,
                    rem[rows],
                    candidates,
                    cand_mask,
                    future_total,
                )
                if sample:
                    tile_probs = F.softmax(tile_logits / temp, dim=1)
                    chosen_idx = torch.multinomial(tile_probs, 1).squeeze(1)
                else:
                    chosen_idx = tile_logits.argmax(dim=1)
                chosen = candidates[torch.arange(rows.shape[0], device=device), chosen_idx]

                allocation[rows, :, tile37] = chosen
                state_ids[rows, tile37] = chosen_idx
                rem[rows] = rem[rows] - chosen
                is_masked[rows, tile37] = False

        return allocation

    @torch.inference_mode()
    def sample_allocations(
        self,
        features: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        decode_steps: int | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        context, memory, memory_padding_mask = self.encoder.forward_context_and_memory(features)
        unseen_counts = self._unseen_counts(features)
        tile_bucket_context = self._tile_bucket_context(unseen_counts, memory, memory_padding_mask)
        rem = self._initial_rem(features, unseen_counts)
        context = context.repeat_interleave(num_samples, dim=0)
        tile_bucket_context = tile_bucket_context.repeat_interleave(num_samples, dim=0)
        memory = memory.repeat_interleave(num_samples, dim=0)
        memory_padding_mask = memory_padding_mask.repeat_interleave(num_samples, dim=0)
        unseen_counts = unseen_counts.repeat_interleave(num_samples, dim=0)
        rem = rem.repeat_interleave(num_samples, dim=0)
        sampled = self._iterative_decode(
            context,
            tile_bucket_context,
            memory,
            memory_padding_mask,
            unseen_counts,
            rem,
            sample=True,
            temperature=temperature,
            decode_steps=int(decode_steps or self.decode_steps),
        )
        batch_size = features["visible_tile_counts"].shape[0]
        return sampled.reshape(batch_size, num_samples, BUCKET_COUNT, TILE37_COUNT)
