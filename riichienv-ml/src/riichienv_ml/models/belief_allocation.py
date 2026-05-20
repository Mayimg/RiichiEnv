"""Joint hidden-allocation belief sampler."""

from __future__ import annotations

import itertools
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
    """Autoregressive legal sampler over opponent hands and residual wall counts."""

    def __init__(
        self,
        d_model: int = 384,
        decoder_hidden_dim: int | None = None,
        tile_order: list[int] | None = None,
        **encoder_kwargs: Any,
    ):
        super().__init__()
        encoder_kwargs["d_model"] = d_model
        self.encoder = BeliefObservationEncoder(**encoder_kwargs)
        self.d_model = d_model
        self.tile_order = tuple(tile_order or range(TILE37_COUNT))
        if sorted(self.tile_order) != list(range(TILE37_COUNT)):
            raise ValueError("tile_order must be a permutation of 0..36")

        hidden = int(decoder_hidden_dim or d_model)
        cross_attn_heads = int(encoder_kwargs.get("nhead", 8))
        dropout = float(encoder_kwargs.get("dropout", 0.1))
        self.alloc_tile_embed = nn.Embedding(TILE37_COUNT, d_model)
        self.alloc_u_embed = nn.Embedding(5, d_model)
        self.alloc_bucket_embed = nn.Embedding(BUCKET_COUNT, d_model)
        self.alloc_cross_attn = nn.MultiheadAttention(
            d_model,
            cross_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.alloc_cross_attn_norm = nn.LayerNorm(d_model)
        partial_count_dim = BUCKET_COUNT * (TILE37_COUNT + _TILE34_COUNT)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * (2 + BUCKET_COUNT) + partial_count_dim + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, _MAX_CANDIDATES),
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
        rem_f = rem.float().unsqueeze(1)
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
        bucket_ids = torch.arange(BUCKET_COUNT, dtype=torch.long, device=unseen_counts.device)
        tile_query = self.alloc_tile_embed(tile_ids).unsqueeze(0) + self.alloc_u_embed(unseen_counts.long())
        bucket_query = self.alloc_bucket_embed(bucket_ids).view(1, 1, BUCKET_COUNT, self.d_model)
        query = (tile_query.unsqueeze(2) + bucket_query).reshape(
            batch_size,
            TILE37_COUNT * BUCKET_COUNT,
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
            BUCKET_COUNT,
            self.d_model,
        )

    def _step_logits(
        self,
        context: torch.Tensor,
        tile_bucket_context: torch.Tensor,
        partial37: torch.Tensor,
        partial34: torch.Tensor,
        rem: torch.Tensor,
        tile37: int,
        unseen_for_tile: torch.Tensor,
        future_unseen_total: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self.candidate_tuples[unseen_for_tile]
        cand_mask = self.candidate_mask[unseen_for_tile]
        rem_expanded = rem.unsqueeze(1)
        feasible = (
            cand_mask
            & (candidates <= rem_expanded).all(dim=-1)
            & ((rem_expanded - candidates) <= future_unseen_total.view(-1, 1, 1)).all(dim=-1)
        )

        tile_ids = torch.full((context.shape[0],), int(tile37), dtype=torch.long, device=context.device)
        tile_emb = self.alloc_tile_embed(tile_ids) + self.alloc_u_embed(unseen_for_tile)
        rem_features = torch.cat([rem.float() / 70.0, unseen_for_tile.float().unsqueeze(1) / 4.0], dim=1)
        partial_features = torch.cat(
            [
                partial37.reshape(context.shape[0], -1),
                partial34.reshape(context.shape[0], -1),
            ],
            dim=1,
        ) / 4.0
        decoder_input = torch.cat(
            [
                context,
                tile_emb,
                tile_bucket_context.reshape(context.shape[0], -1),
                partial_features,
                rem_features,
            ],
            dim=1,
        )
        logits = self._base_logits(rem, candidates) + self.decoder(decoder_input)
        logits = logits.masked_fill(~feasible, torch.finfo(logits.dtype).min)
        return logits, feasible, candidates

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
        return self._decode_allocations(
            context,
            tile_bucket_context,
            unseen_counts,
            rem,
            target_counts=target_counts,
            sample=sample,
            temperature=temperature,
        )

    def _decode_allocations(  # noqa: PLR0915
        self,
        context: torch.Tensor,
        tile_bucket_context: torch.Tensor,
        unseen_counts: torch.Tensor,
        rem: torch.Tensor,
        target_counts: torch.Tensor | None = None,
        *,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        target_sample_valid = None
        target_teacher_path_valid = None
        invalid_target_count = None
        if target_counts is not None:
            target_counts = target_counts.long()
            target_sample_valid, target_teacher_path_valid = self._target_validity(
                target_counts,
                rem,
                unseen_counts,
            )
            invalid_target_count = (~target_sample_valid).sum()
            target_rem = target_counts.sum(dim=2)
            use_target_rem = (~target_sample_valid & target_teacher_path_valid).unsqueeze(1)
            rem = torch.where(use_target_rem, target_rem, rem)

        partial37 = torch.zeros(
            context.shape[0],
            BUCKET_COUNT,
            TILE37_COUNT,
            device=context.device,
            dtype=context.dtype,
        )
        partial34 = torch.zeros(
            context.shape[0],
            BUCKET_COUNT,
            _TILE34_COUNT,
            device=context.device,
            dtype=context.dtype,
        )
        allocation = torch.zeros(context.shape[0], BUCKET_COUNT, TILE37_COUNT, device=context.device, dtype=torch.long)

        losses = []
        correct = []
        total_targets = 0
        for order_idx, tile37 in enumerate(self.tile_order):
            unseen_for_tile = unseen_counts[:, tile37].long()
            future_tiles = self.tile_order[order_idx + 1 :]
            if future_tiles:
                future_total = unseen_counts[:, list(future_tiles)].sum(dim=1)
            else:
                future_total = torch.zeros_like(unseen_for_tile)
            logits, feasible, candidates = self._step_logits(
                context,
                tile_bucket_context[:, tile37],
                partial37,
                partial34,
                rem,
                tile37,
                unseen_for_tile,
                future_total,
            )

            if target_counts is not None:
                target_x = target_counts[:, :, tile37].long()
                matches = (candidates == target_x.unsqueeze(1)).all(dim=-1)
                target_idx = matches.float().argmax(dim=1).long()
                target_feasible = feasible.gather(1, target_idx.unsqueeze(1)).squeeze(1) & matches.any(dim=1)
                valid = target_sample_valid & target_feasible
                if valid.any():
                    losses.append(F.cross_entropy(logits[valid], target_idx[valid], reduction="sum"))
                    pred = logits[valid].argmax(dim=1)
                    correct.append((pred == target_idx[valid]).sum())
                    total_targets += int(valid.sum().item())
                fallback_idx = feasible.float().argmax(dim=1)
                fallback = candidates[torch.arange(context.shape[0], device=context.device), fallback_idx]
                chosen = torch.where(target_teacher_path_valid.unsqueeze(1), target_x, fallback)
            elif sample:
                probs = F.softmax(logits / max(float(temperature), 1e-6), dim=1)
                idx = torch.multinomial(probs, 1).squeeze(1)
                chosen = candidates[torch.arange(context.shape[0], device=context.device), idx]
            else:
                idx = logits.argmax(dim=1)
                chosen = candidates[torch.arange(context.shape[0], device=context.device), idx]

            rem = rem - chosen
            chosen_f = chosen.to(context.dtype)
            partial37[:, :, tile37] = chosen_f
            tile34 = self.tile37_to_tile34[int(tile37)]
            partial34[:, :, tile34] = partial34[:, :, tile34] + chosen_f
            allocation[:, :, tile37] = chosen

        out = {"allocation": allocation}
        if target_counts is not None:
            invalid_rate = invalid_target_count.float() / max(int(target_counts.shape[0]), 1)
            out["invalid_target_rate"] = invalid_rate
        if losses:
            loss = torch.stack(losses).sum() / max(total_targets, 1)
            acc = torch.stack(correct).sum().float() / max(total_targets, 1)
            out.update({"loss": loss, "acc": acc})
        elif target_counts is not None:
            zero = context.sum() * 0.0
            out.update({"loss": zero, "acc": zero.detach()})
        return out

    @torch.inference_mode()
    def sample_allocations(
        self,
        features: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        context, memory, memory_padding_mask = self.encoder.forward_context_and_memory(features)
        unseen_counts = self._unseen_counts(features)
        tile_bucket_context = self._tile_bucket_context(unseen_counts, memory, memory_padding_mask)
        rem = self._initial_rem(features, unseen_counts)
        context = context.repeat_interleave(num_samples, dim=0)
        tile_bucket_context = tile_bucket_context.repeat_interleave(num_samples, dim=0)
        unseen_counts = unseen_counts.repeat_interleave(num_samples, dim=0)
        rem = rem.repeat_interleave(num_samples, dim=0)
        sampled = self._decode_allocations(
            context,
            tile_bucket_context,
            unseen_counts,
            rem,
            sample=True,
            temperature=temperature,
        )["allocation"]
        batch_size = features["visible_tile_counts"].shape[0]
        return sampled.reshape(batch_size, num_samples, BUCKET_COUNT, TILE37_COUNT)
