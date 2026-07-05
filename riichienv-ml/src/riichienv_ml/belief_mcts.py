"""Belief-sampled root PUCT agent for self-match generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger

from riichienv import Phase, RiichiEnv, calculate_shanten
from riichienv_ml.agents import (
    Agent,
    PolicyDecision,
    _action_to_policy_payload,
    _action_type_id,
    _policy_decision_from_logits,
)
from riichienv_ml.config import BeliefMctsConfig, import_class, load_config
from riichienv_ml.features.belief_features import (
    BUCKET_COUNT,
    BUCKET_REL_SEATS,
    TILE37_COUNT,
    TOTAL_TILE_COUNTS37,
    allocation_counts_to_tile37_lists,
    collate_belief_features,
    counts37_from_tiles,
    tile_id_to_tile37,
)
from riichienv_ml.features.sequence_features import collate_sequence_features
from riichienv_ml.utils import build_encoder, configure_matmul_precision, load_model_weights

_DISCARD_ACTION_TYPE_ID = 0
_RIICHI_ACTION_TYPE_ID = 5
_PASS_ACTION_TYPE_ID = 7
_RANK_COUNT = 4


def _tile37_to_tile_ids() -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(TILE37_COUNT)]
    for tile_id in range(136):
        out[tile_id_to_tile37(tile_id)].append(tile_id)
    return out


_TILE37_TO_TILE_IDS = _tile37_to_tile_ids()


@dataclass(frozen=True)
class WorldSample:
    hands: list[list[int]]
    wall: list[int]
    dora_indicators: list[int]
    drawable_count: int


@dataclass
class RolloutState:
    env: RiichiEnv
    root_pid: int
    root_action_index: int
    root_discard_decisions: int
    start_round_key: tuple[int, int, int]
    done: bool = False
    reward: float | None = None
    rank_prediction: list[float] | None = None
    terminal_reason: str | None = None


class BeliefMCTSAgent:
    """Self-match agent that uses belief allocation samples and root-only PUCT."""

    requires_sequence_cache = True

    def __init__(
        self,
        *,
        config_path: str,
        model_path: str,
        device: str,
        search_config: BeliefMctsConfig,
    ):
        self.cfg = search_config
        self.bc = Agent(config_path=config_path, model_path=model_path, device=device)
        self.device = self.bc.device
        self.policy_head_is_pointer = getattr(self.bc.model, "policy_head_type", None) == "pointer"
        if not getattr(self.bc.model, "emit_rank", False):
            raise ValueError("belief_mcts requires a BC model with emit_rank=true")

        self.rank_weights = np.asarray(self.cfg.rank_point_weights, dtype=np.float64)
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)

        self.belief_device = torch.device(self.cfg.belief_device or str(self.device))
        self.belief_model, self.belief_encoder = self._load_belief_sampler()

    def reset(self) -> None:
        self.bc.reset()

    def act(self, obs):
        forced_action = self._single_policy_action(obs)
        if forced_action is not None:
            return forced_action
        return self.bc.act(obs)

    def act_from_env(self, env: RiichiEnv, pid: int, obs):
        forced_action = self._single_policy_action(obs)
        if forced_action is not None:
            return forced_action
        return self.act_with_policy_from_env(env, pid, obs).action

    @torch.inference_mode()
    def act_with_policy_from_env(self, env: RiichiEnv, pid: int, obs) -> PolicyDecision:
        root_eval = self._evaluate_policy_rank([obs])
        logits = root_eval["logits"][0]
        root_rank = root_eval["rank_probs"][0]
        root_expected = self._expected_points(root_rank)

        fallback = _policy_decision_from_logits(
            obs,
            logits.unsqueeze(0).to(self.device),
            self.device,
            candidate_logits=self.policy_head_is_pointer,
        )
        self._attach_bc_rank_meta(fallback.meta, root_rank, root_expected)

        fallback_reason = self._search_fallback_reason(env, obs)
        if fallback_reason is not None:
            self._attach_search_fallback(fallback.meta, fallback_reason, 0, 0)
            return fallback

        root_actions, priors, action_keys = self._root_actions_and_priors(obs, logits)
        if not root_actions:
            self._attach_search_fallback(fallback.meta, "no_root_actions", 0, 0)
            return fallback
        if len(root_actions) == 1:
            meta = self._meta_for_selected_action(
                obs=obs,
                logits=logits,
                selected_action=root_actions[0],
                selected_index=0,
                action_keys=action_keys,
            )
            self._attach_bc_rank_meta(meta, root_rank, root_expected)
            self._attach_search_fallback(meta, "single_root_action", 0, 0)
            return PolicyDecision(action=root_actions[0], meta=meta)

        samples, sample_meta = self._sample_worlds(env, obs, pid)
        if not samples:
            self._attach_search_fallback(
                fallback.meta,
                "no_valid_belief_samples",
                sample_meta["requested"],
                sample_meta["accepted"],
            )
            return fallback

        visits, value_sums, rank_sums, search_meta = self._run_root_rollouts(
            env=env,
            root_pid=pid,
            root_actions=root_actions,
            priors=priors,
            samples=samples,
        )
        search_meta.update(sample_meta)

        if int(visits.sum()) <= 0:
            self._attach_search_fallback(
                fallback.meta,
                "no_successful_rollouts",
                sample_meta["requested"],
                sample_meta["accepted"],
            )
            return fallback

        selected_idx = int(np.flatnonzero(visits == visits.max())[0])
        selected_action = root_actions[selected_idx]
        meta = self._meta_for_selected_action(
            obs=obs,
            logits=logits,
            selected_action=selected_action,
            selected_index=selected_idx,
            action_keys=action_keys,
        )
        self._attach_bc_rank_meta(meta, root_rank, root_expected)
        self._attach_search_stats(meta, action_keys, visits, value_sums, rank_sums, search_meta)
        return PolicyDecision(action=selected_action, meta=meta)

    def _load_belief_sampler(self):
        cfg = load_config(self.cfg.belief_config_path).belief_sampler
        configure_matmul_precision(getattr(cfg, "matmul_precision", None))
        model_class = import_class(cfg.model_class)
        model_config = cfg.model.model_dump()
        model = model_class(**model_config).to(self.belief_device)
        load_model_weights(model, self.cfg.belief_model_path, map_location=self.belief_device, strict=False)
        model.eval()

        encoder_class = import_class(cfg.encoder_class)
        encoder = build_encoder(encoder_class, tile_dim=cfg.game.tile_dim, model_config=model_config)
        return model, encoder

    def _search_fallback_reason(self, env: RiichiEnv, obs) -> str | None:
        if int(getattr(env, "num_players", 4)) != 4:
            return "belief_mcts_supports_4p_only"
        if self._is_response_phase(env):
            if not self.cfg.search_response:
                return "response_search_disabled"
            if bool(getattr(env, "has_pending_kan", False)):
                return "chankan_response_not_supported"
        if len(obs.legal_actions()) == 0:
            return "no_legal_actions"
        return None

    def _evaluate_policy_rank(self, obs_list: list[Any]) -> dict[str, list[Any]]:
        features = [self.bc.encoder.encode(obs) for obs in obs_list]
        batch = self._collate_features(features, self.device)
        output = self.bc.model(batch)
        logits = output[0] if isinstance(output, tuple) else output
        rank_logits = output[-1] if isinstance(output, tuple) else None
        if rank_logits is None:
            raise ValueError("BC model did not return rank logits")
        rank_probs = torch.softmax(rank_logits, dim=-1).detach().cpu().tolist()
        return {
            "logits": [logits[i].detach().cpu() for i in range(logits.shape[0])],
            "rank_probs": [[float(x) for x in probs] for probs in rank_probs],
        }

    def _evaluate_rank_features(self, features: list[dict[str, torch.Tensor]]) -> list[list[float]]:
        batch = self._collate_features(features, self.device)
        output = self.bc.model(batch)
        rank_logits = output[-1] if isinstance(output, tuple) else None
        if rank_logits is None:
            raise ValueError("BC model did not return rank logits")
        probs = torch.softmax(rank_logits, dim=-1).detach().cpu().tolist()
        return [[float(x) for x in row] for row in probs]

    def _sample_actions(self, obs_list: list[Any]) -> list[Any]:
        if not obs_list:
            return []

        actions: list[Any | None] = [None] * len(obs_list)
        model_requests: list[tuple[int, Any]] = []
        for idx, obs in enumerate(obs_list):
            forced_action = self._single_policy_action(obs)
            if forced_action is None:
                model_requests.append((idx, obs))
            else:
                actions[idx] = forced_action

        if not model_requests:
            return self._resolved_actions(actions)

        features = [self.bc.encoder.encode(obs) for _idx, obs in model_requests]
        batch = self._collate_features(features, self.device)
        output = self.bc.model(batch)
        logits = output[0] if isinstance(output, tuple) else output
        for row, (idx, obs) in enumerate(model_requests):
            actions[idx] = self._sample_action_from_logits(obs, logits[row].detach().cpu())
        return self._resolved_actions(actions)

    def _sample_action_from_logits(self, obs, logits_1d: torch.Tensor):
        if self.policy_head_is_pointer:
            candidates = obs.candidate_actions()
            valid_count = min(len(candidates), int(logits_1d.shape[-1]))
            if valid_count <= 0:
                return self._fallback_legal_action(obs)
            probs = self._softmax_np(logits_1d[:valid_count].numpy(), self.cfg.policy_temperature)
            idx = int(self.np_rng.choice(valid_count, p=probs))
            action = obs.find_candidate_action(idx)
            return action if action is not None else self._fallback_legal_action(obs)

        fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        valid_indices = [idx for idx, value in enumerate(fixed_mask.tolist()) if value]
        if not valid_indices:
            return self._fallback_legal_action(obs)
        valid_logits = logits_1d.numpy()[valid_indices]
        probs = self._softmax_np(valid_logits, self.cfg.policy_temperature)
        action_idx = int(self.np_rng.choice(valid_indices, p=probs))
        action = obs.find_action(action_idx)
        return action if action is not None else self._fallback_legal_action(obs)

    @staticmethod
    def _fallback_legal_action(obs):
        legals = obs.legal_actions()
        if not legals:
            raise ValueError("No legal actions available")
        return legals[0]

    @staticmethod
    def _resolved_actions(actions: list[Any | None]) -> list[Any]:
        resolved = []
        for action in actions:
            if action is None:
                raise ValueError("No action resolved for rollout request")
            resolved.append(action)
        return resolved

    def _single_policy_action(self, obs):
        """Return the only action when the policy head has no real choice."""
        if self.policy_head_is_pointer:
            candidates = obs.candidate_actions()
            if len(candidates) == 1:
                action = obs.find_candidate_action(0)
                if action is not None:
                    return action
            if not candidates:
                legals = obs.legal_actions()
                if len(legals) == 1:
                    return legals[0]
            return None

        fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        valid_indices = [idx for idx, value in enumerate(fixed_mask.tolist()) if value]
        if len(valid_indices) == 1:
            action = obs.find_action(valid_indices[0])
            if action is not None:
                return action
        legals = obs.legal_actions()
        if len(legals) == 1:
            return legals[0]
        return None

    @staticmethod
    def _collate_features(features: list[Any], device: torch.device):
        if not features:
            raise ValueError("features must not be empty")
        first = features[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(features).to(device)
        if isinstance(first, dict):
            return {key: value.to(device) for key, value in collate_sequence_features(features).items()}
        raise TypeError(f"Unsupported feature type: {type(first).__name__}")

    @staticmethod
    def _collate_belief(features: list[dict[str, torch.Tensor]], device: torch.device):
        return {key: value.to(device) for key, value in collate_belief_features(features).items()}

    @staticmethod
    def _softmax_np(values: np.ndarray, temperature: float) -> np.ndarray:
        values = values.astype(np.float64, copy=False) / float(temperature)
        values = values - np.max(values)
        exp = np.exp(values)
        total = float(exp.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(values.shape, 1.0 / max(1, values.shape[0]), dtype=np.float64)
        return exp / total

    def _root_actions_and_priors(self, obs, logits_1d: torch.Tensor) -> tuple[list[Any], np.ndarray, list[int]]:
        if self.policy_head_is_pointer:
            candidates = obs.candidate_actions()
            valid_count = min(len(candidates), int(logits_1d.shape[-1]))
            root_actions = []
            action_keys = []
            root_logits = []
            for idx in range(valid_count):
                action = obs.find_candidate_action(idx)
                if action is None:
                    continue
                root_actions.append(action)
                action_keys.append(idx)
                root_logits.append(float(logits_1d[idx]))
            if not root_actions:
                return [], np.zeros(0, dtype=np.float64), []
            priors = self._softmax_np(np.asarray(root_logits, dtype=np.float64), 1.0)
            return root_actions, priors, action_keys

        fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        root_actions = []
        action_keys = []
        logits = []
        for legal in obs.legal_actions():
            encoded_id = None
            try:
                encoded_id = int(legal.encode())
            except Exception:
                pass
            if encoded_id is None or encoded_id >= len(fixed_mask) or fixed_mask[encoded_id] == 0:
                continue
            root_actions.append(legal)
            action_keys.append(encoded_id)
            logits.append(float(logits_1d[encoded_id]))
        if not root_actions:
            return [], np.zeros(0, dtype=np.float64), []
        priors = self._softmax_np(np.asarray(logits, dtype=np.float64), 1.0)
        return root_actions, priors, action_keys

    def _sample_worlds(self, env: RiichiEnv, obs, pid: int) -> tuple[list[WorldSample], dict[str, int]]:
        accepted: list[WorldSample] = []
        requested = int(self.cfg.belief_samples)
        attempts = 0

        feature = self.belief_encoder.encode(obs)
        batch = self._collate_belief([feature], self.belief_device)
        sampling_context = self.belief_model.prepare_allocation_sampling_context(batch)

        max_batches = int(self.cfg.max_belief_sample_batches)
        per_batch = int(self.cfg.belief_sample_batch_size)
        for _ in range(max_batches):
            if len(accepted) >= requested:
                break
            allocations = self.belief_model.sample_allocations_from_context(
                sampling_context,
                num_samples=per_batch,
                temperature=float(self.cfg.belief_temperature),
                decode_steps=self.cfg.belief_decode_steps,
            )
            for allocation in allocations[0]:
                attempts += 1
                sample = self._allocation_to_world_sample(env, obs, pid, allocation)
                if sample is None:
                    continue
                accepted.append(sample)
                if len(accepted) >= requested:
                    break

        return accepted, {
            "requested": requested,
            "accepted": len(accepted),
            "attempted": attempts,
        }

    def _allocation_to_world_sample(  # noqa: PLR0911
        self,
        env: RiichiEnv,
        obs,
        pid: int,
        allocation: torch.Tensor,
    ) -> WorldSample | None:
        try:
            bucket_tile37 = allocation_counts_to_tile37_lists(allocation)
        except ValueError:
            return None

        visible_tiles = self._visible_tiles_from_obs(obs, pid)
        visible_counts = counts37_from_tiles(visible_tiles)
        if np.any(visible_counts > TOTAL_TILE_COUNTS37):
            return None

        available_by37 = [list(ids) for ids in _TILE37_TO_TILE_IDS]
        for tile in visible_tiles:
            tile37 = tile_id_to_tile37(tile)
            try:
                available_by37[tile37].remove(int(tile))
            except ValueError:
                return None

        hands = [list(hand) for hand in getattr(obs, "hands", [[] for _ in range(4)])]
        if len(hands) != 4:
            return None
        hands[pid] = list(getattr(obs, "hand", hands[pid]))

        for bucket_idx, rel in enumerate(BUCKET_REL_SEATS):
            abs_seat = (pid + int(rel)) % 4
            assigned = self._draw_concrete_tiles(available_by37, bucket_tile37[bucket_idx])
            if assigned is None:
                return None
            hands[abs_seat] = sorted(assigned)

        residual_wall = self._draw_concrete_tiles(available_by37, bucket_tile37[BUCKET_COUNT - 1])
        if residual_wall is None:
            return None
        if any(available_by37):
            return None

        if not self._riichi_tenpai_filter(obs, pid, hands):
            return None

        wall = self._build_wall_with_known_dora(env, residual_wall)
        if wall is None:
            return None

        return WorldSample(
            hands=[sorted(int(tile) for tile in hand) for hand in hands],
            wall=wall,
            dora_indicators=[int(tile) for tile in getattr(env, "dora_indicators", [])],
            drawable_count=int(getattr(env, "drawable_count", 0)),
        )

    def _visible_tiles_from_obs(self, obs, pid: int) -> list[int]:
        visible: list[int] = []
        hands = getattr(obs, "hands", [])
        if len(hands) > pid and hands[pid]:
            visible.extend(int(tile) for tile in hands[pid])
        else:
            visible.extend(int(tile) for tile in getattr(obs, "hand", []))

        for seat_discards in getattr(obs, "discards", []):
            visible.extend(int(tile) for tile in seat_discards)

        for seat_melds in getattr(obs, "melds", []):
            for meld in seat_melds:
                called_tile = getattr(meld, "called_tile", None)
                from_who = int(getattr(meld, "from_who", -1))
                opened = bool(getattr(meld, "opened", False))
                skipped_called = False
                for raw_tile in getattr(meld, "tiles", []):
                    tile = int(raw_tile)
                    if (
                        opened
                        and from_who >= 0
                        and called_tile is not None
                        and tile == int(called_tile)
                        and not skipped_called
                    ):
                        skipped_called = True
                        continue
                    visible.append(tile)

        visible.extend(int(tile) for tile in getattr(obs, "dora_indicators", []))
        return visible

    def _draw_concrete_tiles(self, available_by37: list[list[int]], tile37_values: list[int]) -> list[int] | None:
        out: list[int] = []
        for tile37 in tile37_values:
            bucket = available_by37[int(tile37)]
            if not bucket:
                return None
            pos = self.rng.randrange(len(bucket))
            out.append(bucket.pop(pos))
        return out

    @staticmethod
    def _riichi_tenpai_filter(obs, pid: int, hands: list[list[int]]) -> bool:
        riichi_declared = getattr(obs, "riichi_declared", [False] * 4)
        for seat, declared in enumerate(riichi_declared):
            if seat == pid or not declared:
                continue
            if int(calculate_shanten([int(tile) for tile in hands[seat]])) != 0:
                return False
        return True

    def _build_wall_with_known_dora(self, env: RiichiEnv, residual_wall: list[int]) -> list[int] | None:
        wall_len = len(getattr(env, "wall", []))
        dora_indicators = [int(tile) for tile in getattr(env, "dora_indicators", [])]
        rinshan_draw_count = int(getattr(env, "rinshan_draw_count", 0))

        pinned: dict[int, int] = {}
        for idx, tile in enumerate(dora_indicators):
            pos = 4 + 2 * idx - rinshan_draw_count
            if not 0 <= pos < wall_len:
                return None
            if pos in pinned:
                return None
            pinned[pos] = tile

        if len(residual_wall) != wall_len - len(pinned):
            return None

        shuffled = list(residual_wall)
        self.rng.shuffle(shuffled)
        wall: list[int | None] = [None] * wall_len
        for pos, tile in pinned.items():
            wall[pos] = tile

        it = iter(shuffled)
        for pos in range(wall_len):
            if wall[pos] is None:
                wall[pos] = next(it)
        return [int(tile) for tile in wall if tile is not None]

    def _run_root_rollouts(
        self,
        *,
        env: RiichiEnv,
        root_pid: int,
        root_actions: list[Any],
        priors: np.ndarray,
        samples: list[WorldSample],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        action_count = len(root_actions)
        visits = np.zeros(action_count, dtype=np.int64)
        value_sums = np.zeros(action_count, dtype=np.float64)
        rank_sums = np.zeros((action_count, _RANK_COUNT), dtype=np.float64)
        terminal_reasons: dict[str, int] = {}
        rollout_errors = 0

        remaining = int(self.cfg.num_simulations)
        while remaining > 0:
            batch_size = min(int(self.cfg.rollout_batch_size), remaining)
            selected_indices = self._select_puct_batch(priors, visits, value_sums, batch_size)
            states = self._prepare_root_rollouts(env, root_pid, root_actions, samples, selected_indices)

            for state in states:
                if state.done:
                    terminal_reasons[state.terminal_reason or "unknown"] = (
                        terminal_reasons.get(state.terminal_reason or "unknown", 0) + 1
                    )
            active_states = [state for state in states if not state.done]
            self._play_rollout_batch(active_states)

            for state in states:
                if state.reward is None or state.rank_prediction is None:
                    rollout_errors += 1
                    continue
                idx = state.root_action_index
                visits[idx] += 1
                value_sums[idx] += float(state.reward)
                rank_sums[idx] += np.asarray(state.rank_prediction, dtype=np.float64)
                terminal_reasons[state.terminal_reason or "unknown"] = (
                    terminal_reasons.get(state.terminal_reason or "unknown", 0) + 1
                )

            remaining -= batch_size

        return visits, value_sums, rank_sums, {
            "enabled": True,
            "num_simulations_requested": int(self.cfg.num_simulations),
            "num_simulations_completed": int(visits.sum()),
            "rollout_batch_size": int(self.cfg.rollout_batch_size),
            "puct_c": float(self.cfg.puct_c),
            "discard_horizon": int(self.cfg.discard_horizon),
            "terminal_reasons": terminal_reasons,
            "rollout_errors": rollout_errors,
        }

    def _select_puct_batch(
        self,
        priors: np.ndarray,
        visits: np.ndarray,
        value_sums: np.ndarray,
        batch_size: int,
    ) -> list[int]:
        selected = []
        scratch_visits = visits.astype(np.float64, copy=True)
        total = float(max(1.0, scratch_visits.sum()))
        q_values = np.zeros_like(priors, dtype=np.float64)
        visited = visits > 0
        q_values[visited] = value_sums[visited] / visits[visited]

        for _ in range(batch_size):
            scores = q_values + float(self.cfg.puct_c) * priors * math.sqrt(total) / (1.0 + scratch_visits)
            idx = int(np.flatnonzero(scores == scores.max())[0])
            selected.append(idx)
            scratch_visits[idx] += 1.0
            total += 1.0
        return selected

    def _prepare_root_rollouts(
        self,
        env: RiichiEnv,
        root_pid: int,
        root_actions: list[Any],
        samples: list[WorldSample],
        selected_indices: list[int],
    ) -> list[RolloutState]:
        states: list[RolloutState] = []
        other_response_requests: list[tuple[RolloutState, int, Any, dict[int, Any]]] = []
        response_step_requests: list[tuple[RolloutState, dict[int, Any]]] = []

        for action_idx in selected_indices:
            sample = samples[self.rng.randrange(len(samples))]
            clone = env.clone_for_simulation()
            state = RolloutState(
                env=clone,
                root_pid=root_pid,
                root_action_index=int(action_idx),
                root_discard_decisions=0,
                start_round_key=self._round_key(env),
            )
            states.append(state)

            try:
                clone.inject_simulation_state(
                    sample.hands,
                    sample.wall,
                    sample.dora_indicators,
                    int(sample.drawable_count),
                )
                if self._is_response_phase(clone):
                    clone.recompute_current_claims()
                obs_dict = clone.get_observations(list(getattr(clone, "active_players", [])))
                if root_pid not in obs_dict:
                    self._finish_error(state, "root_not_active_after_injection")
                    continue

                actions = {root_pid: root_actions[action_idx]}
                if self._is_response_phase(clone):
                    for other_pid, other_obs in obs_dict.items():
                        if int(other_pid) == int(root_pid):
                            continue
                        other_response_requests.append((state, int(other_pid), other_obs, actions))
                    response_step_requests.append((state, actions))
                else:
                    self._step_root_rollout(state, actions, "root_step_error")
            except Exception as exc:
                logger.debug(f"root rollout preparation failed: {exc}")
                self._finish_error(state, "root_preparation_error")

        if other_response_requests:
            obs_list = [item[2] for item in other_response_requests if not item[0].done]
            sampled = self._sample_actions(obs_list)
            sample_idx = 0
            for state, other_pid, _other_obs, actions in other_response_requests:
                if state.done:
                    continue
                actions[other_pid] = sampled[sample_idx]
                sample_idx += 1

        for state, actions in response_step_requests:
            if state.done:
                continue
            self._step_root_rollout(state, actions, "root_response_step_error")

        return states

    def _step_root_rollout(self, state: RolloutState, actions: dict[int, Any], error_reason: str) -> None:
        try:
            state.env.step(actions)
            if getattr(state.env, "last_error", None):
                logger.debug(f"root rollout env error: {state.env.last_error}")
                self._finish_error(state, "root_step_env_last_error")
                return
            self._finish_if_terminal_after_step(state)
        except Exception as exc:
            logger.debug(f"root rollout step failed: {exc}")
            self._finish_error(state, error_reason)

    def _play_rollout_batch(self, states: list[RolloutState]) -> None:
        while any(not state.done for state in states):
            requests: list[tuple[RolloutState, int, Any]] = []
            rank_eval_features: list[tuple[RolloutState, dict[str, torch.Tensor], str]] = []

            for state in states:
                if state.done:
                    continue
                try:
                    obs_dict = state.env.get_observations(list(getattr(state.env, "active_players", [])))
                except Exception as exc:
                    logger.debug(f"rollout observation failed: {exc}")
                    self._finish_error(state, "observation_error")
                    continue

                root_obs = obs_dict.get(state.root_pid)
                if root_obs is not None and self._is_discard_decision(root_obs):
                    state.root_discard_decisions += 1
                    if state.root_discard_decisions >= int(self.cfg.discard_horizon):
                        rank_eval_features.append((state, self.bc.encoder.encode(root_obs), "discard_horizon"))
                        continue

                if not obs_dict:
                    self._finish_error(state, "no_active_observations")
                    continue
                for pid, obs in obs_dict.items():
                    requests.append((state, int(pid), obs))

            if rank_eval_features:
                rank_probs = self._evaluate_rank_features([feature for _state, feature, _reason in rank_eval_features])
                for (state, _feature, reason), probs in zip(rank_eval_features, rank_probs, strict=True):
                    self._finish_with_rank_prediction(state, probs, reason)

            active_requests = [(state, pid, obs) for state, pid, obs in requests if not state.done]
            if not active_requests:
                continue

            sampled_actions = self._sample_actions([obs for _state, _pid, obs in active_requests])
            actions_by_state: dict[int, tuple[RolloutState, dict[int, Any]]] = {}
            for (state, pid, _obs), action in zip(active_requests, sampled_actions, strict=True):
                key = id(state)
                if key not in actions_by_state:
                    actions_by_state[key] = (state, {})
                actions_by_state[key][1][pid] = action

            for state, actions in actions_by_state.values():
                if state.done:
                    continue
                try:
                    state.env.step(actions)
                    if getattr(state.env, "last_error", None):
                        logger.debug(f"rollout env error: {state.env.last_error}")
                        self._finish_error(state, "env_last_error")
                        continue
                    self._finish_if_terminal_after_step(state)
                except Exception as exc:
                    logger.debug(f"rollout step failed: {exc}")
                    self._finish_error(state, "rollout_step_error")

    def _finish_if_terminal_after_step(self, state: RolloutState) -> None:
        if state.done:
            return
        if state.env.done():
            ranks = state.env.ranks()
            rank_idx = max(0, min(_RANK_COUNT - 1, int(ranks[state.root_pid]) - 1))
            rank_prediction = [0.0] * _RANK_COUNT
            rank_prediction[rank_idx] = 1.0
            self._finish_with_rank_prediction(state, rank_prediction, "hanchan_end")
            return
        if self._round_key(state.env) != state.start_round_key:
            feature = self._encode_kyoku_start_or_current_obs(state.env, state.root_pid)
            probs = self._evaluate_rank_features([feature])[0]
            self._finish_with_rank_prediction(state, probs, "next_kyoku_start")

    def _finish_with_rank_prediction(self, state: RolloutState, rank_prediction: list[float], reason: str) -> None:
        if state.done:
            return
        state.rank_prediction = [float(x) for x in rank_prediction]
        state.reward = self._expected_points(state.rank_prediction)
        state.terminal_reason = reason
        state.done = True

    def _finish_error(self, state: RolloutState, reason: str) -> None:
        if state.done:
            return
        state.rank_prediction = [0.25] * _RANK_COUNT
        state.reward = self._expected_points(state.rank_prediction)
        state.terminal_reason = reason
        state.done = True

    def _encode_kyoku_start_or_current_obs(self, env: RiichiEnv, pid: int) -> dict[str, torch.Tensor]:
        if hasattr(self.bc.encoder, "encode_kyoku_start"):
            try:
                return self.bc.encoder.encode_kyoku_start(
                    scores=list(env.scores()),
                    chang=int(getattr(env, "round_wind", 0)),
                    ju=int(getattr(env, "kyoku_idx", 0)),
                    ben=int(getattr(env, "honba", 0)),
                    liqibang=int(getattr(env, "riichi_sticks", 0)),
                    player_id=int(pid),
                )
            except Exception as exc:
                logger.debug(f"encode_kyoku_start failed, falling back to current observation: {exc}")
        obs = env.get_observations([pid])[pid]
        return self.bc.encoder.encode(obs)

    @staticmethod
    def _round_key(env: RiichiEnv) -> tuple[int, int, int]:
        return (
            int(getattr(env, "round_wind", 0)),
            int(getattr(env, "kyoku_idx", 0)),
            int(getattr(env, "honba", 0)),
        )

    @staticmethod
    def _is_response_phase(env: RiichiEnv) -> bool:
        return int(getattr(env, "phase", Phase.WaitAct)) == int(Phase.WaitResponse)

    @staticmethod
    def _is_discard_decision(obs) -> bool:
        return any(
            _action_type_id(action) in {_DISCARD_ACTION_TYPE_ID, _RIICHI_ACTION_TYPE_ID}
            for action in obs.legal_actions()
        )

    def _expected_points(self, rank_prediction: list[float]) -> float:
        probs = np.asarray(rank_prediction, dtype=np.float64)
        if probs.shape[0] != _RANK_COUNT:
            return 0.0
        return float(np.dot(probs, self.rank_weights))

    def _meta_for_selected_action(
        self,
        *,
        obs,
        logits: torch.Tensor,
        selected_action: Any,
        selected_index: int,
        action_keys: list[int],
    ) -> dict[str, Any]:
        decision = _policy_decision_from_logits(
            obs,
            logits.unsqueeze(0).to(self.device),
            self.device,
            candidate_logits=self.policy_head_is_pointer,
        )

        if self.policy_head_is_pointer:
            selected_key = action_keys[selected_index]
            decision.meta["chosen_index"] = int(selected_key)
            decision.meta["chosen_action"] = _action_to_policy_payload(
                selected_action,
                getattr(obs, "drawn_tile", None),
            )
            for entry in decision.meta.get("candidates", []):
                entry["selected"] = int(entry.get("index", -1)) == int(selected_key)
            for entry in decision.meta.get("legal_actions", []):
                entry["selected_candidate"] = entry.get("candidate_index") == int(selected_key)
            return decision.meta

        try:
            selected_key = int(selected_action.encode())
        except Exception:
            selected_key = action_keys[selected_index]
        decision.meta["chosen_index"] = int(selected_key)
        decision.meta["chosen_action"] = _action_to_policy_payload(selected_action, getattr(obs, "drawn_tile", None))
        for entry in decision.meta.get("legal_actions", []):
            entry["selected"] = entry.get("action_id") == int(selected_key)
        return decision.meta

    def _attach_bc_rank_meta(self, meta: dict[str, Any], rank_prediction: list[float], expected_points: float) -> None:
        meta["bc_rank_head"] = {
            "rank_probs": [float(x) for x in rank_prediction],
            "rank_point_weights": [float(x) for x in self.rank_weights.tolist()],
            "expected_points": float(expected_points),
        }

    def _attach_search_fallback(
        self,
        meta: dict[str, Any],
        reason: str,
        requested_samples: int,
        accepted_samples: int,
    ) -> None:
        meta[self.cfg.metadata_key] = {
            "enabled": False,
            "fallback_reason": reason,
            "belief_samples_requested": int(requested_samples),
            "belief_samples_used": int(accepted_samples),
            "num_simulations_requested": int(self.cfg.num_simulations),
            "num_simulations_completed": 0,
        }

    def _attach_search_stats(
        self,
        meta: dict[str, Any],
        action_keys: list[int],
        visits: np.ndarray,
        value_sums: np.ndarray,
        rank_sums: np.ndarray,
        search_meta: dict[str, Any],
    ) -> None:
        total_visits = int(visits.sum())
        stats_by_key: dict[int, dict[str, Any]] = {}
        for idx, key in enumerate(action_keys):
            count = int(visits[idx])
            if count > 0:
                mean_rank = (rank_sums[idx] / float(count)).tolist()
                mean_expected = float(value_sums[idx] / float(count))
            else:
                mean_rank = None
                mean_expected = None
            stats_by_key[int(key)] = {
                "simulation_count": count,
                "simulation_fraction": float(count / total_visits) if total_visits > 0 else 0.0,
                "mean_rank_probs": mean_rank,
                "mean_expected_points": mean_expected,
            }

        search_candidates: list[dict[str, Any]] = []
        search_legal_actions: list[dict[str, Any]] = []
        if self.policy_head_is_pointer:
            for entry in meta.get("candidates", []):
                stats = stats_by_key.get(int(entry.get("index", -1)))
                if stats is not None:
                    search_candidates.append(
                        {
                            "index": int(entry["index"]),
                            "action": entry.get("action"),
                            **stats,
                        }
                    )
            for entry in meta.get("legal_actions", []):
                candidate_index = entry.get("candidate_index")
                stats = stats_by_key.get(int(candidate_index)) if candidate_index is not None else None
                if stats is not None:
                    search_legal_actions.append(
                        {
                            "legal_index": int(entry["legal_index"]),
                            "candidate_index": int(candidate_index),
                            "action": entry.get("action"),
                            **stats,
                        }
                    )
            selected_key = int(meta.get("chosen_index", -1))
            selected_stats = stats_by_key.get(selected_key, {})
        else:
            for entry in meta.get("legal_actions", []):
                action_id = entry.get("action_id")
                stats = stats_by_key.get(int(action_id)) if action_id is not None else None
                if stats is not None:
                    search_legal_actions.append(
                        {
                            "legal_index": int(entry["legal_index"]),
                            "action_id": int(action_id),
                            "action": entry.get("action"),
                            **stats,
                        }
                    )
            selected_stats = stats_by_key.get(int(meta.get("chosen_index", -1)), {})

        simulation_distribution = [
            {
                "action_key": int(key),
                **stats_by_key[int(key)],
            }
            for key in action_keys
        ]
        meta[self.cfg.metadata_key] = {
            **search_meta,
            "belief_samples_requested": int(search_meta.get("requested", 0)),
            "belief_samples_used": int(search_meta.get("accepted", 0)),
            "belief_sample_attempts": int(search_meta.get("attempted", 0)),
            "simulation_distribution": simulation_distribution,
            "candidates": search_candidates,
            "legal_actions": search_legal_actions,
            "selected_simulation_count": int(selected_stats.get("simulation_count", 0)),
            "selected_mean_expected_points": selected_stats.get("mean_expected_points"),
        }
