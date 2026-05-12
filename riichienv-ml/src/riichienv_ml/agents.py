"""Agents for RiichiEnv loaded from YAML training configs.

Usage::

    from riichienv import RiichiEnv
    from riichienv_ml.agents import Agent

    agents = {
        0: Agent("configs/4p/ppo.yml"),
        1: Agent("configs/4p/bc_model.yml", device="cuda"),
    }

    env = RiichiEnv(game_mode="4p-red-half")
    obs_dict = env.reset()
    while not env.done():
        actions = {}
        for pid, obs in obs_dict.items():
            actions[pid] = agents[pid].act(obs)
        obs_dict = env.step(actions)
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml

from riichienv_ml.config import import_class, load_config
from riichienv_ml.utils import build_encoder, configure_matmul_precision


def _feature_to_device_batch(feature, device: torch.device):
    if isinstance(feature, torch.Tensor):
        return feature.to(device).unsqueeze(0)
    if isinstance(feature, dict):
        return {key: value.to(device).unsqueeze(0) for key, value in feature.items()}
    raise TypeError(f"Unsupported feature type: {type(feature).__name__}")


def _candidate_mask(obs, width: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, width, dtype=torch.bool, device=device)
    n_cand = min(len(obs.candidate_actions()), width)
    if n_cand > 0:
        mask[:, :n_cand] = True
    return mask


_ACTION_TYPE_NAMES = {
    0: "discard",
    1: "chi",
    2: "pon",
    3: "daiminkan",
    4: "ron",
    5: "riichi",
    6: "tsumo",
    7: "pass",
    8: "ankan",
    9: "kakan",
    10: "kyushu_kyuhai",
    11: "kita",
}
_DISCARD_ACTION_TYPE_ID = 0
_PASS_ACTION_TYPE_ID = 7
_MOQIE_TEDASHI = 0
_MOQIE_TSUMOGIRI = 1
_MOQIE_NA = 2
_MOQIE_NAMES = {
    _MOQIE_TEDASHI: "tedashi",
    _MOQIE_TSUMOGIRI: "tsumogiri",
    _MOQIE_NA: "na",
}


@dataclass(frozen=True)
class PolicyDecision:
    """Action selected by a model plus serializable policy diagnostics."""

    action: Any
    meta: dict[str, Any]


def _as_single_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.detach()
    if logits.ndim == 2 and logits.shape[0] == 1:
        return logits.detach()[0]
    raise ValueError(f"Expected logits for a single observation, got shape={tuple(logits.shape)}")


def _finite_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    raw = float(value)
    if not np.isfinite(raw):
        return None
    return raw


def _action_type_id(action) -> int | None:
    try:
        return int(action.action_type)
    except (AttributeError, TypeError, ValueError):
        return None


def _discard_tsumogiri(action, drawn_tile: int | None) -> bool | None:
    if _action_type_id(action) != _DISCARD_ACTION_TYPE_ID:
        return None
    tile = getattr(action, "tile", None)
    if tile is None or drawn_tile is None:
        return False
    return int(tile) == int(drawn_tile)


def _action_moqie(action, drawn_tile: int | None) -> int:
    tsumogiri = _discard_tsumogiri(action, drawn_tile)
    if tsumogiri is None:
        return _MOQIE_NA
    return _MOQIE_TSUMOGIRI if tsumogiri else _MOQIE_TEDASHI


def _action_to_mjai_dict(action, drawn_tile: int | None = None) -> dict[str, Any]:
    try:
        mjai = json.loads(action.to_mjai())
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    tsumogiri = _discard_tsumogiri(action, drawn_tile)
    if tsumogiri is not None:
        mjai["tsumogiri"] = tsumogiri
    return mjai


def _action_to_policy_payload(action, drawn_tile: int | None = None) -> dict[str, Any]:
    action_type_id = _action_type_id(action)
    moqie_id = _action_moqie(action, drawn_tile)
    payload = {
        "mjai": _action_to_mjai_dict(action, drawn_tile),
        "action_type": _ACTION_TYPE_NAMES.get(action_type_id, str(action_type_id)),
        "action_type_id": action_type_id,
        "tile": getattr(action, "tile", None),
        "consume_tiles": list(getattr(action, "consume_tiles", [])),
        "actor": getattr(action, "actor", None),
        "tsumogiri": _discard_tsumogiri(action, drawn_tile),
        "moqie": _MOQIE_NAMES[moqie_id],
        "moqie_id": moqie_id,
    }
    try:
        payload["encoded_id"] = int(action.encode())
    except Exception:
        pass
    return payload


def _masked_softmax_values(logits_1d: torch.Tensor, valid_indices: list[int]) -> dict[int, float]:
    if not valid_indices:
        return {}
    index_t = torch.tensor(valid_indices, dtype=torch.long, device=logits_1d.device)
    valid_logits = logits_1d.index_select(0, index_t)
    probs = torch.softmax(valid_logits, dim=0).detach().cpu().tolist()
    return {idx: float(prob) for idx, prob in zip(valid_indices, probs, strict=True)}


def _pointer_policy_decision(obs, logits: torch.Tensor, device: torch.device) -> PolicyDecision:
    logits_1d = _as_single_logits(logits)
    width = int(logits_1d.shape[-1])
    candidates = obs.candidate_actions()
    valid_count = min(len(candidates), width)
    valid_indices = list(range(valid_count))
    probs_by_idx = _masked_softmax_values(logits_1d, valid_indices)

    mask_t = torch.zeros(1, width, dtype=torch.bool, device=device)
    if valid_count > 0:
        mask_t[:, :valid_count] = True
    masked_logits = logits.masked_fill(~mask_t, -1e9)
    action_idx = int(masked_logits.argmax(dim=1).item())
    action = obs.find_candidate_action(action_idx)

    if action is None:
        legals = obs.legal_actions()
        if legals:
            action = legals[0]
        else:
            raise ValueError(f"No legal action for action_idx={action_idx}")

    logits_cpu = logits_1d.detach().cpu()
    drawn_tile = getattr(obs, "drawn_tile", None)
    candidate_entries = []
    for idx, candidate in enumerate(candidates[:valid_count]):
        candidate_entries.append(
            {
                "index": idx,
                "action": _action_to_policy_payload(candidate, drawn_tile),
                "logit": _finite_float(logits_cpu[idx]),
                "prob": probs_by_idx.get(idx),
                "selected": idx == action_idx,
            }
        )

    legal_entries = []
    for legal_index, legal in enumerate(obs.legal_actions()):
        candidate_index = obs.find_candidate_index(legal)
        logit = None
        prob = None
        if candidate_index is not None and 0 <= candidate_index < valid_count:
            logit = _finite_float(logits_cpu[candidate_index])
            prob = probs_by_idx.get(candidate_index)
        legal_entries.append(
            {
                "legal_index": legal_index,
                "candidate_index": candidate_index,
                "action": _action_to_policy_payload(legal, drawn_tile),
                "logit": logit,
                "prob": prob,
                "selected_candidate": candidate_index == action_idx,
            }
        )

    response_decision = any(_action_type_id(candidate) == _PASS_ACTION_TYPE_ID for candidate in candidates)
    meta = {
        "head": "pointer",
        "logit_space": "candidate",
        "candidate_count": valid_count,
        "legal_action_count": len(legal_entries),
        "chosen_index": action_idx,
        "chosen_action": _action_to_policy_payload(action, drawn_tile),
        "response_decision": response_decision,
        "candidates": candidate_entries,
        "legal_actions": legal_entries,
    }
    return PolicyDecision(action=action, meta=meta)


def _fixed_policy_decision(obs, logits: torch.Tensor, device: torch.device) -> PolicyDecision:
    logits_1d = _as_single_logits(logits)
    fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
    mask_t = torch.from_numpy(fixed_mask).to(device).unsqueeze(0)
    masked_logits = logits.masked_fill(mask_t == 0, -1e9)
    action_idx = int(masked_logits.argmax(dim=1).item())
    action = obs.find_action(action_idx)

    if action is None:
        legals = obs.legal_actions()
        if legals:
            action = legals[0]
        else:
            raise ValueError(f"No legal action for action_idx={action_idx}")

    valid_indices = [idx for idx, value in enumerate(fixed_mask.tolist()) if value]
    probs_by_idx = _masked_softmax_values(logits_1d, valid_indices)
    logits_cpu = logits_1d.detach().cpu()
    drawn_tile = getattr(obs, "drawn_tile", None)

    legal_entries = []
    for legal_index, legal in enumerate(obs.legal_actions()):
        encoded_id = None
        try:
            encoded_id = int(legal.encode())
        except Exception:
            pass
        logit = None
        prob = None
        if encoded_id is not None and 0 <= encoded_id < len(logits_cpu):
            logit = _finite_float(logits_cpu[encoded_id])
            prob = probs_by_idx.get(encoded_id)
        legal_entries.append(
            {
                "legal_index": legal_index,
                "action_id": encoded_id,
                "action": _action_to_policy_payload(legal, drawn_tile),
                "logit": logit,
                "prob": prob,
                "selected": encoded_id == action_idx,
            }
        )

    response_decision = any(_action_type_id(legal) == _PASS_ACTION_TYPE_ID for legal in obs.legal_actions())
    meta = {
        "head": "fixed",
        "logit_space": "action_id",
        "legal_action_count": len(legal_entries),
        "chosen_index": action_idx,
        "chosen_action": _action_to_policy_payload(action, drawn_tile),
        "response_decision": response_decision,
        "legal_actions": legal_entries,
    }
    return PolicyDecision(action=action, meta=meta)


def _policy_decision_from_logits(
    obs,
    logits: torch.Tensor,
    device: torch.device,
    *,
    candidate_logits: bool | None = None,
) -> PolicyDecision:
    fixed_mask = None

    if candidate_logits is None:
        fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        candidate_logits = logits.shape[-1] != fixed_mask.shape[0]

    if candidate_logits:
        return _pointer_policy_decision(obs, logits, device)
    return _fixed_policy_decision(obs, logits, device)


def _select_action_from_logits(
    obs,
    logits: torch.Tensor,
    device: torch.device,
    *,
    candidate_logits: bool | None = None,
):
    fixed_mask = None

    if candidate_logits is None:
        fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        candidate_logits = logits.shape[-1] != fixed_mask.shape[0]

    if not candidate_logits:
        if fixed_mask is None:
            fixed_mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        mask_t = torch.from_numpy(fixed_mask).to(device).unsqueeze(0)
        masked_logits = logits.masked_fill(mask_t == 0, -1e9)
        action_idx = masked_logits.argmax(dim=1).item()
        action = obs.find_action(action_idx)
    else:
        mask_t = _candidate_mask(obs, logits.shape[-1], device)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        action_idx = masked_logits.argmax(dim=1).item()
        action = obs.find_candidate_action(action_idx)

    if action is not None:
        return action

    legals = obs.legal_actions()
    if legals:
        return legals[0]
    raise ValueError(f"No legal action for action_idx={action_idx}")


class Agent:
    """Agent that plays in RiichiEnv using a model from a training config.

    Loads model architecture, encoder, and weights from a YAML config file.
    Auto-detects the config section (ppo / bc / cql).

    Args:
        config_path: Path to a training config YAML.
        model_path: Checkpoint path override.  When ``None``, inferred from
            config (``load_model`` for ppo, ``output`` for bc/cql).
        device: Torch device for model inference.
    """

    _SECTIONS = ("ppo", "bc", "cql")

    def __init__(
        self,
        config_path: str,
        model_path: str | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # Detect config section and load
        section, sub_cfg = self._load_section(config_path)
        game = sub_cfg.game
        configure_matmul_precision(getattr(sub_cfg, "matmul_precision", None))

        # Resolve model path
        if model_path is None:
            model_path = self._find_model_path(sub_cfg, section)

        # Build model from config
        model_class = import_class(sub_cfg.model_class)
        model_config = sub_cfg.model.model_dump()
        self.model = model_class(**model_config).to(self.device)

        # Load weights
        state = torch.load(
            model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._load_weights(state)
        self.model.eval()

        # Build encoder
        encoder_class = import_class(sub_cfg.encoder_class)
        self.encoder = build_encoder(encoder_class, tile_dim=game.tile_dim, model_config=model_config)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_section(cls, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config must be a YAML mapping with one of "
                f"{'/'.join(cls._SECTIONS)}, got {type(raw).__name__}")
        for key in cls._SECTIONS:
            if key in raw:
                cfg = load_config(config_path)
                return key, getattr(cfg, key)
        raise ValueError(
            f"No recognized section ({'/'.join(cls._SECTIONS)}) in {config_path}")

    @staticmethod
    def _find_model_path(sub_cfg, section: str) -> str:
        if section == "ppo" and getattr(sub_cfg, "load_model", None):
            return sub_cfg.load_model
        if getattr(sub_cfg, "output", None):
            return sub_cfg.output
        raise ValueError(
            "Cannot determine model path from config; "
            "pass model_path explicitly")

    def _load_weights(self, state: dict):
        """Load state dict with automatic key mapping between model formats.

        Handles conversions between QNetwork (v_head/a_head/head) and
        ActorCriticNetwork (actor_head/critic_head) in both directions.
        """
        model_keys = set(self.model.state_dict().keys())
        model_has_actor = any(k.startswith("actor_head.") for k in model_keys)
        model_has_v = any(k.startswith("v_head.") for k in model_keys)

        ckpt_has_actor = any(k.startswith("actor_head.") for k in state)
        ckpt_has_v = any(k.startswith("v_head.") for k in state)
        ckpt_has_head = any(k.startswith("head.") for k in state)

        if ckpt_has_head and not ckpt_has_actor and model_has_actor:
            # Single-head QNetwork → ActorCriticNetwork
            new_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "actor_head.")] = v
                elif k.startswith("aux_head."):
                    continue
                else:
                    new_state[k] = v
            result = self.model.load_state_dict(new_state, strict=False)
        elif ckpt_has_actor and not ckpt_has_v and model_has_v:
            # ActorCriticNetwork → QNetwork (reverse mapping)
            new_state = {}
            for k, v in state.items():
                if k.startswith("actor_head."):
                    new_state[k.replace("actor_head.", "a_head.")] = v
                elif k.startswith("critic_head."):
                    new_state[k.replace("critic_head.", "v_head.")] = v
                else:
                    new_state[k] = v
            result = self.model.load_state_dict(new_state, strict=False)
        elif ckpt_has_v and not ckpt_has_actor and model_has_actor:
            # Dueling QNetwork → ActorCriticNetwork
            new_state = {}
            for k, v in state.items():
                if k.startswith("a_head."):
                    new_state[k.replace("a_head.", "actor_head.")] = v
                elif k.startswith("v_head."):
                    new_state[k.replace("v_head.", "critic_head.")] = v
                elif k.startswith("aux_head."):
                    continue
                else:
                    new_state[k] = v
            result = self.model.load_state_dict(new_state, strict=False)
        else:
            result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys:
            warnings.warn(
                f"Missing keys when loading weights: {result.missing_keys}",
                stacklevel=2,
            )
        if result.unexpected_keys:
            warnings.warn(
                f"Unexpected keys when loading weights: {result.unexpected_keys}",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Game interface
    # ------------------------------------------------------------------

    def reset(self):
        """Prepare for a new game (no-op for stateless models)."""

    @torch.inference_mode()
    def act(self, obs):
        """Select an action for the given observation.

        Returns:
            A legal action object from the ``riichienv`` environment
            (e.g. ``Action`` for 4P, ``Action3P`` for 3P), chosen by
            greedy argmax over masked logits.
        """
        feat = self.encoder.encode(obs)
        feat_batch = _feature_to_device_batch(feat, self.device)

        output = self.model(feat_batch)
        logits = output[0] if isinstance(output, tuple) else output
        return _select_action_from_logits(
            obs,
            logits,
            self.device,
            candidate_logits=getattr(self.model, "policy_head_type", None) == "pointer",
        )

    @torch.inference_mode()
    def act_with_policy(self, obs) -> PolicyDecision:
        """Select an action and return serializable policy diagnostics."""
        feat = self.encoder.encode(obs)
        feat_batch = _feature_to_device_batch(feat, self.device)

        output = self.model(feat_batch)
        logits = output[0] if isinstance(output, tuple) else output
        return _policy_decision_from_logits(
            obs,
            logits,
            self.device,
            candidate_logits=getattr(self.model, "policy_head_type", None) == "pointer",
        )
