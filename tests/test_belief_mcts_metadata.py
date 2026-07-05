import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from riichienv_ml.belief_mcts import BeliefMCTSAgent, WorldSample
from riichienv_ml.config import BeliefMctsConfig

from riichienv import Action, ActionType, GameRule, Phase, RiichiEnv
from tests.env.helper import helper_setup_env


def _bare_belief_agent() -> BeliefMCTSAgent:
    agent = object.__new__(BeliefMCTSAgent)
    agent.cfg = BeliefMctsConfig(num_simulations=4, rollout_batch_size=2)
    agent.device = torch.device("cpu")
    agent.policy_head_is_pointer = True
    agent.rank_weights = np.asarray(agent.cfg.rank_point_weights, dtype=np.float64)
    agent.rng = random.Random(0)
    return agent


def _initial_obs():
    env = RiichiEnv(seed=1, rule=GameRule.default_tenhou())
    return env.reset()[0]


def _world_sample_from_env(env: RiichiEnv) -> WorldSample:
    return WorldSample(
        hands=[list(hand) for hand in env.hands],
        wall=list(env.wall),
        dora_indicators=list(env.dora_indicators),
        drawable_count=int(env.drawable_count),
    )


def _root_only_chi_response_env() -> RiichiEnv:
    env = helper_setup_env(
        seed=1,
        hands=[
            [16] + [2] * 12,
            [8, 12] + [0] * 11,
            [40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88],
            [92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 133, 134],
        ],
        current_player=0,
        active_players=[0],
        drawn_tile=20,
    )
    env.step({0: Action(ActionType.DISCARD, tile=16)})
    assert env.phase == Phase.WaitResponse
    assert env.active_players == [1]
    return env


def test_belief_mcts_selected_action_meta_preserves_raw_bc_policy_prob():
    agent = _bare_belief_agent()
    obs = _initial_obs()
    candidate_count = len(obs.candidate_actions())
    logits = torch.linspace(-2.0, 2.0, candidate_count)
    raw_probs = torch.softmax(logits, dim=0).tolist()

    selected_key = 0
    selected_action = obs.find_candidate_action(selected_key)
    meta = agent._meta_for_selected_action(
        obs=obs,
        logits=logits,
        selected_action=selected_action,
        selected_index=selected_key,
        action_keys=list(range(candidate_count)),
    )

    selected_entry = meta["candidates"][selected_key]
    argmax_entry = meta["candidates"][candidate_count - 1]
    assert meta["chosen_index"] == selected_key
    assert selected_entry["selected"] is True
    assert argmax_entry["selected"] is False
    assert selected_entry["logit"] == pytest.approx(float(logits[selected_key]))
    assert selected_entry["prob"] == pytest.approx(float(raw_probs[selected_key]))
    assert selected_entry["prob"] < 0.1


def test_belief_mcts_response_root_steps_when_no_other_responder(monkeypatch):
    agent = _bare_belief_agent()
    env = _root_only_chi_response_env()
    root_pid = 1
    root_obs = env.get_observations([root_pid])[root_pid]
    root_chi = next(action for action in root_obs.legal_actions() if action.action_type == ActionType.CHI)

    def fail_sample_actions(_obs_list):
        raise AssertionError("other responders should not be sampled")

    monkeypatch.setattr(agent, "_sample_actions", fail_sample_actions)

    state = agent._prepare_root_rollouts(
        env,
        root_pid,
        [root_chi],
        [_world_sample_from_env(env)],
        [0],
    )[0]

    assert state.done is False
    assert state.env.phase == Phase.WaitAct
    assert state.env.active_players == [root_pid]
    assert len(state.env.melds[root_pid]) == 1


def test_belief_mcts_response_root_pass_steps_when_no_other_responder(monkeypatch):
    agent = _bare_belief_agent()
    env = _root_only_chi_response_env()
    root_pid = 1
    root_obs = env.get_observations([root_pid])[root_pid]
    root_pass = next(action for action in root_obs.legal_actions() if action.action_type == ActionType.PASS)

    def fail_sample_actions(_obs_list):
        raise AssertionError("other responders should not be sampled")

    monkeypatch.setattr(agent, "_sample_actions", fail_sample_actions)

    state = agent._prepare_root_rollouts(
        env,
        root_pid,
        [root_pass],
        [_world_sample_from_env(env)],
        [0],
    )[0]

    assert state.done is False
    assert state.env.phase == Phase.WaitAct
    assert state.env.active_players == [root_pid]
    assert len(state.env.melds[root_pid]) == 0


def test_belief_mcts_response_root_keeps_priority_cancellation_as_valid_rollout(monkeypatch):
    agent = _bare_belief_agent()
    env = helper_setup_env(
        seed=1,
        hands=[
            [57] + [2] * 12,
            [62, 65] + [0] * 11,
            [56, 58] + [1] * 11,
            [12, 16, 19, 21, 48, 59, 64, 77, 81, 89, 104, 130, 133],
        ],
        current_player=0,
        active_players=[0],
        drawn_tile=100,
    )
    env.step({0: Action(ActionType.DISCARD, tile=57)})
    assert env.phase == Phase.WaitResponse
    assert env.active_players == [1, 2]

    root_pid = 1
    root_obs = env.get_observations([root_pid])[root_pid]
    root_chi = next(action for action in root_obs.legal_actions() if action.action_type == ActionType.CHI)

    def sample_other_pon(obs_list):
        assert len(obs_list) == 1
        return [next(action for action in obs_list[0].legal_actions() if action.action_type == ActionType.PON)]

    monkeypatch.setattr(agent, "_sample_actions", sample_other_pon)

    state = agent._prepare_root_rollouts(
        env,
        root_pid,
        [root_chi],
        [_world_sample_from_env(env)],
        [0],
    )[0]

    assert state.done is False
    assert state.env.phase == Phase.WaitAct
    assert state.env.active_players == [2]
    assert len(state.env.melds[root_pid]) == 0
    assert len(state.env.melds[2]) == 1


def test_belief_mcts_search_stats_are_separate_from_bc_policy_entries():
    agent = _bare_belief_agent()
    obs = _initial_obs()
    candidate_count = len(obs.candidate_actions())
    logits = torch.linspace(-2.0, 2.0, candidate_count)
    meta = agent._meta_for_selected_action(
        obs=obs,
        logits=logits,
        selected_action=obs.find_candidate_action(0),
        selected_index=0,
        action_keys=list(range(candidate_count)),
    )
    original_prob = meta["candidates"][0]["prob"]

    agent._attach_bc_rank_meta(meta, [0.1, 0.2, 0.3, 0.4], -0.2)
    visits = np.zeros(candidate_count, dtype=np.int64)
    visits[0] = 3
    visits[1] = 1
    value_sums = np.zeros(candidate_count, dtype=np.float64)
    value_sums[0] = 0.75
    value_sums[1] = -0.5
    rank_sums = np.zeros((candidate_count, 4), dtype=np.float64)
    rank_sums[0] = np.asarray([1.5, 0.75, 0.75, 0.0], dtype=np.float64)
    rank_sums[1] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    agent._attach_search_stats(
        meta,
        list(range(candidate_count)),
        visits,
        value_sums,
        rank_sums,
        {
            "enabled": True,
            "num_simulations_requested": 4,
            "num_simulations_completed": 4,
            "requested": 2,
            "accepted": 2,
            "attempted": 2,
        },
    )

    assert meta["candidates"][0]["prob"] == original_prob
    assert "simulation_count" not in meta["candidates"][0]
    assert "mean_rank_probs" not in meta["legal_actions"][0]

    rank_meta = meta["bc_rank_head"]
    assert rank_meta["rank_probs"] == [0.1, 0.2, 0.3, 0.4]
    assert rank_meta["rank_point_weights"] == agent.cfg.rank_point_weights
    assert rank_meta["expected_points"] == pytest.approx(-0.2)

    search_meta = meta["belief_puct"]
    assert search_meta["simulation_distribution"][0]["simulation_count"] == 3
    assert search_meta["simulation_distribution"][0]["simulation_fraction"] == pytest.approx(0.75)
    assert search_meta["simulation_distribution"][0]["mean_rank_probs"] == [0.5, 0.25, 0.25, 0.0]
    assert search_meta["simulation_distribution"][2]["simulation_count"] == 0
    assert search_meta["simulation_distribution"][2]["mean_rank_probs"] is None
    assert search_meta["simulation_distribution"][2]["mean_expected_points"] is None
    assert search_meta["candidates"][0]["simulation_count"] == 3
    assert search_meta["legal_actions"][0]["simulation_count"] == 3
    json.dumps(meta)


def test_belief_mcts_root_skips_search_when_only_one_root_action(monkeypatch):
    agent = _bare_belief_agent()
    env = RiichiEnv(seed=1, rule=GameRule.default_tenhou())
    obs = env.reset()[0]
    candidate_count = len(obs.candidate_actions())
    root_action = obs.find_candidate_action(0)
    logits = torch.linspace(-1.0, 1.0, candidate_count)

    monkeypatch.setattr(
        agent,
        "_evaluate_policy_rank",
        lambda obs_list: {"logits": [logits], "rank_probs": [[0.25, 0.25, 0.25, 0.25]]},
    )
    monkeypatch.setattr(
        agent,
        "_root_actions_and_priors",
        lambda obs_arg, logits_arg: ([root_action], np.ones(1, dtype=np.float64), [0]),
    )

    def fail_sample_worlds(*_args, **_kwargs):
        raise AssertionError("belief sampling should be skipped for single-action roots")

    monkeypatch.setattr(agent, "_sample_worlds", fail_sample_worlds)

    decision = agent.act_with_policy_from_env(env, 0, obs)

    assert decision.action.to_mjai() == root_action.to_mjai()
    search_meta = decision.meta["belief_puct"]
    assert search_meta["enabled"] is False
    assert search_meta["fallback_reason"] == "single_root_action"
    assert search_meta["num_simulations_completed"] == 0


def test_belief_mcts_act_from_env_skips_model_when_policy_has_single_action(monkeypatch):
    agent = _bare_belief_agent()
    forced_action = object()
    monkeypatch.setattr(agent, "_single_policy_action", lambda obs: forced_action)

    def fail_act_with_policy(*_args, **_kwargs):
        raise AssertionError("model-backed root policy should be skipped")

    monkeypatch.setattr(agent, "act_with_policy_from_env", fail_act_with_policy)

    assert agent.act_from_env(object(), 0, object()) is forced_action


def test_belief_mcts_rollout_action_sampling_skips_model_for_single_action_obs(monkeypatch):
    agent = _bare_belief_agent()
    forced_obs = SimpleNamespace(name="forced")
    model_obs = SimpleNamespace(name="model")
    forced_action = object()
    sampled_action = object()
    encoded_obs = []

    class CountingEncoder:
        def encode(self, obs):
            encoded_obs.append(obs.name)
            return torch.tensor([1.0])

    class CountingModel:
        def __call__(self, batch):
            assert tuple(batch.shape) == (1, 1)
            return torch.tensor([[0.0, 1.0]])

    agent.bc = SimpleNamespace(encoder=CountingEncoder(), model=CountingModel())
    monkeypatch.setattr(agent, "_single_policy_action", lambda obs: forced_action if obs is forced_obs else None)

    def sample_action_from_logits(obs, logits):
        assert obs is model_obs
        assert logits.tolist() == [0.0, 1.0]
        return sampled_action

    monkeypatch.setattr(agent, "_sample_action_from_logits", sample_action_from_logits)

    actions = agent._sample_actions([forced_obs, model_obs, forced_obs])

    assert actions == [forced_action, sampled_action, forced_action]
    assert encoded_obs == ["model"]
