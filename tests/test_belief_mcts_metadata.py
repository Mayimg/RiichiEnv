import json

import numpy as np
import pytest
import torch
from riichienv_ml.belief_mcts import BeliefMCTSAgent
from riichienv_ml.config import BeliefMctsConfig

from riichienv import GameRule, RiichiEnv


def _bare_belief_agent() -> BeliefMCTSAgent:
    agent = object.__new__(BeliefMCTSAgent)
    agent.cfg = BeliefMctsConfig(num_simulations=4, rollout_batch_size=2)
    agent.device = torch.device("cpu")
    agent.policy_head_is_pointer = True
    agent.rank_weights = np.asarray(agent.cfg.rank_point_weights, dtype=np.float64)
    return agent


def _initial_obs():
    env = RiichiEnv(seed=1, rule=GameRule.default_tenhou())
    return env.reset()[0]


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
