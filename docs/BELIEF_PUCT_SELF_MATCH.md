# Belief-PUCT Self-Match

`riichienv_ml.belief_mcts.BeliefMCTSAgent` is a 4-player self-match agent that combines:

- a behavior-cloning Transformer policy with `emit_rank=true`
- a `JointHiddenAllocationSampler` belief model
- root-only PUCT over the current legal action set
- stochastic BC rollouts after the root action

The ready-to-run config is:

```sh
uv run --package riichienv-ml python riichienv-ml/scripts/run_self_match.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/self_match_belief_mcts_test01.yml
```

The default config uses:

- BC model: `models/behavior_cloning/test45/model.pth`
- belief model: `models/belief_sampler/test35/model.pth`
- hidden-allocation samples per decision: `128`
- root simulations per decision: `512`
- rollout batch size: `64`
- reward weights for predicted final rank: `[1, 1/3, -1/3, -1]`

## Algorithm

At each decision point, the agent first evaluates the root observation with the BC model. The policy head
becomes the root prior, and the rank head is logged as `bc_rank_head.rank_probs` plus
`bc_rank_head.expected_points`.

The belief sampler then generates hidden allocations for the three non-acting seats plus the residual
unknown wall. Riichi players' sampled concealed hands are filtered with `calculate_shanten`; non-tenpai
samples are discarded. If fewer than the requested samples survive after the configured maximum number of
sampling batches, the agent runs with the surviving samples. If no samples survive, it falls back to the BC
policy action.

Each root simulation samples one valid world, fixes one root action selected by batched PUCT, randomizes the
remaining unknown wall tiles while preserving known dora indicator positions, and rolls the hand forward.
After the root action, all rollout choices are sampled from the BC policy distribution. No search tree is
expanded below the root.

Rollout environments are created with `clone_for_simulation()`. These clones do not copy or store MJAI log
strings, so their simulated events are not written to the saved self-match log. They still keep the sequence
feature progression cache enabled, which lets the BC Transformer consume rollout action history without
paying the full MJAI JSON logging cost. The parent self-match environment remains a normal MJAI-logging
environment, so the external saved log format is unchanged apart from metadata additions.

Rollouts stop at the first of:

- the acting player's 6th later discard-capable decision, excluding the current root decision
- hanchan end, in which case the actual final rank is used
- a new kyoku start, in which case the BC rank head is evaluated from the new round state

## Response Decisions

Discard response decisions (`chi`, `pon`, `daiminkan`, `ron`, `pass`) are included in the root action set.
When multiple players can respond, the searched player's root action is fixed and the other responders are
sampled from the BC policy in the rollout batch.

Chankan response search is currently treated as a BC fallback. The Rust engine precomputes kan response
claims through a different path than ordinary discard claims, so recomputing those claims after hidden-hand
injection is not yet supported.

## MJAI Metadata

The saved MJAI log format remains replay-compatible. Existing event fields are not changed; self-match only
adds metadata under `meta.policy` or, for pass/response windows, under `meta.response_policies`.

Additional fields include:

- `bc_rank_head.rank_probs`
- `bc_rank_head.rank_point_weights`
- `bc_rank_head.expected_points`
- `belief_puct.num_simulations_requested`
- `belief_puct.num_simulations_completed`
- `belief_puct.belief_samples_requested`
- `belief_puct.belief_samples_used`
- `belief_puct.simulation_distribution`
- `belief_puct.candidates`
- `belief_puct.legal_actions`
- per-candidate and per-legal-action `simulation_count`
- per-candidate and per-legal-action `simulation_fraction`
- per-candidate and per-legal-action `mean_rank_probs`
- per-candidate and per-legal-action `mean_expected_points`

The existing BC policy metadata fields, including `candidates[].logit`, `candidates[].prob`,
`legal_actions[].logit`, and `legal_actions[].prob`, keep the same format as BC self-match logs. Search
statistics are stored separately under `belief_puct` so they do not overwrite BC policy probabilities.

For actions that receive no simulation visits, `mean_rank_probs` and `mean_expected_points` are `null`.
