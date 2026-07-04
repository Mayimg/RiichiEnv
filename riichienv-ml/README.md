# riichienv-ml

Mahjong RL training pipeline for RiichiEnv.

## Setup

```sh
uv sync --dev --all-packages
uv run maturin develop --release

# CQL+PPO
uv run --package riichienv-ml python riichienv-ml/scripts/train_grp.py -c riichienv-ml/src/riichienv_ml/configs/4p/grp.yml
uv run --package riichienv-ml python riichienv-ml/scripts/train_cql.py -c riichienv-ml/src/riichienv_ml/configs/4p/cql.yml
uv run --package riichienv-ml python riichienv-ml/scripts/train_ppo.py -c riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml

# BC+PPO (requires online teacher, not included in repo)
uv run --package riichienv-ml python riichienv-ml/scripts/train_bc.py -c riichienv-ml/src/riichienv_ml/configs/4p/bc_model.yml
uv run --package riichienv-ml python riichienv-ml/scripts/train_ppo.py -c riichienv-ml/src/riichienv_ml/configs/4p/bc_ppo.yml

# Offline behavior cloning with sequence_features (Tenhou 4P hanchan)
uv run --package riichienv-ml python riichienv-ml/scripts/train_bc.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/bc_tenhou_seq_test01.yml

# Continue training from an existing BC checkpoint
uv run --package riichienv-ml python riichienv-ml/scripts/train_bc.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/bc_tenhou_seq_test02.yml

# Self-match and save MJAI logs
uv run --package riichienv-ml python riichienv-ml/scripts/run_self_match.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/self_match_bc_test01.yml

# Self-match with model-output annotations in each MJAI action event
uv run --package riichienv-ml python riichienv-ml/scripts/run_self_match.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/self_match_bc_test01.yml \
  --log_policy_meta

# Self-match with belief-sampled root PUCT and BC stochastic rollouts
uv run --package riichienv-ml python riichienv-ml/scripts/run_self_match.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/self_match_belief_mcts_test01.yml
```

The Tenhou behavior cloning config writes the model, log file, and offline W&B run data under
`models/behavior_cloning/test01/`.
Offline behavior cloning can shuffle samples across multiple log files per worker with
`bc.shuffle_buffer_files` or `train_bc.py --shuffle_buffer_files`. The default is `1`, which preserves the
original per-file shuffling behavior. Sequence features are memory-heavy, so increase this value cautiously:
the buffer is held independently by each DataLoader worker.
The continued-training config shows how to load an existing BC checkpoint via `load_model` and save the
next run to a separate output path.
The self-match config writes MJAI logs and a summary under `data/self_match/BC/test01/`.
When `self_match.log_policy_meta=true`, selected action events include `meta.policy` with the model's raw
logits and softmax probabilities for every pointer candidate and legal action. Discard payloads include
`tsumogiri` plus `moqie` / `moqie_id` so tedashi and tsumogiri candidates are visible. Response windows are
recorded on the source discard/kan event under `meta.response_policies`, because pass decisions are not
standalone events in the saved MJAI timeline. The base MJAI event type and required fields are unchanged;
tools that ignore unknown fields can still replay the annotated logs.

The belief-PUCT self-match config uses `models/behavior_cloning/test45/model.pth` and
`models/belief_sampler/test35/model.pth`. It samples hidden allocations, filters non-tenpai riichi-player
hands, runs root-only PUCT for the current legal action set, and uses stochastic BC policy rollouts. Its
metadata extends `meta.policy` with `value_head`, per-action visit counts, and per-action mean rollout rank
predictions / expected points. See [BELIEF_PUCT_SELF_MATCH.md](../docs/BELIEF_PUCT_SELF_MATCH.md) for details
and current limitations.
