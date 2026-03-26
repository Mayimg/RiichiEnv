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
```

The Tenhou behavior cloning config writes the model, log file, and offline W&B run data under
`models/behavior_cloning/test01/`.
