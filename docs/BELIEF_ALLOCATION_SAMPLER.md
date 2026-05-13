# Joint Hidden Allocation Belief Sampler

This document describes the 4-player hidden-allocation sampler implemented in
`riichienv_ml.models.belief_allocation`.

## Goal

The model samples legal hidden allocations from the observer's public state:

```text
q_theta(hidden allocation | observation)
```

It is not an action policy and does not predict the next discard.  Its output is
a count allocation over four buckets:

```text
shimocha concealed hand, toimen concealed hand, kamicha concealed hand, residual wall
```

The residual wall means every tile that is unseen by all players: live wall,
dead wall, rinshan, and unrevealed indicator candidates are intentionally kept
as one unordered bucket.

## Tile Space

The sampler uses the existing sequence-feature `tile37` ids:

```text
0=red5m, 1-9=1m-9m,
10=red5p, 11-19=1p-9p,
20=red5s, 21-29=1s-9s,
30-36=E/S/W/N/P/F/C
```

Total counts are `1` for each red five, `3` for each non-red five, and `4` for
all other tile37 ids.

## Encoder

`BeliefFeatureEncoder` extends the existing transformer sequence features with:

- `belief_phase`: self action vs response phase.
- `belief_current_actor`: observer-relative current actor, using the shared
  relative-seat embedding in the model.
- `belief_hand_sizes`: four `(relative_seat, concealed_count)` rows. Opponent
  counts are inferred from public state only; the observer's own hand length is
  directly visible.

The belief model reuses the existing transformer feature groups but does not
consume candidate-action tokens in its context encoder.

## Decoder

For each tile37 id, the decoder enumerates all count tuples
`(x_shimocha, x_toimen, x_kamicha, x_wall)` whose sum equals the unseen count for
that tile.  The maximum class count is `C(7, 3)=35`.

Every step applies a legality mask:

- tuple counts cannot exceed remaining bucket capacity;
- after selecting the tuple, every bucket must still be fillable by future
  unseen tiles.

The logit is:

```text
logit = sum_b log C(rem_b, x_b) + neural_residual
```

The first term is the multivariate hypergeometric prior.  The neural residual is
conditioned on the observation CLS context, current tile id, remaining capacity,
unseen count, and partial allocations generated so far.

## Training Data

`BeliefAllocationDataset` reads full-information MJAI JSONL files through
`MjaiReplay`.  It uses the same replay decision iterator as BC training and asks
for `include_hidden=True`, which adds a teacher-only snapshot of the true
concealed hands.  This does not change the external MJAI log format.

For each decision point:

1. encode the masked observation;
2. count hidden hands for relative seats 1, 2, and 3;
3. compute unseen counts from visible counts;
4. set the wall target as the residual.

Training uses teacher forcing over the 37 tile ids with cross entropy per tile.

## Commands

Train with:

```bash
uv run python riichienv-ml/scripts/train_belief.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/belief_allocation.yml
```

The model can sample from any encoded 4P decision observation:

```python
from riichienv_ml.features.belief_features import BeliefFeatureEncoder, collate_belief_features
from riichienv_ml.models.belief_allocation import JointHiddenAllocationSampler

feature = BeliefFeatureEncoder().encode(obs)
batch = collate_belief_features([feature])
allocation = model.sample_allocations(batch, num_samples=8)
```

`allocation` has shape `(batch, samples, 4, 37)`.
