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

For each tile37 id, the sampler enumerates all count tuples
`(x_shimocha, x_toimen, x_kamicha, x_wall)` whose sum equals the unseen count for
that tile.  The maximum class count is `C(7, 3)=35`.

The neural decoder is a MaskGIT-style denoising transformer over 37 tile tokens.
Each token is the sum of:

```text
token = projected shared tile embedding
      + unseen-count embedding
      + current-state embedding
      + mask embedding
      + decode-step embedding
```

The tile component reuses the observation encoder's shared, attribute-based
tile37 embedding table for the current public state, then projects it into the
decoder width.  This keeps decoder tile tokens in the same tile representation
space as encoder hand, visible-count, dora, progression, and meld tile fields.

The current state is represented by the candidate tuple id for that tile, with a
dedicated `[MASK]` state id.  The transformer uses bidirectional self-attention
over the 37 tile tokens and cross-attention to cached public memory.  The public
memory includes the encoder CLS token, the existing `(37, 3, d_model)`
opponent tile-bucket context, and the selected encoder memory tokens.

Every denoising step applies a legality mask:

- tuple counts cannot exceed remaining bucket capacity;
- after selecting the tuple, every bucket must still be fillable by currently
  masked unseen tiles.

The logit is:

```text
logit = sum_b log C(rem_b, x_b) + neural_residual
```

The first term is the multivariate hypergeometric prior.  The transformer
predicts only the neural residual.

Before denoising, the model builds `37 x 3` opponent tile-bucket
queries:

```text
query(tile37, opponent bucket) = tile37 embedding + bucket embedding + unseen-count embedding
```

Here `tile37 embedding` is the same projected shared tile embedding used by the
37 denoising tokens, not a separate allocation-only tile id table.

These queries cross-attend to selected public encoder tokens:

- `player_info` tokens;
- current `sparse_melds` tokens, including owner seats;
- `visible_tile_counts` tokens;
- `progression` tokens.

The resulting `(37, 3, d_model)` context is computed once per observation and
then reused at each tile decoding step.  The residual wall bucket remains part
of the allocation tuple and remaining-capacity state, but it does not receive
its own cross-attended public memory context.  In multi-sample inference, the
encoder and this cross-attention cache are computed once for the input batch
before being repeated across samples.

Current sparse meld memory is a soft public-state signal.  It lets each opponent
tile-bucket query attend directly to owner-aligned chi/pon/kan structures without
adding hard allocation constraints beyond the existing unseen-count legality
mask.

The denoising decoder replaces the earlier autoregressive MLP, explicit
`partial_count37` / `partial_count34` inputs, and allocation-only tile id
embedding. Checkpoints from earlier sampler variants must be retrained.

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

Training uses random masking instead of teacher-forced tile-order decoding.  For
each batch, the model samples a cosine-distributed mask ratio, replaces that many
tile states with `[MASK]`, subtracts the unmasked ground-truth tuples from the
bucket capacities, and applies cross entropy only on masked tile positions.

For training, `BeliefAllocationDataset` can shuffle across more than one replay
file before yielding samples.  Set `shuffle_buffer_files` in the belief sampler
config to the number of half-game files each worker should accumulate before
shuffling and emitting samples.  The default is `1`, which preserves the original
per-file behavior.  Set `sample_keep_prob` to randomly keep only a fraction of
samples from each training shuffle buffer after it is filled.  This is useful
when nearby decision points share nearly identical hidden-hand targets.  The
default is `1.0`, which keeps all samples.

`riichienv_ml/configs/4p/belief_allocation.yml` uses:

```yaml
shuffle_buffer_files: 128
sample_keep_prob: 0.1
```

Training logs include both epoch-running metrics and recent-window metrics:

- `train/loss`, `train/tile_acc`: running averages from the start of the epoch.
- `train/window100_loss`, `train/window100_tile_acc`: averages since the previous
  100-step log line.
- validation can also log sampled-allocation diagnostics:
  `val/sample_allocation_legal_rate`,
  `val/sample_opponent_hand_size_exact_rate`, `val/sample_unique_rate`, and
  `val/sample_pairwise_l1_distance`.

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
During inference sampling, `sample_allocations` runs the transformer observation
encoder once for the input batch, then repeats the resulting context for the
denoising decoder.  It starts from all `[MASK]` states and runs `decode_steps`
iterations.  At each step, confidence selects the next tile positions in
parallel; candidate tuple sampling for those selected tiles is then applied in
confidence order while updating remaining bucket capacities.  This keeps the
sample legal without rerunning the transformer for each selected tile.

## Log Annotation

`BeliefLogSampler` runs the trained sampler on existing MJAI logs and writes
annotated copies.  The original logs are not modified.

```bash
uv run python riichienv-ml/scripts/sample_belief_logs.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/belief_log_sampling.yml
```

Input files are selected by sorted `input_glob` order and truncated to
`num_logs`.  By default, the sampler uses `skip_single_action=true`, so forced
single-action replay states are not annotated.

Actual MJAI decision events receive:

```json
"meta": {
  "belief_allocation": {
    "observer": 0,
    "opponent_seats": [1, 2, 3],
    "bucket_order": ["shimocha", "toimen", "kamicha"],
    "tile_format": "mjai",
    "tile_space": "tile37",
    "sample_count": 10,
    "samples": [
      [["1m", "5pr"], ["E"], ["9s"]]
    ]
  }
}
```

Each `samples[i]` contains three hidden hands in `opponent_seats` order.  The
residual wall bucket is intentionally omitted from log metadata.  Response
decisions that do not have their own MJAI event, such as `none`, are attached to
the preceding discard or kan event under `meta.belief_response_allocations`.
The log-sampling summary includes `sample_diagnostics` with the same sampled
allocation legality and diversity metrics used during validation.  Output logs
are still ordinary MJAI JSONL files with belief samples stored only in `meta`.
