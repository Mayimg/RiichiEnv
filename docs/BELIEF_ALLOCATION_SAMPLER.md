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
consume candidate-action tokens in its context encoder.  It also masks the
observer's hand token group to all padding before collation, because opponent
hidden-allocation prediction should not condition on private observer hand
composition.  The observer hand remains included in `visible_tile_counts`, so
the unseen tile pool and allocation constraints are unchanged.

The belief observation encoder also appends three opponent shanten-query tokens
after the public belief tokens.  Each query token is the sum of the shared
observer-relative seat embedding and a learned shanten-query embedding.  A
shared five-class head predicts opponent concealed-hand shanten as:

```text
0, 1, 2, 3, 4+
```

This head is trained as an auxiliary task from full-information training logs.

## Decoder

For each `(tile37 id, opponent)` cell, the sampler predicts the number of copies
of that tile in that opponent's concealed hand.  Each cell has five classes:
`0, 1, 2, 3, 4`.

The neural decoder is a MaskGIT-style denoising transformer over
`37 x 3 = 111` allocation-cell tokens plus three opponent shanten-condition
tokens.  Each allocation-cell token is the sum of:

```text
token = projected shared tile embedding
      + shared relative-seat embedding
      + current cell-state embedding
      + tile-remaining-count embedding
      + opponent remaining-slot embedding
```

The tile component reuses the observation encoder's shared, attribute-based
tile37 embedding table for the current public state, then projects it into the
decoder width.  This keeps decoder tile tokens in the same tile representation
space as encoder hand, visible-count, dora, progression, and meld tile fields.
The seat component reuses the observation encoder's shared observer-relative
seat embedding.

The current state is represented by the decided count `0..4`, with a dedicated
`[MASK]` state id.  The transformer uses bidirectional self-attention over the
111 allocation-cell tokens and the three shanten-condition tokens.  A
shanten-condition token is the sum of the shared observer-relative seat
embedding and the embedding for its 0/1/2/3/4+ shanten class.  Each decoder
layer cross-attends directly to cached public memory: the encoder CLS token plus
selected encoder memory tokens.

Every denoising step applies a legality mask.  A candidate count is allowed only
if:

- it does not exceed the remaining count of that tile;
- it does not exceed the remaining concealed-hand slots for that opponent;
- after selecting it, the remaining masked cells can still fill every opponent's
  remaining hand size.

The last condition is checked exactly for the three opponent seats using the
seven non-empty seat subsets.  For every subset `S`, the remaining demand of
seats in `S` must be no larger than the remaining tile capacity adjacent to at
least one still-masked cell in `S`.  This prevents MaskGIT sampling from walking
into a dead end.

Cells with exactly one legal candidate under these constraints are resolved
deterministically.  The sampler applies this forced-cell closure before the
first decoder pass and after sampled cell updates, so cells such as
zero-remaining tiles, opponents with no remaining concealed slots, and rare
single-source remaining allocations become decided count states instead of
staying as `[MASK]` targets.

The logit for cell `(tile t, opponent b)` and count `k` is:

```text
logit = log Hypergeom(k; U_b_avail, u_t_rem, N_b_rem) + neural_residual
```

where `u_t_rem` is the currently unallocated count for tile `t`,
`N_b_rem` is opponent `b`'s remaining hand slots, and `U_b_avail` is the sum of
`u_t_rem` over tile cells that are still masked for opponent `b`.  This is a
single-cell hypergeometric marginal conditioned on already decided cells.  The
transformer predicts only the neural residual.

The public memory that decoder cells cross-attend to includes:

- `player_info` tokens;
- current `sparse_melds` tokens, including owner seats;
- `visible_tile_counts` tokens;
- `progression` tokens.

The residual wall bucket is not decoded as tokens.  After all opponent cells are
sampled, wall counts are reconstructed as:

```text
x_wall[t] = unseen[t] - x_shimocha[t] - x_toimen[t] - x_kamicha[t]
```

In multi-sample inference, the observation encoder is computed once for the
input batch before its public memory is repeated across samples.

At inference time, true shanten classes are unknown.  The model samples one hard
shanten class per opponent and per allocation sample from the encoder head's
predicted distribution, then feeds those sampled classes into the decoder
condition tokens.  If an opponent is publicly in riichi, the sampled class is
overridden to 0 because riichi is a public tenpai guarantee.

Current sparse meld memory is a soft public-state signal.  It lets each opponent
allocation cell attend directly to owner-aligned chi/pon/kan structures without
adding hard allocation constraints beyond the existing unseen-count legality and
hand-size masks.

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
3. compute each opponent hidden hand's shanten class as `0, 1, 2, 3, 4+`;
4. compute unseen counts from visible counts;
5. set the wall target as the residual.

Training uses random masking instead of teacher-forced tile-order decoding.  For
each batch, the model samples a cosine-distributed mask ratio, replaces that many
allocation-cell states with `[MASK]`, subtracts the unmasked ground-truth counts
from tile and opponent remaining capacities, and applies cross entropy only on
masked cells.

During training, the decoder receives the true opponent shanten classes as its
three shanten-condition tokens.  The encoder shanten head is optimized with an
auxiliary cross-entropy loss controlled by `shanten_aux_loss_weight`.

After the random mask is sampled, training applies the same forced-cell closure
against the ground-truth allocation.  Any masked cell whose legal count is
already unique is unmasked and treated as a decided input state, so it does not
contribute cross entropy.  If closure removes every masked cell for a valid
sample, the mask is resampled up to a bounded retry count; samples that remain
fully unmasked after retry simply contribute no masked-cell loss.

For training, `BeliefAllocationDataset` can shuffle across more than one replay
file before yielding samples.  Set `shuffle_buffer_files` in the belief sampler
config to the number of half-game files each worker should accumulate before
shuffling and emitting samples.  The default is `1`, which preserves the original
per-file behavior.  Set `sample_keep_prob` to randomly keep only a fraction of
samples from each training shuffle buffer after it is filled.  This is useful
when nearby decision points share nearly identical hidden-hand targets.  The
default is `1.0`, which keeps all samples.

Training can also use `stratified_sample_keep_prob` to oversample decision
points with advanced target-opponent states.  When enabled, the dataset computes
a keep probability for each of the three opponents and keeps the decision using
the maximum value.  Riichi and meld-count states use fixed probabilities; closed
hands use a linear schedule by that opponent's discard count.  For ordinary
belief-allocation samples this replaces the global `sample_keep_prob`.

`riichienv_ml/configs/4p/belief_allocation.yml` uses:

```yaml
shuffle_buffer_files: 128
sample_keep_prob: 0.1
stratified_sample_keep_prob:
  enabled: true
  riichi: 0.4
  meld1: 0.1
  meld2: 0.2
  meld3plus: 0.5
  closed_start: 0.01
  closed_end: 0.3
  closed_start_discard_count: 0
  closed_end_discard_count: 20
```

Training logs include both epoch-running metrics and recent-window metrics:

- `train/loss`, `train/allocation_loss`, `train/cell_acc`: running averages from
  the start of the epoch.
- `train/shanten_loss`, `train/shanten_acc`: auxiliary shanten-head metrics when
  full-information shanten labels are available.
- `train/window100_loss`, `train/window100_cell_acc`: averages since the previous
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
encoder once for the input batch, then repeats the resulting public memory for
the denoising decoder.  It starts from all `[MASK]` states and runs
`decode_steps` iterations, after first resolving any cells that are already
forced by public counts and hand-size constraints.  For the cosine schedule
path, confidence selects the next allocation-cell positions in parallel; 5-way
count sampling for those selected cells is then applied in confidence order
while updating tile and opponent remaining capacities.  After each sampled cell
update, any newly forced cells are resolved before further decoding continues.
This keeps the sample legal without rerunning the transformer for each selected
cell.

The 4-player sampler has 111 allocation cells: 37 tile types for each of the 3
opponents.  The decoder also receives 3 shanten-condition tokens, but the
confidence schedule applies only to the 111 allocation cells.  When
`decode_steps=111`, the confidence schedule selects exactly one cell per decode
step.  Each step first chooses the next cell from still-masked cells using the
configured `confidence_method`, then samples that cell's count from its legal
5-way output while updating tile and opponent remaining capacities.  Other
`decode_steps` values keep the cosine schedule and may select zero or multiple
cells in a single decode step.

Supported confidence methods are:

- `max_prob`: select cells with the largest legal-output probability.
- `neg_entropy`: select cells with the lowest entropy over the 5-way output.
- `legal_normalized_entropy`: select cells by `1 - H(p) / log(M)`, where `M`
  is the number of legal count candidates for that cell.  This ranks cells by
  how peaked the distribution is relative to the legal candidate count.  Cells
  with `M=1` are forced decisions and receive a priority score above the normal
  `[0, 1]` range.

For repeated sampling from the same fixed observation, callers can split this
work explicitly:

```python
context = model.prepare_allocation_sampling_context(batch)
allocation = model.sample_allocations_from_context(context, num_samples=8)
```

The prepared context contains the encoder output, public-memory tensors, shanten
logits, and public riichi mask, so subsequent calls only rerun the allocation
decoder and per-sample shanten-class draw.

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

## Throughput Benchmark

`BeliefSamplingBenchmark` measures sampler throughput without writing sampled
allocations back to logs.  It replays input MJAI files in sorted `input_glob`
order, collects the same decision observations used by log annotation, and runs
fixed-observation sampling at each decision point.

```bash
uv run python riichienv-ml/scripts/benchmark_belief_sampling.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/belief_sampling_benchmark.yml
```

The default benchmark uses `samples_per_call: 256` and
`target_duration_ms: 100.0`.  Each decision measurement first runs the encoder
once and caches its public-memory context, then repeatedly samples from that
cached context until the target duration is reached.  This models the intended
AI usage pattern: one fixed observation produces many samples without rerunning
the observation encoder.  The reported elapsed time and samples/sec include that
single encoder-cache preparation plus all decoder sampling calls.  CUDA
measurements synchronize the device around timed calls so the reported
samples/sec includes actual GPU execution time.

Outputs:

- `summary.json`: config, device info, per-decision measurements, overall speed,
  and aggregates by `prog_len`, `prog_len` bucket, phase, observer, and log.
- `decisions.csv`: one row per measured decision, including `prog_len`,
  `sparse_meld_len`, actual elapsed time, encoder-cache preparation time,
  decoder elapsed time, number of calls, total samples, samples/sec, call
  durations, and peak CUDA memory when available.
- `prog_len.csv`: exact `prog_len` speed aggregates.
- `prog_len_buckets.csv`: bucketed `prog_len` speed aggregates for quick
  comparison of early, middle, and late hand states.

## Quality Evaluation

`BeliefAllocationEvaluator` evaluates sampled hidden allocations against the
full-information hands in ordinary MJAI logs.  It does not modify input logs or
the MJAI format.  The evaluator replays files in filename-sorted order, keeps a
deterministic subset of decision observations, samples `S` allocations for each
selected observation, and writes decision-level and opponent-level metrics.

```bash
uv run python riichienv-ml/scripts/evaluate_belief_allocation.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/belief_allocation_evaluation.yml
```

The default evaluation config uses `samples_per_decision: 64`,
`batch_size: 4`, and `sample_keep_prob: 0.002`.  This favors broad log coverage
over very large per-observation sample counts.  For quick model screening,
`samples_per_decision: 32` is usually enough; use `128` or `256` for deeper
final checks of diversity-sensitive metrics.

Primary metrics are opponent-only because online inference uses the sampled
opponent hands and fills the residual wall separately:

- legality and diversity: allocation legal rate, bucket exact rate, unique
  sample rate, pairwise L1, and samples/sec;
- raw distribution quality: fair sample Energy Score over opponent buckets in
  tile37, collapsed tile34, and dora/red-weighted tile37 spaces;
- mahjong meaning: opponent-hand shanten CRPS plus a compact semantic Energy
  Score over shanten, tenpai flags, dora/aka counts, suit counts, terminal/honor
  count, pair count, and triplet count.

Outputs:

- `report.md`: a compact Markdown report for quickly reading the run setup,
  sampler-health checks, overall scores, and tenpai/shanten bias by stratum.
- `summary.json`: config, selected files, overall metrics, sampling speed,
  stratified metrics, and shanten distributions.
- `decisions.csv`: one row per selected decision observation.
- `opponents.csv`: one row per target opponent per selected decision.
- `opponent_state.csv`: metrics stratified by `riichi`, `closed`, `meld1`,
  `meld2`, and `meld3plus`.
- `opponent_discard_count.csv`: metrics stratified by the target opponent's
  discard count.
- `opponent_state_discard_count.csv`: metrics stratified by both state and
  discard count.
- `shanten_distributions.csv`: true and sampled shanten distributions for the
  overall set and each stratum.

Use a fixed `seed`, `input_glob`, `num_logs`, `sample_keep_prob`, and
`samples_per_decision` when comparing models.  This keeps the decision subset
identical across runs, so model differences are not hidden by evaluation
sampling noise.

### Profiling

Use the PyTorch profiler script to inspect one fixed decision observation in
detail:

```bash
uv run python riichienv-ml/scripts/profile_belief_sampling.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/belief_sampling_benchmark.yml \
  --decision_index 0 \
  --samples_per_call 256 \
  --profile_calls 1
```

By default, profiling includes one encoder-context preparation followed by
sampling from the cached context, matching the benchmark's intended online
usage pattern.  The script writes:

- `*.table.txt`: aggregated PyTorch operator timings sorted by CUDA time when
  CUDA is active.
- `*.trace.json`: Chrome trace file for `chrome://tracing` or Perfetto.
- `*.metadata.json`: selected decision metadata, profiler settings, elapsed
  time, and output paths.

The sampler adds profiler ranges such as `belief/prepare_allocation_sampling_context`,
`belief/denoise_decoder`, `belief/apply_prior_legality`, and
`belief/cell_sampling_loop` so traces can be read at the model-operation level
instead of only as raw PyTorch operators.  Use `--no-profile_context` to exclude
encoder-context preparation from the profiled region when isolating decoder cost.
