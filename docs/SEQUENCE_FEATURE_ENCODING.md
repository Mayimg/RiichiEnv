# Sequence Feature Encoding (Transformer)

This document describes the sequence feature encoding for transformer models, implemented in `riichienv-core/src/observation/sequence_features.rs` with a Python wrapper at `riichienv-ml/src/riichienv_ml/features/sequence_features.py`.

The encoding design is based on [Kanachan v3](https://github.com/Cryolite/kanachan/wiki/%5Bv3%5DNotes-on-Training-Data) as a subset — `Room` (5 values) and `Grade` (4x16=64 values) are removed since they are online-platform-dependent and unavailable via MJAI protocol.

## Overview

Unlike the CNN encoder (`obs.encode()`) which produces spatial `(C, 34)` tensors, the sequence feature encoding produces heterogeneous feature groups designed for embedding-based transformer architectures:

| Feature Group | Shape | Type | Description |
|---------------|-------|------|-------------|
| **Sparse** | `(8,)` | int64 | Table metadata, tiles remaining, and dora indicators |
| **Dealer** | `()` | int64 | Dealer seat relative to the observing player |
| **Player Stats** | `(4, 5)` | int64 | Per-player public summary tokens in observer-relative seat order |
| **Sparse Melds** | `(16, 9)` | int64 | Current visible melds for all players in factorized meld layout |
| **Sparse Meld Owners** | `(16,)` | int64 | Owner seats aligned with sparse meld rows |
| **Hand** | `(14, 2)` | int64 | Hand tiles as `(tile37, draw_state)` tuples |
| **Visible Tile Counts** | `(37, 2)` | int64 | Per-tile37 visible counts as `(tile37, visible_count)` tuples |
| **Numeric** | `(6,)` | float32 | Continuous scalar features |
| **Agari Overtakes** | `(4, 96, 4)` | float32 | Pairwise rank-overtake flags for standard 4P win patterns |
| **Progression** | `(P, 5)` | int64 | Action history and dora-reveal history as 5-tuple sequences |
| **Progression Melds** | `(P, 9)` | int64 | Factorized meld sidecar aligned with progression rows |
| **Candidates** | `(C, 3)` | int64 | Legal actions as 3-tuple sets |
| **Candidate Melds** | `(C, 9)` | int64 | Factorized meld sidecar aligned with candidate rows |

Progression and candidate groups stay variable-length for a single observation and are padded only to the maximum length inside the current batch. Their boolean masks indicate real vs. batch-padding rows. Smaller fixed-format groups such as sparse metadata, hand, and current visible melds keep their fixed padded shapes.

## Current Transformer Embedding Strategy

The current default transformer implementation (`riichienv-ml/src/riichienv_ml/models/transformer.py`) factorizes tile-only tokens with a **shared tile embedding module**, factorizes visible tile-count tokens with the shared tile embedding plus a count embedding shared by all tile types, factorizes all melds with a **shared meld embedding module**, routes all real relative-seat fields through a **shared relative-seat embedding module**, embeds four per-player public summary tokens, and reshapes agari-overtake features into four winner-relative-seat tokens. Sinusoidal positional encoding is applied only to progression tokens, starting at position 0 for the first progression row.

The feature vocabulary and external MJAI log format are independent of the projection implementation described here. The optimized transformer follows the encoded feature groups and vocabularies listed in this document, but evaluates several projections with fixed-shape table/gather operations to reduce small CUDA kernels and GPU-to-CPU synchronization points.

### Shared tile attributes

For tile-only tokens, a tile embedding is built as:

```text
attribute embeddings -> field-wise projected component sum -> LayerNorm
```

This is mathematically equivalent to the previous `Linear(concat(attributes)) + LayerNorm` form because the linear weight can be split by field and summed as `W1 e1 + W2 e2 + ... + b`.

with the following attributes:

| Attribute | Values |
|-----------|--------|
| `tile34` | `0-33`, padding |
| `suit` | `man / pin / sou / honor / padding` |
| `rank` | `1-9 / none / padding` |
| `honor_kind` | `E / S / W / N / white / green / red / none / padding` |
| `red_flag` | `normal / red / padding` |
| `tile_class` | `simple / terminal / wind / dragon / padding` |
| `dora_flag` | `dora / none / padding` |
| `wind_owner_relative_seat` | shared relative-seat embedding for whose seat wind the wind tile is; zero for non-winds/padding |
| `round_wind_flag` | `round wind / non-round wind / padding`; non-winds and padding use zero |

Notes:
- `tile34` collapses red fives onto their non-red 5 tile type.
- `dora_flag` is computed from the **current observation state**, not from the historical state at each progression event.
- Red fives are treated as `dora` for `dora_flag`.
- Progression includes explicit dora-reveal rows so the transformer can recover when each dora marker became visible while still using the current-state `dora_flag`.
- Wind tile ownership is computed as `(dealer_relative_seat + wind_index) % 4`, where `wind_index` is
  `0=East, 1=South, 2=West, 3=North`.
- `round_wind_flag` is computed only for wind tiles from the current round-wind sparse token.
- Meld slots use `tile37`, so red fives can be represented inside chi/pon/kan structures.

### Where the shared tile embedding is used

The shared tile embedding is applied to single-tile fields and to the tile slots inside the shared meld embedding:

| Feature group | Field / token range | Uses shared tile embedding |
|---------------|---------------------|----------------------------|
| Hand | `tile37` | Yes |
| Visible Tile Counts | `tile37` | Yes, with an extra shared count embedding |
| Sparse | dora-indicator tokens (`75-259`) | Yes, with an extra dora-slot embedding |
| Progression | discard and dora-reveal type ranges | Yes |
| Candidates | discard type range | Yes |
| Sparse / progression / candidate meld sidecars | 4 tile slots per meld | Yes, via shared meld embedding |
| Progression / candidates | pass / ron / tsumo / markers | No |

For sparse dora-indicator tokens, the model keeps the existing dora-slot distinction (`1st` indicator, `2nd` indicator, etc.) as a separate embedding and combines it with the shared tile embedding for the indicator tile itself.

### Shared relative-seat embedding

All real relative-seat values (`0=self`, `1=shimocha`, `2=toimen`, `3=kamicha`) share one base embedding table. Role-specific context is added for dealer, player-summary seat, progression actor, progression from, candidate from, sparse meld owner, agari winner, and wind-tile owner before projection to the required sub-dimension. Value `4` is treated as padding/N/A/marker and maps to zero in the seat module; the aligned type field or padding mask carries the special meaning.

At model forward time, the relative-seat module builds `role x seat` tables once for both sub-dimension and model-dimension outputs. Downstream feature groups gather from these tables instead of repeatedly projecting the same role/seat combinations.

### Projection execution

The Python model uses a split projection helper for feature groups that are conceptually `concat -> Linear -> LayerNorm`. It stores the same full linear weight and bias shape as the concatenated formulation, but applies the relevant weight slice to each field and sums the results before LayerNorm. For categorical fields, small projected tables are built from the embedding weights and then gathered by id.

Sparse melds, progression meld sidecars, and candidate meld sidecars are embedded in one fixed-shape shared meld pass per forward and then split back into their feature groups. Padding rows remain represented by the same `(5, 37, 3, 37, 3, 37, 3, 37, 3)` row and are selected with `torch.where`, so this changes execution shape rather than feature semantics.

## Tile Encodings

### kan37 (37 tiles, red fives distinct)

Used for discard, dora, drawn tile, and all meld tile slots.

| Range | Tiles |
|-------|-------|
| 0 | Red 5m |
| 1-9 | 1m-9m |
| 10 | Red 5p |
| 11-19 | 1p-9p |
| 20 | Red 5s |
| 21-29 | 1s-9s |
| 30-36 | E, S, W, N, P (white), F (green), C (red) |

Conversion from 136-tile ID:
- `tile_id == 16` -> 0 (red 5m)
- `tile_id == 52` -> 10 (red 5p)
- `tile_id == 88` -> 20 (red 5s)
- Otherwise: `tile_type = tile_id / 4`, then `tile_type + 1` (man), `+2` (pin), `+3` (sou/honor)

### Relative Seat

All seat fields used by the transformer are observer-relative:

`(target - observer + 4) % 4`:
- 0 = self
- 1 = shimocha (right / downstream)
- 2 = toimen (across)
- 3 = kamicha (left / upstream)
- 4 = N/A or padding, where supported

External MJAI logs still use absolute seats. The conversion to observer-relative seats happens only inside the sequence feature encoder.

## 1. Sparse Features

**Vocabulary size: 261, max tokens: 8, padding index: 260**

Each observation produces 3-8 sparse tokens. Each token is an index into an embedding table. Dealer is encoded separately as a relative-seat scalar so it can share the relative-seat embedding with other seat fields.

| Offset | Count | Feature | Source |
|--------|-------|---------|--------|
| 0-1 | 2 | Game style (0=tonpuusen, 1=hanchan) | parameter |
| 2-4 | 3 | Chang / round wind (E/S/W) | `obs.round_wind` |
| 5-74 | 70 | Tiles remaining (0-69) | derived from visible tiles |
| 75-259 | 185 | Dora indicators (5 slots x 37 tiles) | `obs.dora_indicators` |
| 260 | 1 | Padding | - |

**Token composition per observation:**
- 2 fixed tokens (game style + round wind)
- 1 tiles-remaining token
- 1-5 dora indicator tokens
- Total: typically 4-8 tokens

### Dealer Feature

Dealer is encoded as one scalar:

| Field | Values | Source |
|-------|--------|--------|
| dealer | 0=self, 1=shimocha, 2=toimen, 3=kamicha | `relative_seat(obs.player_id, obs.oya)` |

### Player Stats Feature

Player stats are encoded as four always-present rows in observer-relative seat order:
`self`, `shimocha`, `toimen`, `kamicha`.

Each row is:

```text
(relative_seat, riichi_active, meld_count, discard_count, tedashi_count)
```

| Field | Values | Source |
|-------|--------|--------|
| `relative_seat` | 0=self, 1=shimocha, 2=toimen, 3=kamicha | row owner |
| `riichi_active` | 0=no, 1=yes | `riichi_declared || riichi_stage` |
| `meld_count` | 0-4 | current meld count, including closed kan; kakan does not add a second meld |
| `discard_count` | 0-12 | all discards, clipped at 12 |
| `tedashi_count` | 0-12 | hand discards, clipped at 12 |

For `tedashi_count`, a discard with MJAI `tsumogiri=false` is counted as tedashi.
Chi and pon are followed by a discard without an intervening draw, so their post-call discard is naturally encoded as tedashi. Kans are followed by a rinshan draw; the subsequent discard is treated as a normal drawn-turn discard. External MJAI logs are unchanged; these rows are internal observation features derived from public state.

Current visible melds are encoded separately by `encode_seq_sparse_melds()` with the shared factorized meld layout.
For training throughput, the Python wrapper reads the bundled `encode_seq_sparse_meld_features()` API,
which returns each 9-field meld row plus its owner sidecar as a 10-field row.
Sparse meld rows include all players' current melds in observer-relative owner order.

### Rust API

```rust
obs.encode_seq_sparse(game_style: u8) -> Vec<u16>
obs.encode_seq_dealer() -> u16
obs.encode_seq_player_stats() -> Vec<[u16; 5]>
```

### Python API (raw)

```python
sparse_bytes = obs.encode_seq_sparse(game_style=1)
sparse = np.frombuffer(sparse_bytes, dtype=np.uint16)  # variable length
dealer = obs.encode_seq_dealer()
player_stats = np.frombuffer(obs.encode_seq_player_stats(), dtype=np.uint16).reshape(4, 5)
sparse_meld_features = np.frombuffer(
    obs.encode_seq_sparse_meld_features(), dtype=np.uint16
).reshape(-1, 10)
melds = sparse_meld_features[:, :9]
owners = sparse_meld_features[:, 9]
```

## 1a. Meld Sidecar Features

**Tuple sequence, max 16 current meld entries; aligned sidecars for progression and candidates**

Each meld row has 9 fields:

```text
(kind, slot0_tile37, slot0_role, slot1_tile37, slot1_role,
       slot2_tile37, slot2_role, slot3_tile37, slot3_role)
```

| Field | Values |
|-------|--------|
| kind | 0=chi, 1=pon, 2=daiminkan, 3=ankan, 4=kakan, 5=padding |
| slot tile | 0-36=tile37, 37=padding |
| slot role | 0=called, 1=consumed, 2=added_tile, 3=padding |

**Padding row:** `(5, 37, 3, 37, 3, 37, 3, 37, 3)`

Sparse current meld rows also have an aligned owner sidecar:

| Field | Values |
|-------|--------|
| sparse_meld_owner | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding |

Rows are ordered by owner relative to the observing player:
`self`, `shimocha`, `toimen`, `kamicha`, with each owner contributing up to 4 melds in state order.
The transformer embeds each row with the shared meld embedding and adds the shared relative-seat embedding with the `meld_owner` role only for sparse current melds.
Progression and candidate meld sidecars keep the 9-field meld row and do not include an owner sidecar.

Slot order:

| Meld Kind | slot_0 | slot_1 | slot_2 | slot_3 |
|-----------|--------|--------|--------|--------|
| chi | called | consumed | consumed | padding |
| pon | called | consumed | consumed | padding |
| daiminkan | called | consumed | consumed | consumed |
| ankan | consumed | consumed | consumed | consumed |
| kakan | added_tile | original pon called | original pon consumed | original pon consumed |

For kakan, `Meld.added_tile` preserves the added tile so current sparse melds can distinguish it from the original pon.

## 2. Hand Features

**Tuple sequence, max 14 entries**

Each hand tile is encoded as a 2-tuple `(tile37, draw_state)`.

| Field | Vocab | Values |
|-------|-------|--------|
| tile37 | 38 | 0-36 = kan37 tile, 37 = padding |
| draw_state | 3 | 0=concealed, 1=drawn, 2=padding |

**Padding tuple:** `(37, 2)`

The sequence is ordered as:
- concealed tiles in hand order
- the optional drawn tile last

### Rust API

```rust
obs.encode_seq_hand() -> Vec<[u16; 2]>
```

### Python API (raw)

```python
hand_bytes = obs.encode_seq_hand()
hand = np.frombuffer(hand_bytes, dtype=np.uint16).reshape(-1, 2)  # variable length
```

## 2a. Visible Tile Count Features

**Fixed 37 tuple tokens**

Each tile37 id is encoded as a 2-tuple `(tile37, visible_count)`.

| Field | Vocab | Values |
|-------|-------|--------|
| tile37 | 37 | 0-36 = kan37 tile |
| visible_count | 5 | 0-4 visible copies, clipped at 4 |

Rows are always ordered by `tile37 = 0..36`. Counts are tile37-specific, so red fives are counted separately from normal fives. For example, red 5m (`tile37=0`) has its own count and normal 5m (`tile37=5`) has a separate count; no combined "all 5m" count is included.

Visible counts include:

- the observing player's hand
- all players' visible meld tiles
- all players' discards
- currently visible dora indicators

For chi/pon/daiminkan/kakan, the claimed tile is counted from the discard river and skipped once inside the meld row, so a consumed/called tile is not double-counted. For kakan, the added tile is counted as visible. Dora indicators count only the indicator tiles currently revealed in the observation, not hidden kan-dora or ura-dora indicators.

The transformer embeds each row by combining the shared tile embedding for `tile37` with a shared `visible_count` embedding:

```text
shared tile embedding + count embedding -> field-wise projected component sum -> LayerNorm
```

External MJAI logs are unchanged; this is an internal observation feature derived from the current visible state.

### Rust API

```rust
obs.encode_seq_visible_tile_counts() -> Vec<[u16; 2]>  // length 37
```

### Python API (raw)

```python
count_bytes = obs.encode_seq_visible_tile_counts()
visible_counts = np.frombuffer(count_bytes, dtype=np.uint16).reshape(37, 2)
```

## 3. Numeric Features

**Fixed: 6 floats**

| Index | Feature | Source |
|-------|---------|--------|
| 0 | Honba (current) | `obs.honba` |
| 1 | Riichi deposits (current) | `obs.riichi_sticks` |
| 2 | Normalized score (self) | `(obs.scores[player_id] - 25000) / 10000` |
| 3 | Normalized score (right / shimocha) | `(obs.scores[(player_id+1)%4] - 25000) / 10000` |
| 4 | Normalized score (across / toimen) | `(obs.scores[(player_id+2)%4] - 25000) / 10000` |
| 5 | Normalized score (left / kamicha) | `(obs.scores[(player_id+3)%4] - 25000) / 10000` |

Round-start numeric features are intentionally omitted. The transformer policy uses the current score state for action decisions.

### Rust API

```rust
obs.encode_seq_numeric() -> [f32; 6]
```

### Python API (raw)

```python
numeric_bytes = obs.encode_seq_numeric()
numeric = np.frombuffer(numeric_bytes, dtype=np.float32)  # shape (6,)
```

## 3a. Agari Overtake Features

**Fixed: 4 x 96 x 4 floats**

These features summarize the current score situation as pairwise rank-overtake
flags under the same standard Tenhou 4P non-PAO settlement patterns used by the
GRP rank-prediction model.

| Axis | Size | Meaning |
|------|------|---------|
| winner_relative_seat | 4 | 0=self, 1=shimocha, 2=toimen, 3=kamicha |
| standard_agari_pattern | 96 | 24 tsumo patterns, then 3 ron-target blocks x 24 ron patterns |
| target_relative_seat | 4 | 0=self, 1=shimocha, 2=toimen, 3=kamicha |

A feature is `1.0` when the winner starts below the target in current rank and
finishes above the target after that settlement pattern. It is otherwise `0.0`.
The diagonal `winner_relative_seat == target_relative_seat` is always `0.0`.
Ranks use the engine's stable seat-order tie-break rule. Dealer/non-dealer
payments, honba, and riichi deposits are included in the settlement simulation.

Pattern ordering matches `riichienv-core/src/grp.rs`:

- indices `0-23`: tsumo patterns
- indices `24-95`: ron patterns, grouped by absolute target seat order while
  skipping the winner

External MJAI logs are unchanged. This is an internal observation feature
derived from current scores, dealer, honba, and riichi deposits.

In the transformer, the flat 1536-float vector is reshaped to four tokens:

```text
(winner_relative_seat=0..3, 96 patterns x 4 targets)
```

Each winner token uses a shared `Linear(384 -> d_model)` projection plus the
shared relative-seat embedding for that winner seat. The packed feature layout
is still flat for Ray worker compatibility.

### Rust API

```rust
obs.encode_seq_agari_overtakes() -> Vec<f32>  // length 1536
```

### Python API (raw)

```python
agari_bytes = obs.encode_seq_agari_overtakes()
agari = np.frombuffer(agari_bytes, dtype=np.float32).reshape(4, 96, 4)
```

## 4. Progression Features (Action History)

**Variable-length 5-tuple sequence**

Each action or dora reveal from the kyoku start to the current decision point is encoded as a 5-tuple `(actor, type, moqie, liqi, from)`.

### Tuple Fields

| Field | Vocab | Values |
|-------|-------|--------|
| actor | 5 | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding/marker |
| type | 80 | see table below |
| moqie | 3 | 0=tedashi (hand tile), 1=tsumogiri (drawn tile), 2=N/A |
| liqi | 3 | 0=no riichi, 1=with riichi declaration, 2=N/A |
| from | 5 | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=N/A |

Batch padding uses `(4, 0, 2, 2, 4)` and is ignored by `prog_mask`.

### Type Encoding (80 values)

| Range | Count | Action | Encoding |
|-------|-------|--------|----------|
| 0 | 1 | Beginning-of-round marker | Fixed value |
| 1-37 | 37 | Discard | `1 + kan37(tile)` |
| 38 | 1 | Chi | Details in aligned progression meld row |
| 39 | 1 | Pon | Details in aligned progression meld row |
| 40 | 1 | Daiminkan | Details in aligned progression meld row |
| 41 | 1 | Ankan | Details in aligned progression meld row |
| 42 | 1 | Kakan | Details in aligned progression meld row |
| 43-79 | 37 | Dora reveal | `43 + kan37(dora_marker)` |

### MJAI Event to Tuple Mapping

| Event | Tuple |
|-------|-------|
| `start_kyoku` | `(4, 0, 2, 2, 4)`, then `(4, 43+kan37(dora_marker), 2, 2, 4)` when `dora_marker` is present |
| `dahai` | `(actor_rel, 1+kan37, moqie, liqi, 4)` |
| `chi` | `(actor_rel, 38, 2, 2, target_rel)` |
| `pon` | `(actor_rel, 39, 2, 2, target_rel)` |
| `daiminkan` | `(actor_rel, 40, 2, 2, target_rel)` |
| `ankan` | `(actor_rel, 41, 2, 2, 4)` |
| `kakan` | `(actor_rel, 42, 2, 2, 4)` |
| `dora` | `(4, 43+kan37(dora_marker), 2, 2, 4)` |

- For `dahai`: `liqi=1` if preceded by a `reach` event from the same actor
- `tsumo`, `reach_accepted` events are **not** included in progression
- `actor_rel` and `target_rel` are relative to the observing player, not to the event actor.
- Dora reveal rows use the external MJAI `dora_marker` tile. They do not change the MJAI log format; this is only an internal sequence-feature expansion.

### Rust API

```rust
obs.encode_seq_progression() -> Vec<[u16; 5]>
```

### Python API (raw)

```python
prog_bytes = obs.encode_seq_progression()
prog = np.frombuffer(prog_bytes, dtype=np.uint16).reshape(-1, 5)  # variable length
prog_melds = np.frombuffer(obs.encode_seq_progression_melds(), dtype=np.uint16).reshape(-1, 9)
```

## 5. Candidate Features (Legal Actions)

**Variable-length 3-tuple set**

Each legal action candidate is encoded as a 3-tuple `(type, moqie, from)`. The candidate list is collapsed by the strict candidate tuple plus its aligned meld sidecar row. Physical copies with identical visible semantics still collapse, but red-five discards, tedashi/tsumogiri discards, and red/non-red consumed chi/pon candidates remain distinct pointer targets.

### Tuple Fields

| Field | Vocab | Values |
|-------|-------|--------|
| type | 47 | see table below |
| moqie | 3 | 0=tedashi, 1=tsumogiri, 2=N/A |
| from | 5 | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding |

Batch padding uses `(42, 2, 4)` and is ignored by `cand_mask`.

### Type Encoding (47 values)

| Range | Count | Action | Encoding |
|-------|-------|--------|----------|
| 0-36 | 37 | Discard | `kan37(tile)`; red fives are separate from normal fives |
| 37 | 1 | Riichi | Fixed |
| 38 | 1 | Ankan | Details in aligned candidate meld row |
| 39 | 1 | Kakan | Details in aligned candidate meld row |
| 40 | 1 | Tsumo (win) | Fixed |
| 41 | 1 | Kyushu kyuhai (9 terminals draw) | Fixed |
| 42 | 1 | Pass | Fixed |
| 43 | 1 | Chi | Details in aligned candidate meld row |
| 44 | 1 | Pon | Details in aligned candidate meld row |
| 45 | 1 | Daiminkan | Details in aligned candidate meld row |
| 46 | 1 | Ron (win) | Fixed |

For chi/pon, the tuple may be identical between red-consume and non-red-consume variants. That distinction is represented by the aligned candidate meld row, whose tile slots use `tile37`.

### Rust API

```rust
obs.encode_seq_candidates() -> Vec<[u16; 3]>
```

### Python API (raw)

```python
cand_bytes = obs.encode_seq_candidates()
cand = np.frombuffer(cand_bytes, dtype=np.uint16).reshape(-1, 3)  # variable length
cand_melds = np.frombuffer(obs.encode_seq_candidate_melds(), dtype=np.uint16).reshape(-1, 9)
```

## Python Wrapper: SequenceFeatureEncoder

`riichienv_ml.features.sequence_features.SequenceFeatureEncoder` provides torch tensors with masks. `collate_sequence_features()` pads progression and candidate groups to the maximum length in the current batch.

BC training uses `pack_sequence_features()` as the DataLoader fast path. It applies the same batch-local P/C padding, then flattens the batch into one `float32` tensor plus `(prog_len, cand_len)` metadata so worker IPC, pinned-memory handling, and host-to-device transfer do not move many small tensors per batch. The transformer accepts both this packed tuple and the collated dictionary form.

`SequenceFeaturePackedEncoder` remains as a deprecated config alias, but it now returns the same dynamic feature dictionary rather than a flat packed tensor.

### Usage

```python
from riichienv import RiichiEnv
from riichienv_ml.features.sequence_features import SequenceFeatureEncoder

env = RiichiEnv(game_mode="4p-red-half")
obs_dict = env.reset()
enc = SequenceFeatureEncoder(n_players=4, game_style=1)

for pid, obs in obs_dict.items():
    features = enc.encode(obs)
    # features["sparse"]      -- (8,) int64, padded with 260
    # features["dealer"]      -- () int64, relative dealer seat
    # features["player_stats"]-- (4, 5) int64, per-player public summaries
    # features["sparse_melds"]-- (16, 9) int64, padded with (5, 37, 3, ...)
    # features["sparse_meld_owners"]-- (16,) int64, padded with 4
    # features["hand"]        -- (14, 2) int64, padded with (37, 2)
    # features["visible_tile_counts"] -- (37, 2) int64, rows (tile37, visible_count)
    # features["numeric"]     -- (6,) float32
    # features["agari_overtakes"] -- (1536,) float32, reshapeable to (4, 96, 4)
    # features["progression"] -- (P, 5) int64
    # features["prog_melds"]  -- (P, 9) int64, aligned with progression
    # features["candidates"]  -- (C, 3) int64
    # features["cand_melds"]  -- (C, 9) int64, aligned with candidates
    # features["sparse_mask"] -- (8,) bool, True for real tokens
    # features["hand_mask"]   -- (14,) bool, True for real entries
    # features["prog_mask"]   -- (P,) bool, True for real entries
    # features["cand_mask"]   -- (C,) bool, True for real entries
```

### Constants

```python
SequenceFeatureEncoder.SPARSE_VOCAB_SIZE  # 261
SequenceFeatureEncoder.MAX_SPARSE_LEN     # 8
SequenceFeatureEncoder.DEALER_DIMS         # 4
SequenceFeatureEncoder.PLAYER_INFO_DIMS    # (4, 2, 5, 13, 13)
SequenceFeatureEncoder.PLAYER_INFO_TOKENS  # 4
SequenceFeatureEncoder.PLAYER_INFO_WIDTH   # 5
SequenceFeatureEncoder.MAX_SPARSE_MELDS   # 16
SequenceFeatureEncoder.MELD_DIMS           # (6, 38, 4, 38, 4, 38, 4, 38, 4)
SequenceFeatureEncoder.SPARSE_MELD_FEATURE_WIDTH  # 10
SequenceFeatureEncoder.SPARSE_MELD_OWNER_DIMS  # 5
SequenceFeatureEncoder.HAND_DIMS          # (38, 3)
SequenceFeatureEncoder.MAX_HAND_LEN       # 14
SequenceFeatureEncoder.VISIBLE_TILE_COUNT_DIMS    # (37, 5)
SequenceFeatureEncoder.VISIBLE_TILE_COUNT_TOKENS  # 37
SequenceFeatureEncoder.VISIBLE_TILE_COUNT_WIDTH   # 2
SequenceFeatureEncoder.NUM_NUMERIC         # 6
SequenceFeatureEncoder.AGARI_OVERTAKE_DIMS # (4, 96, 4)
SequenceFeatureEncoder.AGARI_OVERTAKE_DIM  # 1536
SequenceFeatureEncoder.PROG_DIMS           # (5, 80, 3, 3, 5)
SequenceFeatureEncoder.CAND_DIMS           # (47, 3, 5)
```

## Implementation

| File | Package | Description |
|------|---------|-------------|
| `riichienv-core/src/observation/sequence_features.rs` | riichienv-core | Rust encoding logic (~470 lines) |
| `riichienv-core/src/observation/python.rs` | riichienv-core | PyO3 bindings |
| `riichienv-ml/src/riichienv_ml/features/sequence_features.py` | riichienv-ml | Python wrapper |
