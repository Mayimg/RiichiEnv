# Sequence Feature Encoding (Transformer)

This document describes the sequence feature encoding for transformer models, implemented in `riichienv-core/src/observation/sequence_features.rs` with a Python wrapper at `riichienv-ml/src/riichienv_ml/features/sequence_features.py`.

The encoding design is based on [Kanachan v3](https://github.com/Cryolite/kanachan/wiki/%5Bv3%5DNotes-on-Training-Data) as a subset — `Room` (5 values) and `Grade` (4x16=64 values) are removed since they are online-platform-dependent and unavailable via MJAI protocol.

## Overview

Unlike the CNN encoder (`obs.encode()`) which produces spatial `(C, 34)` tensors, the sequence feature encoding produces padded heterogeneous feature groups designed for embedding-based transformer architectures:

| Feature Group | Shape | Type | Description |
|---------------|-------|------|-------------|
| **Sparse** | `(9,)` | int64 | Table metadata, tiles remaining, and dora indicators |
| **Sparse Melds** | `(4, 9)` | int64 | Current player's melds in factorized meld layout |
| **Hand** | `(14, 2)` | int64 | Hand tiles as `(tile37, draw_state)` tuples |
| **Numeric** | `(6,)` | float32 | Continuous scalar features |
| **Progression** | `(256, 5)` | int64 | Action history as 5-tuple sequences |
| **Progression Melds** | `(256, 9)` | int64 | Factorized meld sidecar aligned with progression rows |
| **Candidates** | `(32, 2)` | int64 | Legal actions as 2-tuple sets |
| **Candidate Melds** | `(32, 9)` | int64 | Factorized meld sidecar aligned with candidate rows |

Each variable-length group is padded to its maximum length, with accompanying boolean masks indicating real vs. padding entries.

## Current Transformer Embedding Strategy

The current default transformer implementation (`riichienv-ml/src/riichienv_ml/models/transformer.py`) factorizes tile-only tokens with a **shared tile embedding module** and factorizes all melds with a **shared meld embedding module**.

### Shared tile attributes

For tile-only tokens, a tile embedding is built as:

```text
attribute embeddings -> concat -> linear projection
```

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

Notes:
- `tile34` collapses red fives onto their non-red 5 tile type.
- `dora_flag` is computed from the **current observation state**, not from the historical state at each progression event.
- Red fives are treated as `dora` for `dora_flag`.
- Meld slots use `tile37`, so red fives can be represented inside chi/pon/kan structures.

### Where the shared tile embedding is used

The shared tile embedding is applied to single-tile fields and to the tile slots inside the shared meld embedding:

| Feature group | Field / token range | Uses shared tile embedding |
|---------------|---------------------|----------------------------|
| Hand | `tile37` | Yes |
| Sparse | dora-indicator tokens (`79-263`) | Yes, with an extra dora-slot embedding |
| Progression | discard type range | Yes |
| Candidates | discard type range | Yes |
| Sparse / progression / candidate meld sidecars | 4 tile slots per meld | Yes, via shared meld embedding |
| Progression / candidates | pass / ron / tsumo / markers | No |

For sparse dora-indicator tokens, the model keeps the existing dora-slot distinction (`1st` indicator, `2nd` indicator, etc.) as a separate embedding and combines it with the shared tile embedding for the indicator tile itself.

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

**Vocabulary size: 265, max tokens: 9, padding index: 264**

Each observation produces 4-9 sparse tokens. Each token is an index into an embedding table.

| Offset | Count | Feature | Source |
|--------|-------|---------|--------|
| 0-1 | 2 | Game style (0=tonpuusen, 1=hanchan) | parameter |
| 2-4 | 3 | Chang / round wind (E/S/W) | `obs.round_wind` |
| 5-8 | 4 | Dealer relative to observer | `relative_seat(obs.player_id, obs.oya)` |
| 9-78 | 70 | Tiles remaining (0-69) | derived from visible tiles |
| 79-263 | 185 | Dora indicators (5 slots x 37 tiles) | `obs.dora_indicators` |
| 264 | 1 | Padding | - |

**Token composition per observation:**
- 3 fixed tokens (game style + round wind + dealer-relative)
- 1 tiles-remaining token
- 1-5 dora indicator tokens
- Total: typically 5-9 tokens

Current melds are encoded separately by `encode_seq_sparse_melds()` with the shared factorized meld layout.

### Rust API

```rust
obs.encode_seq_sparse(game_style: u8) -> Vec<u16>
```

### Python API (raw)

```python
sparse_bytes = obs.encode_seq_sparse(game_style=1)
sparse = np.frombuffer(sparse_bytes, dtype=np.uint16)  # variable length
```

## 1a. Meld Sidecar Features

**Tuple sequence, max 4 current meld entries; aligned sidecars for progression and candidates**

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

## 4. Progression Features (Action History)

**5-tuple sequence, max 256 entries (default)**

Each action from the kyoku start to the current decision point is encoded as a 5-tuple `(actor, type, moqie, liqi, from)`.

### Tuple Fields

| Field | Vocab | Values |
|-------|-------|--------|
| actor | 5 | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding/marker |
| type | 44 | see table below |
| moqie | 3 | 0=tedashi (hand tile), 1=tsumogiri (drawn tile), 2=N/A |
| liqi | 3 | 0=no riichi, 1=with riichi declaration, 2=N/A |
| from | 5 | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=N/A |

**Padding tuple:** `(4, 43, 2, 2, 4)`

### Type Encoding (44 values)

| Range | Count | Action | Encoding |
|-------|-------|--------|----------|
| 0 | 1 | Beginning-of-round marker | Fixed value |
| 1-37 | 37 | Discard | `1 + kan37(tile)` |
| 38 | 1 | Chi | Details in aligned progression meld row |
| 39 | 1 | Pon | Details in aligned progression meld row |
| 40 | 1 | Daiminkan | Details in aligned progression meld row |
| 41 | 1 | Ankan | Details in aligned progression meld row |
| 42 | 1 | Kakan | Details in aligned progression meld row |
| 43 | 1 | Padding | - |

### MJAI Event to Tuple Mapping

| Event | Tuple |
|-------|-------|
| `start_kyoku` | `(4, 0, 2, 2, 4)` |
| `dahai` | `(actor_rel, 1+kan37, moqie, liqi, 4)` |
| `chi` | `(actor_rel, 38, 2, 2, target_rel)` |
| `pon` | `(actor_rel, 39, 2, 2, target_rel)` |
| `daiminkan` | `(actor_rel, 40, 2, 2, target_rel)` |
| `ankan` | `(actor_rel, 41, 2, 2, 4)` |
| `kakan` | `(actor_rel, 42, 2, 2, 4)` |

- For `dahai`: `liqi=1` if preceded by a `reach` event from the same actor
- `tsumo`, `dora`, `reach_accepted` events are **not** included in progression
- `actor_rel` and `target_rel` are relative to the observing player, not to the event actor.

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

**3-tuple set, max 32 entries (default)**

Each legal action candidate is encoded as a 3-tuple `(type, moqie, from)`. The candidate list is collapsed by the strict candidate tuple plus its aligned meld sidecar row. Physical copies with identical visible semantics still collapse, but red-five discards, tedashi/tsumogiri discards, and red/non-red consumed chi/pon candidates remain distinct pointer targets.

### Tuple Fields

| Field | Vocab | Values |
|-------|-------|--------|
| type | 48 | see table below |
| moqie | 3 | 0=tedashi, 1=tsumogiri, 2=N/A |
| from | 5 | 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding |

**Padding tuple:** `(47, 2, 4)`

### Type Encoding (48 values)

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
| 47 | 1 | Padding | - |

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

`riichienv_ml.features.sequence_features.SequenceFeatureEncoder` provides padded torch tensors with masks.

### Usage

```python
from riichienv import RiichiEnv
from riichienv_ml.features.sequence_features import SequenceFeatureEncoder

env = RiichiEnv(game_mode="4p-red-half")
obs_dict = env.reset()
enc = SequenceFeatureEncoder(n_players=4, game_style=1)

for pid, obs in obs_dict.items():
    features = enc.encode(obs)
    # features["sparse"]      -- (9,) int64, padded with 264
    # features["sparse_melds"]-- (4, 9) int64, padded with (5, 37, 3, ...)
    # features["hand"]        -- (14, 2) int64, padded with (37, 2)
    # features["numeric"]     -- (6,) float32
    # features["progression"] -- (256, 5) int64, padded with (4, 43, 2, 2, 4)
    # features["prog_melds"]  -- (256, 9) int64, aligned with progression
    # features["candidates"]  -- (32, 3) int64, padded with (47, 2, 4)
    # features["cand_melds"]  -- (32, 9) int64, aligned with candidates
    # features["sparse_mask"] -- (9,) bool, True for real tokens
    # features["hand_mask"]   -- (14,) bool, True for real entries
    # features["prog_mask"]   -- (256,) bool, True for real entries
    # features["cand_mask"]   -- (32,) bool, True for real entries
```

### Constants

```python
SequenceFeatureEncoder.SPARSE_VOCAB_SIZE  # 265
SequenceFeatureEncoder.MAX_SPARSE_LEN     # 9
SequenceFeatureEncoder.MAX_SPARSE_MELDS   # 4
SequenceFeatureEncoder.MELD_DIMS           # (6, 38, 4, 38, 4, 38, 4, 38, 4)
SequenceFeatureEncoder.HAND_DIMS          # (38, 3)
SequenceFeatureEncoder.MAX_HAND_LEN       # 14
SequenceFeatureEncoder.MAX_PROG_LEN       # 256 (default; V1 compat: 512)
SequenceFeatureEncoder.MAX_CAND_LEN       # 32  (default; V1 compat: 64)
SequenceFeatureEncoder.NUM_NUMERIC         # 6
SequenceFeatureEncoder.PROG_DIMS           # (5, 44, 3, 3, 5)
SequenceFeatureEncoder.CAND_DIMS           # (48, 3, 5)
```

## Implementation

| File | Package | Description |
|------|---------|-------------|
| `riichienv-core/src/observation/sequence_features.rs` | riichienv-core | Rust encoding logic (~470 lines) |
| `riichienv-core/src/observation/python.rs` | riichienv-core | PyO3 bindings |
| `riichienv-ml/src/riichienv_ml/features/sequence_features.py` | riichienv-ml | Python wrapper |
