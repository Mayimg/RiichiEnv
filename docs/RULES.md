# Game Rules Configuration

Detailed game mechanics can be configured using the `GameRule` struct.

> [!NOTE]
> Configurable rule options are still under development and not yet exhaustive. If you find missing rules or have suggestions for additional configuration options, please report them via [GitHub Issues #43](https://github.com/smly/RiichiEnv/issues/43).

## "Responsibility Payment" behavior for composite Yakumans

- **Tenhou**: Pao player pays the full amount for Tsumo (including non-Pao Yakumans) and splits the full amount for Ron.
- **Mahjong Soul**: Pao player is liable only for the Pao-triggering Yakuman part. The remaining Yakumans are treated as normal settlement (split among all for Tsumo, or paid by deal-in player for Ron).

| Flag | Description |
|------|-------------|
| `.yakuman_pao_is_liability_only` | Whether to limit Pao liability to the specific Pao-triggering Yakuman only (Mahjong Soul style). If false, Pao covers the full amount (Tenhou style). |

## Double Ron

| Flag | Description |
|------|-------------|
| `.allow_double_ron` | Whether to allow double ron (two players declaring Ron on the same discard). |

## Kuikae (Swap Calling) Mode

Controls whether players can discard a tile that completes the same sequence they just called.

| Mode | Description |
|------|-------------|
| `KuikaeMode.None` | No kuikae restriction. |
| `KuikaeMode.Basic` | Basic kuikae restriction (cannot discard the called tile). |
| `KuikaeMode.StrictFlank` | Strict kuikae restriction including flank tiles (Tenhou/MJSoul standard). |

## Kokushi Musou Rules

| Flag | Description |
|------|-------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | Whether to allow Ron on a closed Kan (Ankan) for Kokushi Musou (Chankan). |
| `.is_kokushi_musou_13machi_double` | Whether to treat Kokushi Musou 13-sided wait as a Double Yakuman. |

## Kan Dora Reveal Timing

Controls when dora indicators are revealed after a Kan (quad) declaration.

| Mode | Description |
|------|-------------|
| `KanDoraTimingMode.TenhouImmediate` | **Tenhou style**: Ankan reveals dora before rinshan tsumo (rinshan kaihou includes kan dora). Daiminkan/Kakan reveal dora before discard (Ron includes kan dora, but rinshan kaihou does not). |
| `KanDoraTimingMode.MajsoulImmediate` | **Mahjong Soul style (段位戦ルール)**: Ankan reveals dora immediately after kan declaration (即めくり). Daiminkan/Kakan reveals dora after the discard, in chronological order of kan declarations. |
| `KanDoraTimingMode.AfterDiscard` | Same as TenhouImmediate. Ankan before rinshan tsumo, Daiminkan/Kakan before discard. |

### Event Order Examples

**TenhouImmediate / AfterDiscard - Ankan (Closed Kan)**
```
ankan → dora → tsumo (rinshan) → dahai
```
Note: Rinshan kaihou (winning on rinshan draw) includes the kan dora.

**TenhouImmediate / AfterDiscard - Kakan/Daiminkan (Open/Added Kan)**
```
kakan → tsumo (rinshan) → dahai → dora
daiminkan → tsumo (rinshan) → dahai → dora
```
Note: Ron on the discard includes the kan dora. Rinshan kaihou does not include the kan dora.

**MajsoulImmediate - Ankan (Closed Kan)**
```
ankan → dora → tsumo (rinshan) → dahai
```

**MajsoulImmediate - Kakan/Daiminkan (Open/Added Kan)**
```
kakan → tsumo (rinshan) → dahai → dora
daiminkan → tsumo (rinshan) → dahai → dora
```

### Usage

```python
from riichienv import RiichiEnv, GameRule, KanDoraTimingMode

# Tenhou timing (default for default_tenhou())
rule = GameRule(kan_dora_timing=KanDoraTimingMode.TenhouImmediate)
env = RiichiEnv(rule=rule)

# Mahjong Soul timing (default for default_mjsoul())
rule = GameRule(kan_dora_timing=KanDoraTimingMode.MajsoulImmediate)
env = RiichiEnv(rule=rule)

# All kans reveal after discard
rule = GameRule(kan_dora_timing=KanDoraTimingMode.AfterDiscard)
env = RiichiEnv(rule=rule)
```

## Three-Player (Sanma) Rules

| Flag | Description |
|------|-------------|
| `.is_sanma` | Whether the game is a 3-player (sanma) game. |
| `.allow_kita` | Whether Kita (BaBei / 北抜き) declarations are allowed. |
| `.sanma_tsumo_zon` | Whether tsumo-zon (ツモ損) scoring is used. When enabled, the non-dealing winner pays the tsumo amount without the absent player's share being distributed. |

## Platform-Specific Rule Presets

Differences in standard ranked match rules across major platforms.

### 4-Player Presets

| Flag | `default_tenhou()` | `default_mjsoul()` |
|------|--------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` |
| `.yakuman_pao_is_liability_only` | `False` | `True` |
| `.allow_double_ron` | `True` | `True` |
| `.kuikae_mode` | `StrictFlank` | `StrictFlank` |
| `.kan_dora_timing` | `TenhouImmediate` | `MajsoulImmediate` |
| `.is_sanma` | `False` | `False` |
| `.allow_kita` | `False` | `False` |
| `.sanma_tsumo_zon` | `False` | `False` |

### 3-Player (Sanma) Presets

| Flag | `default_tenhou_sanma()` | `default_mjsoul_sanma()` |
|------|--------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` |
| `.yakuman_pao_is_liability_only` | `False` | `True` |
| `.allow_double_ron` | `True` | `True` |
| `.kuikae_mode` | `StrictFlank` | `StrictFlank` |
| `.kan_dora_timing` | `TenhouImmediate` | `MajsoulImmediate` |
| `.is_sanma` | `True` | `True` |
| `.allow_kita` | `True` | `True` |
| `.sanma_tsumo_zon` | `True` | `True` |
