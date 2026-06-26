//! Sequence feature encoding for transformer models.
//!
//! Produces sparse tokens, numeric features, progression (action and dora-reveal history),
//! and candidate (legal action) features suitable for embedding-based
//! transformer architectures.
//!
//! Based on the Kanachan v3 encoding (subset — Room and Grade removed):
//! <https://github.com/Cryolite/kanachan/wiki/%5Bv3%5DNotes-on-Training-Data>

use crate::action::{Action, ActionType};
use crate::parser::mjai_to_tid;
use crate::types::{Meld, MeldType};

use super::Observation;

// ── Constants ────────────────────────────────────────────────────────────────
// These constants are `pub` for Python-side consumption (see riichienv-ml).
#[allow(dead_code)]
pub const SPARSE_VOCAB_SIZE: usize = 261;
#[allow(dead_code)]
pub const SPARSE_PAD: u16 = 260;
pub const MAX_SPARSE_LEN: usize = 8;

/// Dealer relative seat dimension: 0=self, 1=shimocha, 2=toimen, 3=kamicha.
#[allow(dead_code)]
pub const DEALER_DIMS: u16 = 4;

/// Player summary tuple dimensions:
/// (relative_seat, riichi_active, meld_count, discard_count, tedashi_count).
#[allow(dead_code)]
pub const PLAYER_INFO_DIMS: [u16; 5] = [4, 2, 5, 21, 21];
#[allow(dead_code)]
pub const PLAYER_INFO_TOKENS: usize = 4;
#[allow(dead_code)]
pub const PLAYER_INFO_WIDTH: usize = 5;

/// Hand tuple dimensions: (tile37, draw_state)
#[allow(dead_code)]
pub const HAND_DIMS: [u16; 2] = [38, 3];
pub const MAX_HAND_LEN: usize = 14;
#[allow(dead_code)]
pub const HAND_PAD: [u16; 2] = [37, 2];

/// Visible tile count tuple dimensions: (tile37, visible_count)
#[allow(dead_code)]
pub const VISIBLE_TILE_COUNT_DIMS: [u16; 2] = [37, 5];
#[allow(dead_code)]
pub const VISIBLE_TILE_COUNT_TOKENS: usize = 37;
#[allow(dead_code)]
pub const VISIBLE_TILE_COUNT_WIDTH: usize = 2;

/// Progression tuple dimensions: (actor, type, moqie, liqi, from)
#[allow(dead_code)]
pub const PROG_DIMS: [u16; 5] = [5, 80, 3, 3, 5];
#[allow(dead_code)]
pub const PROG_PAD: [u16; 5] = [4, 0, 2, 2, 4];

/// Candidate tuple dimensions: (type, moqie, from)
#[allow(dead_code)]
pub const CAND_DIMS: [u16; 3] = [47, 3, 5];
#[allow(dead_code)]
pub const CAND_PAD: [u16; 3] = [42, 2, 4];

/// Meld feature row: (kind, slot0_tile37, slot0_role, ..., slot3_tile37, slot3_role)
#[allow(dead_code)]
pub const MELD_FEATURE_WIDTH: usize = 9;
#[allow(dead_code)]
pub const CANDIDATE_FEATURE_WIDTH: usize = 3 + MELD_FEATURE_WIDTH;
#[allow(dead_code)]
pub const SPARSE_MELD_FEATURE_WIDTH: usize = MELD_FEATURE_WIDTH + 1;
#[allow(dead_code)]
pub const MELD_KIND_DIMS: u16 = 6;
#[allow(dead_code)]
pub const MELD_ROLE_DIMS: u16 = 4;
#[allow(dead_code)]
pub const MAX_SPARSE_MELDS: usize = 16;
#[allow(dead_code)]
pub const MELD_PAD: [u16; MELD_FEATURE_WIDTH] = [5, 37, 3, 37, 3, 37, 3, 37, 3];
#[allow(dead_code)]
pub const SPARSE_MELD_OWNER_DIMS: u16 = 5;
#[allow(dead_code)]
pub const SPARSE_MELD_OWNER_PAD: u16 = 4;

pub const NUM_NUMERIC: usize = 6;
#[allow(dead_code)]
pub const AGARI_OVERTAKE_FEATURE_DIM: usize = crate::grp::TENHOU_4P_PAIRWISE_OVERTAKE_DIM;

const SCORE_NORM_BASE: f32 = 25000.0;
const SCORE_NORM_SCALE: f32 = 10000.0;

const SPARSE_DORA_OFFSET: u16 = 75;
const MELD_KIND_CHI: u16 = 0;
const MELD_KIND_PON: u16 = 1;
const MELD_KIND_DAIMINKAN: u16 = 2;
const MELD_KIND_ANKAN: u16 = 3;
const MELD_KIND_KAKAN: u16 = 4;

const MELD_ROLE_CALLED: u16 = 0;
const MELD_ROLE_CONSUMED: u16 = 1;
const MELD_ROLE_ADDED: u16 = 2;

const CAND_TYPE_RIICHI: u16 = 37;
const CAND_TYPE_ANKAN: u16 = 38;
const CAND_TYPE_KAKAN: u16 = 39;
const CAND_TYPE_TSUMO: u16 = 40;
const CAND_TYPE_KYUSHU_KYUHAI: u16 = 41;
const CAND_TYPE_PASS: u16 = 42;
const CAND_TYPE_CHI: u16 = 43;
const CAND_TYPE_PON: u16 = 44;
const CAND_TYPE_DAIMINKAN: u16 = 45;
const CAND_TYPE_RON: u16 = 46;

const CAND_MOQIE_TEDASHI: u16 = 0;
const CAND_MOQIE_TSUMOGIRI: u16 = 1;
const CAND_MOQIE_NA: u16 = 2;

const PROG_TYPE_START: u16 = 0;
const PROG_TYPE_DISCARD_OFFSET: u16 = 1;
const PROG_TYPE_CHI: u16 = 38;
const PROG_TYPE_PON: u16 = 39;
const PROG_TYPE_DAIMINKAN: u16 = 40;
const PROG_TYPE_ANKAN: u16 = 41;
const PROG_TYPE_KAKAN: u16 = 42;
const PROG_TYPE_DORA_OFFSET: u16 = 43;
const PLAYER_MELD_COUNT_CAP: u16 = 4;
const PLAYER_DISCARD_COUNT_CAP: u16 = 20;

fn normalize_score(score: i32) -> f32 {
    (score as f32 - SCORE_NORM_BASE) / SCORE_NORM_SCALE
}

// ── Tile conversions ─────────────────────────────────────────────────────────

/// Convert a 136-tile ID to kan37 (37 tiles, red fives distinct).
///
/// Layout: 0=red5m, 1-9=1m-9m, 10=red5p, 11-19=1p-9p,
///         20=red5s, 21-29=1s-9s, 30-36=E/S/W/N/P/F/C
pub fn tile_id_to_kan37(tile_id: u32) -> u8 {
    // Red fives
    if tile_id == 16 {
        return 0; // red 5m
    }
    if tile_id == 52 {
        return 10; // red 5p
    }
    if tile_id == 88 {
        return 20; // red 5s
    }

    let tile_type = (tile_id / 4) as u8; // 0-33
    tile_type_to_kan37(tile_type)
}

/// Convert a tile type (0-33, no red distinction) to kan37.
/// For 5m/5p/5s this returns the non-red version.
fn tile_type_to_kan37(tile_type: u8) -> u8 {
    match tile_type {
        0..=8 => tile_type + 1,   // 1m-9m → 1-9
        9..=17 => tile_type + 2,  // 1p-9p → 11-19
        18..=26 => tile_type + 3, // 1s-9s → 21-29
        27..=33 => tile_type + 3, // honors → 30-36
        _ => 0,
    }
}

/// Convert MJAI tile string to kan37.
fn mjai_tile_to_kan37(mjai: &str) -> Option<u8> {
    let tid = mjai_to_tid(mjai)?;
    Some(tile_id_to_kan37(tid as u32))
}

fn increment_visible_tile37_count(counts: &mut [u16; VISIBLE_TILE_COUNT_TOKENS], tile: u32) {
    if tile < 136 {
        let k37 = tile_id_to_kan37(tile);
        counts[k37 as usize] = counts[k37 as usize].saturating_add(1);
    }
}

fn count_meld_visible_tiles(counts: &mut [u16; VISIBLE_TILE_COUNT_TOKENS], meld: &Meld) {
    let called_idx = meld.called_tile.and_then(|called| {
        meld.tiles
            .iter()
            .position(|&tile| tile == called)
            .or_else(|| {
                let called_k37 = tile_id_to_kan37(called as u32);
                meld.tiles
                    .iter()
                    .position(|&tile| tile_id_to_kan37(tile as u32) == called_k37)
            })
    });

    for (idx, &tile) in meld.tiles.iter().enumerate() {
        if called_idx == Some(idx) {
            continue;
        }
        increment_visible_tile37_count(counts, tile as u32);
    }
}

// ── Factorized meld encoding ─────────────────────────────────────────────────

fn meld_pad() -> [u16; MELD_FEATURE_WIDTH] {
    MELD_PAD
}

fn set_meld_slot(row: &mut [u16; MELD_FEATURE_WIDTH], slot: usize, tile: u8, role: u16) {
    if slot >= 4 {
        return;
    }
    row[1 + slot * 2] = tile_id_to_kan37(tile as u32) as u16;
    row[2 + slot * 2] = role;
}

fn sorted_tiles(mut tiles: Vec<u8>) -> Vec<u8> {
    tiles.sort_by_key(|&t| (t / 4, tile_id_to_kan37(t as u32), t));
    tiles
}

fn remove_exact_or_kan37(tiles: &mut Vec<u8>, target: u8) -> Option<u8> {
    if let Some(idx) = tiles.iter().position(|&t| t == target) {
        return Some(tiles.remove(idx));
    }
    let target_k37 = tile_id_to_kan37(target as u32);
    tiles
        .iter()
        .position(|&t| tile_id_to_kan37(t as u32) == target_k37)
        .map(|idx| tiles.remove(idx))
}

fn encode_called_consumed_meld(
    kind: u16,
    called: u8,
    consumed: Vec<u8>,
) -> [u16; MELD_FEATURE_WIDTH] {
    let mut row = meld_pad();
    row[0] = kind;
    set_meld_slot(&mut row, 0, called, MELD_ROLE_CALLED);
    for (idx, tile) in sorted_tiles(consumed).into_iter().take(3).enumerate() {
        set_meld_slot(&mut row, idx + 1, tile, MELD_ROLE_CONSUMED);
    }
    row
}

fn encode_consumed_meld(kind: u16, consumed: Vec<u8>) -> [u16; MELD_FEATURE_WIDTH] {
    let mut row = meld_pad();
    row[0] = kind;
    for (idx, tile) in sorted_tiles(consumed).into_iter().take(4).enumerate() {
        set_meld_slot(&mut row, idx, tile, MELD_ROLE_CONSUMED);
    }
    row
}

fn encode_kakan_meld(
    added: u8,
    called: Option<u8>,
    existing_tiles: Vec<u8>,
) -> [u16; MELD_FEATURE_WIDTH] {
    let mut row = meld_pad();
    row[0] = MELD_KIND_KAKAN;
    set_meld_slot(&mut row, 0, added, MELD_ROLE_ADDED);

    let mut remaining = existing_tiles;
    if let Some(called_tile) = called {
        let removed = remove_exact_or_kan37(&mut remaining, called_tile).unwrap_or(called_tile);
        set_meld_slot(&mut row, 1, removed, MELD_ROLE_CALLED);
    } else if let Some(fallback_called) = remaining.first().copied() {
        let removed =
            remove_exact_or_kan37(&mut remaining, fallback_called).unwrap_or(fallback_called);
        set_meld_slot(&mut row, 1, removed, MELD_ROLE_CALLED);
    }

    for (idx, tile) in sorted_tiles(remaining).into_iter().take(2).enumerate() {
        set_meld_slot(&mut row, idx + 2, tile, MELD_ROLE_CONSUMED);
    }
    row
}

fn split_called_consumed_tiles(meld: &Meld) -> Option<(u8, Vec<u8>)> {
    let called_tile = meld.called_tile?;
    let mut consumed = meld.tiles.clone();
    remove_exact_or_kan37(&mut consumed, called_tile)?;
    Some((called_tile, consumed))
}

fn encode_meld_feature(meld: &Meld) -> [u16; MELD_FEATURE_WIDTH] {
    match meld.meld_type {
        MeldType::Chi => split_called_consumed_tiles(meld)
            .map(|(called, consumed)| encode_called_consumed_meld(MELD_KIND_CHI, called, consumed))
            .unwrap_or_else(meld_pad),
        MeldType::Pon => split_called_consumed_tiles(meld)
            .map(|(called, consumed)| encode_called_consumed_meld(MELD_KIND_PON, called, consumed))
            .unwrap_or_else(meld_pad),
        MeldType::Daiminkan => split_called_consumed_tiles(meld)
            .map(|(called, consumed)| {
                encode_called_consumed_meld(MELD_KIND_DAIMINKAN, called, consumed)
            })
            .unwrap_or_else(meld_pad),
        MeldType::Ankan => encode_consumed_meld(MELD_KIND_ANKAN, meld.tiles.clone()),
        MeldType::Kakan => {
            let added = meld.added_tile.or_else(|| meld.tiles.first().copied());
            if let Some(added_tile) = added {
                let mut existing = meld.tiles.clone();
                remove_exact_or_kan37(&mut existing, added_tile);
                encode_kakan_meld(added_tile, meld.called_tile, existing)
            } else {
                meld_pad()
            }
        }
    }
}

fn parse_optional_mjai_tid(v: &serde_json::Value, key: &str) -> Option<u8> {
    v[key].as_str().and_then(mjai_to_tid)
}

/// Observer-relative seat.
///
/// 0=self, 1=shimocha(right/downstream), 2=toimen(across), 3=kamicha(left/upstream).
fn relative_seat(observer: u8, target: u8) -> u8 {
    ((target as i16 - observer as i16 + 4) % 4) as u8
}

/// Parse "consumed" array from MJAI event JSON → Vec<u8> of tile IDs.
fn parse_consumed_tids_from_value(v: &serde_json::Value) -> Vec<u8> {
    let mut tids = Vec::new();
    if let Some(arr) = v["consumed"].as_array() {
        for item in arr {
            if let Some(s) = item.as_str()
                && let Some(tid) = mjai_to_tid(s)
            {
                tids.push(tid);
            }
        }
    }
    tids
}

fn encode_dora_progression_row(dora_marker: &str) -> Option<[u16; 5]> {
    let k37 = mjai_tile_to_kan37(dora_marker)?;
    Some([4, PROG_TYPE_DORA_OFFSET + k37 as u16, 2, 2, 4])
}

/// Process a single MJAI event for canonical progression and aligned meld sidecar caches.
///
/// Seat fields in the returned progression row are absolute MJAI seats. They are
/// converted to observer-relative seats by `Observation::encode_seq_progression()`.
pub fn process_single_event_progression_with_meld(
    event: &serde_json::Value,
    pending_reach_actor: &mut Option<u8>,
) -> Option<([u16; 5], [u16; MELD_FEATURE_WIDTH])> {
    let event_type = event["type"].as_str()?;

    match event_type {
        "start_kyoku" => Some(([4, PROG_TYPE_START, 2, 2, 4], meld_pad())),
        "reach" => {
            if let Some(actor) = event["actor"].as_u64() {
                *pending_reach_actor = Some(actor as u8);
            }
            None
        }
        "dahai" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let k37 = mjai_tile_to_kan37(pai)?;
            let type_idx = PROG_TYPE_DISCARD_OFFSET + k37 as u16;
            let moqie = if event["tsumogiri"].as_bool().unwrap_or(false) {
                1
            } else {
                0
            };
            let liqi = if *pending_reach_actor == Some(actor) {
                *pending_reach_actor = None;
                1
            } else {
                0
            };
            Some(([actor as u16, type_idx, moqie, liqi, 4], meld_pad()))
        }
        "chi" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let target = event["target"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let called_tid = mjai_to_tid(pai)?;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.len() < 2 {
                return None;
            }
            let meld = encode_called_consumed_meld(MELD_KIND_CHI, called_tid, consumed);
            Some(([actor as u16, PROG_TYPE_CHI, 2, 2, target as u16], meld))
        }
        "pon" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let target = event["target"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let called_tid = mjai_to_tid(pai)?;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.len() < 2 {
                return None;
            }
            let meld = encode_called_consumed_meld(MELD_KIND_PON, called_tid, consumed);
            Some(([actor as u16, PROG_TYPE_PON, 2, 2, target as u16], meld))
        }
        "daiminkan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let target = event["target"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let called_tid = mjai_to_tid(pai)?;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.len() < 3 {
                return None;
            }
            let meld = encode_called_consumed_meld(MELD_KIND_DAIMINKAN, called_tid, consumed);
            Some((
                [actor as u16, PROG_TYPE_DAIMINKAN, 2, 2, target as u16],
                meld,
            ))
        }
        "ankan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.is_empty() {
                return None;
            }
            let meld = encode_consumed_meld(MELD_KIND_ANKAN, consumed);
            Some(([actor as u16, PROG_TYPE_ANKAN, 2, 2, 4], meld))
        }
        "kakan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let added = mjai_to_tid(pai)?;
            let called = parse_optional_mjai_tid(event, "called");
            let existing_tiles = parse_consumed_tids_from_value(event);
            let meld = encode_kakan_meld(added, called, existing_tiles);
            Some(([actor as u16, PROG_TYPE_KAKAN, 2, 2, 4], meld))
        }
        "dora" => {
            let dora_marker = event["dora_marker"].as_str().unwrap_or("?");
            if dora_marker == "?" {
                return None;
            }
            encode_dora_progression_row(dora_marker).map(|row| (row, meld_pad()))
        }
        _ => None,
    }
}

/// Process one MJAI event into zero or more progression rows.
///
/// `start_kyoku` yields both the beginning-of-round marker and, when present,
/// the initial dora reveal row. External MJAI logs stay unchanged; this only
/// expands the internal sequence representation.
pub fn process_event_progression_entries_with_melds(
    event: &serde_json::Value,
    pending_reach_actor: &mut Option<u8>,
) -> Vec<([u16; 5], [u16; MELD_FEATURE_WIDTH])> {
    let Some(event_type) = event["type"].as_str() else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(2);
    if let Some(entry) = process_single_event_progression_with_meld(event, pending_reach_actor) {
        out.push(entry);
    }

    if event_type == "start_kyoku"
        && let Some(dora_marker) = event["dora_marker"].as_str()
        && dora_marker != "?"
        && let Some(row) = encode_dora_progression_row(dora_marker)
    {
        out.push((row, meld_pad()));
    }

    out
}

// ── Sparse features ──────────────────────────────────────────────────────────

impl Observation {
    /// Encode sparse features: variable-length u16 indices (max 8).
    ///
    /// Offsets:
    /// - 0-1: game style (0=tonpuusen, 1=hanchan)
    /// - 2-4: chang / round wind (E/S/W)
    /// - 5-74: tiles remaining (0-69)
    /// - 75-259: dora indicators (5 slots × 37 tiles)
    /// - 260: padding
    pub fn encode_seq_sparse(&self, game_style: u8) -> Vec<u16> {
        let mut tokens: Vec<u16> = Vec::with_capacity(MAX_SPARSE_LEN);

        // 1. Game style (offset 0-1)
        tokens.push(game_style.min(1) as u16);

        // 2. Chang / round wind (offset 2-4)
        tokens.push(2 + self.round_wind.min(2) as u16);

        // 3. Tiles remaining (offset 5-74)
        let tiles_remaining = self.count_tiles_remaining();
        tokens.push(5 + tiles_remaining.min(69));

        // 4. Dora indicators (offset 75-259, 5 slots × 37)
        for (i, &dora_tid) in self.dora_indicators.iter().enumerate() {
            if i >= 5 {
                break;
            }
            let k37 = tile_id_to_kan37(dora_tid);
            tokens.push(SPARSE_DORA_OFFSET + (i as u16) * 37 + k37 as u16);
        }

        if tokens.len() > MAX_SPARSE_LEN {
            tokens.truncate(MAX_SPARSE_LEN);
        }

        tokens
    }

    /// Encode dealer seat relative to the observing player.
    ///
    /// Values: 0=self, 1=shimocha, 2=toimen, 3=kamicha.
    pub fn encode_seq_dealer(&self) -> u16 {
        relative_seat(self.player_id.min(3), self.oya.min(3)) as u16
    }

    /// Encode per-player public summary rows in observer-relative seat order.
    ///
    /// Each row is `(relative_seat, riichi_active, meld_count, discard_count, tedashi_count)`.
    /// Counts are clipped to the embedding vocabularies: meld count 0-4, discard
    /// and tedashi counts 0-20. `riichi_active` includes the transient reach
    /// stage between a `reach` event and the committing discard.
    pub fn encode_seq_player_stats(&self) -> Vec<[u16; PLAYER_INFO_WIDTH]> {
        let mut out = Vec::with_capacity(PLAYER_INFO_TOKENS);

        for (owner_rel, &abs_idx) in self.rel_order().iter().enumerate() {
            let riichi_active = self.riichi_declared[abs_idx] || self.riichi_stage[abs_idx];
            let meld_count = (self.melds[abs_idx].len() as u16).min(PLAYER_MELD_COUNT_CAP);
            let discard_count = (self.discards[abs_idx].len() as u16).min(PLAYER_DISCARD_COUNT_CAP);
            let tedashi_count = self
                .count_tedashi_discards(abs_idx)
                .min(PLAYER_DISCARD_COUNT_CAP);

            out.push([
                owner_rel as u16,
                if riichi_active { 1 } else { 0 },
                meld_count,
                discard_count,
                tedashi_count,
            ]);
        }

        out
    }

    fn count_tedashi_discards(&self, abs_idx: usize) -> u16 {
        let discards_len = self.discards[abs_idx].len();
        let from_hand = &self.discard_from_hand[abs_idx];
        if from_hand.len() == discards_len {
            return from_hand.iter().filter(|&&v| v).count() as u16;
        }

        let tsumogiri = &self.tsumogiri_flags[abs_idx];
        if tsumogiri.len() == discards_len {
            return tsumogiri.iter().filter(|&&v| !v).count() as u16;
        }

        self.events
            .iter()
            .filter_map(|event_str| serde_json::from_str::<serde_json::Value>(event_str).ok())
            .filter(|v| {
                v["type"].as_str() == Some("dahai")
                    && v["actor"].as_u64() == Some(abs_idx as u64)
                    && !v["tsumogiri"].as_bool().unwrap_or(false)
            })
            .count() as u16
    }

    fn encode_sparse_meld_entries(&self) -> Vec<([u16; MELD_FEATURE_WIDTH], u16)> {
        let mut out: Vec<([u16; MELD_FEATURE_WIDTH], u16)> = Vec::with_capacity(MAX_SPARSE_MELDS);

        for (owner_rel, &abs_idx) in self.rel_order().iter().enumerate() {
            for meld in self.melds[abs_idx].iter().take(4) {
                if out.len() >= MAX_SPARSE_MELDS {
                    return out;
                }
                out.push((encode_meld_feature(meld), owner_rel as u16));
            }
        }

        out
    }

    /// Encode all players' current melds with the shared factorized layout.
    ///
    /// Melds are ordered by observer-relative owner seat:
    /// self, shimocha, toimen, kamicha; each owner contributes up to 4 melds.
    pub fn encode_seq_sparse_melds(&self) -> Vec<[u16; MELD_FEATURE_WIDTH]> {
        self.encode_sparse_meld_entries()
            .into_iter()
            .map(|(meld, _owner)| meld)
            .collect()
    }

    /// Encode owner seats aligned with `encode_seq_sparse_melds()`.
    ///
    /// Values: 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding.
    pub fn encode_seq_sparse_meld_owners(&self) -> Vec<u16> {
        self.encode_sparse_meld_entries()
            .into_iter()
            .map(|(_meld, owner)| owner)
            .collect()
    }

    /// Encode current meld rows and aligned owner seats in one pass.
    pub fn encode_seq_sparse_meld_features(&self) -> Vec<[u16; SPARSE_MELD_FEATURE_WIDTH]> {
        self.encode_sparse_meld_entries()
            .into_iter()
            .map(|(meld, owner)| {
                let mut row = [0u16; SPARSE_MELD_FEATURE_WIDTH];
                row[..MELD_FEATURE_WIDTH].copy_from_slice(&meld);
                row[MELD_FEATURE_WIDTH] = owner;
                row
            })
            .collect()
    }

    /// Encode the hand as variable-length (tile37, draw_state) tuples.
    ///
    /// The hand is ordered as concealed tiles followed by the drawn tile, if
    /// present. `draw_state`: 0=concealed, 1=drawn.
    pub fn encode_seq_hand(&self) -> Vec<[u16; 2]> {
        let mut hand: Vec<[u16; 2]> = Vec::with_capacity(MAX_HAND_LEN);
        let my_hand = &self.hands[self.player_id as usize];
        let drawn_k37 = self
            .get_drawn_tile()
            .map(|drawn| tile_id_to_kan37(drawn as u32));
        let mut drawn_consumed = false;

        for &tid in my_hand {
            if tid < 136 {
                let k37 = tile_id_to_kan37(tid);
                if !drawn_consumed && drawn_k37 == Some(k37) {
                    drawn_consumed = true;
                    continue;
                }
                hand.push([k37 as u16, 0]);
            }
        }

        if let Some(k37) = drawn_k37 {
            hand.push([k37 as u16, 1]);
        }

        if hand.len() > MAX_HAND_LEN {
            hand.truncate(MAX_HAND_LEN);
        }

        hand
    }

    fn visible_tile37_counts(&self) -> [u16; VISIBLE_TILE_COUNT_TOKENS] {
        let mut counts = [0u16; VISIBLE_TILE_COUNT_TOKENS];

        for &tile in &self.hands[self.player_id as usize] {
            increment_visible_tile37_count(&mut counts, tile);
        }
        for melds in &self.melds {
            for meld in melds {
                count_meld_visible_tiles(&mut counts, meld);
            }
        }
        for discards in &self.discards {
            for &tile in discards {
                increment_visible_tile37_count(&mut counts, tile);
            }
        }
        for &tile in &self.dora_indicators {
            increment_visible_tile37_count(&mut counts, tile);
        }

        counts
    }

    /// Encode visible tile counts as fixed 37 `(tile37, visible_count)` rows.
    ///
    /// Counts include the observing player's hand, all visible meld tiles, all
    /// discards, and currently visible dora indicators. Claimed tiles are counted
    /// from the discard river and skipped once inside their meld to avoid double
    /// counting the same visible tile.
    pub fn encode_seq_visible_tile_counts(&self) -> Vec<[u16; VISIBLE_TILE_COUNT_WIDTH]> {
        self.visible_tile37_counts()
            .into_iter()
            .enumerate()
            .map(|(tile37, count)| [tile37 as u16, count.min(4)])
            .collect()
    }

    /// Count approximate tiles remaining in the wall.
    fn count_tiles_remaining(&self) -> u16 {
        let n = 4; // 4 players
        let total_tiles: u32 = 136; // 4P

        let mut used: u32 = 0;
        // Hands
        for i in 0..n {
            used += self.hands[i].len() as u32;
        }
        // Discards
        for i in 0..n {
            used += self.discards[i].len() as u32;
        }
        // Melds (only count tiles not already in hand)
        for i in 0..n {
            for meld in &self.melds[i] {
                used += meld.tiles.len() as u32;
            }
        }
        // Dora indicators
        used += self.dora_indicators.len() as u32;

        // Dead wall has 14 tiles (minus dora indicators already counted)
        // Initial deal: 13*4 = 52 tiles. Remaining = total - 14(dead) - used
        let wall_size = total_tiles.saturating_sub(14 + used);
        wall_size as u16
    }

    /// Get the last drawn tile for the current player (from tsumo event).
    fn get_drawn_tile(&self) -> Option<u8> {
        // Walk events backwards to find last tsumo for this player
        for event_str in self.events.iter().rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) {
                let event_type = v["type"].as_str().unwrap_or("");
                if event_type == "tsumo" {
                    let actor = v["actor"].as_u64();
                    if actor == Some(self.player_id as u64)
                        && let Some(pai) = v["pai"].as_str()
                        && pai != "?"
                    {
                        return mjai_to_tid(pai);
                    }
                }
                // Stop at decision-relevant events
                if event_type == "dahai"
                    || event_type == "chi"
                    || event_type == "pon"
                    || event_type == "daiminkan"
                {
                    break;
                }
            }
        }
        None
    }

    // ── Numeric features ─────────────────────────────────────────────────

    /// Encode numeric features: 6 floats.
    ///
    /// [0] honba (current)
    /// [1] riichi deposits (current)
    /// [2-5] normalized scores (self, right, opposite, left) relative to player_id
    pub fn encode_seq_numeric(&self) -> [f32; NUM_NUMERIC] {
        let mut out = [0.0f32; NUM_NUMERIC];
        let pid = self.player_id as usize;

        // Current state
        out[0] = self.honba as f32;
        out[1] = self.riichi_sticks as f32;

        // Scores (self-relative rotation)
        for i in 0..4 {
            let seat = (pid + i) % 4;
            out[2 + i] = normalize_score(self.scores[seat]);
        }

        out
    }

    /// Encode pairwise agari-rank-overtake features as 4 * 96 * 4 floats.
    ///
    /// Layout: `winner_rel * 96 * 4 + pattern_index * 4 + target_rel`.
    /// Winner and target axes are observer-relative. The settlement patterns
    /// match the Tenhou 4P standard non-PAO patterns used by the GRP model.
    pub fn encode_seq_agari_overtakes(&self) -> Vec<f32> {
        crate::grp::encode_tenhou_4p_pairwise_overtakes(
            self.scores,
            self.oya as usize,
            self.honba as u32,
            self.riichi_sticks,
            self.player_id.min(3) as usize,
        )
    }

    // ── Progression features ─────────────────────────────────────────────

    /// Encode progression (action history) as variable-length 5-tuples.
    ///
    /// Each tuple: (actor, type, moqie, liqi, from)
    /// - actor: 0-3 (seats), 4 (marker/padding)
    /// - type: 0=start, 1-37=discard, 38=chi, 39=pon, 40=daiminkan,
    ///   41=ankan, 42=kakan, 43-79=dora reveal
    /// - moqie: 0=tedashi, 1=tsumogiri, 2=N/A
    /// - liqi: 0=no riichi, 1=with riichi, 2=N/A
    /// - from: 0-3 (observer-relative seat), 4=N/A
    pub fn encode_seq_progression(&self) -> Vec<[u16; 5]> {
        // Fast path: use pre-computed progression from GameState
        if let Some(ref cached) = self.cached_progression {
            return self.relativize_progression(cached.clone());
        }

        // Fallback: parse events from JSON (for deserialized Observations, replays, etc.)
        let mut prog: Vec<[u16; 5]> = Vec::with_capacity(128);
        let mut pending_reach_actor: Option<u8> = None;

        for event_str in &self.events {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) else {
                continue;
            };
            for (entry, _meld) in
                process_event_progression_entries_with_melds(&v, &mut pending_reach_actor)
            {
                prog.push(entry);
            }
        }

        self.relativize_progression(prog)
    }

    fn relativize_progression(&self, prog: Vec<[u16; 5]>) -> Vec<[u16; 5]> {
        prog.into_iter()
            .map(|mut row| {
                if row[0] < 4 {
                    row[0] = relative_seat(self.player_id, row[0] as u8) as u16;
                }
                if row[4] < 4 {
                    row[4] = relative_seat(self.player_id, row[4] as u8) as u16;
                }
                row
            })
            .collect()
    }

    /// Encode factorized meld sidecar rows aligned with encode_seq_progression().
    pub fn encode_seq_progression_melds(&self) -> Vec<[u16; MELD_FEATURE_WIDTH]> {
        if let Some(ref cached) = self.cached_progression_melds {
            return cached.clone();
        }

        let mut melds: Vec<[u16; MELD_FEATURE_WIDTH]> = Vec::with_capacity(128);
        let mut pending_reach_actor: Option<u8> = None;

        for event_str in &self.events {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) else {
                continue;
            };
            for (_entry, meld) in
                process_event_progression_entries_with_melds(&v, &mut pending_reach_actor)
            {
                melds.push(meld);
            }
        }

        melds
    }

    // ── Candidate features ───────────────────────────────────────────────

    /// Return pointer-policy candidate actions in one-to-one correspondence
    /// with the strict candidate feature rows.
    ///
    /// Physical copies that have the same visible semantics are collapsed, but
    /// red fives, tedashi/tsumogiri discards, and red/non-red consumed melds
    /// remain distinct candidates.
    pub fn candidate_actions(&self) -> Vec<Action> {
        let pid = self.player_id;
        let mut keys: Vec<([u16; 3], [u16; MELD_FEATURE_WIDTH])> =
            Vec::with_capacity(self._legal_actions.len());
        let mut actions: Vec<Action> = Vec::with_capacity(self._legal_actions.len());

        for action in &self._legal_actions {
            let Some(tuple) = self.encode_candidate_action(action, pid) else {
                continue;
            };
            let meld = self.encode_candidate_action_meld(action, pid);
            let key = (tuple, meld);

            if !keys.contains(&key) {
                keys.push(key);
                actions.push(action.clone());
            }
        }

        actions
    }

    pub fn find_candidate_action(&self, candidate_index: usize) -> Option<Action> {
        self.candidate_actions().get(candidate_index).cloned()
    }

    pub fn find_candidate_index(&self, action: &Action) -> Option<usize> {
        let pid = self.player_id;
        let target = (
            self.encode_candidate_action(action, pid)?,
            self.encode_candidate_action_meld(action, pid),
        );
        self.candidate_actions().iter().position(|candidate| {
            self.encode_candidate_action(candidate, pid)
                .map(|tuple| (tuple, self.encode_candidate_action_meld(candidate, pid)) == target)
                .unwrap_or(false)
        })
    }

    /// Encode candidate (legal action) features as variable-length 3-tuples.
    ///
    /// Each tuple: (type, moqie, from)
    /// - type: 0-36=discard tile37, 37-46=special/call
    /// - moqie: 0=tedashi, 1=tsumogiri, 2=N/A
    /// - from: 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=padding
    pub fn encode_seq_candidates(&self) -> Vec<[u16; 3]> {
        let mut cands: Vec<[u16; 3]> = Vec::with_capacity(64);
        let pid = self.player_id;

        for action in self.candidate_actions() {
            let tuple = self.encode_candidate_action(&action, pid);
            if let Some(t) = tuple {
                cands.push(t);
            }
        }

        cands
    }

    /// Encode factorized meld sidecar rows aligned with encode_seq_candidates().
    pub fn encode_seq_candidate_melds(&self) -> Vec<[u16; MELD_FEATURE_WIDTH]> {
        let mut melds: Vec<[u16; MELD_FEATURE_WIDTH]> = Vec::with_capacity(64);
        let pid = self.player_id;

        for action in self.candidate_actions() {
            if self.encode_candidate_action(&action, pid).is_some() {
                melds.push(self.encode_candidate_action_meld(&action, pid));
            }
        }

        melds
    }

    /// Encode candidate tuples and aligned factorized meld rows in one pass.
    pub fn encode_seq_candidate_features(&self) -> Vec<[u16; CANDIDATE_FEATURE_WIDTH]> {
        self.candidate_action_features()
            .into_iter()
            .map(|(candidate, meld)| {
                let mut row = [0u16; CANDIDATE_FEATURE_WIDTH];
                row[..3].copy_from_slice(&candidate);
                row[3..].copy_from_slice(&meld);
                row
            })
            .collect()
    }

    fn candidate_action_features(&self) -> Vec<([u16; 3], [u16; MELD_FEATURE_WIDTH])> {
        let mut features: Vec<([u16; 3], [u16; MELD_FEATURE_WIDTH])> = Vec::with_capacity(64);
        let pid = self.player_id;

        for action in self.candidate_actions() {
            if let Some(candidate) = self.encode_candidate_action(&action, pid) {
                features.push((candidate, self.encode_candidate_action_meld(&action, pid)));
            }
        }

        features
    }

    /// Encode a single legal action as a candidate 3-tuple.
    fn encode_candidate_action(&self, action: &Action, pid: u8) -> Option<[u16; 3]> {
        match action.action_type {
            ActionType::Discard => {
                let tile = action.tile?;
                let type_idx = tile_id_to_kan37(tile as u32) as u16;
                let moqie = if Some(tile) == self.drawn_tile {
                    CAND_MOQIE_TSUMOGIRI
                } else {
                    CAND_MOQIE_TEDASHI
                };
                Some([type_idx, moqie, 0]) // from=0 (self)
            }
            ActionType::Riichi => Some([CAND_TYPE_RIICHI, CAND_MOQIE_NA, 0]),
            ActionType::Ankan => {
                action.consume_tiles.first()?;
                Some([CAND_TYPE_ANKAN, CAND_MOQIE_NA, 0])
            }
            ActionType::Kakan => {
                action
                    .tile
                    .or_else(|| action.consume_tiles.first().copied())?;
                Some([CAND_TYPE_KAKAN, CAND_MOQIE_NA, 0])
            }
            ActionType::Tsumo => Some([CAND_TYPE_TSUMO, CAND_MOQIE_NA, 0]),
            ActionType::KyushuKyuhai => Some([CAND_TYPE_KYUSHU_KYUHAI, CAND_MOQIE_NA, 0]),
            ActionType::Pass => Some([CAND_TYPE_PASS, CAND_MOQIE_NA, 0]),
            ActionType::Chi => {
                action.tile?;
                if action.consume_tiles.len() < 2 {
                    return None;
                }

                // from = relative seat of the discard source
                let target = self.find_last_discard_actor()?;
                let rel = relative_seat(pid, target);

                Some([CAND_TYPE_CHI, CAND_MOQIE_NA, rel as u16])
            }
            ActionType::Pon => {
                action.tile?;
                if action.consume_tiles.len() < 2 {
                    return None;
                }

                let target = self.find_last_discard_actor()?;
                let rel = relative_seat(pid, target);

                Some([CAND_TYPE_PON, CAND_MOQIE_NA, rel as u16])
            }
            ActionType::Daiminkan => {
                action.tile?;

                let target = self.find_last_discard_actor()?;
                let rel = relative_seat(pid, target);

                Some([CAND_TYPE_DAIMINKAN, CAND_MOQIE_NA, rel as u16])
            }
            ActionType::Ron => {
                let target = self.find_last_discard_actor()?;
                let rel = relative_seat(pid, target);
                Some([CAND_TYPE_RON, CAND_MOQIE_NA, rel as u16])
            }
            ActionType::Kita => None, // 3P only, not supported
        }
    }

    fn encode_candidate_action_meld(&self, action: &Action, pid: u8) -> [u16; MELD_FEATURE_WIDTH] {
        match action.action_type {
            ActionType::Chi => action
                .tile
                .map(|called| {
                    encode_called_consumed_meld(MELD_KIND_CHI, called, action.consume_tiles.clone())
                })
                .unwrap_or_else(meld_pad),
            ActionType::Pon => action
                .tile
                .map(|called| {
                    encode_called_consumed_meld(MELD_KIND_PON, called, action.consume_tiles.clone())
                })
                .unwrap_or_else(meld_pad),
            ActionType::Daiminkan => action
                .tile
                .map(|called| {
                    encode_called_consumed_meld(
                        MELD_KIND_DAIMINKAN,
                        called,
                        action.consume_tiles.clone(),
                    )
                })
                .unwrap_or_else(meld_pad),
            ActionType::Ankan => {
                encode_consumed_meld(MELD_KIND_ANKAN, action.consume_tiles.clone())
            }
            ActionType::Kakan => {
                let Some(added) = action
                    .tile
                    .or_else(|| action.consume_tiles.first().copied())
                else {
                    return meld_pad();
                };
                let pon = self.melds[pid as usize]
                    .iter()
                    .find(|m| m.meld_type == MeldType::Pon && m.tiles[0] / 4 == added / 4);
                let called = pon.and_then(|m| m.called_tile);
                let existing = pon
                    .map(|m| m.tiles.clone())
                    .unwrap_or_else(|| action.consume_tiles.clone());
                encode_kakan_meld(added, called, existing)
            }
            _ => meld_pad(),
        }
    }

    /// Find the actor of the last discard (for chi/pon/kan/ron response).
    fn find_last_discard_actor(&self) -> Option<u8> {
        if let Some(actor) = self.last_discard_actor {
            return Some(actor);
        }

        for event_str in self.events.iter().rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) {
                let event_type = v["type"].as_str().unwrap_or("");
                if event_type == "dahai" || event_type == "kakan" {
                    return v["actor"].as_u64().map(|a| a as u8);
                }
            }
        }
        None
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Meld, MeldType};

    #[test]
    fn test_tile_id_to_kan37() {
        // Red fives
        assert_eq!(tile_id_to_kan37(16), 0); // red 5m → 0
        assert_eq!(tile_id_to_kan37(52), 10); // red 5p → 10
        assert_eq!(tile_id_to_kan37(88), 20); // red 5s → 20

        // 1m (tile_id 0-3, type 0) → kan37 = 1
        assert_eq!(tile_id_to_kan37(0), 1);
        assert_eq!(tile_id_to_kan37(3), 1);

        // 5m non-red (tile_id 17-19, type 4) → kan37 = 5
        assert_eq!(tile_id_to_kan37(17), 5);

        // 9m (tile_id 32-35, type 8) → kan37 = 9
        assert_eq!(tile_id_to_kan37(32), 9);

        // 1p (tile_id 36-39, type 9) → kan37 = 11
        assert_eq!(tile_id_to_kan37(36), 11);

        // 1s (tile_id 72-75, type 18) → kan37 = 21
        assert_eq!(tile_id_to_kan37(72), 21);

        // East wind (tile_id 108-111, type 27) → kan37 = 30
        assert_eq!(tile_id_to_kan37(108), 30);

        // Chun / Red dragon (tile_id 132-135, type 33) → kan37 = 36
        assert_eq!(tile_id_to_kan37(132), 36);
    }

    #[test]
    fn test_relative_seat() {
        // Player 0 observing player 0 -> self
        assert_eq!(relative_seat(0, 0), 0);
        // Player 0 observing player 1 -> shimocha
        assert_eq!(relative_seat(0, 1), 1);
        // Player 0 observing player 2 -> toimen
        assert_eq!(relative_seat(0, 2), 2);
        // Player 0 observing player 3 -> kamicha
        assert_eq!(relative_seat(0, 3), 3);
        // Player 2 observing player 3 -> shimocha
        assert_eq!(relative_seat(2, 3), 1);
    }

    #[test]
    fn test_sparse_vocab_bounds() {
        // Verify all sparse offsets are within vocab
        assert!(SPARSE_PAD < SPARSE_VOCAB_SIZE as u16);

        // Dora max: 75 + 4*37 + 36 = 75 + 148 + 36 = 259
        assert!(SPARSE_DORA_OFFSET + 4 * 37 + 36 < SPARSE_VOCAB_SIZE as u16);

        assert_eq!(SPARSE_PAD as usize + 1, SPARSE_VOCAB_SIZE);
    }

    #[test]
    fn test_hand_tuple_tile_values_use_tile_type_not_instance() {
        // 1m copies should collapse to the same tile37 field.
        let t0 = tile_id_to_kan37(0) as u16;
        let t1 = tile_id_to_kan37(1) as u16;
        let t3 = tile_id_to_kan37(3) as u16;
        assert_eq!(t0, t1);
        assert_eq!(t1, t3);

        // Red 5m and normal 5m should remain distinct.
        let red_5m = tile_id_to_kan37(16) as u16;
        let normal_5m = tile_id_to_kan37(17) as u16;
        assert_ne!(red_5m, normal_5m);
    }

    #[test]
    fn test_encode_meld_feature_roles() {
        let chi = Meld::new(MeldType::Chi, vec![0, 4, 8], true, 1, Some(0));
        assert_eq!(
            encode_meld_feature(&chi),
            [
                MELD_KIND_CHI,
                1,
                MELD_ROLE_CALLED,
                2,
                MELD_ROLE_CONSUMED,
                3,
                MELD_ROLE_CONSUMED,
                37,
                3
            ]
        );

        let pon = Meld::new(MeldType::Pon, vec![108, 109, 110], true, 1, Some(108));
        assert_eq!(
            encode_meld_feature(&pon),
            [
                MELD_KIND_PON,
                30,
                MELD_ROLE_CALLED,
                30,
                MELD_ROLE_CONSUMED,
                30,
                MELD_ROLE_CONSUMED,
                37,
                3
            ]
        );

        let daiminkan = Meld::new(MeldType::Daiminkan, vec![0, 1, 2, 3], true, 1, Some(0));
        assert_eq!(
            encode_meld_feature(&daiminkan),
            [
                MELD_KIND_DAIMINKAN,
                1,
                MELD_ROLE_CALLED,
                1,
                MELD_ROLE_CONSUMED,
                1,
                MELD_ROLE_CONSUMED,
                1,
                MELD_ROLE_CONSUMED,
            ]
        );

        let ankan = Meld::new(MeldType::Ankan, vec![72, 73, 74, 75], false, -1, None);
        assert_eq!(
            encode_meld_feature(&ankan),
            [
                MELD_KIND_ANKAN,
                21,
                MELD_ROLE_CONSUMED,
                21,
                MELD_ROLE_CONSUMED,
                21,
                MELD_ROLE_CONSUMED,
                21,
                MELD_ROLE_CONSUMED,
            ]
        );

        let kakan = Meld::new(MeldType::Kakan, vec![16, 17, 18, 19], true, -1, Some(17))
            .with_added_tile(16);
        assert_eq!(
            encode_meld_feature(&kakan),
            [
                MELD_KIND_KAKAN,
                0,
                MELD_ROLE_ADDED,
                5,
                MELD_ROLE_CALLED,
                5,
                MELD_ROLE_CONSUMED,
                5,
                MELD_ROLE_CONSUMED,
            ]
        );
    }

    #[test]
    fn test_encode_seq_hand_moves_drawn_tile_to_tail() {
        let obs = Observation::new(
            0,
            [vec![0, 1, 4], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![r#"{"type":"tsumo","actor":0,"pai":"1m"}"#.to_string()],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let hand = obs.encode_seq_hand();
        assert_eq!(hand, vec![[1, 0], [2, 0], [1, 1]]);
    }

    #[test]
    fn test_encode_seq_hand_without_drawn_is_all_concealed() {
        let obs = Observation::new(
            0,
            [vec![0, 4, 8], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let hand = obs.encode_seq_hand();
        assert_eq!(hand, vec![[1, 0], [2, 0], [3, 0]]);
    }

    #[test]
    fn test_encode_seq_visible_tile_counts_uses_tile37_and_skips_called_meld_tile() {
        let obs = Observation::new(
            0,
            [vec![16, 17], vec![], vec![], vec![]],
            [
                vec![],
                vec![Meld::new(MeldType::Pon, vec![0, 1, 2], true, 0, Some(0))],
                vec![],
                vec![],
            ],
            [vec![], vec![], vec![0], vec![]],
            vec![4],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let counts = obs.encode_seq_visible_tile_counts();

        assert_eq!(counts.len(), VISIBLE_TILE_COUNT_TOKENS);
        for (tile37, row) in counts.iter().enumerate() {
            assert_eq!(row[0], tile37 as u16);
        }
        assert_eq!(counts[0], [0, 1]); // red 5m in self hand
        assert_eq!(counts[1], [1, 3]); // discard + two consumed pon tiles; called tile is skipped in meld
        assert_eq!(counts[2], [2, 1]); // visible dora indicator 2m
        assert_eq!(counts[5], [5, 1]); // normal 5m in self hand
        assert_eq!(counts.iter().map(|row| row[1]).sum::<u16>(), 6);
    }

    #[test]
    fn test_encode_seq_player_stats_uses_relative_order_and_clipped_counts() {
        let mut obs = Observation::new(
            2,
            [vec![], vec![], vec![], vec![]],
            [
                vec![Meld::new(
                    MeldType::Pon,
                    vec![108, 109, 110],
                    true,
                    1,
                    Some(108),
                )],
                vec![],
                vec![],
                vec![
                    Meld::new(MeldType::Pon, vec![112, 113, 114], true, 2, Some(112)),
                    Meld::new(
                        MeldType::Kakan,
                        vec![116, 117, 118, 119],
                        true,
                        1,
                        Some(116),
                    ),
                ],
            ],
            [vec![0, 4, 8], vec![12, 16], vec![], (0..21).collect()],
            vec![],
            [25000, 25000, 25000, 25000],
            [false, true, false, false],
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );
        obs.riichi_stage = [false, false, false, true];
        obs.discard_from_hand = [
            vec![true, false, true],
            vec![true, true],
            vec![],
            vec![true; 21],
        ];

        assert_eq!(
            obs.encode_seq_player_stats(),
            vec![
                [0, 0, 0, 0, 0],
                [1, 1, 2, 20, 20],
                [2, 0, 1, 3, 2],
                [3, 1, 0, 2, 2],
            ]
        );
    }

    #[test]
    fn test_encode_seq_numeric_uses_current_normalized_scores_only() {
        let obs = Observation::new(
            2,
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [35000, 15000, 25000, 42000],
            [false; 4],
            vec![],
            vec![
                r#"{"type":"start_kyoku","honba":9,"kyotaku":8,"scores":[1000,2000,3000,4000]}"#
                    .to_string(),
            ],
            3,
            2,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let numeric = obs.encode_seq_numeric();
        let expected = [3.0, 2.0, 0.0, 1.7, 1.0, -1.0];
        for (actual, expected) in numeric.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sparse_melds_use_factorized_rows() {
        let obs = Observation::new(
            0,
            [vec![36, 40, 44, 48], vec![], vec![], vec![]],
            [
                vec![
                    Meld::new(MeldType::Pon, vec![108, 109, 110], true, 1, Some(108)),
                    Meld::new(MeldType::Chi, vec![36, 40, 44], true, 1, Some(36)),
                ],
                vec![Meld::new(
                    MeldType::Pon,
                    vec![112, 113, 114],
                    true,
                    0,
                    Some(112),
                )],
                vec![],
                vec![],
            ],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let sparse = obs.encode_seq_sparse(1);
        assert!(sparse.iter().all(|&t| t < SPARSE_PAD));

        let melds = obs.encode_seq_sparse_melds();
        let owners = obs.encode_seq_sparse_meld_owners();
        assert_eq!(melds.len(), 3);
        assert_eq!(owners, vec![0, 0, 1]);
        assert_eq!(melds[0][0], MELD_KIND_PON);
        assert_eq!(melds[1][0], MELD_KIND_CHI);
        assert_eq!(melds[2][0], MELD_KIND_PON);
    }

    #[test]
    fn test_dealer_feature_uses_relative_seat_and_sparse_excludes_dealer() {
        let obs = Observation::new(
            2,
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![],
            0,
            0,
            1,
            1,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let sparse = obs.encode_seq_sparse(1);
        let dealer = obs.encode_seq_dealer();

        assert_eq!(sparse[0], 1); // game style
        assert_eq!(sparse[1], 3); // round wind S: 2 + 1
        assert_eq!(sparse[2], 74); // max-clamped tiles remaining: 5 + 69
        assert_eq!(dealer, 3); // dealer seat 1 from observer seat 2: kamicha
        assert!(dealer < DEALER_DIMS);
        assert!(sparse.iter().all(|&t| t < SPARSE_PAD));
    }

    #[test]
    fn test_progression_actor_and_from_are_observer_relative() {
        let obs = Observation::new(
            2,
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![
                r#"{"type":"start_kyoku"}"#.to_string(),
                r#"{"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}"#.to_string(),
                r#"{"type":"pon","actor":3,"target":1,"pai":"5m","consumed":["5m","5m"]}"#
                    .to_string(),
            ],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let progression = obs.encode_seq_progression();

        assert_eq!(progression[0], [4, 0, 2, 2, 4]);
        assert_eq!(progression[1][0], 3); // actor 1 from observer 2: kamicha
        assert_eq!(progression[1][4], 4);
        assert_eq!(progression[2][0], 1); // actor 3 from observer 2: shimocha
        assert_eq!(progression[2][4], 3); // target 1 from observer 2: kamicha
    }

    #[test]
    fn test_progression_includes_initial_and_kan_dora_reveals() {
        let obs = Observation::new(
            0,
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![
                r#"{"type":"start_kyoku","dora_marker":"5m"}"#.to_string(),
                r#"{"type":"dora","dora_marker":"E"}"#.to_string(),
            ],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let progression = obs.encode_seq_progression();
        let prog_melds = obs.encode_seq_progression_melds();

        assert_eq!(progression[0], [4, PROG_TYPE_START, 2, 2, 4]);
        assert_eq!(progression[1], [4, PROG_TYPE_DORA_OFFSET + 5, 2, 2, 4]);
        assert_eq!(progression[2], [4, PROG_TYPE_DORA_OFFSET + 30, 2, 2, 4]);
        assert_eq!(prog_melds.len(), progression.len());
        assert!(prog_melds.iter().all(|row| *row == MELD_PAD));
    }

    #[test]
    fn test_candidate_features_bundle_matches_separate_encoders() {
        let obs = Observation::new(
            1,
            [vec![], vec![16, 17, 18, 20], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [[19].to_vec(), vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false; 4],
            vec![
                Action::new(ActionType::Discard, Some(16), vec![], None),
                Action::new(ActionType::Pon, Some(19), vec![16, 17], None),
            ],
            vec![r#"{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}"#.to_string()],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
            None,
        );

        let candidates = obs.encode_seq_candidates();
        let melds = obs.encode_seq_candidate_melds();
        let bundled = obs.encode_seq_candidate_features();

        assert_eq!(bundled.len(), candidates.len());
        assert_eq!(bundled.len(), melds.len());
        for (i, row) in bundled.iter().enumerate() {
            assert_eq!(row[..3], candidates[i]);
            assert_eq!(row[3..], melds[i]);
        }
    }

    #[test]
    fn test_progression_type_bounds() {
        let prog_type_max = PROG_DIMS[1];
        assert!(79 < prog_type_max); // dora reveal max
        assert!(42 < prog_type_max); // kakan
    }

    #[test]
    fn test_candidate_type_bounds() {
        let cand_type_max = CAND_DIMS[0];
        assert!(36 < cand_type_max); // discard max: 36
        assert!(46 < cand_type_max); // ron
    }
}
