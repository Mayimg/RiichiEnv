//! Sequence feature encoding for transformer models.
//!
//! Produces sparse tokens, numeric features, progression (action history),
//! and candidate (legal action) features suitable for embedding-based
//! transformer architectures.
//!
//! Based on the Kanachan v3 encoding (subset — Room and Grade removed):
//! <https://github.com/Cryolite/kanachan/wiki/%5Bv3%5DNotes-on-Training-Data>

use crate::action::{Action, ActionEncoder, ActionType};
use crate::parser::mjai_to_tid;
use crate::types::{Meld, MeldType};

use super::Observation;

// ── Constants ────────────────────────────────────────────────────────────────
// These constants are `pub` for Python-side consumption (see riichienv-ml).
#[allow(dead_code)]
pub const SPARSE_VOCAB_SIZE: usize = 269;
#[allow(dead_code)]
pub const SPARSE_PAD: u16 = 268;
pub const MAX_SPARSE_LEN: usize = 10;

/// Hand tuple dimensions: (tile37, draw_state)
#[allow(dead_code)]
pub const HAND_DIMS: [u16; 2] = [38, 3];
pub const MAX_HAND_LEN: usize = 14;
#[allow(dead_code)]
pub const HAND_PAD: [u16; 2] = [37, 2];

/// Progression tuple dimensions: (actor, type, moqie, liqi, from)
#[allow(dead_code)]
pub const PROG_DIMS: [u16; 5] = [5, 44, 3, 3, 5];
pub const MAX_PROG_LEN: usize = 512;
#[allow(dead_code)]
pub const PROG_PAD: [u16; 5] = [4, 43, 2, 2, 4];

/// Candidate tuple dimensions: (type, from)
#[allow(dead_code)]
pub const CAND_DIMS: [u16; 2] = [45, 4];
#[allow(dead_code)]
pub const MAX_CAND_LEN: usize = 64;
#[allow(dead_code)]
pub const CAND_PAD: [u16; 2] = [44, 3];

/// Meld feature row: (kind, slot0_tile37, slot0_role, ..., slot3_tile37, slot3_role)
#[allow(dead_code)]
pub const MELD_FEATURE_WIDTH: usize = 9;
#[allow(dead_code)]
pub const MELD_KIND_DIMS: u16 = 6;
#[allow(dead_code)]
pub const MELD_ROLE_DIMS: u16 = 4;
#[allow(dead_code)]
pub const MAX_SPARSE_MELDS: usize = 4;
#[allow(dead_code)]
pub const MELD_PAD: [u16; MELD_FEATURE_WIDTH] = [5, 37, 3, 37, 3, 37, 3, 37, 3];

pub const NUM_NUMERIC: usize = 12;

const SPARSE_DORA_OFFSET: u16 = 83;
const MELD_KIND_CHI: u16 = 0;
const MELD_KIND_PON: u16 = 1;
const MELD_KIND_DAIMINKAN: u16 = 2;
const MELD_KIND_ANKAN: u16 = 3;
const MELD_KIND_KAKAN: u16 = 4;

const MELD_ROLE_CALLED: u16 = 0;
const MELD_ROLE_CONSUMED: u16 = 1;
const MELD_ROLE_ADDED: u16 = 2;

const CAND_TYPE_RIICHI: u16 = 34;
const CAND_TYPE_ANKAN: u16 = 35;
const CAND_TYPE_KAKAN: u16 = 36;
const CAND_TYPE_TSUMO: u16 = 37;
const CAND_TYPE_KYUSHU_KYUHAI: u16 = 38;
const CAND_TYPE_PASS: u16 = 39;
const CAND_TYPE_CHI: u16 = 40;
const CAND_TYPE_PON: u16 = 41;
const CAND_TYPE_DAIMINKAN: u16 = 42;
const CAND_TYPE_RON: u16 = 43;

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

fn is_red_five(tile: u8) -> bool {
    matches!(tile, 16 | 52 | 88)
}

fn consumed_red_count(action: &Action) -> usize {
    action
        .consume_tiles
        .iter()
        .filter(|&&tile| is_red_five(tile))
        .count()
}

fn prefer_candidate_action(candidate: &Action, current: &Action) -> bool {
    match candidate.action_type {
        ActionType::Discard => {
            let candidate_is_red = candidate.tile.is_some_and(is_red_five);
            let current_is_red = current.tile.is_some_and(is_red_five);
            !candidate_is_red && current_is_red
        }
        ActionType::Chi | ActionType::Pon => {
            consumed_red_count(candidate) > consumed_red_count(current)
        }
        _ => false,
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

/// Relative seat: (target - actor + n_players - 1) % n_players
/// For 4P: 0=shimocha(right), 1=toimen(across), 2=kamicha(left)
fn relative_from(actor: u8, target: u8) -> u8 {
    ((target as i8 - actor as i8 + 3) % 4) as u8
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

/// Process a single MJAI event for progression and aligned meld sidecar caches.
pub fn process_single_event_progression_with_meld(
    event: &serde_json::Value,
    pending_reach_actor: &mut Option<u8>,
) -> Option<([u16; 5], [u16; MELD_FEATURE_WIDTH])> {
    let event_type = event["type"].as_str()?;

    match event_type {
        "start_kyoku" => Some(([4, 0, 2, 2, 4], meld_pad())),
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
            let type_idx = 1 + k37 as u16;
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
            let rel = relative_from(actor, target);
            let meld = encode_called_consumed_meld(MELD_KIND_CHI, called_tid, consumed);
            Some(([actor as u16, 38, 2, 2, rel as u16], meld))
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
            let rel = relative_from(actor, target);
            let meld = encode_called_consumed_meld(MELD_KIND_PON, called_tid, consumed);
            Some(([actor as u16, 39, 2, 2, rel as u16], meld))
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
            let rel = relative_from(actor, target);
            let meld = encode_called_consumed_meld(MELD_KIND_DAIMINKAN, called_tid, consumed);
            Some(([actor as u16, 40, 2, 2, rel as u16], meld))
        }
        "ankan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.is_empty() {
                return None;
            }
            let meld = encode_consumed_meld(MELD_KIND_ANKAN, consumed);
            Some(([actor as u16, 41, 2, 2, 4], meld))
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
            Some(([actor as u16, 42, 2, 2, 4], meld))
        }
        _ => None,
    }
}

// ── Sparse features ──────────────────────────────────────────────────────────

impl Observation {
    /// Encode sparse features: variable-length u16 indices (max 10).
    ///
    /// Offsets:
    /// - 0-1: game style (0=tonpuusen, 1=hanchan)
    /// - 2-5: seat (player_id)
    /// - 6-8: chang / round wind (E/S/W)
    /// - 9-12: ju / dealer round (0-3)
    /// - 13-82: tiles remaining (0-69)
    /// - 83-267: dora indicators (5 slots × 37 tiles)
    /// - 268: padding
    pub fn encode_seq_sparse(&self, game_style: u8) -> Vec<u16> {
        let mut tokens: Vec<u16> = Vec::with_capacity(MAX_SPARSE_LEN);

        // 1. Game style (offset 0-1)
        tokens.push(game_style.min(1) as u16);

        // 2. Seat (offset 2-5)
        tokens.push(2 + self.player_id.min(3) as u16);

        // 3. Chang / round wind (offset 6-8)
        tokens.push(6 + self.round_wind.min(2) as u16);

        // 4. Ju / dealer (offset 9-12)
        tokens.push(9 + self.oya.min(3) as u16);

        // 5. Tiles remaining (offset 13-82)
        let tiles_remaining = self.count_tiles_remaining();
        tokens.push(13 + tiles_remaining.min(69));

        // 6. Dora indicators (offset 83-267, 5 slots × 37)
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

    /// Encode the current player's melds with the shared factorized layout.
    pub fn encode_seq_sparse_melds(&self) -> Vec<[u16; MELD_FEATURE_WIDTH]> {
        self.melds[self.player_id as usize]
            .iter()
            .take(MAX_SPARSE_MELDS)
            .map(encode_meld_feature)
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

    /// Encode numeric features: 12 floats.
    ///
    /// [0] honba (current)
    /// [1] riichi deposits (current)
    /// [2-5] scores (self, right, opposite, left) relative to player_id
    /// [6] honba (round start)
    /// [7] riichi deposits (round start)
    /// [8-11] scores at round start (self-relative)
    pub fn encode_seq_numeric(&self) -> [f32; NUM_NUMERIC] {
        let mut out = [0.0f32; NUM_NUMERIC];
        let pid = self.player_id as usize;

        // Current state
        out[0] = self.honba as f32;
        out[1] = self.riichi_sticks as f32;

        // Scores (self-relative rotation)
        for i in 0..4 {
            let seat = (pid + i) % 4;
            out[2 + i] = self.scores[seat] as f32;
        }

        // Round-start values from start_kyoku event
        let (start_honba, start_riichi, start_scores) = self.parse_start_kyoku_info();
        out[6] = start_honba as f32;
        out[7] = start_riichi as f32;
        for i in 0..4 {
            let seat = (pid + i) % 4;
            out[8 + i] = start_scores[seat] as f32;
        }

        out
    }

    /// Parse start_kyoku event for initial round state.
    fn parse_start_kyoku_info(&self) -> (u32, u32, [i32; 4]) {
        for event_str in &self.events {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str)
                && v["type"].as_str() == Some("start_kyoku")
            {
                let honba = v["honba"].as_u64().unwrap_or(0) as u32;
                let kyotaku = v["kyotaku"].as_u64().unwrap_or(0) as u32;
                let mut scores = [0i32; 4];
                if let Some(arr) = v["scores"].as_array() {
                    for (i, val) in arr.iter().enumerate().take(4) {
                        scores[i] = val.as_i64().unwrap_or(0) as i32;
                    }
                }
                return (honba, kyotaku, scores);
            }
        }
        (self.honba as u32, self.riichi_sticks, self.scores)
    }

    // ── Progression features ─────────────────────────────────────────────

    /// Encode progression (action history) as variable-length 5-tuples.
    ///
    /// Each tuple: (actor, type, moqie, liqi, from)
    /// - actor: 0-3 (seats), 4 (marker/padding)
    /// - type: 0=start, 1-37=discard, 38=chi, 39=pon, 40=daiminkan,
    ///   41=ankan, 42=kakan, 43=padding
    /// - moqie: 0=tedashi, 1=tsumogiri, 2=N/A
    /// - liqi: 0=no riichi, 1=with riichi, 2=N/A
    /// - from: 0-2 (relative seat), 4=N/A
    pub fn encode_seq_progression(&self) -> Vec<[u16; 5]> {
        // Fast path: use pre-computed progression from GameState
        if let Some(ref cached) = self.cached_progression {
            return cached.clone();
        }

        // Fallback: parse events from JSON (for deserialized Observations, replays, etc.)
        let mut prog: Vec<[u16; 5]> = Vec::with_capacity(128);
        let mut pending_reach_actor: Option<u8> = None;

        for event_str in &self.events {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) else {
                continue;
            };
            if let Some((entry, _meld)) =
                process_single_event_progression_with_meld(&v, &mut pending_reach_actor)
            {
                prog.push(entry);
            }

            if prog.len() >= MAX_PROG_LEN {
                break;
            }
        }

        prog
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
            if let Some((_entry, meld)) =
                process_single_event_progression_with_meld(&v, &mut pending_reach_actor)
            {
                melds.push(meld);
            }

            if melds.len() >= MAX_PROG_LEN {
                break;
            }
        }

        melds
    }

    // ── Candidate features ───────────────────────────────────────────────

    /// Return candidate actions in one-to-one correspondence with the current
    /// 4-player 82-action space legal mask.
    ///
    /// The engine keeps physical tile choices in legal_actions.  For pointer
    /// policy training we collapse those choices by Action::encode():
    /// - Discard candidates prefer a non-red tile when red and non-red copies
    ///   map to the same tile34 action.
    /// - Chi/Pon candidates prefer representatives that consume red fives when
    ///   several physical consumes map to the same 82-action id.
    pub fn candidate_actions(&self) -> Vec<Action> {
        let encoder = ActionEncoder::FourPlayer;
        let mut ids: Vec<i32> = Vec::with_capacity(self._legal_actions.len());
        let mut actions: Vec<Action> = Vec::with_capacity(self._legal_actions.len());

        for action in &self._legal_actions {
            let Ok(action_id) = encoder.encode(action) else {
                continue;
            };

            if let Some(pos) = ids.iter().position(|&id| id == action_id) {
                if prefer_candidate_action(action, &actions[pos]) {
                    actions[pos] = action.clone();
                }
            } else {
                ids.push(action_id);
                actions.push(action.clone());
            }
        }

        actions
    }

    pub fn find_candidate_action(&self, candidate_index: usize) -> Option<Action> {
        self.candidate_actions().get(candidate_index).cloned()
    }

    pub fn find_candidate_index(&self, action: &Action) -> Option<usize> {
        let encoder = ActionEncoder::FourPlayer;
        let target_id = encoder.encode(action).ok()?;
        self.candidate_actions().iter().position(|candidate| {
            encoder
                .encode(candidate)
                .is_ok_and(|candidate_id| candidate_id == target_id)
        })
    }

    /// Encode candidate (legal action) features as variable-length 2-tuples.
    ///
    /// Each tuple: (type, from)
    /// - type: 0-44
    /// - from: 0-2 (relative seat), 3=self
    pub fn encode_seq_candidates(&self) -> Vec<[u16; 2]> {
        let mut cands: Vec<[u16; 2]> = Vec::with_capacity(64);
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

    /// Encode a single legal action as a candidate 2-tuple.
    fn encode_candidate_action(&self, action: &Action, pid: u8) -> Option<[u16; 2]> {
        match action.action_type {
            ActionType::Discard => {
                let tile = action.tile?;
                let type_idx = (tile / 4) as u16; // 0-33 tile34
                Some([type_idx, 3]) // from=3 (self)
            }
            ActionType::Riichi => Some([CAND_TYPE_RIICHI, 3]),
            ActionType::Ankan => {
                action.consume_tiles.first()?;
                Some([CAND_TYPE_ANKAN, 3])
            }
            ActionType::Kakan => {
                action
                    .tile
                    .or_else(|| action.consume_tiles.first().copied())?;
                Some([CAND_TYPE_KAKAN, 3])
            }
            ActionType::Tsumo => Some([CAND_TYPE_TSUMO, 3]),
            ActionType::KyushuKyuhai => Some([CAND_TYPE_KYUSHU_KYUHAI, 3]),
            ActionType::Pass => Some([CAND_TYPE_PASS, 3]),
            ActionType::Chi => {
                action.tile?;
                if action.consume_tiles.len() < 2 {
                    return None;
                }

                // from = relative seat of the discard source
                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);

                Some([CAND_TYPE_CHI, rel as u16])
            }
            ActionType::Pon => {
                action.tile?;
                if action.consume_tiles.len() < 2 {
                    return None;
                }

                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);

                Some([CAND_TYPE_PON, rel as u16])
            }
            ActionType::Daiminkan => {
                action.tile?;

                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);

                Some([CAND_TYPE_DAIMINKAN, rel as u16])
            }
            ActionType::Ron => {
                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);
                Some([CAND_TYPE_RON, rel as u16])
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
    fn test_relative_from() {
        // Player 0 calling from player 3 → kamicha (left) = 2
        assert_eq!(relative_from(0, 3), 2);
        // Player 0 calling from player 1 → shimocha (right) = 0
        assert_eq!(relative_from(0, 1), 0);
        // Player 0 calling from player 2 → toimen (across) = 1
        assert_eq!(relative_from(0, 2), 1);
        // Player 2 calling from player 3 → shimocha = 0
        assert_eq!(relative_from(2, 3), 0);
    }

    #[test]
    fn test_sparse_vocab_bounds() {
        // Verify all sparse offsets are within vocab
        assert!(SPARSE_PAD < SPARSE_VOCAB_SIZE as u16);

        // Dora max: 83 + 4*37 + 36 = 83 + 148 + 36 = 267
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
        );

        let hand = obs.encode_seq_hand();
        assert_eq!(hand, vec![[1, 0], [2, 0], [3, 0]]);
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
                vec![],
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
        );

        let sparse = obs.encode_seq_sparse(1);
        assert!(sparse.iter().all(|&t| t < SPARSE_PAD));

        let melds = obs.encode_seq_sparse_melds();
        assert_eq!(melds.len(), 2);
        assert_eq!(melds[0][0], MELD_KIND_PON);
        assert_eq!(melds[1][0], MELD_KIND_CHI);
    }

    #[test]
    fn test_progression_type_bounds() {
        let prog_type_max = PROG_DIMS[1];
        assert!(43 < prog_type_max); // padding
        assert!(42 < prog_type_max); // kakan
    }

    #[test]
    fn test_candidate_type_bounds() {
        let cand_type_max = CAND_DIMS[0];
        assert!(33 < cand_type_max); // discard max: 33
        assert!(44 < cand_type_max); // padding
        assert!(43 < cand_type_max); // ron
    }
}
