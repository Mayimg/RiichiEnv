use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::score::calculate_score;

const N_PLAYERS: usize = 4;
pub const TENHOU_4P_SELF_AGARI_DIM: usize = 96;
pub const TENHOU_4P_OTHER_OVERTAKE_DIM: usize = 288;
pub const TENHOU_4P_GRP_INPUT_DIM: usize = 404;
const TENHOU_4P_TSUMO_PATTERN_COUNT: usize = 24;
const TENHOU_4P_RON_PATTERN_COUNT: usize = 24;

const STANDARD_TSUMO_HAN_FU: &[(u8, u8)] = &[
    (1, 30),
    (1, 40),
    (1, 50),
    (1, 60),
    (1, 70),
    (1, 80),
    (1, 90),
    (1, 100),
    (1, 110),
    (2, 20),
    (2, 25),
    (2, 30),
    (2, 40),
    (2, 50),
    (2, 60),
    (2, 70),
    (2, 80),
    (2, 90),
    (2, 100),
    (2, 110),
    (3, 20),
    (3, 25),
    (3, 30),
    (3, 40),
    (3, 50),
    (3, 60),
    (4, 20),
    (4, 25),
    (4, 30),
];
const STANDARD_RON_HAN_FU: &[(u8, u8)] = &[
    (1, 30),
    (1, 40),
    (1, 50),
    (1, 60),
    (1, 70),
    (1, 80),
    (1, 90),
    (1, 100),
    (1, 110),
    (2, 25),
    (2, 30),
    (2, 40),
    (2, 50),
    (2, 60),
    (2, 70),
    (2, 80),
    (2, 90),
    (2, 100),
    (2, 110),
    (3, 25),
    (3, 30),
    (3, 40),
    (3, 50),
    (3, 60),
    (4, 25),
    (4, 30),
];
const LIMIT_HAND_HANS: &[u8] = &[5, 6, 8, 11];
const MAX_YAKUMAN_MULTIPLIER: u8 = 4;

#[derive(Clone, Copy)]
struct PatternTables {
    tsumo: [(i32, i32); TENHOU_4P_TSUMO_PATTERN_COUNT],
    ron: [i32; TENHOU_4P_RON_PATTERN_COUNT],
}

pub fn encode_tenhou_4p_rows(
    start_scores: [i32; N_PLAYERS],
    delta_scores: [i32; N_PLAYERS],
    chang: u8,
    ju: u8,
    ben: u8,
    liqibang: u8,
) -> Vec<f32> {
    let current_ranks = stable_ranks(&start_scores);
    let oya = (ju as usize) % N_PLAYERS;
    let winner_next_ranks = compute_all_winner_next_ranks(&start_scores, oya, ben, liqibang);

    let mut out = vec![0.0f32; N_PLAYERS * TENHOU_4P_GRP_INPUT_DIM];
    for focus in 0..N_PLAYERS {
        let row = &mut out[focus * TENHOU_4P_GRP_INPUT_DIM..(focus + 1) * TENHOU_4P_GRP_INPUT_DIM];
        let mut idx = 0;

        for score in start_scores {
            row[idx] = score as f32 / 25000.0;
            idx += 1;
        }
        for delta in delta_scores {
            row[idx] = delta as f32 / 12000.0;
            idx += 1;
        }
        row[idx] = chang as f32 / 3.0;
        idx += 1;
        row[idx] = ju as f32 / 3.0;
        idx += 1;
        row[idx] = ben as f32 / 4.0;
        idx += 1;
        row[idx] = liqibang as f32 / 4.0;
        idx += 1;

        row[idx + focus] = 1.0;
        idx += N_PLAYERS;

        row[idx + current_ranks[focus] as usize] = 1.0;
        idx += N_PLAYERS;

        let focus_rank = current_ranks[focus];
        for next_ranks in &winner_next_ranks[focus] {
            let gain = (focus_rank as i32 - next_ranks[focus] as i32).max(0) as f32 / 3.0;
            row[idx] = gain;
            idx += 1;
        }

        for other in 0..N_PLAYERS {
            if other == focus {
                continue;
            }
            let started_below = current_ranks[other] > focus_rank;
            for next_ranks in &winner_next_ranks[other] {
                let overtook = started_below && next_ranks[other] < next_ranks[focus];
                row[idx] = if overtook { 1.0 } else { 0.0 };
                idx += 1;
            }
        }

        debug_assert_eq!(idx, TENHOU_4P_GRP_INPUT_DIM);
    }

    out
}

pub fn stable_ranks(scores: &[i32; N_PLAYERS]) -> [u8; N_PLAYERS] {
    let mut indexed = [(0usize, 0i32); N_PLAYERS];
    for (seat, score) in scores.iter().copied().enumerate() {
        indexed[seat] = (seat, score);
    }
    indexed.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut ranks = [0u8; N_PLAYERS];
    for (rank, (seat, _)) in indexed.into_iter().enumerate() {
        ranks[seat] = rank as u8;
    }
    ranks
}

fn compute_all_winner_next_ranks(
    start_scores: &[i32; N_PLAYERS],
    oya: usize,
    ben: u8,
    liqibang: u8,
) -> [[[u8; N_PLAYERS]; TENHOU_4P_SELF_AGARI_DIM]; N_PLAYERS] {
    let mut out = [[[0u8; N_PLAYERS]; TENHOU_4P_SELF_AGARI_DIM]; N_PLAYERS];
    for (winner, winner_out) in out.iter_mut().enumerate() {
        *winner_out = compute_winner_next_ranks(start_scores, oya, ben, liqibang, winner);
    }
    out
}

fn compute_winner_next_ranks(
    start_scores: &[i32; N_PLAYERS],
    oya: usize,
    ben: u8,
    liqibang: u8,
    winner: usize,
) -> [[u8; N_PLAYERS]; TENHOU_4P_SELF_AGARI_DIM] {
    let patterns = tenhou_4p_base_agari_patterns(winner == oya);
    let mut out = [[0u8; N_PLAYERS]; TENHOU_4P_SELF_AGARI_DIM];
    let deposit = liqibang as i32 * 1000;
    let honba = ben as i32;
    let mut idx = 0;

    for &(pay_oya, pay_ko) in &patterns.tsumo {
        let mut next_scores = *start_scores;
        let mut total_win = deposit;
        for (seat, next_score) in next_scores.iter_mut().enumerate() {
            if seat == winner {
                continue;
            }
            let mut pay = if winner == oya || seat != oya {
                pay_ko
            } else {
                pay_oya
            };
            pay += honba * 100;
            *next_score -= pay;
            total_win += pay;
        }
        next_scores[winner] += total_win;
        out[idx] = stable_ranks(&next_scores);
        idx += 1;
    }

    for target in 0..N_PLAYERS {
        if target == winner {
            continue;
        }
        for &pay_ron in &patterns.ron {
            let mut next_scores = *start_scores;
            let total_gain = pay_ron + honba * 300;
            next_scores[target] -= total_gain;
            next_scores[winner] += total_gain + deposit;
            out[idx] = stable_ranks(&next_scores);
            idx += 1;
        }
    }

    debug_assert_eq!(idx, TENHOU_4P_SELF_AGARI_DIM);
    out
}

fn tenhou_4p_base_agari_patterns(is_oya: bool) -> &'static PatternTables {
    static KO_PATTERNS: OnceLock<PatternTables> = OnceLock::new();
    static OYA_PATTERNS: OnceLock<PatternTables> = OnceLock::new();

    if is_oya {
        OYA_PATTERNS.get_or_init(|| build_pattern_tables(true))
    } else {
        KO_PATTERNS.get_or_init(|| build_pattern_tables(false))
    }
}

fn build_pattern_tables(is_oya: bool) -> PatternTables {
    let mut tsumo = BTreeSet::new();
    let mut ron = BTreeSet::new();

    for &(han, fu) in STANDARD_TSUMO_HAN_FU {
        let score = calculate_score(han, fu, is_oya, true, 0, 4);
        tsumo.insert((score.pay_tsumo_oya as i32, score.pay_tsumo_ko as i32));
    }
    for &(han, fu) in STANDARD_RON_HAN_FU {
        let score = calculate_score(han, fu, is_oya, false, 0, 4);
        ron.insert(score.pay_ron as i32);
    }
    for &han in LIMIT_HAND_HANS {
        let tsumo_score = calculate_score(han, 30, is_oya, true, 0, 4);
        let ron_score = calculate_score(han, 30, is_oya, false, 0, 4);
        tsumo.insert((
            tsumo_score.pay_tsumo_oya as i32,
            tsumo_score.pay_tsumo_ko as i32,
        ));
        ron.insert(ron_score.pay_ron as i32);
    }
    for yakuman_count in 1..=MAX_YAKUMAN_MULTIPLIER {
        let han = 13 * yakuman_count;
        let tsumo_score = calculate_score(han, 0, is_oya, true, 0, 4);
        let ron_score = calculate_score(han, 0, is_oya, false, 0, 4);
        tsumo.insert((
            tsumo_score.pay_tsumo_oya as i32,
            tsumo_score.pay_tsumo_ko as i32,
        ));
        ron.insert(ron_score.pay_ron as i32);
    }

    let tsumo_vec: Vec<_> = tsumo.into_iter().collect();
    let ron_vec: Vec<_> = ron.into_iter().collect();
    debug_assert_eq!(tsumo_vec.len(), TENHOU_4P_TSUMO_PATTERN_COUNT);
    debug_assert_eq!(ron_vec.len(), TENHOU_4P_RON_PATTERN_COUNT);

    PatternTables {
        tsumo: tsumo_vec
            .try_into()
            .expect("tenhou 4p tsumo pattern count must stay fixed"),
        ron: ron_vec
            .try_into()
            .expect("tenhou 4p ron pattern count must stay fixed"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TENHOU_4P_GRP_INPUT_DIM, TENHOU_4P_OTHER_OVERTAKE_DIM, TENHOU_4P_SELF_AGARI_DIM,
        encode_tenhou_4p_rows, stable_ranks,
    };

    #[test]
    fn stable_ranks_follow_engine_tiebreak() {
        let ranks = stable_ranks(&[25000, 25000, 24000, 24000]);
        assert_eq!(ranks, [0, 1, 2, 3]);
    }

    #[test]
    fn encode_tenhou_4p_rows_has_expected_shape_and_signals() {
        let rows = encode_tenhou_4p_rows([25000, 24000, 24000, 24000], [0, 0, 0, 0], 0, 0, 0, 0);
        assert_eq!(rows.len(), 4 * TENHOU_4P_GRP_INPUT_DIM);

        let focus_p3 = &rows[3 * TENHOU_4P_GRP_INPUT_DIM..4 * TENHOU_4P_GRP_INPUT_DIM];
        let agari_start = 20;
        let overtake_start = agari_start + TENHOU_4P_SELF_AGARI_DIM;
        assert_eq!(focus_p3[agari_start + 24], 1.0);
        assert_eq!(focus_p3[agari_start + 48], 2.0 / 3.0);
        assert_eq!(focus_p3[overtake_start], 0.0);

        let focus_p0 = &rows[..TENHOU_4P_GRP_INPUT_DIM];
        assert_eq!(focus_p0[overtake_start], 1.0);
        assert_eq!(focus_p0[overtake_start + 96], 1.0);
        assert_eq!(focus_p0[overtake_start + 192], 1.0);
        assert_eq!(TENHOU_4P_OTHER_OVERTAKE_DIM, 288);
    }
}
