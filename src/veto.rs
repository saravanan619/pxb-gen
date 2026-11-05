//! src/veto.rs
//!
//! Rust port of the JS `vetoPenaltyPass` logic, upgraded with fast early-exit
//! guards, pin-aware SEE, Pixie-specific hazard checks (PhaseRook, Bouncer,
//! Bishopper/BisHop, Basilisk paralysis, WarAutomaton drift, SumoRook), and
//! mating-net / smothered-mate detection.
//!
//!   * `opp_has_mate_in_1`   — does the opponent have an immediate mate?
//!   * `king_move_safe`      — is the king safe after this move?
//!   * `xray_hang_penalty`   — is the moved piece x-ray hung?
//!   * `see_capture`         — static exchange evaluation for a capture
//!   * `veto_score`          — combined penalty; below VETO_THRESHOLD = reject
//!
//! All helpers use the exact signatures and types from `pos.rs`, `types.rs`,
//! `attacks.rs`, `movegen.rs`, and `eval.rs`.

use crate::apply_full::{apply_full, apply_full_no_wa};
use crate::attacks::{av_suppressed, find_king, is_checkmate, king_attacked, square_attacked};
use crate::eval::{init_eval_sum, opp_attacking_phase, piece_value};
use crate::movegen::{family_of, gen_moves, piece_attacks};
use crate::pos::{is_paralyzed, occ, slot_of, Piece, Pos};
use crate::types::{
    is_non_capturer, is_pawn_like, sq, sq_x, sq_y, xy_on_board, Move, PieceKey,
    MF_CASTLE_K, MF_CASTLE_Q, MF_EP, MF_PROMO, PROVISIONAL_ICICLE_FREEZE,
    PROVISIONAL_IRON_INVULN, SQ_NONE,
};

// ═══════════════════════════════════════════════════════════════════════════
// Public constants
// ═══════════════════════════════════════════════════════════════════════════

/// A root move whose `veto_score` is strictly below this value is rejected.
pub const VETO_THRESHOLD: i32 = -50_000;

/// Hard veto: opponent has an immediate mate after this move.
const VETO_MATE_IN_1: i32 = -1_000_000;

/// Hard veto: the move leaves our king-like piece on an attacked square.
const VETO_KING_UNSAFE: i32 = -100_000;

/// Hard veto (DeepSeek): moved piece is clearly hanging (SEE net loss >= 300cp).
const VETO_HANGING: i32 = -60_000;

/// Hard veto (DeepSeek): move creates a pin that enables mate threat.
const VETO_PIN_MATE: i32 = -70_000;

/// Hanging: net loss threshold (cp) that triggers hard veto.
const HANG_HARD_LOSS: i32 = 300;

/// Hanging: soft penalty range (scaled by loss size).
const HANG_SOFT_MIN: i32 = 150;
const HANG_SOFT_MAX: i32 = 900;

/// Soft penalty: SEE shows a losing exchange worth > 100 cp.
const SEE_HARD_PEN: i32 = 150;

/// Soft penalty: SEE shows a losing exchange of 1..=100 cp.
const SEE_SOFT_PEN: i32 = 40;

/// Soft penalty: piece worth >= 300 is x-ray hung behind a single pawn.
const XRAY_PEN: i32 = 250;

/// Soft penalty: valuable piece steps into Basilisk paralysis or WarAutomaton drift danger.
const PIXIE_TRAP_PEN: i32 = 220;

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Is this piece a king-like piece? Matches the "king"/"rocketman" set from
/// the JS veto, broadened to include the extra king variants this engine has.
#[inline]
fn is_king_piece(p: &Piece) -> bool {
    use PieceKey::*;
    let k = p.key;
    let s = p.sub_key;
    matches!(
        k,
        King | WarKing | MeditatingKing | PopeKing | IrreverentKing | Rocketman
    ) || matches!(
        s,
        King | WarKing | MeditatingKing | PopeKing | IrreverentKing | Rocketman
    )
}

/// Get the value of the piece currently sitting on `sq`, or 0 if empty.
#[inline]
fn value_at(pos: &Pos, sq_idx: u8) -> i32 {
    if sq_idx >= 64 {
        return 0;
    }
    slot_of(pos, sq_idx)
        .map(|i| piece_value(&pos.pieces[i]))
        .unwrap_or(0)
}

#[inline]
fn is_cardinal_slider(k: PieceKey, sk: PieceKey) -> bool {
    let f1 = family_of(k);
    let f2 = family_of(sk);
    f1 == 4 || f1 == 5 || f2 == 4 || f2 == 5 || k == PieceKey::Pilgrim || sk == PieceKey::Pilgrim
}

#[inline]
fn is_diagonal_slider(k: PieceKey, sk: PieceKey) -> bool {
    let f1 = family_of(k);
    let f2 = family_of(sk);
    f1 == 3 || f1 == 5 || f2 == 3 || f2 == 5
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek safety patches: pin detection, hanging pieces, pin-mate threats
// ═══════════════════════════════════════════════════════════════════════════

/// Internal pin check that reuses a precomputed occupancy array `base_occ` to
/// avoid re-allocating/re-building occupancy on every candidate attacker.
/// Also handles Pixie phasing/leaping sliders (`PhaseRook`, crouched
/// `Bishopper`/`BisHop`, `Bouncer`).
#[inline]
fn is_absolutely_pinned_with_occ(
    after: &Pos,
    sq_idx: u8,
    color: u8,
    base_occ: &[bool; 64],
) -> bool {
    if sq_idx >= 64 {
        return false; // v131: guard SQ_NONE (255)
    }
    let king_sq = match find_king(after, color) {
        Some(idx) => {
            let k = &after.pieces[idx];
            if !xy_on_board(k.x, k.y) {
                return false;
            }
            sq(k.x as i8, k.y as i8)
        }
        None => return false,
    };
    if sq_idx == king_sq {
        return false;
    }

    let kx = sq_x(king_sq) as i8;
    let ky = sq_y(king_sq) as i8;
    let px = sq_x(sq_idx) as i8;
    let py = sq_y(sq_idx) as i8;
    let dx = (px - kx).signum();
    let dy = (py - ky).signum();

    let aligned = (dx == 0 && dy != 0)
        || (dy == 0 && dx != 0)
        || (dx != 0 && dy != 0 && (px - kx).abs() == (py - ky).abs());

    let mut occ2 = *base_occ;
    occ2[sq_idx as usize] = false;

    if aligned {
        // Walk from king through sq_idx and beyond, allowing up to 1 additional blocker
        // in case an enemy PhaseRook or crouched Bishopper/BisHop is pinning through a blocker.
        let mut x = kx + dx;
        let mut y = ky + dy;
        let mut blockers_seen = 0u8;
        while (x as u8) < 8 && (y as u8) < 8 {
            let s = sq(x, y);
            if occ2[s as usize] {
                if let Some(pi) = slot_of(after, s) {
                    let q = &after.pieces[pi];
                    if q.alive && q.player != color {
                        if piece_attacks(after, pi, &occ2, true, false).has(king_sq) {
                            return true;
                        }
                    }
                }
                blockers_seen += 1;
                if blockers_seen >= 2 {
                    break;
                }
            }
            x += dx;
            y += dy;
        }
    }

    // Check if an enemy Bouncer (which reflects off board edges) pins `sq_idx` to `king_sq`.
    for i in 0..after.pieces.len() {
        let q = &after.pieces[i];
        if !q.alive || q.player == color || !xy_on_board(q.x, q.y) {
            continue;
        }
        if q.key == PieceKey::Bouncer || q.sub_key == PieceKey::Bouncer {
            if !piece_attacks(after, i, base_occ, true, false).has(king_sq)
                && piece_attacks(after, i, &occ2, true, false).has(king_sq)
            {
                return true;
            }
        }
    }

    false
}

/// Is the piece on `sq_idx` (belonging to `color`) absolutely pinned to its
/// own king-like piece? A pinned piece cannot legally move off the pin line,
/// so it cannot capture our hanging piece.
fn is_absolutely_pinned(after: &Pos, sq_idx: u8, color: u8) -> bool {
    if sq_idx >= 64 {
        return false; // v131: guard SQ_NONE (255)
    }
    let occ_local = occ(after);
    is_absolutely_pinned_with_occ(after, sq_idx, color, &occ_local)
}

/// Single-pinner info for an absolutely pinned piece: (pinner square, king
/// square, pin direction dx/dy from the king, pinner distance from the king).
fn pin_info(
    pos: &Pos,
    qsq: u8,
    color: u8,
    base_occ: &[bool; 64],
) -> Option<(u8, u8, i8, i8, i32)> {
    if qsq >= 64 {
        return None;
    }
    let king_idx = find_king(pos, color)?;
    let k = &pos.pieces[king_idx];
    if !k.alive || !xy_on_board(k.x, k.y) {
        return None;
    }
    let king_sq = sq(k.x as i8, k.y as i8);
    if qsq == king_sq {
        return None;
    }
    let kx = sq_x(king_sq) as i8;
    let ky = sq_y(king_sq) as i8;
    let px = sq_x(qsq) as i8;
    let py = sq_y(qsq) as i8;
    let dx = (px - kx).signum();
    let dy = (py - ky).signum();
    let aligned = (dx == 0 && dy != 0)
        || (dy == 0 && dx != 0)
        || (dx != 0 && dy != 0 && (px - kx).abs() == (py - ky).abs());
    if !aligned {
        return None;
    }
    let dq = (px - kx).abs().max((py - ky).abs()) as i32;
    let mut occ2 = *base_occ;
    occ2[qsq as usize] = false;
    let mut x = kx + dx;
    let mut y = ky + dy;
    let mut dist = 1i32;
    while x >= 0 && x < 8 && y >= 0 && y < 8 {
        let s = sq(x, y);
        if occ2[s as usize] {
            if dist < dq {
                return None;
            }
            let pi = slot_of(pos, s)?;
            let q = &pos.pieces[pi];
            if !q.alive || q.player == color {
                return None;
            }
            if piece_attacks(pos, pi, &occ2, true, false).has(king_sq) {
                return Some((s, king_sq, dx, dy, dist));
            }
            return None;
        }
        x += dx;
        y += dy;
        dist += 1;
    }
    None
}

/// Target-aware capture legality. A pinned piece CAN capture the pinner and
/// CAN capture along the pin segment [king, pinner].
fn can_capture_piece_to(
    pos: &Pos,
    qi: usize,
    target_sq: u8,
    occ_local: &[bool; 64],
) -> bool {
    let q = &pos.pieces[qi];
    if !q.alive || !xy_on_board(q.x, q.y) {
        return false;
    }
    let qk = q.key;
    let qs = q.sub_key;
    if is_non_capturer(qk) || is_non_capturer(qs) {
        return false;
    }
    if qk == PieceKey::Icicle || qs == PieceKey::Icicle {
        return false;
    }
    if qk == PieceKey::BladeRunner || qs == PieceKey::BladeRunner {
        return false;
    }
    if av_suppressed(pos, qi) {
        return false;
    }
    if PROVISIONAL_ICICLE_FREEZE && q.frozen() {
        return false;
    }
    if is_paralyzed(pos, qi, occ_local) {
        return false;
    }
    let qsq = sq(q.x as i8, q.y as i8);
    if is_absolutely_pinned_with_occ(pos, qsq, q.player, occ_local) {
        if let Some((pinner_sq, king_sq, dx, dy, dq)) =
            pin_info(pos, qsq, q.player, occ_local)
        {
            if target_sq == pinner_sq {
                return true;
            }
            let tx = sq_x(target_sq) as i8;
            let ty = sq_y(target_sq) as i8;
            let kx = sq_x(king_sq) as i8;
            let ky = sq_y(king_sq) as i8;
            let tdx = tx - kx;
            let tdy = ty - ky;
            let on_ray = (dx == 0 && tdx == 0 && tdy.signum() == dy)
                || (dy == 0 && tdy == 0 && tdx.signum() == dx)
                || (dx != 0
                    && dy != 0
                    && tdx.signum() == dx
                    && tdy.signum() == dy
                    && tdx.abs() == tdy.abs());
            if on_ray {
                let dt = tdx.abs().max(tdy.abs()) as i32;
                if dt < dq {
                    return true;
                }
            }
        }
        return false;
    }
    true
}

/// Can piece `q` (at index `qi`) legally capture `target_sq`? Skips non-capturers,
/// frozen, paralyzed, suppressed, and pieces pinned off the capture line.
#[inline]
fn can_capture_piece(after: &Pos, qi: usize, occ_local: &[bool; 64]) -> bool {
    let q = &after.pieces[qi];
    if !q.alive || !xy_on_board(q.x, q.y) {
        return false;
    }
    let qk = q.key;
    let qs = q.sub_key;
    if is_non_capturer(qk) || is_non_capturer(qs) {
        return false;
    }
    if qk == PieceKey::Icicle || qs == PieceKey::Icicle {
        return false;
    }
    if qk == PieceKey::BladeRunner || qs == PieceKey::BladeRunner {
        return false;
    }
    if av_suppressed(after, qi) {
        return false;
    }
    if PROVISIONAL_ICICLE_FREEZE && q.frozen() {
        return false;
    }
    if is_paralyzed(after, qi, occ_local) {
        return false;
    }
    let qsq = sq(q.x as i8, q.y as i8);
    if is_absolutely_pinned_with_occ(after, qsq, q.player, occ_local) {
        return false;
    }
    true
}

/// DeepSeek `hang_penalty`: detects the moved piece left hanging on `after`.
///
/// Returns a negative penalty (0 = safe). Returns `VETO_HANGING` (hard veto)
/// when the SEE net loss >= `HANG_HARD_LOSS` (300cp) after netting the
/// material the move itself captured.
pub fn hang_penalty(pos: &Pos, after: &Pos, m: &Move, me: u8, opp: u8) -> i32 {
    let occ_local = occ(after);
    hang_penalty_with_occ(pos, after, m, me, opp, &occ_local)
}

fn hang_penalty_with_occ(
    pos: &Pos,
    after: &Pos,
    m: &Move,
    me: u8,
    opp: u8,
    occ_local: &[bool; 64],
) -> i32 {
    let to = m.to;
    if to >= 64 {
        return 0;
    }
    let piece_idx = match slot_of(after, to) {
        Some(i) => i,
        None => return 0,
    };
    let piece = &after.pieces[piece_idx];
    if piece.player != me || !piece.alive {
        return 0;
    }
    // King-like movers are handled by the king-safety path.
    if is_king_piece(piece) {
        return 0;
    }
    // Invulnerable pieces cannot be captured — nothing to lose.
    if PROVISIONAL_IRON_INVULN
        && (piece.key == PieceKey::IronPawn || piece.sub_key == PieceKey::IronPawn)
    {
        return 0;
    }

    let mover_val = piece_value(piece);

    // Find cheapest enemy attacker that can legally capture `to` (including WarAutomaton drift).
    let mut cheapest_attacker_val = i32::MAX;
    let mut any_attacker = false;
    for i in 0..after.pieces.len() {
        let q = &after.pieces[i];
        if !xy_on_board(q.x, q.y) {
            continue;
        } // v130
        if !q.alive || q.player != opp {
            continue;
        }
        // Skip kings (king captures handled separately).
        if q.key == PieceKey::King || q.sub_key == PieceKey::King {
            continue;
        }
        if !can_capture_piece(after, i, occ_local) {
            continue;
        }
        if !piece_attacks(after, i, occ_local, false, false).has(to) {
            continue;
        }
        any_attacker = true;
        // HordeMother captures also spawn an 80cp Hordeling for the opponent,
        // so treat HordeMother as a cheaper/more eager attacker in hang checks.
        let v = if q.key == PieceKey::HordeMother || q.sub_key == PieceKey::HordeMother {
            (piece_value(q) - 80).max(100)
        } else {
            piece_value(q)
        };
        if v < cheapest_attacker_val {
            cheapest_attacker_val = v;
        }
    }
    if !any_attacker {
        return 0;
    }

    // Is the square defended by a non-pinned friendly piece?
    let mut defended = false;
    for i in 0..after.pieces.len() {
        if i == piece_idx {
            continue;
        }
        let q = &after.pieces[i];
        if !xy_on_board(q.x, q.y) || !q.alive || q.player != me {
            continue;
        }
        if is_king_piece(q) {
            if !av_suppressed(after, i)
                && !is_paralyzed(after, i, occ_local)
                && piece_attacks(after, i, occ_local, true, false).has(to)
            {
                defended = true;
                break;
            }
            continue;
        }
        if !can_capture_piece(after, i, occ_local) {
            continue;
        }
        if piece_attacks(after, i, occ_local, true, false).has(to) {
            defended = true;
            break;
        }
    }

    // Approximate net loss.
    let net_loss = if defended {
        (mover_val - cheapest_attacker_val).max(0)
    } else {
        mover_val
    };
    if net_loss <= 0 {
        return 0;
    }

    // Net against material the move itself won (capture gain).
    let capture_gain = if m.flags & MF_EP != 0 {
        100
    } else {
        value_at(pos, m.to)
    };
    let effective_loss = net_loss - capture_gain;
    if effective_loss <= 0 {
        return 0;
    }

    if effective_loss >= HANG_HARD_LOSS {
        return VETO_HANGING; // hard veto: clearly hanging
    }

    // Soft penalty scaled by loss size, clamped to [150, 900].
    -(effective_loss.clamp(HANG_SOFT_MIN, HANG_SOFT_MAX))
}

/// DeepSeek `pin_mate_threat`: does our move create an absolute pin where
/// the pinned piece was the sole defender of a king-adjacent mating square?
///
/// Hard veto (`VETO_PIN_MATE`) only for the strong signal: pinned piece was
/// the only defender of an empty square adjacent to our king, and an enemy
/// piece attacks that square.
pub fn pin_mate_threat(after: &Pos, m: &Move, me: u8, opp: u8) -> bool {
    let occ_local = occ(after);
    pin_mate_threat_with_occ(after, m, me, opp, &occ_local)
}

fn pin_mate_threat_with_occ(
    after: &Pos,
    m: &Move,
    me: u8,
    opp: u8,
    occ_local: &[bool; 64],
) -> bool {
    let to = m.to;
    if to >= 64 {
        return false;
    }
    let king_sq = match find_king(after, me) {
        Some(idx) => {
            let k = &after.pieces[idx];
            if !xy_on_board(k.x, k.y) {
                return false;
            }
            sq(k.x as i8, k.y as i8)
        }
        None => return false,
    };
    let kx = sq_x(king_sq) as i8;
    let ky = sq_y(king_sq) as i8;
    let px = sq_x(to) as i8;
    let py = sq_y(to) as i8;
    let dx = (px - kx).signum();
    let dy = (py - ky).signum();
    if dx != 0 && dy != 0 && (px - kx).abs() != (py - ky).abs() {
        return false;
    }
    if dx == 0 && dy == 0 {
        return false;
    }
    if !is_absolutely_pinned_with_occ(after, to, me, occ_local) {
        return false;
    }
    for ax in (kx - 1).max(0)..=(kx + 1).min(7) {
        for ay in (ky - 1).max(0)..=(ky + 1).min(7) {
            let adj = sq(ax, ay);
            if adj == king_sq || occ_local[adj as usize] {
                continue;
            }
            let mut defenders = 0;
            for i in 0..after.pieces.len() {
                let q = &after.pieces[i];
                if !xy_on_board(q.x, q.y) {
                    continue;
                } // v130
                if !q.alive || q.player != me {
                    continue;
                }
                let qsq = sq(q.x as i8, q.y as i8);
                if qsq == to {
                    continue; // exclude pinned piece
                }
                if !can_capture_piece(after, i, occ_local) {
                    continue;
                }
                if piece_attacks(after, i, occ_local, true, false).has(adj) {
                    defenders += 1;
                    break;
                }
            }
            if defenders == 0 && square_attacked(after, adj, opp, occ_local) {
                return true;
            }
        }
    }
    false
}

// DeepSeek v116: hyper-aggressive paranoid veto
//
// Paranoid material guard. Three phases:
//   1. The move is already simulated by the caller (`after`). Compute its
//      concrete immediate gain: captured value + promotion bonus, minus a
//      100cp tempo cost (the move must pay for itself).
//   2. Paranoid SEE scan: from the opponent's perspective, run
//      `see_recapture` on EVERY own piece worth >= 300cp on `after`.
//      The deficit is the worst NEWLY-created exposure (after minus before)
//      minus the concrete gain. Pre-existing exposure is not this move's
//      fault, so it is subtracted.
//   3. Nuclear penalty schedule on the deficit (cp):
//      >= 900: -1_000_000 - deficit*500   (queen-tier)
//      >= 500: -1_000_000 - deficit*300   (rook-tier)
//      >= 300: -1_000_000 - deficit*200   (minor-tier: the Q/N/B case)
//      >= 200: -1_000_000 - deficit*100
//      >= 100:   -150_000 - deficit*100
//      >=  50:    -50_000 - deficit*200
//      <   50:             -deficit*100   (soft: re-ranks only)
//
// Returns <= -1_000_000 for hard rejects (skipped without recursing).
pub fn hyper_aggressive_veto(pos: &Pos, after: &Pos, m: &Move, me: u8, opp: u8) -> i32 {
    let before_occ = occ(pos);
    let after_occ = occ(after);
    hyper_aggressive_veto_with_occ(pos, after, m, me, opp, &before_occ, &after_occ)
}

fn hyper_aggressive_veto_with_occ(
    pos: &Pos,
    after: &Pos,
    m: &Move,
    me: u8,
    opp: u8,
    before_occ: &[bool; 64],
    after_occ: &[bool; 64],
) -> i32 {
    // RUN16 game-1 fix (turn 71): a king move made while in check must not
    // be nuked for "newly exposing" a piece the king's old square was
    // shielding. The king on d7 blocked Rook f7's ray onto Marauder c7,
    // so EVERY king departure — step or teleport — showed a new exposure
    // and the veto rejected all 26 root moves, leaving zero ranked
    // candidates and an abandonment despite four server-legal escapes.
    // When in check the king cannot stay: the "before" exposure is not an
    // available alternative, and the search's own evaluation of the
    // resulting position is the right judge. Non-king moves in check
    // (blocks/captures) do not vacate the king square, so their exposure
    // accounting is unaffected and stays vetoed as before.
    if m.from < 64 {
        if let Some(s) = slot_of(pos, m.from) {
            if is_king_piece(&pos.pieces[s]) && king_attacked(pos, me, opp, Some(before_occ)) {
                return 0;
            }
        }
    }
    // Phase 1: concrete immediate gain.
    let mut gain = value_at(pos, m.to);
    if m.flags & MF_EP != 0 {
        gain += 100;
    }
    if m.flags & MF_PROMO != 0 && m.promo != 0 {
        gain += 800;
    }
    gain -= 100; // tempo cost: the move must pay for itself
    let gain = gain.max(0);

    // False-positive guard: if our move delivers check to the enemy king AND the
    // checking piece itself does not hang on `m.to`, the opponent must answer the
    // check first and cannot freely cash in on an unrelated square.
    let delivers_check = king_attacked(after, opp, me, Some(after_occ));

    // Phase 2: per-piece new exposure of valuable own pieces (>= 300cp).
    // Replaces board-wide global max subtraction so already-hanging pieces cannot
    // mask newly created exposures on other pieces (sunk-cost masking / Case E).
    let mut new_exposure = 0i32;
    if delivers_check {
        if m.to < 64 {
            let after_loss = see_recapture_with_occ(after, m.to, me, opp, after_occ);
            let before_loss = if m.from < 64 {
                see_recapture_with_occ(pos, m.from, me, opp, before_occ)
            } else {
                0
            };
            new_exposure = after_loss.saturating_sub(before_loss);
        }
    } else {
        for i in 0..after.pieces.len() {
            let q = &after.pieces[i];
            if !q.alive || q.player != me || is_king_piece(q) {
                continue;
            }
            if piece_value(q) < 300 || !xy_on_board(q.x, q.y) {
                continue;
            }
            let after_sq = sq(q.x as i8, q.y as i8);
            let after_loss = see_recapture_with_occ(after, after_sq, me, opp, after_occ);
            if after_loss <= 0 {
                continue;
            }
            if i >= pos.pieces.len() {
                continue;
            }
            let b = &pos.pieces[i];
            if !b.alive || b.player != me || !xy_on_board(b.x, b.y) {
                continue;
            }
            let before_loss =
                see_recapture_with_occ(pos, sq(b.x as i8, b.y as i8), me, opp, before_occ);
            let delta = after_loss.saturating_sub(before_loss);
            if delta > new_exposure {
                new_exposure = delta;
            }
        }
    }
    if new_exposure <= 0 {
        return 0;
    }
    let deficit = new_exposure - gain;
    if deficit <= 0 {
        return 0;
    }

    // Phase 3: nuclear schedule.
    if deficit >= 900 {
        -1_000_000 - deficit * 500
    } else if deficit >= 500 {
        -1_000_000 - deficit * 300
    } else if deficit >= 300 {
        -1_000_000 - deficit * 200
    } else if deficit >= 200 {
        -1_000_000 - deficit * 100
    } else if deficit >= 100 {
        -150_000 - deficit * 100
    } else if deficit >= 50 {
        -50_000 - deficit * 200
    } else {
        -deficit * 100
    }
}

/// Helper: worst `see_recapture` loss among own pieces worth >= 300cp
/// (king-like pieces excluded - the king-safety path owns those).
fn worst_valuable_exposure_with_occ(p: &Pos, me: u8, opp: u8, occ_local: &[bool; 64]) -> i32 {
    let mut worst = 0i32;
    for i in 0..p.pieces.len() {
        let q = &p.pieces[i];
        if !xy_on_board(q.x, q.y) {
            continue;
        } // v130
        if !q.alive || q.player != me {
            continue;
        }
        if is_king_piece(q) {
            continue;
        }
        let val = piece_value(q);
        if val < 300 || val <= worst {
            continue;
        }
        let qsq = sq(q.x as i8, q.y as i8);
        let loss = see_recapture_with_occ(p, qsq, me, opp, occ_local);
        if loss > worst {
            worst = loss;
        }
    }
    worst
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Mate-in-1 guard
// ═══════════════════════════════════════════════════════════════════════════

/// Turn-38 fix (94E99u): Exposure on `m.from` BEFORE the move was played.
pub fn pre_move_exposure(pos: &Pos, m: &Move, me: u8, opp: u8) -> i32 {
    if m.from >= 64 {
        return 0;
    }
    let occ_before = occ(pos);
    see_recapture_with_occ(pos, m.from, me, opp, &occ_before)
}

/// Does `opp` have an immediate mate-in-1 against `me`, given that `after`
/// is the position immediately after our candidate move (opp to move)?
///
/// Faithful port of JS `oppHasMateIn1(p2, me, opp)`.
pub fn opp_has_mate_in_1(after: &Pos, me: u8, opp: u8) -> bool {
    let replies = gen_moves(after, opp, None);
    for m in replies {
        let p3 = match apply_full(after, &m, opp, me) {
            Some(p) => p,
            None => continue,
        };
        if p3.king_thrown == Some(opp) || p3.gold_win == Some(opp) {
            return true;
        }
        if is_checkmate(&p3, me, opp) {
            return true;
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. King safety
// ═══════════════════════════════════════════════════════════════════════════

/// After playing `m`, is `me`'s king-like piece on a square `opp` does not
/// attack? Covers "leave the king in check" and "move king into attack".
/// Judged pre-WarAutomaton-advance per the wiki: a capture is legal even
/// if the advance exposes the King, and illegal even if the advance
/// would block the check.
pub fn king_move_safe(pos: &Pos, m: &Move, me: u8, opp: u8) -> bool {
    let after = match apply_full_no_wa(pos, m, me, opp) {
        Some(p) => p,
        None => return false,
    };
    !king_attacked(&after, me, opp, None)
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. X-ray hang detection (broadened to all Pixie sliders)
// ═══════════════════════════════════════════════════════════════════════════

/// Public wrapper: returns `XRAY_PEN` if the piece moved to `m.to` is worth
/// >= 300 and would be attacked by an enemy slider once the single blocking
/// pawn advances; otherwise 0. Port of JS `xrayHangingValue`.
pub fn xray_hang_penalty(pos: &Pos, m: &Move, me: u8, opp: u8) -> i32 {
    let after = match apply_full(pos, m, me, opp) {
        Some(p) => p,
        None => return 0,
    };
    let occ_local = occ(&after);
    xray_hang_penalty_after(pos, &after, m, opp, &occ_local)
}

fn xray_hang_penalty_after(
    before: &Pos,
    after: &Pos,
    m: &Move,
    opp: u8,
    occ_local: &[bool; 64],
) -> i32 {
    if m.to >= 64 {
        return 0;
    }
    let mover_val = if m.from != SQ_NONE {
        value_at(before, m.from)
    } else {
        value_at(after, m.to)
    };
    if mover_val < 300 {
        return 0;
    }

    let target = m.to;
    let tx = sq_x(target) as i8;
    let ty = sq_y(target) as i8;

    for i in 0..after.pieces.len() {
        let q = &after.pieces[i];
        if !xy_on_board(q.x, q.y) {
            continue;
        } // v130
        if !q.alive || q.player != opp {
            continue;
        }
        if PROVISIONAL_ICICLE_FREEZE && q.frozen() {
            continue;
        }
        if is_non_capturer(q.key) || is_non_capturer(q.sub_key) {
            continue;
        }
        if av_suppressed(after, i) {
            continue;
        }

        let k = q.key;
        let sk = q.sub_key;
        let is_card = is_cardinal_slider(k, sk);
        let is_diag = is_diagonal_slider(k, sk);
        if !is_card && !is_diag {
            continue;
        }

        let qx = q.x as i8;
        let qy = q.y as i8;
        let dx = tx - qx;
        let dy = ty - qy;

        let (sx, sy) = if dx == 0 && dy != 0 && is_card {
            (0i8, if dy > 0 { 1 } else { -1 })
        } else if dy == 0 && dx != 0 && is_card {
            (if dx > 0 { 1 } else { -1 }, 0i8)
        } else if dx != 0 && dx.abs() == dy.abs() && is_diag {
            (if dx > 0 { 1 } else { -1 }, if dy > 0 { 1 } else { -1 })
        } else {
            continue;
        };

        // Walk the ray, tracking up to 2 blockers without heap allocation.
        let mut blocker_count = 0u8;
        let mut single_blocker_idx = 0usize;
        let mut cx = qx + sx;
        let mut cy = qy + sy;
        let mut reached = false;
        loop {
            if cx == tx && cy == ty {
                reached = true;
                break;
            }
            if (cx as u8) >= 8 || (cy as u8) >= 8 {
                break;
            }
            let csq = sq(cx, cy);
            if occ_local[csq as usize] {
                if let Some(bi) = slot_of(after, csq) {
                    blocker_count += 1;
                    single_blocker_idx = bi;
                    if blocker_count > 1 {
                        break;
                    }
                }
            }
            cx += sx;
            cy += sy;
        }
        if !reached || blocker_count != 1 {
            continue;
        }

        let bl = &after.pieces[single_blocker_idx];
        let bl_pawn = is_pawn_like(bl.key) || is_pawn_like(bl.sub_key);
        if !bl_pawn {
            continue;
        }

        let pdir: i8 = if bl.player == 0 { 1 } else { -1 };
        let fx = bl.x as i8;
        let fy = bl.y as i8 + pdir;
        if (fy as u8) >= 8 {
            continue;
        }
        if occ_local[sq(fx, fy) as usize] {
            continue;
        }

        if piece_value(q) > 0 {
            return XRAY_PEN;
        }
    }
    0
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. SEE (Static Exchange Evaluation — Fast Early-Exit & Pin-Aware)
// ═══════════════════════════════════════════════════════════════════════════

/// Port of JS `seeRecapture`. Returns the net material that the side to move
/// in `pos` (i.e. the OPPONENT of `me`) can extract by capturing the piece
/// sitting on `target` (= `me`'s piece). Returns 0 if `target` is empty,
/// belongs to `opp`, or is an invulnerable iron pawn.
fn see_recapture(pos: &Pos, target: u8, me: u8, opp: u8) -> i32 {
    let occ_local = occ(pos);
    see_recapture_with_occ(pos, target, me, opp, &occ_local)
}

fn see_recapture_with_occ(
    pos: &Pos,
    target: u8,
    me: u8,
    opp: u8,
    occ_local: &[bool; 64],
) -> i32 {
    if target >= 64 {
        return 0;
    }
    let victim_idx = match slot_of(pos, target) {
        Some(i) => i,
        None => return 0,
    };
    let victim = &pos.pieces[victim_idx];
    if victim.player != me || !victim.alive {
        return 0;
    }
    if PROVISIONAL_IRON_INVULN
        && (victim.key == PieceKey::IronPawn || victim.sub_key == PieceKey::IronPawn)
    {
        return 0;
    }

    // Phase 1: Collect ONLY enemy (`opp`) attackers first.
    // In most positions a piece is not attacked at all — early-exiting before
    // scanning friendly defenders cuts SEE runtime by 3x-5x.
    let mut opp_atk: [i32; 16] = [0; 16];
    let mut n_opp = 0usize;

    for i in 0..pos.pieces.len() {
        if i == victim_idx {
            continue;
        }
        let q = &pos.pieces[i];
        if !xy_on_board(q.x, q.y) || !q.alive || q.player != opp {
            continue;
        }
        let qk = q.key;
        let qs = q.sub_key;
        if qk == PieceKey::King || qs == PieceKey::King {
            continue;
        }
        if !can_capture_piece_to(pos, i, target, occ_local) {
            continue;
        }
        if !piece_attacks(pos, i, occ_local, true, false).has(target) {
            continue;
        }
        if n_opp < 16 {
            opp_atk[n_opp] = piece_value(q);
            n_opp += 1;
        }
    }

    // Check if opponent king attacks `target`.
    let mut opp_king_attacks = false;
    if let Some(ki) = find_king(pos, opp) {
        let kp = &pos.pieces[ki];
        if kp.alive
            && xy_on_board(kp.x, kp.y)
            && !av_suppressed(pos, ki)
            && !is_paralyzed(pos, ki, occ_local)
            && piece_attacks(pos, ki, occ_local, true, false).has(target)
        {
            opp_king_attacks = true;
        }
    }

    // Fast early exit: if no enemy piece and no enemy king attacks `target`, loss is 0.
    if n_opp == 0 && !opp_king_attacks {
        return 0;
    }

    // Phase 2: Collect friendly (`me`) defenders of `target`.
    let mut me_atk: [i32; 16] = [0; 16];
    let mut n_me = 0usize;

    for i in 0..pos.pieces.len() {
        if i == victim_idx {
            continue;
        }
        let q = &pos.pieces[i];
        if !xy_on_board(q.x, q.y) || !q.alive || q.player != me {
            continue;
        }
        let qk = q.key;
        let qs = q.sub_key;
        if qk == PieceKey::King || qs == PieceKey::King {
            continue;
        }
        if !can_capture_piece_to(pos, i, target, occ_local) {
            continue;
        }
        if !piece_attacks(pos, i, occ_local, true, false).has(target) {
            continue;
        }
        if n_me < 16 {
            me_atk[n_me] = piece_value(q);
            n_me += 1;
        }
    }

    let victim_val = piece_value(victim);
    let king_free = if opp_king_attacks && n_me == 0 {
        victim_val
    } else {
        0
    };

    if n_opp == 0 {
        return king_free;
    }

    opp_atk[..n_opp].sort_unstable();
    me_atk[..n_me].sort_unstable();

    // Build swap list on stack: [victim, opp_atk[0], me_atk[0], opp_atk[1], me_atk[1], ...]
    let mut w: [i32; 33] = [0; 33];
    w[0] = victim_val;
    let mut w_len = 1usize;
    let mut n = 0usize;
    loop {
        let next = if n % 2 == 0 {
            let idx = n / 2;
            if idx < n_opp {
                Some(opp_atk[idx])
            } else {
                None
            }
        } else {
            let idx = (n - 1) / 2;
            if idx < n_me {
                Some(me_atk[idx])
            } else {
                None
            }
        };
        match next {
            Some(v) if w_len < 33 => {
                w[w_len] = v;
                w_len += 1;
            }
            _ => break,
        }
        n += 1;
    }

    let mut val = 0i32;
    if w_len >= 2 {
        for d in (0..w_len - 1).rev() {
            val = (w[d] - val).max(0);
        }
    }
    val.max(king_free)
}

/// Internal: uses the already-applied `after` and its `after_occ` to avoid
/// redundant `apply_full` and `occ` calls when called from `veto_score`.
fn see_capture_internal(
    before: &Pos,
    after: &Pos,
    m: &Move,
    me: u8,
    opp: u8,
    after_occ: &[bool; 64],
) -> i32 {
    if m.flags & MF_EP != 0 {
        return 0;
    }
    if m.flags & MF_PROMO != 0 && m.promo != 0 {
        return 0;
    }
    let victim_val = value_at(before, m.to);
    if victim_val == 0 {
        return 0;
    }
    let opp_gain = see_recapture_with_occ(after, m.to, me, opp, after_occ);
    victim_val - opp_gain
}

/// SEE for capture `m`. Returns the net material swing (centipawns) for the
/// mover, assuming the best play in the recapture sequence. Returns 0 for
/// non-captures, promotions, and en-passant.
pub fn see_capture(pos: &Pos, m: &Move) -> i32 {
    let me = pos.stm;
    let opp = 1 - me;
    let after = match apply_full(pos, m, me, opp) {
        Some(p) => p,
        None => return 0,
    };
    let after_occ = occ(&after);
    see_capture_internal(pos, &after, m, me, opp, &after_occ)
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Pixie-specific hazard penalty (Basilisk paralysis & WarAutomaton drift)
// ═══════════════════════════════════════════════════════════════════════════

fn pixie_hazard_penalty(after: &Pos, m: &Move, me: u8, opp: u8, after_occ: &[bool; 64]) -> i32 {
    let to = m.to;
    if to >= 64 {
        return 0;
    }
    let piece_idx = match slot_of(after, to) {
        Some(i) => i,
        None => return 0,
    };
    let p = &after.pieces[piece_idx];
    if !p.alive || p.player != me || is_king_piece(p) {
        return 0;
    }
    let val = piece_value(p);
    if val < 300 {
        return 0;
    }

    let mut pen = 0i32;

    // 1. Stepping a valuable piece into enemy Basilisk paralysis range
    if is_paralyzed(after, piece_idx, after_occ) {
        pen -= PIXIE_TRAP_PEN;
    }

    // 2. Stepping a valuable piece onto a square threatened by an enemy WarAutomaton's drift+attack
    for i in 0..after.pieces.len() {
        let q = &after.pieces[i];
        if !q.alive || q.player != opp || !xy_on_board(q.x, q.y) {
            continue;
        }
        if q.key == PieceKey::WarAutomaton || q.sub_key == PieceKey::WarAutomaton {
            if !piece_attacks(after, i, after_occ, true, false).has(to)
                && piece_attacks(after, i, after_occ, false, false).has(to)
            {
                pen -= PIXIE_TRAP_PEN;
                break;
            }
        }
    }

    pen
}

// ═══════════════════════════════════════════════════════════════════════════
// Combined veto score
// ═══════════════════════════════════════════════════════════════════════════

/// Combined veto penalty for `m`.
///
/// Returns a value in one of these ranges:
///   * `VETO_MATE_IN_1`   (-1_000_000): opponent has an immediate mate-in-1
///   * hyper-aggressive   (<= -1_000_000): paranoid material guard (v116)
///   * `VETO_KING_UNSAFE` (-100_000):   king ends up on an attacked square
///   * `VETO_PIN_MATE`    (-70_000):    pin creates a mating-square threat
///   * `VETO_HANGING`     (-60_000):    moved piece clearly hanging (>=300cp)
///   * `[-900, 0]`: soft penalty (SEE + x-ray hang + hang soft + pixie hazard)
///
/// `pxb_search_root` rejects any move whose score is below `VETO_THRESHOLD`.
pub fn veto_score(pos: &Pos, m: &Move, me: u8, opp: u8) -> i32 {
    let after = match apply_full(pos, m, me, opp) {
        Some(p) => p,
        None => return i32::MIN,
    };

    if after.king_thrown == Some(opp) || after.gold_win == Some(opp) {
        return VETO_MATE_IN_1;
    }

    let before_occ = occ(pos);
    let after_occ = occ(&after);

    // 1. King safety — judged on the PRE-WarAutomaton-advance position
    // (wiki WarAutomaton: check is judged before the advance, so a
    // capture is illegal even if the advance would block the check,
    // and legal even if the advance exposes the King). All other veto
    // checks below (hang, mate-in-1, SEE) use the post-advance `after`,
    // which is the real resulting position. Root-only cost: one extra
    // apply per vetoed root move (veto never runs in search's hot loop).
    let pre_wa = apply_full_no_wa(pos, m, me, opp);
    let king_pos = pre_wa.as_ref().unwrap_or(&after);
    let king_occ = occ(king_pos);
    if king_attacked(king_pos, me, opp, Some(&king_occ)) {
        return VETO_KING_UNSAFE;
    }

    // Winning moves end the game immediately; material exposure after them is moot.
    if after.king_thrown == Some(me)
        || after.gold_win == Some(me)
        || is_checkmate(&after, opp, me)
    {
        return 0;
    }

    // 2. Hanging-piece detection (fast check on `m.to` before full mate-in-1 scan).
    let h = hang_penalty_with_occ(pos, &after, m, me, opp, &after_occ);
    if h <= VETO_HANGING {
        return h; // hard veto: clearly hanging
    }

    // 3. DeepSeek v116: hyper-aggressive paranoid veto.
    let ha = hyper_aggressive_veto_with_occ(pos, &after, m, me, opp, &before_occ, &after_occ);
    if ha <= -1_000_000 {
        return ha; // hard veto: paranoid material guard
    }

    // 4. Pin-mate threat (pinned sole defender of mating square).
    if pin_mate_threat_with_occ(&after, m, me, opp, &after_occ) {
        return VETO_PIN_MATE;
    }

    // 5. Mate-in-1 guard — hard veto (runs after fast hard-veto filters so blunders
    // that already failed king-safety / hanging / paranoid SEE don't waste 40 apply_full calls).
    if opp_has_mate_in_1(&after, me, opp) {
        return VETO_MATE_IN_1;
    }

    let mut score = h + ha;

    // 6. SEE soft penalty (netted against pre_move_exposure).
    if m.flags & MF_EP == 0 && (m.flags & MF_PROMO == 0 || m.promo == 0) {
        let victim_val = value_at(pos, m.to);
        if victim_val > 0 {
            let vloss = -see_capture_internal(pos, &after, m, me, opp, &after_occ);
            let adj_vloss = vloss.saturating_sub(pre_move_exposure(pos, m, me, opp));
            if adj_vloss > 100 {
                score -= SEE_HARD_PEN;
            } else if adj_vloss > 0 {
                score -= SEE_SOFT_PEN;
            }
        }
    }

    // 7. X-ray hang soft penalty.
    let mover_val = if m.from != SQ_NONE {
        value_at(pos, m.from)
    } else {
        value_at(&after, m.to)
    };
    if mover_val >= 300 {
        let pen = xray_hang_penalty_after(pos, &after, m, opp, &after_occ);
        if pen > 0 {
            score -= pen;
        }
    }

    // 8. Pixie-specific trap soft penalty (Basilisk paralysis & WarAutomaton drift).
    score += pixie_hazard_penalty(&after, m, me, opp, &after_occ);

    // 9. Rank 4 Fix: Phase-scaled King shelter delta.
    // Penalizes voluntary midgame King walks out of back-rank shelter (such as
    // Game 152's `2. Ke1-d2`, `3. Kd2-e3`, `4. Ke3-f3`) proportional to the
    // opponent's remaining non-pawn attacking firepower (`opp_attacking_phase`),
    // while allowing Endgame Kings (`opp_phase <= 3`) to centralize freely.
    score += king_shelter_delta(pos, &after, m, me, opp, &before_occ);

    score
}

/// Phase-scaled King shelter delta for `veto_score`.
/// Returns a negative centipawn penalty when a non-castling King move walks
/// into ranks 2..7 (`after_ky >= 1`) during the opening/middlegame (`opp_phase >= 4`)
/// without escaping check, and `0` in endgames (`opp_phase <= 3`).
pub fn king_shelter_delta(
    pos: &Pos,
    after: &Pos,
    m: &Move,
    me: u8,
    opp: u8,
    before_occ: &[bool; 64],
) -> i32 {
    if (m.flags & (MF_CASTLE_K | MF_CASTLE_Q)) != 0 {
        return 25; // Castling improves shelter
    }
    if m.from >= 64 || m.to >= 64 {
        return 0;
    }
    let mover_is_king = slot_of(pos, m.from)
        .map(|s| is_king_piece(&pos.pieces[s]))
        .unwrap_or(false);
    if !mover_is_king {
        return 0;
    }
    let opp_phase = opp_attacking_phase(after, opp);
    if opp_phase <= 3 {
        // Endgame: allow King centralization without shelter penalty
        return 0;
    }
    let was_in_check = king_attacked(pos, me, opp, Some(before_occ));
    let before_ky = if me == 0 {
        sq_y(m.from) as i32
    } else {
        7 - (sq_y(m.from) as i32)
    };
    let after_ky = if me == 0 {
        sq_y(m.to) as i32
    } else {
        7 - (sq_y(m.to) as i32)
    };
    if after_ky <= 0 {
        return 0;
    }
    let step_forward = (after_ky - before_ky).max(0);
    if was_in_check && step_forward == 0 {
        return 0;
    }
    let after_kx = sq_x(m.to) as i32;
    let center_exposed = if (2..=5).contains(&after_kx) { 35 * after_ky } else { 0 };
    let raw_pen = 110 * after_ky * after_ky + 65 * step_forward + center_exposed;
    -((raw_pen * opp_phase) / 12)
}

#[cfg(test)]
mod veto_rank4_tests {
    use super::*;

    fn make_pos(pieces: &[(PieceKey, u8, u8, u8)], stm: u8) -> Pos {
        let mut p = Pos::default();
        p.stm = stm;
        for (i, &(k, pl, x, y)) in pieces.iter().enumerate() {
            p.pieces.push(Piece {
                id: (i + 1) as u32,
                key: k,
                sub_key: k,
                player: pl,
                x,
                y,
                alive: true,
                ..Default::default()
            });
            p.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        init_eval_sum(&mut p);
        p
    }

    #[test]
    fn test_midgame_king_walk_rejected() {
        let p = make_pos(
            &[
                (PieceKey::Rook, 0, 0, 0),
                (PieceKey::Knight, 0, 1, 0),
                (PieceKey::Bishop, 0, 2, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Bishop, 0, 5, 0),
                (PieceKey::Knight, 0, 6, 0),
                (PieceKey::Rook, 0, 7, 0),
                (PieceKey::Pawn, 0, 3, 2), // d3
                (PieceKey::Pawn, 0, 4, 1), // e2
                (PieceKey::Rook, 1, 0, 7),
                (PieceKey::Knight, 1, 1, 7),
                (PieceKey::Bishop, 1, 2, 7),
                (PieceKey::Queen, 1, 3, 7),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Bishop, 1, 5, 7),
                (PieceKey::Knight, 1, 6, 7),
                (PieceKey::Rook, 1, 7, 7),
                (PieceKey::Pawn, 1, 4, 4), // e5
            ],
            0,
        );
        let kd2 = Move { from: sq(4, 0), to: sq(3, 1), ..Default::default() };
        let nf3 = Move { from: sq(6, 0), to: sq(5, 2), ..Default::default() };
        let v_kd2 = veto_score(&p, &kd2, 0, 1);
        let v_nf3 = veto_score(&p, &nf3, 0, 1);
        assert!(
            v_kd2 <= -180 && v_nf3 > v_kd2 + 150,
            "Midgame King walk Ke1-d2 must receive heavy shelter penalty (v_kd2={}, v_nf3={})",
            v_kd2,
            v_nf3
        );
    }

    #[test]
    fn test_endgame_king_centralization_allowed() {
        let p = make_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Bishop, 0, 2, 0),
                (PieceKey::Pawn, 0, 4, 1),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Knight, 1, 1, 7),
                (PieceKey::Pawn, 1, 4, 6),
            ],
            0,
        );
        let kd2 = Move { from: sq(4, 0), to: sq(3, 1), ..Default::default() };
        let after = apply_full(&p, &kd2, 0, 1).unwrap();
        let b_occ = occ(&p);
        let delta = king_shelter_delta(&p, &after, &kd2, 0, 1, &b_occ);
        assert_eq!(delta, 0, "Endgame King centralization must have 0 shelter penalty");
    }

    #[test]
    fn test_sunk_cost_masking_does_not_hide_new_queen_hang() {
        // White Rook on a4 is already attacked by Black Rook on a8 (470cp exposure).
        // White Bishop on e2 shields White Queen on d1 from Black Rook on h5 (or e8).
        // Moving Bishop e2->f3 newly exposes Queen on e1 (873cp) to Black Rook on e8.
        // Even if another piece had high exposure, per-piece delta must catch the new exposure.
        let p = make_pos(
            &[
                (PieceKey::King, 0, 6, 0),
                (PieceKey::Queen, 0, 3, 0), // d1
                (PieceKey::Bishop, 0, 3, 1), // d2 shielding d1
                (PieceKey::Rook, 0, 0, 3), // a4 already hanging to a8
                (PieceKey::King, 1, 6, 7),
                (PieceKey::Rook, 1, 3, 7), // d8 pinning/x-raying d1
                (PieceKey::Rook, 1, 0, 7), // a8 attacking a4
            ],
            0,
        );
        let bd2_e3 = Move { from: sq(3, 1), to: sq(4, 2), ..Default::default() };
        let vs = veto_score(&p, &bd2_e3, 0, 1);
        assert!(vs < VETO_THRESHOLD, "Unshielding Queen on d1 must be hard-vetoed even when Rook a4 is already hanging, got {vs}");
    }

    #[test]
    fn test_already_hanging_mover_capture_nets_pre_move_exposure() {
        // White Rook on d4 is already undefended and attacked by Black Rook on d8.
        // Capturing Black Knight on b4 (defended by a5 pawn) trades an already-doomed Rook (470)
        // for a Knight (340), which is a net recovery, not a new blunder.
        let p = make_pos(
            &[
                (PieceKey::King, 0, 6, 0),
                (PieceKey::Rook, 0, 3, 3), // d4
                (PieceKey::King, 1, 6, 7),
                (PieceKey::Rook, 1, 3, 7), // d8 attacking d4
                (PieceKey::Knight, 1, 1, 3), // b4
                (PieceKey::Pawn, 1, 0, 4), // a5 defending b4
            ],
            0,
        );
        let rxb4 = Move { from: sq(3, 3), to: sq(1, 3), flags: crate::types::MF_CAPTURE, ..Default::default() };
        let vs = veto_score(&p, &rxb4, 0, 1);
        assert!(vs >= VETO_THRESHOLD, "Desperado capture by already-hanging Rook must survive veto, got {vs}");
    }
}
