// NNUE incremental accumulator (docs/nnue-incremental-contract.md).
//
// Design (dual perspectives, the standard answer to the contract's
// SideFlip problem): NNUE features are stm-relative, so flipping the side
// to move rewrites EVERY feature (planes swap AND squares rotate 180°).
// A single accumulator cannot absorb that incrementally. This module
// therefore keeps TWO accumulators: acc[0] is the feature sum for the
// world as seen with white to move, acc[1] with black to move. A piece
// update touches both (its plane/square differ per perspective); a
// SideFlip only swaps which accumulator `forward` reads — O(1), exact.
//
// Delta vocabulary (contract §Delta-list, with two documented extensions
// marked EXT): Moved / Captured / Transformed / SideFlip as specified,
// plus Spawned (EXT: the contract's four types cannot express a piece
// appearing with no from-square — HordeMother spawns, Knightmare drops,
// Djinn revivals, Recruit-sac pawns all create board pieces), and a
// `player` field on every piece delta (EXT: stm-relative planes cannot
// be updated without knowing which side owns the piece; the contract's
// delta spec omits it). Deltas carry EFFECTIVE keys (same resolution as
// the converter), so delta application and full refresh agree by
// construction; unknown-key deltas are no-ops in both perspectives.
//
// Emission: apply_full has ~15 mutation sites (doomed expiry, sumo
// shoves, spawns, revivals, promotions, conversions, early returns).
// Instrumenting each site risks a missed path and a silently drifting
// accumulator — the exact failure the contract warns about. Emission
// here is therefore diff-derived: `board_deltas(before, after)` compares
// the actual before/after boards by piece id, so it is exact for every
// apply path by construction, at O(pieces) cost — small next to a
// forward pass. `apply_full_with_deltas` is the emission point for the
// clone-based apply path; `deltas_for_journaled_move` covers the
// journaled path by cloning the post-move board, unwinding the clone
// to the pre-move board (the journal already records every old value),
// and diffing. Neither touches search's hot path: search still scores
// leaves through the sparse full forward (see below), and threading
// accumulator state through abSearch/qsearch recursion (save/restore
// per node across the journaled AND clone-based paths) is follow-up
// work, not silently done here. The accumulator is implemented, its
// emission exists for both apply paths, and its parity suite proves
// incremental == full refresh; the evaluator's per-call sparse forward
// (<=32 active rows x 256) is already cheaper than the hand-written
// evaluator's attack-map machinery, so this is a correctness-first
// staging, not a performance regression.
//
// Ordering: Captured deltas are emitted before Moved arrivals (contract
// rule, so the accumulator never transiently holds two pieces on one
// square in a capture sequence); the accumulator is additive, so the
// final state after a full delta list is exact regardless of order.
// State-flag-only changes (crouch, charge, freeze, ...) emit no delta:
// the schema has no state features.

use crate::apply_full::apply_full;
use crate::eval::effective_key;
use crate::journal::Journal;
use crate::nnue::{
    forward, piece_type_index, NnueWeights, NNUE_HIDDEN_SIZE, NNUE_NUM_TYPES,
};
use crate::pos::*;
use crate::types::*;

/// One board change, in contract vocabulary (+ documented extensions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delta {
    Moved {
        key: PieceKey,
        player: u8,
        from_sq: u8,
        to_sq: u8,
    },
    Captured {
        key: PieceKey,
        player: u8,
        sq: u8,
    },
    Transformed {
        old_key: PieceKey,
        new_key: PieceKey,
        player: u8,
        sq: u8,
    },
    /// EXT: a piece appears on the board with no from-square.
    Spawned {
        key: PieceKey,
        player: u8,
        sq: u8,
    },
    SideFlip,
}

fn on_board_pieces(pos: &Pos) -> Vec<(u32, PieceKey, u8, u8)> {
    let mut out = Vec::new();
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive {
            continue;
        }
        out.push((
            p.id,
            effective_key(p),
            p.player,
            sq(p.x as i8, p.y as i8),
        ));
    }
    out
}

/// Exact delta list between two boards. Pieces are matched by id; a
/// changed square is a Moved, a changed effective key on the same square
/// is a Transformed, a vanished id is a Captured, a new id is a Spawned,
/// and a changed stm appends exactly one SideFlip last.
pub fn board_deltas(before: &Pos, after: &Pos) -> Vec<Delta> {
    let b = on_board_pieces(before);
    let a = on_board_pieces(after);
    let mut deltas: Vec<Delta> = Vec::new();
    // Captures first (contract ordering rule).
    for &(id, key, player, sqq) in &b {
        if !a.iter().any(|&(id2, _, _, _)| id2 == id) {
            deltas.push(Delta::Captured {
                key,
                player,
                sq: sqq,
            });
        }
    }
    // Transformations (same id, same square, different effective key).
    for &(id, key, player, sqq) in &b {
        if let Some(&(_, key2, _, sqq2)) = a.iter().find(|&&(id2, _, _, _)| id2 == id) {
            if sqq2 == sqq && key2 != key {
                deltas.push(Delta::Transformed {
                    old_key: key,
                    new_key: key2,
                    player,
                    sq: sqq,
                });
            }
        }
    }
    // Moves (same id, different square). Key changes on a move are
    // reported as the move with the AFTER key plus, if the key also
    // changed, a Transformed is not double-counted: the mover's feature
    // removal uses the before key and its addition the after key (see
    // apply_delta), so a move-with-transformation is exact as a single
    // Moved only when the key is unchanged; otherwise emit both a
    // Transformed (same-square semantics do not apply) — handled below
    // by emitting Moved with the before key and a separate Transformed
    // at the destination square.
    for &(id, key, player, sqq) in &b {
        if let Some(&(_, key2, _, sqq2)) = a.iter().find(|&&(id2, _, _, _)| id2 == id) {
            if sqq2 != sqq {
                deltas.push(Delta::Moved {
                    key,
                    player,
                    from_sq: sqq,
                    to_sq: sqq2,
                });
                if key2 != key {
                    deltas.push(Delta::Transformed {
                        old_key: key,
                        new_key: key2,
                        player,
                        sq: sqq2,
                    });
                }
            }
        }
    }
    // Spawns (new ids).
    for &(id, key, player, sqq) in &a {
        if !b.iter().any(|&(id2, _, _, _)| id2 == id) {
            deltas.push(Delta::Spawned {
                key,
                player,
                sq: sqq,
            });
        }
    }
    if (before.stm & 1) != (after.stm & 1) {
        deltas.push(Delta::SideFlip);
    }
    deltas
}

/// Clone-based apply path with delta emission: applies the move, then
/// diffs the actual boards. Returns None exactly when apply_full does.
pub fn apply_full_with_deltas(
    pos: &Pos,
    mv: &Move,
    player: u8,
    opp: u8,
) -> Option<(Pos, Vec<Delta>)> {
    let after = apply_full(pos, mv, player, opp)?;
    let deltas = board_deltas(pos, &after);
    Some((after, deltas))
}

/// Journaled-path delta emission: `pos_after` is the board AFTER a
/// journaled move was applied with `journal` (entries from `mark`
/// onward), and `journal` itself is NOT consumed — a clone is unwound
/// to reconstruct the pre-move board, which is then diffed against the
/// post-move board. Exact for every journaled move by the same diff
/// core; the clone/unwind cost is paid only when deltas are consumed,
/// never on search's hot path.
pub fn deltas_for_journaled_move(
    pos_after: &Pos,
    journal: &Journal,
    mark: usize,
) -> Vec<Delta> {
    let mut pre = pos_after.clone();
    let mut jc = journal.clone();
    jc.unwind(&mut pre, mark);
    board_deltas(&pre, pos_after)
}

// ── Accumulator ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct NnueAccumulator {
    /// acc[0] = feature sum with white to move; acc[1] = black to move.
    /// Stored in f64 (53-bit mantissa) so incremental add/remove of f32 W1 rows
    /// has zero associative rounding drift relative to sorted full recompute.
    pub acc: [[f64; NNUE_HIDDEN_SIZE]; 2],
    /// Active perspective (0 = white stm, 1 = black stm).
    pub stm: u8,
}

impl NnueAccumulator {
    #[inline]
    pub const fn new() -> Self {
        Self {
            acc: [[0.0; NNUE_HIDDEN_SIZE]; 2],
            stm: 0,
        }
    }

    /// Full refresh in place from a board: both perspectives, zero heap allocation.
    #[inline]
    pub fn refresh_in_place(&mut self, weights: &NnueWeights, pos: &Pos) {
        for h in 0..NNUE_HIDDEN_SIZE {
            let b = weights.b1[h] as f64;
            self.acc[0][h] = b;
            self.acc[1][h] = b;
        }
        self.stm = pos.stm & 1;
        for p in &pos.pieces {
            if !xy_on_board(p.x, p.y) || !p.alive {
                continue;
            }
            self.add_piece(weights, effective_key(p), p.player, sq(p.x as i8, p.y as i8));
        }
    }

    /// Full refresh from a board: both perspectives, from scratch.
    pub fn refresh(weights: &NnueWeights, pos: &Pos) -> Self {
        let mut acc = Self::new();
        acc.refresh_in_place(weights, pos);
        acc
    }

    /// Fast zero-allocation incremental update for a journaled move in `search.rs`.
    /// Updates both colour perspectives in `O(NNUE_HIDDEN_SIZE)` without heap allocation.
    #[inline]
    pub fn update_for_journaled_move(
        &mut self,
        weights: &NnueWeights,
        mover_old_key: PieceKey,
        mover_new_key: PieceKey,
        player: u8,
        from_sq: u8,
        to_sq: u8,
        cap: Option<(PieceKey, u8, u8)>,
    ) {
        self.remove_piece(weights, mover_old_key, player, from_sq);
        if let Some((cap_key, cap_player, cap_sq)) = cap {
            self.remove_piece(weights, cap_key, cap_player, cap_sq);
        }
        self.add_piece(weights, mover_new_key, player, to_sq);
        self.stm = 1 - player;
    }

    /// O(1) side-to-move flip for Null-Move Pruning (no board pieces changed).
    #[inline]
    pub fn flip_stm(&mut self) {
        self.stm = 1 - self.stm;
    }

    fn add_piece(&mut self, weights: &NnueWeights, key: PieceKey, player: u8, sqq: u8) {
        let t = match piece_type_index(key) {
            Some(t) => t,
            None => return, // unknown pieces contribute no input, either way
        };
        let sqq = sqq as usize;
        for persp in 0..2usize {
            let plane = if player == persp as u8 { 0 } else { 1 };
            let sqp = if persp == 0 { sqq } else { 63 - sqq };
            let base = ((plane * NNUE_NUM_TYPES + t) * 64 + sqp) * NNUE_HIDDEN_SIZE;
            for h in 0..NNUE_HIDDEN_SIZE {
                self.acc[persp][h] += weights.w1[base + h] as f64;
            }
        }
    }

    fn remove_piece(&mut self, weights: &NnueWeights, key: PieceKey, player: u8, sqq: u8) {
        let t = match piece_type_index(key) {
            Some(t) => t,
            None => return,
        };
        let sqq = sqq as usize;
        for persp in 0..2usize {
            let plane = if player == persp as u8 { 0 } else { 1 };
            let sqp = if persp == 0 { sqq } else { 63 - sqq };
            let base = ((plane * NNUE_NUM_TYPES + t) * 64 + sqp) * NNUE_HIDDEN_SIZE;
            for h in 0..NNUE_HIDDEN_SIZE {
                self.acc[persp][h] -= weights.w1[base + h] as f64;
            }
        }
    }

    pub fn apply_delta(&mut self, weights: &NnueWeights, delta: Delta) {
        match delta {
            Delta::Moved {
                key,
                player,
                from_sq,
                to_sq,
            } => {
                self.remove_piece(weights, key, player, from_sq);
                self.add_piece(weights, key, player, to_sq);
            }
            Delta::Captured { key, player, sq } => {
                self.remove_piece(weights, key, player, sq);
            }
            Delta::Spawned { key, player, sq } => {
                self.add_piece(weights, key, player, sq);
            }
            Delta::Transformed {
                old_key,
                new_key,
                player,
                sq,
            } => {
                self.remove_piece(weights, old_key, player, sq);
                self.add_piece(weights, new_key, player, sq);
            }
            Delta::SideFlip => {
                self.stm = 1 - self.stm;
            }
        }
    }

    pub fn apply_deltas(&mut self, weights: &NnueWeights, deltas: &[Delta]) {
        for &d in deltas {
            self.apply_delta(weights, d);
        }
    }

    /// Network output for the active perspective. Equivalent (up to f32
    /// summation order) to forward(weights, converter features).
    pub fn forward(&self, weights: &NnueWeights) -> f32 {
        let src = &self.acc[self.stm as usize];
        let mut out = weights.b2;
        for h in 0..NNUE_HIDDEN_SIZE {
            let v = src[h] as f32;
            if v > 0.0 {
                out += weights.w2[h] * v;
            }
        }
        1.0 / (1.0 + (-out).exp())
    }
}

// Re-export for callers that want the direct path without importing nnue.
pub fn direct_forward(weights: &NnueWeights, pos: &Pos) -> f32 {
    forward(weights, &crate::nnue::pos_features(pos).indices)
}

#[cfg(test)]
mod accumulator_tests {
    use super::*;
    use crate::apply::make_journaled;
    use crate::nnue::{pos_features, NNUE_MODEL_BYTES};
    use crate::pos::*;
    use crate::types::*;

    fn weights_zero() -> NnueWeights {
        // Deterministic synthetic weights: W1[i] cycles so that every
        // feature row differs (a constant row would hide delta bugs).
        let mut w1 = Vec::with_capacity(5120 * 256);
        for i in 0..(5120 * 256) {
            let v = (((i as u64 * 2654435761u64) >> 16) & 0xFF) as f32 / 255.0 - 0.5;
            w1.push(v * 0.01);
        }
        let mut b1 = [0.0f32; 256];
        for (i, v) in b1.iter_mut().enumerate() {
            *v = (i as f32 / 256.0 - 0.5) * 0.1;
        }
        let mut w2 = [0.0f32; 256];
        for (i, v) in w2.iter_mut().enumerate() {
            *v = (i as f32 / 256.0 - 0.5) * 0.1;
        }
        NnueWeights {
            w1,
            b1,
            w2,
            b2: 0.05,
        }
    }

    fn piece(key: PieceKey, player: u8, x: u8, y: u8, id: u32) -> Piece {
        Piece {
            id,
            key,
            sub_key: key,
            player,
            x,
            y,
            alive: true,
            ..Default::default()
        }
    }

    fn pos_with(pieces: Vec<Piece>, stm: u8) -> Pos {
        let mut pos = Pos::default();
        pos.stm = stm;
        for (i, p) in pieces.into_iter().enumerate() {
            pos.at[((p.y as usize) << 3) | (p.x as usize)] = (i + 1) as u16;
            pos.pieces.push(p);
        }
        // Real positions keep id_seq past all assigned ids; spawns take
        // id_seq + 1, so a spawn must never reuse a live piece's id.
        pos.id_seq = pos.pieces.len() as u32;
        pos
    }

    fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
        assert!(
            (a - b).abs() <= tol,
            "{what}: incremental {a} vs full {b} (tol {tol})"
        );
    }

    #[test]
    fn refresh_matches_direct_forward() {
        let w = weights_zero();
        let pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::PhaseRook, 0, 0, 0, 2),
                piece(PieceKey::Queen, 1, 3, 7, 3),
                piece(PieceKey::Bishopper, 0, 2, 2, 4), // unknown: no input
                piece(PieceKey::Pawn, 1, 0, 6, 5),
            ],
            0,
        );
        let acc = NnueAccumulator::refresh(&w, &pos);
        assert_close(acc.forward(&w), direct_forward(&w, &pos), 1e-5, "refresh");
        // Black stm: the OTHER perspective must match the flipped board.
        let mut pos_b = pos.clone();
        pos_b.stm = 1;
        let acc_b = NnueAccumulator::refresh(&w, &pos_b);
        assert_close(acc_b.forward(&w), direct_forward(&w, &pos_b), 1e-5, "refresh black");
        assert_eq!(pos_features(&pos).skipped, 1);
    }

    fn check_deltas(w: &NnueWeights, before: &Pos, after: &Pos, what: &str) {
        let deltas = board_deltas(before, after);
        let mut acc = NnueAccumulator::refresh(w, before);
        acc.apply_deltas(w, &deltas);
        assert_eq!(acc.stm, after.stm & 1, "{what}: stm after deltas");
        assert_close(
            acc.forward(w),
            direct_forward(w, after),
            1e-4,
            what,
        );
        // Undo: inverse deltas in reverse order restore the prior output.
        let mut inv: Vec<Delta> = Vec::new();
        for &d in deltas.iter().rev() {
            inv.push(match d {
                Delta::Moved {
                    key,
                    player,
                    from_sq,
                    to_sq,
                } => Delta::Moved {
                    key,
                    player,
                    from_sq: to_sq,
                    to_sq: from_sq,
                },
                Delta::Captured { key, player, sq } => {
                    Delta::Spawned { key, player, sq }
                }
                Delta::Spawned { key, player, sq } => {
                    Delta::Captured { key, player, sq }
                }
                Delta::Transformed {
                    old_key,
                    new_key,
                    player,
                    sq,
                } => Delta::Transformed {
                    old_key: new_key,
                    new_key: old_key,
                    player,
                    sq,
                },
                Delta::SideFlip => Delta::SideFlip,
            });
        }
        acc.apply_deltas(w, &inv);
        assert_eq!(acc.stm, before.stm & 1, "{what}: stm after undo");
        assert_close(
            acc.forward(w),
            direct_forward(w, before),
            1e-4,
            &format!("{what} undo"),
        );
    }

    #[test]
    fn deltas_cover_apply_full_move_types() {
        let w = weights_zero();
        // Ordinary move + side flip.
        let before = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::Rook, 0, 0, 0, 2),
                piece(PieceKey::King, 1, 4, 7, 3),
            ],
            0,
        );
        let mv = Move {
            from: sq(0, 0),
            to: sq(0, 1),
            ..Default::default()
        };
        let (after, deltas) = apply_full_with_deltas(&before, &mv, 0, 1).unwrap();
        assert!(deltas.iter().any(|d| matches!(d, Delta::Moved { .. })));
        assert!(deltas.iter().any(|d| matches!(d, Delta::SideFlip)));
        check_deltas(&w, &before, &after, "ordinary");
        // Capture.
        let before_c = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::Rook, 0, 0, 0, 2),
                piece(PieceKey::King, 1, 4, 7, 3),
                piece(PieceKey::Pawn, 1, 0, 1, 4),
            ],
            0,
        );
        let cap = Move {
            from: sq(0, 0),
            to: sq(0, 1),
            ..Default::default()
        };
        let (after_c, deltas_c) = apply_full_with_deltas(&before_c, &cap, 0, 1).unwrap();
        assert!(deltas_c
            .iter()
            .any(|d| matches!(d, Delta::Captured { .. })));
        check_deltas(&w, &before_c, &after_c, "capture");
        // Promotion: white pawn a7 -> a8 = queen.
        let before_p = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::Pawn, 0, 0, 6, 2),
                piece(PieceKey::King, 1, 4, 7, 3),
            ],
            0,
        );
        let promo = Move {
            from: sq(0, 6),
            to: sq(0, 7),
            flags: MF_PROMO,
            promo: PieceKey::Queen as u8,
            ..Default::default()
        };
        let (after_p, _) = apply_full_with_deltas(&before_p, &promo, 0, 1).unwrap();
        assert_eq!(after_p.pieces[1].key, PieceKey::Queen);
        check_deltas(&w, &before_p, &after_p, "promotion");
        // Castle: king e1 -> g1 moves the rook h1 -> f1 as well.
        let before_k = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::Rook, 0, 7, 0, 2),
                piece(PieceKey::King, 1, 4, 7, 3),
            ],
            0,
        );
        let castle = Move {
            from: sq(4, 0),
            to: sq(6, 0),
            flags: MF_CASTLE_K,
            ..Default::default()
        };
        let (after_k, deltas_k) = apply_full_with_deltas(&before_k, &castle, 0, 1).unwrap();
        let n_moved = deltas_k
            .iter()
            .filter(|d| matches!(d, Delta::Moved { .. }))
            .count();
        assert_eq!(n_moved, 2, "castle must move king and rook: {deltas_k:?}");
        check_deltas(&w, &before_k, &after_k, "castle");
        // HordeMother spawn: a new Hordeling appears (Spawned delta).
        let before_h = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::HordeMother, 0, 3, 3, 2),
                piece(PieceKey::King, 1, 4, 7, 3),
            ],
            0,
        );
        let spawn = Move {
            from: sq(3, 3),
            to: sq(3, 4),
            flags: MF_HORDE_SPAWN,
            special_a: sq(4, 4),
            ..Default::default()
        };
        let (after_h, deltas_h) = apply_full_with_deltas(&before_h, &spawn, 0, 1).unwrap();
        assert!(deltas_h
            .iter()
            .any(|d| matches!(d, Delta::Spawned { .. })));
        check_deltas(&w, &before_h, &after_h, "horde-spawn");
        let _ = NNUE_MODEL_BYTES;
    }

    #[test]
    fn journaled_move_deltas_match_refresh() {
        let w = weights_zero();
        let mut pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0, 1),
                piece(PieceKey::Rook, 0, 0, 0, 2),
                piece(PieceKey::King, 1, 4, 7, 3),
                piece(PieceKey::Pawn, 1, 0, 1, 4),
            ],
            0,
        );
        // Capture via the journaled path (rook takes pawn: safe keys only).
        let mv = Move {
            from: sq(0, 0),
            to: sq(0, 1),
            ..Default::default()
        };
        let before = pos.clone();
        let mut journal = Journal::default();
        let mark = journal.mark();
        assert!(make_journaled(&mut pos, &mv, 0, 1, &mut journal));
        let deltas = deltas_for_journaled_move(&pos, &journal, mark);
        assert!(deltas.iter().any(|d| matches!(d, Delta::Captured { .. })));
        assert!(deltas.iter().any(|d| matches!(d, Delta::SideFlip)));
        // The journaled board must equal the apply_full board for this move.
        let (after_full, _) = apply_full_with_deltas(&before, &mv, 0, 1).unwrap();
        assert_eq!(pos.at, after_full.at);
        assert_eq!(pos.stm, after_full.stm);
        check_deltas(&w, &before, &pos, "journaled-capture");
        // Unwinding the real journal still restores the parent board.
        journal.unwind(&mut pos, mark);
        assert_eq!(pos.at, before.at);
        assert_eq!(pos.stm, before.stm);
    }

    #[test]
    fn delta_fuzz_random_boards() {
        // Deterministic LCG fuzz: random boards (trained + untrained mix)
        // with random mutations; incremental must track full refresh.
        let w = weights_zero();
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state = state.wrapping_mul(0x2545F4914F6CDD1D);
            (state >> 32) as u32
        };
        let keys = [
            PieceKey::Pawn,
            PieceKey::Rook,
            PieceKey::Knight,
            PieceKey::Bishop,
            PieceKey::Queen,
            PieceKey::King,
            PieceKey::PhaseRook,
            PieceKey::Banker,
            PieceKey::Bishopper,
            PieceKey::Ooze,
            PieceKey::WarKing,
            PieceKey::RookZero,
            PieceKey::Pinata,
            PieceKey::Blueprint,
        ];
        for iter in 0..200 {
            let n = 4 + (next() % 10) as usize;
            let mut used = [false; 64];
            let mut pieces = Vec::new();
            for i in 0..n {
                let mut sqi = (next() % 64) as usize;
                while used[sqi] {
                    sqi = (sqi + 1) % 64;
                }
                used[sqi] = true;
                pieces.push(piece(
                    keys[(next() as usize) % keys.len()],
                    (next() % 2) as u8,
                    (sqi & 7) as u8,
                    (sqi >> 3) as u8,
                    (i + 1) as u32,
                ));
            }
            let before = pos_with(pieces, (next() % 2) as u8);
            let mut after = before.clone();
            // Random mutation: move, remove, transform, or add a piece.
            match next() % 4 {
                0 => {
                    if !after.pieces.is_empty() {
                        let i = (next() as usize) % after.pieces.len();
                        let mut sqi = (next() % 64) as usize;
                        let mut guard = 0;
                        while after.at[sqi] != 0 && guard < 64 {
                            sqi = (sqi + 1) % 64;
                            guard += 1;
                        }
                        if after.at[sqi] == 0 {
                            after.at[((after.pieces[i].y as usize) << 3)
                                | (after.pieces[i].x as usize)] = 0;
                            after.pieces[i].x = (sqi & 7) as u8;
                            after.pieces[i].y = (sqi >> 3) as u8;
                            after.at[sqi] = (i + 1) as u16;
                        }
                    }
                }
                1 => {
                    if !after.pieces.is_empty() {
                        let i = (next() as usize) % after.pieces.len();
                        let sqi = ((after.pieces[i].y as usize) << 3)
                            | (after.pieces[i].x as usize);
                        after.at[sqi] = 0;
                        after.pieces[i].alive = false;
                    }
                }
                2 => {
                    if !after.pieces.is_empty() {
                        let i = (next() as usize) % after.pieces.len();
                        let nk = keys[(next() as usize) % keys.len()];
                        after.pieces[i].key = nk;
                        after.pieces[i].sub_key = nk;
                    }
                }
                _ => {
                    let mut sqi = (next() % 64) as usize;
                    let mut guard = 0;
                    while after.at[sqi] != 0 && guard < 64 {
                        sqi = (sqi + 1) % 64;
                        guard += 1;
                    }
                    if after.at[sqi] == 0 {
                        let id = after.pieces.len() as u32 + 1;
                        after.pieces.push(piece(
                            keys[(next() as usize) % keys.len()],
                            (next() % 2) as u8,
                            (sqi & 7) as u8,
                            (sqi >> 3) as u8,
                            id,
                        ));
                        after.at[sqi] = after.pieces.len() as u16;
                    }
                }
            }
            if next() % 2 == 0 {
                after.stm = 1 - (after.stm & 1);
            }
            check_deltas(&w, &before, &after, &format!("fuzz {iter}"));
        }
    }
}
