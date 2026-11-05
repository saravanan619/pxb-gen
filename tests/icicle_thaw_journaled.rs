//! Minimal regression test for the Phase Rook movegen panic (gate match,
//! 2026-11-04, seed 777, game 4).
//!
//! Root cause: a frozen piece's stay-in-place thaw move (from == to,
//! MF_ICICLE_THAW) was journaled-eligible. `make_journaled` writes
//! at[to] = mover+1 and then at[from] = 0 on the SAME square (net at = 0),
//! while `occ_after_journaled` sets from = false and then to = true
//! (net occ = true), and it also marks the mover dead as its own victim.
//! Any later PhaseRook ray over that square with the derived occupancy
//! reads `pieces[(0 - 1) as usize]` = `pieces[65535]` -> bounds panic.
//!
//! Pre-fix this test PANICS inside `gives_check` (via `phase_rook_into`).
//! Post-fix `make_journaled` refuses the thaw, the search falls back to
//! `apply_full` (which implements thaw correctly), and all assertions pass.

use pxb_engine::apply::make_journaled;
use pxb_engine::apply_full::apply_full;
use pxb_engine::attacks::{gives_check, king_attacked};
use pxb_engine::journal::Journal;
use pxb_engine::pos::{occ_after_journaled, Piece, Pos};
use pxb_engine::types::{sq, Move, PieceKey, MF_ICICLE_THAW};

fn build_frozen_knight_pos() -> (Pos, [bool; 64]) {
    // White: frozen Knight at (3,5), PhaseRook at (3,2) whose north ray
    // runs (3,3), (3,4), (3,5). Black: King at (7,7) (gives_check target).
    let mut pos = Pos::default();
    pos.at = [0; 64];
    let mut occ = [false; 64];
    let mut push = |pos: &mut Pos, occ: &mut [bool; 64], key: PieceKey, player: u8, x: u8, y: u8, frozen: bool| {
        let mut p = Piece {
            key,
            player,
            x,
            y,
            alive: true,
            ..Default::default()
        };
        if frozen {
            p.frost = 2;
            p.state_flags = 1 << 4; // Frozen bit, as set by apply_full frost
        }
        pos.pieces.push(p);
        let s = sq(x as i8, y as i8) as usize;
        pos.at[s] = pos.pieces.len() as u16;
        occ[s] = true;
    };
    push(&mut pos, &mut occ, PieceKey::Knight, 0, 3, 5, true);
    push(&mut pos, &mut occ, PieceKey::PhaseRook, 0, 3, 2, false);
    push(&mut pos, &mut occ, PieceKey::King, 1, 7, 7, false);
    pos.stm = 0; // White to move: the thaw is White's move
    (pos, occ)
}

#[test]
fn icicle_thaw_is_never_journaled() {
    let (mut pos, occ) = build_frozen_knight_pos();
    let knight_sq = sq(3, 5);
    let thaw = Move {
        from: knight_sq,
        to: knight_sq,
        flags: MF_ICICLE_THAW,
        ..Default::default()
    };
    let p2_occ = occ_after_journaled(&occ, &thaw);
    let mut j = Journal::default();
    if make_journaled(&mut pos, &thaw, 0, 1, &mut j) {
        // BUG PATH (pre-fix): the journaled thaw desyncs at[]/occ, and the
        // White PhaseRook's north ray over (3,5) must panic here.
        let _ = gives_check(&pos, 0, 1, Some(&p2_occ));
        panic!("journaled thaw did not panic in gives_check — bug not reproduced");
    }
    // FIXED PATH: thaw refused; position untouched.
    assert_eq!(pos.at[knight_sq as usize], 1, "at[] must be untouched");
    assert!(pos.pieces[0].alive, "frozen knight must stay alive");
    assert!(pos.pieces[0].frozen(), "knight must stay frozen");
    // apply_full implements the thaw correctly: frozen cleared, frost
    // reset, piece alive, and fresh-occ attack queries do not panic.
    let p2 = apply_full(&pos, &thaw, 0, 1).expect("thaw must apply");
    assert!(p2.pieces[0].alive, "thawed knight must stay alive");
    assert!(!p2.pieces[0].frozen(), "thaw must clear Frozen");
    assert_eq!(p2.pieces[0].frost, 0, "thaw must reset frost");
    let _ = gives_check(&p2, 0, 1, None);
    let _ = king_attacked(&p2, 0, 1, None);
}
