//! Game-4 replay: gate-match crash (2026-11-04, seed 777) reproduction.
//!
//! Builds the game-4 ply-22 root position from the instrumented rerun log
//! (dumpKey, run3) and runs Black's search exactly as match.js did
//! (200ms budget, depth 8). Pre-fix this panics with a bounds check
//! inside `phase_rook_into`; post-fix it completes and returns a move.
//!
//! Played-game frost state at the root (from the Black Icicle's moves):
//! White Knight (3,5) frost=1, White Rocketman (0,7) frost=1, nobody
//! frozen. The panic needs one more in-tree Icicle move next to the
//! Knight (frost 1 -> 2 -> frozen), then a White-node thaw of the Knight
//! taken through the journaled search path.

use pxb_engine::pos::{Piece, Pos};
use pxb_engine::search::Search;
use pxb_engine::think::run_ranked_root_search;
use pxb_engine::types::{sq, PieceKey};

fn piece(key: PieceKey, player: u8, x: u8, y: u8, frost: u8) -> Piece {
    Piece {
        key,
        player,
        x,
        y,
        frost,
        alive: true,
        ..Default::default()
    }
}

fn build_root() -> Pos {
    use PieceKey::*;
    // (key, player, x, y, frost) — live pieces only, from the ply-22 dumpKey.
    let spec: &[(PieceKey, u8, u8, u8, u8)] = &[
        // White
        (Pawn, 0, 0, 1, 0),
        (Pawn, 0, 1, 1, 0),
        (Pawn, 0, 2, 3, 0),
        (Pawn, 0, 3, 1, 0),
        (Pawn, 0, 4, 1, 0),
        (Pawn, 0, 5, 1, 0),
        (Pawn, 0, 6, 1, 0),
        (Pawn, 0, 7, 2, 0),
        (Aristocrat, 0, 2, 0, 0),
        (Knight, 0, 3, 5, 1),
        (PhaseRook, 0, 4, 2, 0),
        (Bishop, 0, 5, 0, 0),
        (Rocketman, 0, 0, 7, 1),
        (Pinata, 0, 5, 2, 0),
        (Rook, 0, 0, 0, 0),
        (Queen, 0, 4, 0, 0),
        // Black
        (Shrike, 1, 6, 6, 0),
        (Pawn, 1, 0, 3, 0),
        (Pawn, 1, 2, 4, 0),
        (Pawn, 1, 3, 6, 0),
        (Pawn, 1, 4, 4, 0),
        (Pawn, 1, 5, 6, 0),
        (Pawn, 1, 7, 6, 0),
        (Basilisk, 1, 5, 7, 0),
        (Knight, 1, 2, 5, 0),
        (Knight, 1, 7, 5, 0),
        (FissionReactor, 1, 3, 7, 0),
        (Icicle, 1, 4, 5, 0),
        (Rook, 1, 7, 7, 0),
        (King, 1, 4, 7, 0),
    ];
    let mut pos = Pos::default();
    pos.at = [0; 64];
    for (i, &(key, player, x, y, frost)) in spec.iter().enumerate() {
        pos.pieces.push(piece(key, player, x, y, frost));
        pos.at[sq(x as i8, y as i8) as usize] = (i + 1) as u16;
    }
    pos.stm = 1; // Black to move at ply 22
    pos
}

#[test]
fn game4_ply22_black_search_completes() {
    // Match the crashing side's weights when available (NEW = epoch-10).
    let epoch10 =
        "/home/hatch/workspace/pxb-train-data-prep/runs/2026-11-04-morning-fixed/train/nnue-phase2-epoch10.bin";
    if let Ok(bytes) = std::fs::read(epoch10) {
        assert!(
            pxb_engine::nnue::nnue_load(&bytes),
            "epoch-10 weights must parse"
        );
    } else {
        eprintln!("epoch-10 weights not found; using embedded weights");
    }
    let mut pos = build_root();
    let mut s = Search::new();
    let outcome = run_ranked_root_search(&mut s, &mut pos, 1, 200, 8);
    assert!(
        !outcome.no_moves,
        "Black must have a move at the ply-22 root"
    );
    assert!(
        outcome.legacy_pick.is_some(),
        "search must return a root move"
    );
}
