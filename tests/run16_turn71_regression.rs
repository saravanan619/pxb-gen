// RUN16 game-1 turn-71 regression (2026-11-03):
// Black to move, king-slot Rocketman on d7 in check from Rook f7.
// The king on d7 blocks the rook's rank ray onto Marauder c7, so the
// hyper-aggressive veto used to nuke EVERY king departure (step or
// teleport) for "newly exposing" c7 — all 26 root moves rejected,
// zero ranked candidates, cluster status 2, abandonment — despite
// four server-legal escapes (d8/e8/c8/c6). A king move made while in
// check must survive the exposure veto: staying is not an option.
use pxb_engine::pos::Pos;
use pxb_engine::think::{run_ranked_root_search, think_candidates, RankedCandidate, RankedOutcome};
use pxb_engine::types::{Move, PieceKey};

fn sqn(x: i8, y: i8) -> u8 { (y * 8 + x) as u8 }
fn alg(s: u8) -> String {
    let f = (b'a' + (s % 8)) as char;
    format!("{}{}", f, s / 8 + 1)
}

fn turn71_board() -> Pos {
    let mut pos = Pos::default();
    let mut add = |key: PieceKey, sub: PieceKey, pl: u8, x: i8, y: i8| {
        let mut p = pxb_engine::pos::Piece::default();
        p.key = key; p.sub_key = sub; p.player = pl;
        p.x = x as u8; p.y = y as u8; p.alive = true;
        p.home_x = x; p.home_y = y;
        let idx = pos.pieces.len();
        pos.pieces.push(p);
        pos.at[sqn(x, y) as usize] = (idx + 1) as u16;
    };
    let w = 0u8; let b = 1u8;
    add(PieceKey::Dancer, PieceKey::Dancer, w, 5, 7);
    add(PieceKey::Djinn, PieceKey::Djinn, w, 7, 7);
    add(PieceKey::King, PieceKey::King, w, 5, 1);
    add(PieceKey::Pawn, PieceKey::Pawn, w, 4, 3);
    add(PieceKey::Pawn, PieceKey::Pawn, w, 6, 1);
    add(PieceKey::Pawn, PieceKey::Pawn, w, 7, 2);
    add(PieceKey::Queen, PieceKey::Queen, w, 6, 6);
    add(PieceKey::Rook, PieceKey::Rook, w, 5, 6);
    add(PieceKey::Marauder, PieceKey::Marauder, b, 2, 6);
    add(PieceKey::Pawn, PieceKey::Pawn, b, 0, 6);
    add(PieceKey::Pawn, PieceKey::Pawn, b, 2, 3);
    add(PieceKey::Pawn, PieceKey::Pawn, b, 3, 4);
    add(PieceKey::Pawn, PieceKey::Pawn, b, 4, 5);
    add(PieceKey::Queen, PieceKey::Queen, b, 3, 2);
    add(PieceKey::Rook, PieceKey::Rook, b, 1, 0);
    // King-slot piece: live game encoding carries subKey=King.
    add(PieceKey::Rocketman, PieceKey::King, b, 3, 6);
    pos
}

#[test]
fn king_escapes_survive_veto_when_in_check() {
    let pos = turn71_board();
    assert!(pxb_engine::attacks::king_attacked(&pos, 1, 0, None), "d7 must be in check (rook f7)");
    let gen = pxb_engine::movegen::gen_moves(&pos, 1, None);
    assert_eq!(gen.len(), 95, "live gen count was 95");
    for dest in [(3i8, 7i8), (4, 7), (2, 7), (2, 5)] {
        let m = gen.iter().find(|m| m.from == sqn(3, 6) && m.to == sqn(dest.0, dest.1) && m.flags == 0)
            .unwrap_or_else(|| panic!("king step to {} missing from gen", alg(sqn(dest.0, dest.1))));
        let vs = pxb_engine::veto::veto_score(&pos, m, 1, 0);
        assert!(vs >= pxb_engine::veto::VETO_THRESHOLD,
            "king step to {} vetoed at {} (was -1079200 before fix)", alg(sqn(dest.0, dest.1)), vs);
    }
}

#[test]
fn server_pair_filter_keeps_a_king_escape() {
    // Ranked list containing a king step must survive the server filter.
    let step = Move { from: sqn(3, 6), to: sqn(2, 5), ..Default::default() };
    let outcome = RankedOutcome {
        ranked: vec![RankedCandidate { mv: step, search_score: 143, veto_score: 0 }],
        legacy_pick: None, legacy_score: 0, fallback: None,
        depth: 1, nodes: 1, veto_checks: 1, veto_rejects: 0, no_moves: false,
    };
    let server: Vec<Move> = [[3u8, 6u8, 3u8, 7u8], [3, 6, 4, 7], [3, 6, 2, 7], [3, 6, 2, 5], [3, 6, 3, 6]]
        .iter().map(|q| Move { from: sqn(q[0] as i8, q[1] as i8), to: sqn(q[2] as i8, q[3] as i8), ..Default::default() }).collect();
    let pos = turn71_board();
    let s = pxb_engine::search::Search::new();
    let (cands, status) = think_candidates(&s, &pos, 1, &outcome, Some(&server));
    assert_eq!(status, pxb_engine::think::THINK_OK);
    assert!(!cands.is_empty());
    assert_eq!((cands[0].mv.from, cands[0].mv.to), (sqn(3, 6), sqn(2, 5)));
}

#[test]
fn ranked_search_returns_candidates_on_turn71() {
    let mut pos = turn71_board();
    let mut s = pxb_engine::search::Search::new();
    let out = run_ranked_root_search(&mut s, &mut pos, 1, 5000, 2);
    assert!(!out.ranked.is_empty(), "ranked list must not be empty after the veto fix");
    let server: Vec<Move> = [[3u8, 6u8, 3u8, 7u8], [3, 6, 4, 7], [3, 6, 2, 7], [3, 6, 2, 5], [3, 6, 3, 6]]
        .iter().map(|q| Move { from: sqn(q[0] as i8, q[1] as i8), to: sqn(q[2] as i8, q[3] as i8), ..Default::default() }).collect();
    let (cands, status) = think_candidates(&s, &pos, 1, &out, Some(&server));
    assert_eq!(status, pxb_engine::think::THINK_OK, "must not be a server mismatch");
    assert!(!cands.is_empty());
}
