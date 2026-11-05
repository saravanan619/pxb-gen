// Sandbox GAP probe tests (2026-11-03): Shrike, BisHop, Djinn, Gunslinger,
// plus batch 2 (Pilgrim, Golden Pawn, Hero Pawn, Horde Mother, BisHop recharge).
// Ground truth: ~/workspace/pxb-puzzles-site/sandbox-probe-4-pieces.md
// and the batch-2 play-test findings (played Sandbox moves, 2026-11-03).
// These tests assert the Sandbox-verified behavior. Some FAIL on the
// pre-fix engine on purpose (they pin the bug); the fix job turns them green.
use pxb_engine::apply_full::apply_full;
use pxb_engine::eval::{evaluate, MATE_CONST};
use pxb_engine::movegen::gen_moves;
use pxb_engine::pos::{Piece, Pos};
use pxb_engine::types::{
    Move, PieceKey, MF_BISHOPPER_CROUCH, MF_BISHOPPER_LEAP, MF_CAPTURE,
    MF_DJINN_DISSIPATE, MF_GUN_STANDOFF, MF_HORDE_SPAWN, MF_PILGRIM_RESURRECT,
    MF_PROMO, MF_SHRIKE_FIRST,
};

fn sqn(x: i8, y: i8) -> u8 {
    (y * 8 + x) as u8
}
fn alg(s: u8) -> String {
    let f = (b'a' + (s % 8)) as char;
    format!("{}{}", f, s / 8 + 1)
}

fn build(specs: &[(PieceKey, u8, i8, i8, u16)]) -> Pos {
    let mut pos = Pos::default();
    for &(k, pl, x, y, flags) in specs {
        let mut p = Piece::default();
        p.key = k;
        p.sub_key = k;
        p.player = pl;
        p.x = x as u8;
        p.y = y as u8;
        p.alive = true;
        p.home_x = x;
        p.home_y = y;
        p.state_flags = flags;
        let idx = pos.pieces.len();
        // Unique ids: apply_full's doom cleanup and the Gunslinger timer
        // look pieces up by id. (An earlier version left every id at 0,
        // which made the doom cleanup kill the white king and the timer
        // read the king's threats instead of the Gunslinger's.)
        p.id = (idx + 1) as u32;
        pos.pieces.push(p);
        pos.at[sqn(x, y) as usize] = (idx + 1) as u16;
    }
    pos
}

fn piece_moves(pos: &Pos, player: u8, from: u8) -> Vec<Move> {
    gen_moves(pos, player, None)
        .into_iter()
        .filter(|m| m.from == from)
        .collect()
}

fn find<'a>(moves: &'a [Move], to: u8) -> Option<&'a Move> {
    moves.iter().find(|m| m.to == to)
}

fn alive_at(pos: &Pos, s: u8) -> bool {
    pos.at[s as usize] != 0
        && pos.pieces[(pos.at[s as usize] - 1) as usize].alive
}

// ---------- Shrike ----------
const KINGS: [(PieceKey, u8, i8, i8, u16); 2] = [
    (PieceKey::King, 0, 0, 0, 0),
    (PieceKey::King, 1, 7, 7, 0),
];

#[test]
fn shrike_setup_a_both_pawns_only_e2xe4() {
    // White Shrike e2, black pawns e3+e4: ONLY e2xe4, kills BOTH.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Shrike, 0, 4, 1, 0));
    specs.push((PieceKey::Pawn, 1, 4, 2, 0));
    specs.push((PieceKey::Pawn, 1, 4, 3, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 1));
    assert_eq!(ms.len(), 1, "setup A shrike moves: {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    let m = ms[0].clone();
    assert_eq!(m.to, sqn(4, 3));
    assert_ne!(m.flags & MF_SHRIKE_FIRST, 0);
    assert_ne!(m.flags & MF_CAPTURE, 0);
    let c = apply_full(&pos, &m, 0, 1).expect("shrike A applies");
    assert!(!alive_at(&c, sqn(4, 2)), "e3 pawn must die");
    assert!(!alive_at(&c, sqn(4, 3)) || {
        // e4 now holds the Shrike itself, not the pawn
        let s = &c.pieces[(c.at[sqn(4, 3) as usize] - 1) as usize];
        s.key == PieceKey::Shrike && s.player == 0
    }, "e4 pawn must die, shrike lands e4");
    let s = &c.pieces[(c.at[sqn(4, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Shrike);
}

#[test]
fn shrike_setup_b_intermediate_dies_on_empty_landing() {
    // White Shrike e2, black pawn e3 only, e4 empty: ONLY e2xe4, e3 dies.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Shrike, 0, 4, 1, 0));
    specs.push((PieceKey::Pawn, 1, 4, 2, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 1));
    assert_eq!(ms.len(), 1, "setup B shrike moves: {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    let m = ms[0].clone();
    assert_eq!(m.to, sqn(4, 3));
    let c = apply_full(&pos, &m, 0, 1).expect("shrike B applies");
    assert!(!alive_at(&c, sqn(4, 2)), "e3 pawn must die even on empty landing");
    let s = &c.pieces[(c.at[sqn(4, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Shrike);
}

#[test]
fn probe_shrike_diagonal_captures() {
    // INVESTIGATIVE (no Sandbox verdict): does Shrike get diagonal pawn
    // captures? Prints the move list; run with --nocapture.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Shrike, 0, 4, 1, 0));
    specs.push((PieceKey::Pawn, 1, 3, 2, 0));
    specs.push((PieceKey::Pawn, 1, 5, 2, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 1));
    println!("SHRIKE-DIAG moves: {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
}

// ---------- BisHop ----------
const CROUCHED: u16 = 1 << 0;

#[test]
fn bishopper_crouch_is_turn_costing_and_leap_is_one_shot() {
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::BisHop, 0, 3, 3, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    let crouch = ms.iter().find(|m| m.flags & MF_BISHOPPER_CROUCH != 0)
        .expect("standing bishopper must offer crouch");
    assert_eq!(crouch.from, crouch.to, "crouch is stay-in-place");
    let c = apply_full(&pos, crouch, 0, 1).expect("crouch applies");
    assert_eq!(c.stm, 1, "crouch must cost the turn");
    let slot = c.at[sqn(3, 3) as usize] as usize - 1;
    assert!(c.pieces[slot].crouched(), "crouched bit set after crouch");
}

#[test]
fn bishopper_leap_only_adjacent_beyond_screen() {
    // Crouched BisHop d4, screen pawn e5, empty f6+g7: leap ONLY to f6.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::BisHop, 0, 3, 3, CROUCHED));
    specs.push((PieceKey::Pawn, 1, 4, 4, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    let leap = find(&ms, sqn(5, 5)).expect("leap to f6 (adjacent beyond) must exist");
    assert_ne!(leap.flags & MF_BISHOPPER_LEAP, 0);
    assert!(find(&ms, sqn(6, 6)).is_none(),
        "no double leap to g7; moves: {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    // Normal screen capture still offered as a plain bishop capture.
    let cap = find(&ms, sqn(4, 4)).expect("screen capture d4xe5 must exist");
    assert_eq!(cap.flags & MF_BISHOPPER_LEAP, 0, "screen capture is not a leap");
    // Leap consumes crouch (one-shot).
    let c = apply_full(&pos, leap, 0, 1).expect("leap applies");
    let slot = c.at[sqn(5, 5) as usize] as usize - 1;
    assert!(!c.pieces[slot].crouched(), "leap must consume crouch");
}

#[test]
fn bishopper_leap_capture_beyond_screen() {
    // Crouched BisHop d4, screen e5, ENEMY f6: capture leap to f6, no g7.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::BisHop, 0, 3, 3, CROUCHED));
    specs.push((PieceKey::Pawn, 1, 4, 4, 0));
    specs.push((PieceKey::Pawn, 1, 5, 5, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    let leap = find(&ms, sqn(5, 5)).expect("capture leap to f6 must exist");
    assert_ne!(leap.flags & MF_BISHOPPER_LEAP, 0);
    assert_ne!(leap.flags & MF_CAPTURE, 0);
    assert!(find(&ms, sqn(6, 6)).is_none(), "no leap past f6");
    let c = apply_full(&pos, leap, 0, 1).expect("capture leap applies");
    assert!(alive_at(&c, sqn(4, 4)), "screen pawn survives a leap");
    let s = &c.pieces[(c.at[sqn(5, 5) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::BisHop);
}

// ---------- Djinn ----------
#[test]
fn djinn_dissipate_costs_turn_and_reforms_on_next_capture() {
    // White Djinn d4 dissipates; Black then captures elsewhere; Djinn returns.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Djinn, 0, 3, 3, 0));
    specs.push((PieceKey::Pawn, 0, 0, 6, 0)); // white pawn a7, capture fodder
    specs.push((PieceKey::Queen, 1, 1, 6, 0)); // black queen b7
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    let dis = ms.iter().find(|m| m.flags & MF_DJINN_DISSIPATE != 0)
        .expect("djinn must offer dissipate");
    let c = apply_full(&pos, dis, 0, 1).expect("dissipate applies");
    assert_eq!(c.stm, 1, "dissipate must cost the turn");
    assert!(!alive_at(&c, sqn(3, 3)), "djinn leaves d4");
    assert_eq!(c.limbo.len(), 1, "djinn in limbo");
    // Black queen b7 captures white pawn a7.
    let bm = piece_moves(&c, 1, sqn(1, 6));
    let cap = find(&bm, sqn(0, 6)).expect("queen capture must exist");
    let c2 = apply_full(&c, cap, 1, 0).expect("capture applies");
    assert!(alive_at(&c2, sqn(3, 3)), "djinn reforms on d4 after next capture");
    let s = &c2.pieces[(c2.at[sqn(3, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Djinn);
    assert_eq!(s.player, 0);
}

#[test]
fn djinn_reform_captures_enemy_on_home_square() {
    // Black rook sits on d4; Black captures elsewhere; Djinn reforms, rook dies.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Djinn, 0, 3, 3, 0));
    specs.push((PieceKey::Pawn, 0, 0, 6, 0));
    specs.push((PieceKey::Queen, 1, 1, 6, 0));
    let pos = build(&specs);
    let dis = piece_moves(&pos, 0, sqn(3, 3)).into_iter()
        .find(|m| m.flags & MF_DJINN_DISSIPATE != 0).unwrap();
    let c = apply_full(&pos, &dis, 0, 1).unwrap();
    // Place black rook on d4 after dissipate.
    let mut c = c;
    let mut r = Piece::default();
    r.key = PieceKey::Rook; r.sub_key = PieceKey::Rook; r.player = 1;
    r.x = 3; r.y = 3; r.alive = true; r.home_x = 3; r.home_y = 3;
    let idx = c.pieces.len();
    c.pieces.push(r);
    c.at[sqn(3, 3) as usize] = (idx + 1) as u16;
    let bm = piece_moves(&c, 1, sqn(1, 6));
    let cap = find(&bm, sqn(0, 6)).expect("queen capture must exist").clone();
    let c2 = apply_full(&c, &cap, 1, 0).expect("capture applies");
    let s = &c2.pieces[(c2.at[sqn(3, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Djinn, "djinn reforms on d4");
    assert_eq!(s.player, 0, "rook must die to the reform");
}

#[test]
fn djinn_capture_does_not_revive_same_move() {
    // Capturing a Djinn must NOT revive it in the same apply (killing capturer).
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Djinn, 0, 3, 3, 0));
    specs.push((PieceKey::Queen, 1, 4, 4, 0)); // black queen e5
    let pos = build(&specs);
    let bm = piece_moves(&pos, 1, sqn(4, 4));
    let cap = find(&bm, sqn(3, 3)).expect("queen takes djinn must exist").clone();
    let c = apply_full(&pos, &cap, 1, 0).expect("capture applies");
    let s = &c.pieces[(c.at[sqn(3, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Queen, "capturer must survive on d4");
    assert_eq!(s.player, 1);
    assert!(c.pieces.iter().all(|p| !(p.alive && p.key == PieceKey::Djinn)),
        "djinn must stay dead this move (limbo only)");
}

// ---------- Gunslinger ----------
fn gunslinger_board() -> Pos {
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Gunslinger, 0, 3, 3, 0)); // white d4
    specs.push((PieceKey::Bishop, 1, 6, 6, 0)); // black bishop g7, mutual diagonal
    build(&specs)
}

#[test]
fn gunslinger_standoff_kills_only_enemy() {
    // Standoff (once armed) removes ONLY the bishop; Gunslinger survives.
    let pos = gunslinger_board();
    // Arm via full round: white king move, black king move (threat intact).
    let wk = piece_moves(&pos, 0, sqn(0, 0)).into_iter()
        .find(|m| m.to == sqn(0, 1)).expect("white king step").clone();
    let c = apply_full(&pos, &wk, 0, 1).unwrap();
    let bk = piece_moves(&c, 1, sqn(7, 7)).into_iter()
        .find(|m| m.to == sqn(7, 6)).expect("black king step").clone();
    let c2 = apply_full(&c, &bk, 1, 0).unwrap();
    let ms = piece_moves(&c2, 0, sqn(3, 3));
    let so = ms.iter().find(|m| m.flags & MF_GUN_STANDOFF != 0)
        .expect("armed standoff must be offered after a full round");
    assert_eq!(so.special_a, sqn(6, 6), "standoff targets g7 bishop");
    let c3 = apply_full(&c2, so, 0, 1).expect("standoff applies");
    assert!(alive_at(&c3, sqn(3, 3)), "gunslinger must SURVIVE standoff");
    let s = &c3.pieces[(c3.at[sqn(3, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Gunslinger);
    assert!(!alive_at(&c3, sqn(6, 6)), "bishop must die");
}

#[test]
fn gunslinger_standoff_not_offered_before_full_round() {
    // Fresh mutual threat: standoff must NOT be offered yet.
    let pos = gunslinger_board();
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    assert!(ms.iter().all(|m| m.flags & MF_GUN_STANDOFF == 0),
        "no standoff before a full round; moves: {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
}

#[test]
fn gunslinger_timer_resets_when_threat_breaks() {
    // Arm it, then break the threat (bishop moves away); standoff must go.
    let pos = gunslinger_board();
    let wk = piece_moves(&pos, 0, sqn(0, 0)).into_iter()
        .find(|m| m.to == sqn(0, 1)).unwrap().clone();
    let c = apply_full(&pos, &wk, 0, 1).unwrap();
    let bk = piece_moves(&c, 1, sqn(7, 7)).into_iter()
        .find(|m| m.to == sqn(7, 6)).unwrap().clone();
    let c2 = apply_full(&c, &bk, 1, 0).unwrap();
    assert!(piece_moves(&c2, 0, sqn(3, 3)).iter()
        .any(|m| m.flags & MF_GUN_STANDOFF != 0), "armed after round");
    // Black bishop g7 moves off the diagonal (g7->h6 is off d4 diagonal? g7-h6:
    // h6 is not on d4's diagonal, so threat breaks).
    let bm = piece_moves(&c2, 1, sqn(6, 6)).into_iter()
        .find(|m| m.to == sqn(7, 5)).expect("bishop retreat").clone();
    let c3 = apply_full(&c2, &bm, 1, 0).unwrap();
    // White moves king again to get back to white's turn.
    let wk2 = piece_moves(&c3, 0, sqn(0, 1)).into_iter()
        .find(|m| m.to == sqn(0, 2)).expect("white king step 2").clone();
    let c4 = apply_full(&c3, &wk2, 0, 1).unwrap();
    // Bishop back to g7? Not needed: threat is broken, and even if the
    // bishop returned, the timer must have reset (no instant re-arm).
    let _ = c4;
    let ms = piece_moves(&c3, 0, sqn(3, 3));
    assert!(ms.iter().all(|m| m.flags & MF_GUN_STANDOFF == 0),
        "threat broken: no standoff on white's next turn");
}

// ---------- ElectroKnight (Sandbox zap probe, 2026-11-03) ----------
// Charged capture zaps exactly ONE extra enemy: first destructible piece
// scanning the LANDING square's neighbours clockwise from north. No
// player choice, not random. Iron immune (skipped, not absorbing).
// AV as victim is destroyed, never blocks. Charge burns on any charged
// capture, even when the zap kills nothing.
use pxb_engine::types::{off_code, MF_ELECTRO, MF_ICICLE_THAW,
    MF_KNIGHTMARE_DROP, MF_KNIGHTMARE_HOP, MF_KNIGHTMARE_OFF, SQ_NONE};

const ECHARGED: u16 = 1 << 1;

#[test]
fn ek_zap_is_clockwise_first_destructible_only() {
    // EK e4 charged, pawn f6 target, knight g6, bishop f5, rook g5 all
    // adjacent to f6. f6 neighbours clockwise from north: f7, g7 (empty),
    // g6 (knight) -> victim is g6 ONLY.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::ElectroKnight, 0, 4, 3, ECHARGED));
    specs.push((PieceKey::Pawn, 1, 5, 5, 0));
    specs.push((PieceKey::Knight, 1, 6, 5, 0));
    specs.push((PieceKey::Bishop, 1, 5, 4, 0));
    specs.push((PieceKey::Rook, 1, 6, 4, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 3));
    let zaps: Vec<&Move> = ms.iter()
        .filter(|m| m.flags & MF_ELECTRO != 0 && m.to == sqn(5, 5)).collect();
    assert_eq!(zaps.len(), 1, "exactly one zap move to f6, got {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags, m.special_a)).collect::<Vec<_>>());
    assert_eq!(zaps[0].special_a, sqn(6, 5), "victim must be g6 knight");
    let c = apply_full(&pos, zaps[0], 0, 1).expect("ek zap applies");
    assert!(!alive_at(&c, sqn(6, 5)), "g6 knight must die to the zap");
    assert!(alive_at(&c, sqn(5, 4)), "f5 bishop must survive");
    assert!(alive_at(&c, sqn(6, 4)), "g5 rook must survive");
    let slot = c.at[sqn(5, 5) as usize] as usize - 1;
    assert_eq!(c.pieces[slot].key, PieceKey::ElectroKnight);
    assert_eq!(c.pieces[slot].quiet, 0, "charge must burn");
    assert!(!c.pieces[slot].echarged(), "charge bit must clear");
}

#[test]
fn ek_zap_skips_iron_and_burns_charge_with_no_victim() {
    // Iron on g7 (NE of f6) is skipped for the pawn on g6 (E of f6).
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::ElectroKnight, 0, 4, 3, ECHARGED));
    specs.push((PieceKey::Pawn, 1, 5, 5, 0));
    specs.push((PieceKey::IronPawn, 1, 6, 6, 0));
    specs.push((PieceKey::Pawn, 1, 6, 5, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 3));
    let zaps: Vec<&Move> = ms.iter()
        .filter(|m| m.flags & MF_ELECTRO != 0 && m.to == sqn(5, 5)).collect();
    assert_eq!(zaps.len(), 1);
    assert_eq!(zaps[0].special_a, sqn(6, 5), "iron g7 skipped, pawn g6 zapped");
    let c = apply_full(&pos, zaps[0], 0, 1).unwrap();
    assert!(alive_at(&c, sqn(6, 6)), "iron pawn must survive the zap");

    // Iron ONLY adjacent: no zap move at all, but the plain charged
    // capture still burns the charge (Sandbox test 4b).
    let mut specs2 = KINGS.to_vec();
    specs2.push((PieceKey::ElectroKnight, 0, 4, 3, ECHARGED));
    specs2.push((PieceKey::Pawn, 1, 5, 5, 0));
    specs2.push((PieceKey::IronPawn, 1, 6, 5, 0));
    let pos2 = build(&specs2);
    let ms2 = piece_moves(&pos2, 0, sqn(4, 3));
    assert!(ms2.iter().all(|m| m.flags & MF_ELECTRO == 0),
        "no zap move when only Iron is adjacent");
    let plain = ms2.iter().find(|m| m.to == sqn(5, 5) && m.flags & MF_CAPTURE != 0)
        .expect("plain capture must exist").clone();
    let c2 = apply_full(&pos2, &plain, 0, 1).unwrap();
    assert!(alive_at(&c2, sqn(6, 5)), "iron survives the plain capture too");
    let slot = c2.at[sqn(5, 5) as usize] as usize - 1;
    assert_eq!(c2.pieces[slot].quiet, 0, "charged capture burns charge even with no zap");
    assert!(!c2.pieces[slot].echarged());
}

#[test]
fn ek_zap_destroys_av_victim_no_block() {
    // AV on g6 (E of f6) is the zap victim: it dies, it does not block.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::ElectroKnight, 0, 4, 3, ECHARGED));
    specs.push((PieceKey::Pawn, 1, 5, 5, 0));
    specs.push((PieceKey::AntiViolence, 1, 6, 5, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 3));
    let zap = ms.iter().find(|m| m.flags & MF_ELECTRO != 0 && m.to == sqn(5, 5))
        .expect("zap move must exist even with AV victim").clone();
    assert_eq!(zap.special_a, sqn(6, 5));
    let c = apply_full(&pos, &zap, 0, 1).unwrap();
    assert!(!alive_at(&c, sqn(6, 5)), "AV victim must die, not block");
}

// ---------- Knightmare (Sandbox 14x14 probe, 2026-11-03) ----------

fn pos_with_offboard(ox: i8, oy: i8, extra: &[(PieceKey, u8, i8, i8, u16)]) -> Pos {
    use pxb_engine::pos::OffboardPiece;
    let mut specs = KINGS.to_vec();
    specs.extend_from_slice(extra);
    let mut pos = build(&specs);
    pos.offboard.push(OffboardPiece {
        id: 77, key: PieceKey::Knightmare, sub_key: PieceKey::Knightmare,
        player: 0, off_since: 1, ox, oy,
    });
    pos
}

fn offboard_moves(pos: &Pos) -> Vec<Move> {
    gen_moves(pos, 0, None).into_iter().filter(|m| m.from == SQ_NONE).collect()
}

#[test]
fn knightmare_off_jumps_from_b4_are_knight_jumps_to_ring() {
    // b4: 6 on-board knight moves + 2 ring jumps (off:-1,4 and off:-1,2).
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Knightmare, 0, 1, 3, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(1, 3));
    assert_eq!(ms.len(), 8, "b4 must offer 8 moves, got {:?}",
        ms.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    let offs: Vec<u8> = ms.iter()
        .filter(|m| m.flags & MF_KNIGHTMARE_OFF != 0).map(|m| m.to).collect();
    assert_eq!(offs.len(), 2);
    assert!(offs.contains(&off_code(-1, 4)));
    assert!(offs.contains(&off_code(-1, 2)));
}

#[test]
fn knightmare_onboard_d4_has_no_off_jump() {
    // Central d4: all 8 knight moves land on-board, none off.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Knightmare, 0, 3, 3, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    assert_eq!(ms.len(), 8);
    assert!(ms.iter().all(|m| m.flags & MF_KNIGHTMARE_OFF == 0));
}

#[test]
fn knightmare_offboard_corner_has_two_hops() {
    // off:-3,10: only (-1,9) and (-2,8) stay on the stage.
    let pos = pos_with_offboard(-3, 10, &[]);
    let ms = offboard_moves(&pos);
    assert_eq!(ms.len(), 2, "corner must offer 2 hops, got {:?}",
        ms.iter().map(|m| (m.to, m.flags)).collect::<Vec<_>>());
    assert!(ms.iter().all(|m| m.flags & MF_KNIGHTMARE_HOP != 0));
    let tos: Vec<u8> = ms.iter().map(|m| m.to).collect();
    assert!(tos.contains(&off_code(-1, 9)));
    assert!(tos.contains(&off_code(-2, 8)));
}

#[test]
fn knightmare_drop_capture_and_no_own_half_limit() {
    // off:-1,2 with black pawn b4: drop-capture b4, quiet drops a5/b2,
    // a1 excluded (own king), 4 ring hops.
    let pos = pos_with_offboard(-1, 2, &[(PieceKey::Pawn, 1, 1, 3, 0)]);
    let ms = offboard_moves(&pos);
    let drop_cap = ms.iter().find(|m| m.to == sqn(1, 3))
        .expect("drop-capture b4 must exist").clone();
    assert_ne!(drop_cap.flags & MF_KNIGHTMARE_DROP, 0);
    assert_ne!(drop_cap.flags & MF_CAPTURE, 0);
    assert!(ms.iter().any(|m| m.to == sqn(0, 4) && m.flags & MF_CAPTURE == 0));
    assert!(ms.iter().any(|m| m.to == sqn(1, 1) && m.flags & MF_CAPTURE == 0));
    assert!(ms.iter().all(|m| m.to != sqn(0, 0)), "no drop onto own king");
    let c = apply_full(&pos, &drop_cap, 0, 1).expect("drop-capture applies");
    assert!(!c.pieces.iter().any(|p| p.alive && p.key == PieceKey::Pawn && p.player == 1),
        "b4 pawn must die to the drop");
    let s = &c.pieces[(c.at[sqn(1, 3) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Knightmare);
    assert_eq!(s.player, 0);

    // No own-half restriction: off:-1,7 offers a drop to b7, deep in
    // Black's half, and it plays.
    let pos2 = pos_with_offboard(-1, 7, &[]);
    let ms2 = offboard_moves(&pos2);
    let b7 = ms2.iter().find(|m| m.to == sqn(1, 6))
        .expect("b7 drop must be offered (no own-half limit)").clone();
    let c2 = apply_full(&pos2, &b7, 0, 1).expect("b7 drop applies");
    let s2 = &c2.pieces[(c2.at[sqn(1, 6) as usize] - 1) as usize];
    assert_eq!(s2.key, PieceKey::Knightmare);
}

// ---------- Anti-violence aura (Sandbox play-test, 2026-11-03) ----------
// Settled rule: an enemy piece ADJACENT to an AV has its captures disabled
// entirely. The AV itself cannot capture. (The earlier "aura did not
// manifest" note was a test-setup error: the pieces were not adjacent.)

#[test]
fn av_pacifies_adjacent_enemy_after_it_lands() {
    // AV d4, black knight e6 (NOT adjacent: knight-move away), white pawn
    // c5 (adjacent to AV). Knight is offered x c5 and x d4; it plays x c5
    // and lands adjacent to the AV, where its captures collapse to none.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::AntiViolence, 0, 3, 3, 0));
    specs.push((PieceKey::Knight, 1, 4, 5, 0));
    specs.push((PieceKey::Pawn, 0, 2, 4, 0));
    let pos = build(&specs);
    let bm = piece_moves(&pos, 1, sqn(4, 5));
    let cap = bm.iter().find(|m| m.to == sqn(2, 4) && m.flags & MF_CAPTURE != 0)
        .expect("non-adjacent knight must be offered x c5").clone();
    let c = apply_full(&pos, &cap, 1, 0).expect("knight capture applies");
    let nm = piece_moves(&c, 1, sqn(2, 4));
    assert_eq!(nm.len(), 8, "pacified knight: 8 quiet moves, got {:?}",
        nm.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    assert!(nm.iter().all(|m| m.flags & MF_CAPTURE == 0),
        "pacified knight must have NO captures");
    // AV itself never captures, even with the knight a knight-move away.
    let av = piece_moves(&c, 0, sqn(3, 3));
    assert!(av.iter().all(|m| m.flags & MF_CAPTURE == 0),
        "AV itself must never capture");
}

// ---------- Icicle thaw (Sandbox play-test, 2026-11-03) ----------
// Frost 1 = badge only, moves normal. Frost 2 = Frozen: the ONLY legal
// move is stay-in-place, which thaws (turn consumed) and restores moves.

#[test]
fn icicle_frost1_moves_normal_frost2_only_thaw() {
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Icicle, 0, 3, 3, 0));
    specs.push((PieceKey::Knight, 1, 4, 4, 0));
    let mut pos = build(&specs);
    // Frost 1, not frozen: full knight moves.
    let slot = pos.at[sqn(4, 4) as usize] as usize - 1;
    pos.pieces[slot].frost = 1;
    let ms = piece_moves(&pos, 1, sqn(4, 4));
    assert_eq!(ms.len(), 8, "frost-1 knight moves normally");
    // Frost 2 + frozen bit: exactly one move, stay-in-place thaw.
    pos.pieces[slot].frost = 2;
    pos.pieces[slot].state_flags |= 1 << 4;
    let ms2 = piece_moves(&pos, 1, sqn(4, 4));
    assert_eq!(ms2.len(), 1, "frozen knight: only thaw, got {:?}",
        ms2.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    assert_eq!(ms2[0].from, ms2[0].to);
    assert_ne!(ms2[0].flags & MF_ICICLE_THAW, 0);
    let c = apply_full(&pos, &ms2[0], 1, 0).expect("thaw applies");
    let slot2 = c.at[sqn(4, 4) as usize] as usize - 1;
    assert!(!c.pieces[slot2].frozen(), "thaw must clear frozen");
    assert_eq!(c.pieces[slot2].frost, 0, "thaw must reset frost");
    let ms3 = piece_moves(&c, 1, sqn(4, 4));
    assert_eq!(ms3.len(), 8, "normal moves return after thaw");
}

// ---------- War Automaton any-capture advance (Sandbox, 2026-11-03) ----------

#[test]
fn wa_advances_on_another_piece_capture() {
    // White rook h1 captures black pawn h5; the white WA on d4 must step
    // forward to d5 (any capture triggers the advance, not just the WA's
    // own capture).
    let mut specs2 = KINGS.to_vec();
    specs2.push((PieceKey::WarAutomaton, 0, 3, 3, 0));
    specs2.push((PieceKey::Rook, 0, 7, 0, 0));
    specs2.push((PieceKey::Pawn, 1, 7, 4, 0));
    let pos = build(&specs2);
    let rm = piece_moves(&pos, 0, sqn(7, 0)).into_iter()
        .find(|m| m.to == sqn(7, 4)).expect("rook capture must exist").clone();
    let c = apply_full(&pos, &rm, 0, 1).expect("capture applies");
    let wa = c.pieces.iter().find(|p| p.alive
        && (p.key == PieceKey::WarAutomaton || p.sub_key == PieceKey::WarAutomaton)
        && p.player == 0).expect("WA must survive");
    assert_eq!((wa.x, wa.y), (3, 4), "WA must step d4->d5 on another piece's capture");
}

// ---------- Fission 5th-capture explosion (Sandbox, 2026-11-03) ----------

#[test]
fn fission_explodes_on_fifth_capture() {
    // Reactor d4 with 4 caps captures pawn e5 (5th): reactor dies and both
    // diagonally adjacent knights (d6, f4) die; a far pawn survives.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::FissionReactor, 0, 3, 3, 0));
    specs.push((PieceKey::Pawn, 1, 4, 4, 0));
    specs.push((PieceKey::Knight, 1, 3, 5, 0));
    specs.push((PieceKey::Knight, 1, 5, 3, 0));
    specs.push((PieceKey::Pawn, 1, 1, 1, 0));
    let mut pos = build(&specs);
    let slot = pos.at[sqn(3, 3) as usize] as usize - 1;
    pos.pieces[slot].caps = 4;
    let mv = piece_moves(&pos, 0, sqn(3, 3)).into_iter()
        .find(|m| m.to == sqn(4, 4)).expect("reactor capture must exist").clone();
    let c = apply_full(&pos, &mv, 0, 1).expect("capture applies");
    assert!(!c.pieces.iter().any(|p| p.alive
        && (p.key == PieceKey::FissionReactor || p.sub_key == PieceKey::FissionReactor)),
        "reactor must die in its own explosion");
    assert!(!alive_at(&c, sqn(3, 5)), "d6 knight must die");
    assert!(!alive_at(&c, sqn(5, 3)), "f4 knight must die");
    assert!(alive_at(&c, sqn(1, 1)), "far pawn must survive");
}

// ---------- Blade Runner delayed kill (Sandbox, 2026-11-03) ----------

#[test]
fn blade_runner_victim_dies_a_turn_later() {
    // White BR d4 passes through black pawn e5 to f6. The pawn survives
    // Black's next move and is gone after White's following move.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::BladeRunner, 0, 3, 3, 0));
    specs.push((PieceKey::Pawn, 1, 4, 4, 0));
    let pos = build(&specs);
    let mv = piece_moves(&pos, 0, sqn(3, 3)).into_iter()
        .find(|m| m.to == sqn(5, 5)).expect("BR pass-through must exist").clone();
    let c = apply_full(&pos, &mv, 0, 1).expect("pass-through applies");
    assert!(alive_at(&c, sqn(4, 4)), "victim stays (marked) right after pass-through");
    // Black moves king h8->g8; victim still alive.
    let bk = piece_moves(&c, 1, sqn(7, 7)).into_iter()
        .find(|m| m.to == sqn(6, 7)).expect("black king step").clone();
    let c2 = apply_full(&c, &bk, 1, 0).expect("black move applies");
    assert!(alive_at(&c2, sqn(4, 4)), "victim survives the opponent's move");
    // White moves king a1->a2; doom cleanup kills the victim first.
    let wk = piece_moves(&c2, 0, sqn(0, 0)).into_iter()
        .find(|m| m.to == sqn(0, 1)).expect("white king step").clone();
    let c3 = apply_full(&c2, &wk, 0, 1).expect("white move applies");
    assert!(!alive_at(&c3, sqn(4, 4)), "victim must be gone a turn later");
}

// ---------- Basilisk petrify (Sandbox, 2026-11-03) ----------

#[test]
fn basilisk_petrifies_only_the_threatened_piece() {
    // White Basilisk d4 threatens black knight e5 (diagonal, unblocked):
    // the knight gets 0 moves; the black king moves normally. Basilisk
    // itself never captures.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::Basilisk, 0, 3, 3, 0));
    specs.push((PieceKey::Knight, 1, 4, 4, 0));
    let pos = build(&specs);
    let nm = piece_moves(&pos, 1, sqn(4, 4));
    assert!(nm.is_empty(), "petrified knight must have 0 moves, got {:?}",
        nm.iter().map(|m| (alg(m.to), m.flags)).collect::<Vec<_>>());
    let km = piece_moves(&pos, 1, sqn(7, 7));
    assert!(!km.is_empty(), "unthreatened king must move normally");
    let bm = piece_moves(&pos, 0, sqn(3, 3));
    assert!(bm.iter().all(|m| m.flags & MF_CAPTURE == 0),
        "Basilisk must never capture");
}

// ---------- Sandbox batch 2 (2026-11-03, played moves) ----------

#[test]
fn pilgrim_resurrects_on_the_move_reaching_20_and_counter_resets() {
    // Sandbox sequence, compressed: Pilgrim has travelled 16 squares, one
    // dead white pawn exists. The move taking it to exactly 20 (a1-e5,
    // 4 squares) must resurrect ON THAT MOVE, onto the origin square a1,
    // and the counter must reset to 0/20 afterwards.
    let mut specs = vec![
        (PieceKey::King, 0, 7, 0, 0), // white king h1
        (PieceKey::King, 1, 7, 7, 0), // black king h8
        (PieceKey::Pilgrim, 0, 0, 0, 0), // pilgrim a1
        (PieceKey::Pawn, 0, 1, 6, 0), // dead white pawn (square cleared below)
    ];
    let mut pos = build(&specs);
    // Travelled 16 so far.
    pos.pieces[2].odo = 16;
    // The pawn is dead: not on the board, but still a revival candidate.
    pos.pieces[3].alive = false;
    pos.at[sqn(1, 6) as usize] = 0;
    let ms = piece_moves(&pos, 0, sqn(0, 0));
    let to_e5: Vec<&Move> = ms.iter().filter(|m| m.to == sqn(4, 4)).collect();
    assert!(!to_e5.is_empty(), "a1-e5 must be offered");
    assert!(to_e5.iter().all(|m| m.flags & MF_PILGRIM_RESURRECT != 0),
        "the threshold-crossing move must resurrect (no plain variant): {:?}",
        to_e5.iter().map(|m| m.flags).collect::<Vec<_>>());
    let rm = to_e5[0].clone();
    let c = apply_full(&pos, &rm, 0, 1).expect("pilgrim resurrect applies");
    // Revived pawn appears on the origin square a1, fresh.
    assert!(alive_at(&c, sqn(0, 0)), "dead pawn must reappear on a1");
    let rev = &c.pieces[(c.at[sqn(0, 0) as usize] - 1) as usize];
    assert_eq!(rev.key, PieceKey::Pawn);
    assert_eq!(rev.player, 0);
    // Pilgrim counter reset to 0/20, charge cleared.
    let pg = &c.pieces[(c.at[sqn(4, 4) as usize] - 1) as usize];
    assert_eq!(pg.key, PieceKey::Pilgrim);
    assert_eq!(pg.odo, 0, "counter must reset to 0 after resurrecting");
    assert_eq!(pg.state_flags & (1 << 8), 0, "charged bit must clear");
}

#[test]
fn pilgrim_move_to_19_does_not_resurrect() {
    // Same setup, but the move only reaches 19: no resurrect flag.
    let mut pos = build(&[
        (PieceKey::King, 0, 7, 0, 0),
        (PieceKey::King, 1, 7, 7, 0),
        (PieceKey::Pilgrim, 0, 0, 0, 0),
        (PieceKey::Pawn, 0, 1, 6, 0),
    ]);
    pos.pieces[2].odo = 16;
    pos.pieces[3].alive = false;
    pos.at[sqn(1, 6) as usize] = 0;
    let ms = piece_moves(&pos, 0, sqn(0, 0));
    let to_d4: Vec<&Move> = ms.iter().filter(|m| m.to == sqn(3, 3)).collect();
    assert!(!to_d4.is_empty(), "a1-d4 must be offered");
    assert!(to_d4.iter().all(|m| m.flags & MF_PILGRIM_RESURRECT == 0),
        "a move reaching only 19 must not resurrect");
}

#[test]
fn golden_pawn_last_rank_is_instant_win_not_promotion() {
    // Sandbox: Golden Pawn e7-e8 ends the game at once, no dialog, the
    // pawn standing on e8. Engine: gold_win set, piece unchanged.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::GoldenPawn, 0, 4, 6, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(4, 6));
    let promo = ms.iter().find(|m| m.to == sqn(4, 7) && m.flags & MF_PROMO != 0)
        .expect("golden pawn must offer e7-e8 as a promotion move");
    let c = apply_full(&pos, promo, 0, 1).expect("golden move applies");
    assert_eq!(c.gold_win, Some(0), "golden pawn on last rank must win");
    let s = &c.pieces[(c.at[sqn(4, 7) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::GoldenPawn, "no piece choice: pawn stays golden on e8");
    assert_eq!(evaluate(&c, 0), MATE_CONST, "gold win must evaluate as mate");
    assert_eq!(evaluate(&c, 1), -MATE_CONST);
}

#[test]
fn hero_pawn_promotes_immediately_on_giving_check_from_7th_rank() {
    // Sandbox: Hero Pawn d6-d7+ promotes at once, before Black moves,
    // even though d7 is only the 7th rank. Engine auto-queens on apply.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::HeroPawn, 0, 3, 5, 0));
    // Black king e8 (KINGS has it on h8; move it to e8 for the check).
    let mut pos = build(&specs);
    pos.pieces[1].x = 4;
    pos.pieces[1].y = 7;
    pos.at[sqn(7, 7) as usize] = 0;
    pos.at[sqn(4, 7) as usize] = 2;
    let ms = piece_moves(&pos, 0, sqn(3, 5));
    let m = find(&ms, sqn(3, 6)).expect("hero pawn must offer d6-d7");
    assert_eq!(m.flags & MF_PROMO, 0, "d7 is not the last rank");
    let c = apply_full(&pos, m, 0, 1).expect("hero move applies");
    let s = &c.pieces[(c.at[sqn(3, 6) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::Queen, "check-giving hero pawn must promote at once");
}

#[test]
fn hero_pawn_quiet_move_does_not_promote() {
    // Same piece, no check given: stays a Hero Pawn.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::HeroPawn, 0, 3, 4, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 4));
    let m = find(&ms, sqn(3, 5)).expect("hero pawn must offer d5-d6");
    let c = apply_full(&pos, m, 0, 1).expect("hero move applies");
    let s = &c.pieces[(c.at[sqn(3, 5) as usize] - 1) as usize];
    assert_eq!(s.key, PieceKey::HeroPawn, "no check, no promotion");
}

#[test]
fn horde_mother_spawn_set_is_all_free_ranks_2_to_7() {
    // Sandbox: on capture, EVERY free square on ranks 2-7 is offered;
    // ranks 1/8 and occupied squares are excluded. Origin (vacated) and
    // victim square (freed) count as free: 48 squares on an open board.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::HordeMother, 0, 3, 3, 0));
    specs.push((PieceKey::Pawn, 1, 4, 4, 0));
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    let spawns: Vec<u8> = ms.iter()
        .filter(|m| m.to == sqn(4, 4) && m.flags & MF_HORDE_SPAWN != 0)
        .map(|m| m.special_a)
        .collect();
    assert_eq!(spawns.len(), 48, "all free ranks 2-7 squares: {:?}",
        spawns.iter().map(|s| alg(*s)).collect::<Vec<_>>());
    let mut sorted = spawns.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 48, "spawn squares must be distinct");
    for s in &spawns {
        let y = s / 8;
        assert!((1..=6).contains(&y), "spawn square {} on rank {} outside 2-7", alg(*s), y + 1);
    }
    assert!(spawns.contains(&sqn(3, 3)), "origin d4 (vacated) must be offered");
    assert!(spawns.contains(&sqn(4, 4)), "victim square e5 (freed) must be offered");
    assert!(spawns.contains(&sqn(0, 1)), "a2 must be offered");
    assert!(spawns.contains(&sqn(7, 6)), "h7 must be offered");
}

#[test]
fn horde_shared_death_capturing_one_kills_all() {
    // Sandbox: capturing one hordeling removes the mother and all others.
    let mut pos = build(&[
        (PieceKey::King, 0, 0, 0, 0),
        (PieceKey::King, 1, 7, 7, 0),
        (PieceKey::HordeMother, 0, 3, 5, 0),
        (PieceKey::Hordeling, 0, 4, 5, 0),
        (PieceKey::Hordeling, 0, 3, 4, 0),
        (PieceKey::Pawn, 1, 3, 6, 0),
    ]);
    pos.pieces[2].horde_id = 7;
    pos.pieces[3].horde_id = 7;
    pos.pieces[4].horde_id = 7;
    let cap = Move { from: sqn(3, 6), to: sqn(4, 5), flags: MF_CAPTURE, ..Default::default() };
    let c = apply_full(&pos, &cap, 1, 0).expect("hordeling capture applies");
    assert!(!alive_at(&c, sqn(3, 5)), "mother must die with the hordeling");
    assert!(!alive_at(&c, sqn(3, 4)), "other hordeling must die with the hordeling");
    assert!(alive_at(&c, sqn(4, 5)), "capturing pawn stands on e6");
}

#[test]
fn bishopper_crouch_recharges_after_one_opponent_reply() {
    // Sandbox-settled (2026-11-03): one-round cooldown. Crouch, opponent
    // replies, leap, opponent replies once -> crouch is offered again.
    let mut specs = KINGS.to_vec();
    specs.push((PieceKey::BisHop, 0, 3, 3, 0));
    specs.push((PieceKey::Pawn, 1, 4, 4, 0)); // screen e5
    specs.push((PieceKey::Pawn, 1, 0, 6, 0)); // black pawn a7, reply 1
    specs.push((PieceKey::Pawn, 1, 1, 6, 0)); // black pawn b7, reply 2
    let pos = build(&specs);
    let ms = piece_moves(&pos, 0, sqn(3, 3));
    let crouch = ms.iter().find(|m| m.flags & MF_BISHOPPER_CROUCH != 0)
        .expect("standing bishopper must offer crouch").clone();
    let c1 = apply_full(&pos, &crouch, 0, 1).expect("crouch applies");
    assert_eq!(c1.stm, 1);
    let r1 = Move { from: sqn(0, 6), to: sqn(0, 5), ..Default::default() };
    let c2 = apply_full(&c1, &r1, 1, 0).expect("black reply 1 applies");
    assert_eq!(c2.stm, 0);
    let ms2 = piece_moves(&c2, 0, sqn(3, 3));
    let leap = find(&ms2, sqn(5, 5)).expect("leap to f6 must exist").clone();
    assert_ne!(leap.flags & MF_BISHOPPER_LEAP, 0);
    let c3 = apply_full(&c2, &leap, 0, 1).expect("leap applies");
    let slot = c3.at[sqn(5, 5) as usize] as usize - 1;
    assert!(!c3.pieces[slot].crouched(), "leap consumes crouch");
    let r2 = Move { from: sqn(1, 6), to: sqn(1, 5), ..Default::default() };
    let c4 = apply_full(&c3, &r2, 1, 0).expect("black reply 2 applies");
    assert_eq!(c4.stm, 0);
    let ms3 = piece_moves(&c4, 0, sqn(5, 5));
    assert!(ms3.iter().any(|m| m.flags & MF_BISHOPPER_CROUCH != 0),
        "crouch must be offered again after one opponent reply");
}
