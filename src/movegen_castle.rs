use crate::pos::*;
use crate::types::*;
use crate::attacks::*;

pub fn castle_moves(pos: &Pos, player: u8, opp: u8, occ_local: &[bool; 64], out: &mut Vec<Move>) {
    let ky: i8 = if player == 0 { 0 } else { 7 };
    let king = match find_king(pos, player) { Some(k) => k, None => return };
    let kp = &pos.pieces[king];
    if kp.x != 4 || kp.y as i8 != ky { return; }
    if kp.moved() { return; }
    if is_paralyzed(pos, king, occ_local) { return; }
    if king_attacked(pos, player, opp, Some(occ_local)) { return; }

    for &(flag_k, rook_x, king_to, between) in &[
        (true,  7i8, 6i8, [5i8, 6i8, -1i8]),
        (false, 0i8, 2i8, [1i8, 2i8, 3i8]),
    ] {
        let rook_sq = sq(rook_x, ky);
        let ri = match slot_of(pos, rook_sq) { Some(s) => s, None => continue };
        let r = &pos.pieces[ri];
        if r.moved() { continue; }
        let is_castle_rook = matches!(r.key,
            PieceKey::Rook | PieceKey::PhaseRook | PieceKey::RookZero)
            || matches!(r.sub_key,
                PieceKey::Rook | PieceKey::PhaseRook | PieceKey::RookZero);
        if !is_castle_rook { continue; }

        let mut clear = true;
        for &bx in &between {
            if bx < 0 || bx > 7 { continue; }  // v131: skip -1 sentinel (king-side)
            if pos.at[sq(bx, ky) as usize] != 0 { clear = false; break; }
        }
        if !clear { continue; }

        let (tx1, tx2) = if flag_k { (5i8, 6i8) } else { (3i8, 2i8) };
        let mut safe = true;
        for &tx in &[tx1, tx2] {
            if square_attacked(pos, sq(tx, ky), opp, occ_local) { safe = false; break; }
        }
        if !safe { continue; }

        let mv = Move {
            from: sq(4, ky), to: sq(king_to, ky),
            flags: if flag_k { MF_CASTLE_K } else { MF_CASTLE_Q },
            ..Default::default()
        };
        out.push(mv);
    }
}
