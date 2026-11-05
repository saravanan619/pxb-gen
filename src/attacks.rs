use crate::pos::*;
use crate::types::*;
use crate::movegen::piece_attacks;

pub fn find_king(pos: &Pos, player: u8) -> Option<usize> {
    for (i, p) in pos.pieces.iter().enumerate() {
        if !p.alive || p.player != player { continue; }
        if !xy_on_board(p.x, p.y) { continue; } // v130: skip phantom kings
        use PieceKey::*;
        let k = p.key; let s = p.sub_key;
        if matches!(k, King|WarKing|MeditatingKing|PopeKing|IrreverentKing)
        || matches!(s, King|WarKing|MeditatingKing|PopeKing|IrreverentKing) {
            return Some(i);
        }
    }
    None
}

pub fn av_suppressed(pos: &Pos, p_idx: usize) -> bool {
    let p = &pos.pieces[p_idx];
    if !xy_on_board(p.x, p.y) { return false; } // v130
    let x = p.x as i8; let y = p.y as i8;
    for dx in -1i8..=1 {
        for dy in -1i8..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx; let ny = y + dy;
            if !on_board(nx, ny) { continue; }
            if let Some(q_idx) = slot_of(pos, sq(nx, ny)) {
                let q = &pos.pieces[q_idx];
                if q.player != p.player
                   && (q.key == PieceKey::AntiViolence || q.sub_key == PieceKey::AntiViolence) {
                    return true;
                }
            }
        }
    }
    false
}

// ── Lever-3 attack context ────────────────────────────────────────────────
// Pure per-node-cost optimisation, zero behaviour change: the original
// square_attacked / king_attacked / gives_check loops called av_suppressed
// (8 slot_of lookups) AND is_paralyzed (a full piece scan hunting for an
// enemy Basilisk) for EVERY attacker piece — O(attackers x pieces) per
// call, on a path the search walks once per candidate move. This context
// precomputes, in two linear scans, exactly the information those two
// predicates can observe:
//   * av_bb: AntiViolence squares as seen through pos.at (the same view
//     slot_of gives av_suppressed, including any dead-but-still-indexed
//     piece — av_suppressed never checks alive, so neither do we here).
//   * bas: whether an alive, on-board, non-frozen (when
//     PROVISIONAL_ICICLE_FREEZE) Basilisk of each player exists — the
//     only case in which is_paralyzed can return true.
// Per attacker, suppression is then a walk over the (usually empty) enemy
// AV bitboard, and paralysis is skipped outright when no enemy Basilisk
// exists; when one does exist, or when a piece's player is outside {0,1},
// the original predicates run unchanged (exact fallback).
struct AttackCtx {
    av_bb: [u64; 2],
    av_other: bool,
    bas: [bool; 2],
    bas_other: bool,
}

fn attack_ctx(pos: &Pos) -> AttackCtx {
    let mut ctx = AttackCtx { av_bb: [0, 0], av_other: false, bas: [false, false], bas_other: false };
    // Single pieces scan for Basilisk flags, plus an AntiViolence
    // existence gate: if no piece in the whole list carries the AV key,
    // no at[] entry can point to one either, so the AV bitboards are
    // certainly empty and the 64-square at[] scan below is skipped.
    // This keeps the context near-free on small / AV-free boards (where
    // the old per-attacker cost was already tiny); when AV exists, the
    // at[] scan runs for exactness (see the av_bb comment above).
    let mut has_av_piece = false;
    for q in &pos.pieces {
        if q.key == PieceKey::AntiViolence || q.sub_key == PieceKey::AntiViolence {
            has_av_piece = true;
        }
        if !q.alive || !xy_on_board(q.x, q.y) { continue; }
        if q.key != PieceKey::Basilisk && q.sub_key != PieceKey::Basilisk { continue; }
        if PROVISIONAL_ICICLE_FREEZE && q.frozen() { continue; }
        match q.player {
            0 => ctx.bas[0] = true,
            1 => ctx.bas[1] = true,
            _ => ctx.bas_other = true,
        }
    }
    if has_av_piece {
        for s in 0..64 {
            let v = pos.at[s];
            if v != 0 {
                if let Some(q) = pos.pieces.get((v - 1) as usize) {
                    if q.key == PieceKey::AntiViolence || q.sub_key == PieceKey::AntiViolence {
                        match q.player {
                            0 => ctx.av_bb[0] |= 1u64 << s,
                            1 => ctx.av_bb[1] |= 1u64 << s,
                            _ => ctx.av_other = true,
                        }
                    }
                }
            }
        }
    }
    ctx
}

#[inline]
fn ctx_suppressed(ctx: &AttackCtx, pos: &Pos, i: usize) -> bool {
    let p = &pos.pieces[i];
    if ctx.av_other { return av_suppressed(pos, i); }
    let enemy_bb = match p.player {
        0 => ctx.av_bb[1],
        1 => ctx.av_bb[0],
        _ => return av_suppressed(pos, i),
    };
    // av_suppressed checks the 8 Chebyshev neighbours of (p.x, p.y);
    // an enemy AV bit at Chebyshev distance exactly 1 is the same test.
    let ax = p.x as i8;
    let ay = p.y as i8;
    let mut bb = enemy_bb;
    while bb != 0 {
        let t = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if (sq_x(t) as i8 - ax).abs().max((sq_y(t) as i8 - ay).abs()) == 1 {
            return true;
        }
    }
    false
}

#[inline]
fn ctx_paralyzed(ctx: &AttackCtx, pos: &Pos, i: usize, occ: &[bool; 64]) -> bool {
    let pl = pos.pieces[i].player;
    if ctx.bas_other { return is_paralyzed(pos, i, occ); }
    let enemy_bas = match pl {
        0 => ctx.bas[1],
        1 => ctx.bas[0],
        _ => return is_paralyzed(pos, i, occ),
    };
    if !enemy_bas { return false; }
    is_paralyzed(pos, i, occ)
}

pub fn square_attacked(pos: &Pos, target: u8, by_player: u8, occ: &[bool; 64]) -> bool {
    let ctx = attack_ctx(pos);
    for i in 0..pos.pieces.len() {
        let q = &pos.pieces[i];
        if !xy_on_board(q.x, q.y) { continue; } // v130: skip phantom pieces
        if !q.alive || q.player != by_player { continue; }
        if ctx_suppressed(&ctx, pos, i) { continue; }
        if ctx_paralyzed(&ctx, pos, i, occ) { continue; }
        if piece_attacks(pos, i, occ, true, false).has(target) { return true; }
    }
    false
}

pub fn king_attacked(pos: &Pos, player: u8, by_player: u8, pre_occ: Option<&[bool; 64]>) -> bool {
    let occ_owned;
    let occ: &[bool; 64] = match pre_occ {
        Some(o) => o,
        None => { occ_owned = occ(pos); &occ_owned }
    };
    let ks = match find_king(pos, player) {
        Some(k) => sq(pos.pieces[k].x as i8, pos.pieces[k].y as i8),
        None => return false,
    };
    let kx = sq_x(ks) as i8;
    let ky = sq_y(ks) as i8;

    let ctx = attack_ctx(pos);
    if let Some(ei) = find_king(pos, by_player) {
        let ek = &pos.pieces[ei];
        if (ek.x as i8 - kx).abs().max((ek.y as i8 - ky).abs()) <= 1
           && !ctx_suppressed(&ctx, pos, ei) && !ctx_paralyzed(&ctx, pos, ei, occ) {
            return true;
        }
    }
    for i in 0..pos.pieces.len() {
        let q = &pos.pieces[i];
        if !xy_on_board(q.x, q.y) { continue; } // v130: skip phantom pieces
        if !q.alive || q.player != by_player || q.key == PieceKey::King { continue; }
        if ctx_suppressed(&ctx, pos, i) { continue; }
        if ctx_paralyzed(&ctx, pos, i, occ) { continue; }
        if piece_attacks(pos, i, occ, true, false).has(ks) { return true; }
    }
    false
}

pub fn king_attacked_strict(pos: &Pos, player: u8, by_player: u8) -> bool {
    let occ = occ(pos);
    let ks = match find_king(pos, player) {
        Some(k) => sq(pos.pieces[k].x as i8, pos.pieces[k].y as i8),
        None => return false,
    };
    let ctx = attack_ctx(pos);
    if let Some(ei) = find_king(pos, by_player) {
        let ek = &pos.pieces[ei];
        let kx = sq_x(ks) as i8; let ky = sq_y(ks) as i8;
        if (ek.x as i8 - kx).abs().max((ek.y as i8 - ky).abs()) <= 1
           && !ctx_suppressed(&ctx, pos, ei) && !ctx_paralyzed(&ctx, pos, ei, &occ) {
            return true;
        }
    }
    for i in 0..pos.pieces.len() {
        let q = &pos.pieces[i];
        if !xy_on_board(q.x, q.y) { continue; } // v130: skip phantom pieces
        if !q.alive || q.player != by_player || q.key == PieceKey::King { continue; }
        if ctx_suppressed(&ctx, pos, i) || ctx_paralyzed(&ctx, pos, i, &occ) { continue; }
        if piece_attacks(pos, i, &occ, true, true).has(ks) { return true; }
    }
    false
}

pub fn gives_check(p2: &Pos, me: u8, opp: u8, pre_occ: Option<&[bool; 64]>) -> bool {
    let occ_owned;
    let occ = match pre_occ { Some(o) => o, None => { occ_owned = occ(p2); &occ_owned } };
    let ks = match find_king(p2, opp) {
        Some(k) => sq(p2.pieces[k].x as i8, p2.pieces[k].y as i8),
        None => return false,
    };
    let ctx = attack_ctx(p2);
    for i in 0..p2.pieces.len() {
        let q = &p2.pieces[i];
        if !xy_on_board(q.x, q.y) { continue; } // v130: skip phantom pieces
        if !q.alive || q.player != me { continue; }
        if ctx_suppressed(&ctx, p2, i) || ctx_paralyzed(&ctx, p2, i, occ) { continue; }
        if piece_attacks(p2, i, occ, true, false).has(ks) { return true; }
    }
    false
}

pub fn checkers_of(pos: &Pos, player: u8, by_player: u8) -> Bitboard {
    let mut out = Bitboard(0);
    let ks = match find_king(pos, player) {
        Some(k) => sq(pos.pieces[k].x as i8, pos.pieces[k].y as i8),
        None => return out,
    };
    let occ_local = occ(pos);
    if let Some(ei) = find_king(pos, by_player) {
        let ek = &pos.pieces[ei];
        if (ek.x as i8 - sq_x(ks) as i8).abs().max((ek.y as i8 - sq_y(ks) as i8).abs()) <= 1 {
            out.set(sq(ek.x as i8, ek.y as i8));
        }
    }
    for i in 0..pos.pieces.len() {
        let q = &pos.pieces[i];
        if !xy_on_board(q.x, q.y) { continue; } // v130: skip phantom pieces
        if !q.alive || q.player != by_player || q.key == PieceKey::King { continue; }
        if is_paralyzed(pos, i, &occ_local) { continue; }
        if piece_attacks(pos, i, &occ_local, true, false).has(ks) {
            out.set(sq(q.x as i8, q.y as i8));
        }
    }
    out
}

pub fn has_legal_move(pos: &Pos, player: u8, opp: u8) -> bool {
    let moves = crate::movegen::gen_moves(pos, player, None);
    for m in &moves {
        if let Some(p2) = crate::apply_full::apply_full(pos, m, player, opp) {
            if !king_attacked(&p2, player, opp, None) { return true; }
        }
    }
    false
}

pub fn is_checkmate(pos: &Pos, player: u8, opp: u8) -> bool {
    king_attacked(pos, player, opp, None) && !has_legal_move(pos, player, opp)
}
