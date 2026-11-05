use crate::pos::*;
use crate::types::*;
use crate::eval::*;
use crate::attacks::*;
use crate::movegen::{is_king_target, piece_attacks};

pub fn apply_full(pos: &Pos, mv: &Move, player: u8, opp: u8) -> Option<Pos> {
    apply_full_skip_wa(pos, mv, player, opp, false)
}

/// Same as `apply_full` but without the WarAutomaton advance. Used ONLY
/// for the King-safety legality check: wiki (WarAutomaton) judges check
/// on the position after the capture but BEFORE the automatic advance
/// (a capture that leaves the King in check is illegal even if the
/// advance would block it, and a capture is legal even if the advance
/// exposes the King). The resulting position for search/eval is always
/// the full `apply_full` one (with the advance); only legality is
/// pre-advance. Callers: veto King safety + think fallback (root-only;
/// search's hot path still uses post-advance safety for now, which can
/// only miss WA-exposed mates, never play a WA-blocked illegal move
/// once veto gates the root).
pub fn apply_full_no_wa(pos: &Pos, mv: &Move, player: u8, opp: u8) -> Option<Pos> {
    apply_full_skip_wa(pos, mv, player, opp, true)
}

fn apply_full_skip_wa(pos: &Pos, mv: &Move, player: u8, opp: u8, skip_wa: bool) -> Option<Pos> {
    let mut p2 = pos.clone();
    let n0 = alive_count(&p2);

    if !p2.doomed.is_empty() {
        let mut di = p2.doomed.len();
        while di > 0 {
            di -= 1;
            if p2.doomed[di].by != player { continue; }
            let id = p2.doomed[di].id;
            for s in 0..p2.pieces.len() {
                if p2.pieces[s].alive && p2.pieces[s].id == id {
                    p2.pieces[s].alive = false;
                    if xy_on_board(p2.pieces[s].x, p2.pieces[s].y) { // v130
                        let sqq = sq(p2.pieces[s].x as i8, p2.pieces[s].y as i8);
                        p2.at[sqq as usize] = 0;
                    }
                    break;
                }
            }
            p2.doomed.remove(di);
        }
    }

    if mv.flags & MF_KNIGHTMARE_OFF != 0 {
        // Sandbox-verified: jumping off is a knight jump to a 14x14 ring
        // square, carried as a code in `to`. A code decoding on-board (or
        // outside the stage) is not a legal off-move.
        let (ox, oy) = off_xy(mv.to);
        if !off_in_ring(ox, oy) { return None; }
        if p2.offboard.iter().any(|o| o.ox == ox && o.oy == oy) { return None; }
        if let Some(slot) = slot_of(&p2, mv.from) {
            let p = &p2.pieces[slot];
            if (p.key == PieceKey::Knightmare || p.sub_key == PieceKey::Knightmare) && p.player == player {
                p2.offboard.push(OffboardPiece {
                    id: p.id, key: p.key, sub_key: p.sub_key, player: p.player, off_since: 1,
                    ox, oy,
                });
                p2.pieces[slot].alive = false;
                p2.at[mv.from as usize] = 0;
            }
        }
        p2.stm = opp;
        p2.halfmove += 1;
        p2.hist_hashes.push(rep_hash(&p2, opp));
        return Some(p2);
    }
    if mv.flags & MF_KNIGHTMARE_DROP != 0 {
        // Validate before mutating: the target must be empty (quiet drop)
        // or hold an enemy non-King (drop-capture, Sandbox-verified). An
        // own piece or a King on the target makes the move illegal; the
        // old code silently VANISHED the Knightmare in that case.
        if mv.to >= 64 { return None; }
        if let Some(vs) = slot_of(&p2, mv.to) {
            let v = &p2.pieces[vs];
            if v.player == player || v.key == PieceKey::King || v.sub_key == PieceKey::King {
                return None;
            }
        }
        if let Some(idx) = p2.offboard.iter().position(|o| o.id == mv.offboard_id) {
            let ob = p2.offboard.remove(idx);
            if let Some(vs) = slot_of(&p2, mv.to) {
                kill_slot(&mut p2, vs);
            }
            let new_piece = Piece {
                id: ob.id, key: ob.key, sub_key: ob.sub_key, player: ob.player,
                x: sq_x(mv.to), y: sq_y(mv.to),
                state_flags: 1 << 2, alive: true, ..Default::default()
            };
            p2.pieces.push(new_piece);
            p2.at[mv.to as usize] = p2.pieces.len() as u16;
        }
        p2.stm = opp;
        p2.halfmove += 1;
        p2.hist_hashes.push(rep_hash(&p2, opp));
        return Some(p2);
    }
    if mv.flags & MF_KNIGHTMARE_HOP != 0 {
        // Offboard-to-offboard knight hop on the 14x14 staging board.
        let (nx, ny) = off_xy(mv.to);
        if !off_in_ring(nx, ny) { return None; }
        if p2.offboard.iter().any(|o| o.id != mv.offboard_id && o.ox == nx && o.oy == ny) {
            return None;
        }
        match p2.offboard.iter_mut().find(|o| {
            o.id == mv.offboard_id && o.player == player
                && (o.key == PieceKey::Knightmare || o.sub_key == PieceKey::Knightmare)
        }) {
            Some(ob) => { ob.ox = nx; ob.oy = ny; }
            None => return None,
        }
        p2.stm = opp;
        p2.halfmove += 1;
        p2.hist_hashes.push(rep_hash(&p2, opp));
        return Some(p2);
    }
    if mv.flags & MF_BISHOPPER_CROUCH != 0 {
        if let Some(slot) = slot_of(&p2, mv.from) {
            p2.pieces[slot].state_flags |= 1 << 0;
        }
        p2.stm = opp;
        p2.halfmove += 1;
        p2.hist_hashes.push(rep_hash(&p2, opp));
        return Some(p2);
    }
    if mv.flags & MF_DJINN_DISSIPATE != 0 {
        if let Some(slot) = slot_of(&p2, mv.from) {
            let p = p2.pieces[slot].clone();
            p2.limbo.push(LimboPiece {
                id: p.id, key: p.key, sub_key: p.sub_key, player: p.player,
                home_x: p.x as i8, home_y: p.y as i8,
            });
            p2.pieces[slot].alive = false;
            p2.at[mv.from as usize] = 0;
        }
        // Dissipate returns before apply_triggers, but it CAN break a
        // Gunslinger mutual threat (the dissipated Djinn may have been
        // the threat partner), so the standoff timer is updated here.
        update_gunslinger_armed(pos, &mut p2, player);
        p2.stm = opp;
        p2.halfmove += 1;
        p2.hist_hashes.push(rep_hash(&p2, opp));
        return Some(p2);
    }

    if mv.flags & MF_IS_PUSH != 0 {
        if !sumo_push(&mut p2, mv, player) { return None; }
    }

    let mut vic_pre: Option<Piece> = None;
    if mv.flags & MF_IS_PUSH == 0 {
        let cap_sq = if mv.flags & MF_EP != 0 { mv.ep_victim } else { mv.to };
        if let Some(v_slot) = slot_of(&p2, cap_sq) {
            let v = p2.pieces[v_slot].clone();
            if v.player != player { vic_pre = Some(v); }
        }
    }

    if mv.flags & (MF_CASTLE_K | MF_CASTLE_Q) != 0 {
        let ky: i8 = if player == 0 { 0 } else { 7 };
        apply_move_raw(&mut p2, mv.from, mv.to);
        let (rook_from, rook_to) = if mv.flags & MF_CASTLE_K != 0 { (7u8, 5u8) } else { (0u8, 3u8) };
        apply_move_raw(&mut p2, sq(rook_from as i8, ky), sq(rook_to as i8, ky));
    } else if mv.flags & MF_DANCER_DOUBLE != 0 {
        let leg1_vic = slot_of(&p2, mv.to).map(|s| p2.pieces[s].clone());
        let leg2_vic = slot_of(&p2, mv.special_a).map(|s| p2.pieces[s].clone());
        let mut vic1 = leg1_vic.filter(|v| v.player != player);
        let vic2 = leg2_vic.filter(|v| v.player != player);
        if vic1.is_some() && vic2.is_some() {
            let v1 = vic1.clone().unwrap();
            let v2 = vic2.clone().unwrap();
            let drop = if piece_value(&v1) >= piece_value(&v2) { v2 } else { vic1.take().unwrap() };
            kill_piece(&mut p2, drop.id);
            vic_pre = vic1.or(Some(drop));
        } else {
            vic_pre = vic1.or(vic2);
        }
        apply_move_raw(&mut p2, mv.from, mv.to);
        apply_move_raw(&mut p2, mv.to, mv.special_a);
    } else if mv.from == mv.to {
        // Stay-in-place actions (Gunslinger standoff, Icicle thaw): there
        // is no separate victim square. The old code ran
        // apply_move_raw_with_cap with cap_sq == from, which KILLED THE
        // MOVER ITSELF — the standoff self-kill lived here as well as in
        // the explicit block, and it would have killed every thaw move.
        // A from==to move is a no-op displacement; victims (standoff)
        // are handled by their explicit blocks below.
        apply_move_raw(&mut p2, mv.from, mv.to);
    } else {
        let cap_sq = if mv.flags & MF_EP != 0 { mv.ep_victim } else { mv.to };
        apply_move_raw_with_cap(&mut p2, mv.from, mv.to, cap_sq);
    }

    let had_cap = vic_pre.is_some();
    apply_triggers(&mut p2, pos, mv, player, opp, n0, vic_pre, skip_wa);
    init_eval_sum(&mut p2);
    p2.stm = opp;
    let is_pawn_move = mv.from < 64 && slot_of(pos, mv.from).map_or(false, |s| is_piece_pawn_like(&pos.pieces[s]));
    if had_cap || alive_count(&p2) < n0 || (mv.flags & (MF_CAPTURE | MF_PROMO) != 0) || is_pawn_move {
        p2.halfmove = 0;
    } else {
        p2.halfmove += 1;
    }
    p2.hist_hashes.push(rep_hash(&p2, opp));
    Some(p2)
}

// ── helpers ───────────────────────────────────────────────────────────────
pub fn kill_piece(p2: &mut Pos, id: u32) {
    for i in 0..p2.pieces.len() {
        if p2.pieces[i].alive && p2.pieces[i].id == id {
            kill_slot(p2, i);
            return;
        }
    }
}
pub fn kill_slot(p2: &mut Pos, slot: usize) {
    if !p2.pieces[slot].alive { return; }
    p2.pieces[slot].alive = false;
    if xy_on_board(p2.pieces[slot].x, p2.pieces[slot].y) { // v130
        let sqq = sq(p2.pieces[slot].x as i8, p2.pieces[slot].y as i8);
        p2.at[sqq as usize] = 0;
    }
    let pl = p2.pieces[slot].player;
    p2.last_cap[pl as usize] = (p2.pieces[slot].key, p2.pieces[slot].sub_key);
}
fn apply_move_raw(p2: &mut Pos, from: u8, to: u8) {
    if let Some(slot) = slot_of(p2, from) {
        p2.at[from as usize] = 0;
        p2.at[to as usize] = (slot + 1) as u16;
        p2.pieces[slot].x = sq_x(to);
        p2.pieces[slot].y = sq_y(to);
    }
}
fn apply_move_raw_with_cap(p2: &mut Pos, from: u8, to: u8, cap_sq: u8) {
    if let Some(vs) = slot_of(p2, cap_sq) { kill_slot(p2, vs); }
    apply_move_raw(p2, from, to);
}

/// Wiki Sumo ram (pixiewiki Sumo Rook). The Sumo moves to V-1 (one square
/// before the first target V) and the consecutive line starting at V is
/// shoved along the push direction: 1 piece -> 2 squares, 2 pieces -> 1
/// square each, 3+ pieces -> hold ground (no push; movegen generates a
/// quiet move to V-1 instead, so a push move with 3+ is rejected here).
/// The victim is at mv.to + push_dir (NOT mv.to: the Sumo stops before V).
/// Both sides push except the mover's own King, which is never pushed
/// (a push line containing it is rejected). Pushed pawns never promote
/// (keys are untouched); Iron and Knightmare pushed off-board die
/// (Knightmare is captured, not offboarded); a King pushed off-board
/// sets king_thrown (a win, as elsewhere). AV never blocks a push and
/// paralysis never blocks one either (paralysis is positional and is
/// recomputed from the new board). This runs BEFORE the Sumo itself
/// moves, while V is still on its starting square.
pub fn sumo_push(p2: &mut Pos, mv: &Move, player: u8) -> bool {
    if mv.flags & MF_IS_PUSH == 0 { return true; }
    let dx = mv.push_dx; let dy = mv.push_dy;
    let vx = sq_x(mv.to) as i8 + dx;
    let vy = sq_y(mv.to) as i8 + dy;
    if !on_board(vx, vy) { return false; }
    let victim_sq = sq(vx, vy);
    let vs = match slot_of(p2, victim_sq) { Some(s) => s, None => return false };

    // Consecutive line from V along (dx, dy); gaps break the line.
    let mut line: Vec<usize> = vec![vs];
    let mut lx = vx + dx;
    let mut ly = vy + dy;
    while on_board(lx, ly) {
        match slot_of(p2, sq(lx, ly)) {
            Some(s) => { line.push(s); lx += dx; ly += dy; }
            None => break,
        }
    }
    if line.len() >= 3 { return false; }
    for &s in line.iter() {
        let p = &p2.pieces[s];
        if p.player == player && (p.key == PieceKey::King || p.sub_key == PieceKey::King) {
            return false;
        }
    }

    // Move a piece to (nx, ny), or eject it if (nx, ny) is off-board.
    // Ejecting a King sets king_thrown (and kills it, as before).
    let mut move_or_eject = |p2: &mut Pos, slot: usize, nx: i8, ny: i8| {
        let old_sq = sq(p2.pieces[slot].x as i8, p2.pieces[slot].y as i8);
        p2.at[old_sq as usize] = 0;
        if !on_board(nx, ny) {
            let p = &p2.pieces[slot];
            if p.key == PieceKey::King || p.sub_key == PieceKey::King {
                p2.king_thrown = Some(player);
            }
            p2.pieces[slot].alive = false;
        } else {
            let ns = sq(nx, ny);
            p2.at[ns as usize] = (slot + 1) as u16;
            p2.pieces[slot].x = nx as u8;
            p2.pieces[slot].y = ny as u8;
        }
    };

    if line.len() == 1 {
        let dx2 = vx + 2 * dx;
        let dy2 = vy + 2 * dy;
        if on_board(dx2, dy2) && slot_of(p2, sq(dx2, dy2)).is_some() {
            // Gap-behind collision: V+1 is empty but V+2 is occupied by a
            // non-line piece, so the 2-square shove cannot land.
            return false;
        }
        move_or_eject(p2, vs, dx2, dy2);
        return true;
    }
    // Line of 2: second piece moves 1 (off-board -> ejected), victim
    // moves into the second piece's old square. V+2 is empty or
    // off-board by consecutive construction; anything else is corrupt.
    let second = line[1];
    let sx = p2.pieces[second].x as i8;
    let sy = p2.pieces[second].y as i8;
    let sdx = sx + dx;
    let sdy = sy + dy;
    if on_board(sdx, sdy) && slot_of(p2, sq(sdx, sdy)).is_some() {
        return false;
    }
    move_or_eject(p2, second, sdx, sdy);
    move_or_eject(p2, vs, sx, sy);
    true
}

// ── triggers (huge but direct port) ───────────────────────────────────────
fn apply_triggers(p2: &mut Pos, pos_pre: &Pos, mv: &Move, player: u8, opp: u8,
                  n0: usize, vic_pre: Option<Piece>, skip_wa: bool) {
    let _ = pos_pre;
    let last_rank: i8 = if player == 0 { 7 } else { 0 };

    for i in 0..p2.pieces.len() {
        if !p2.pieces[i].alive { continue; }
        if p2.pieces[i].player != player { continue; }
        let k = p2.pieces[i].key; let s = p2.pieces[i].sub_key;
        if k == PieceKey::Dancer || s == PieceKey::Dancer {
            p2.pieces[i].state_flags &= !(1 << 7);
        }
    }

    let mover_slot = slot_of(p2, mv.to);
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if k == PieceKey::Dancer || sk == PieceKey::Dancer {
            if mv.flags & MF_DANCER_DOUBLE != 0 {
                let e = p2.pieces[s].encore.saturating_add(1).min(2);
                p2.pieces[s].encore = e;
            }
            if let Some(kk) = find_king(p2, opp) {
                let atk = piece_attacks(p2, s, &occ(p2), true, true);
                if atk.has(sq(p2.pieces[kk].x as i8, p2.pieces[kk].y as i8)) {
                    p2.pieces[s].state_flags |= 1 << 7;
                }
            }
        }
    }

    if let Some(v) = &vic_pre {
        if v.player != player {
            p2.last_cap[v.player as usize] = (v.key, v.sub_key);
        }
    }

    if let Some(s) = mover_slot {
        p2.pieces[s].state_flags |= 1 << 2;
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if (k == PieceKey::Djinn || sk == PieceKey::Djinn)
           && (p2.pieces[s].home_x < 0 || p2.pieces[s].home_y < 0) {
            p2.pieces[s].home_x = sq_x(mv.to) as i8;
            p2.pieces[s].home_y = sq_y(mv.to) as i8;
        }
    }

    // BladeRunner doom
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if k == PieceKey::BladeRunner || sk == PieceKey::BladeRunner {
            let bdx = sq_x(mv.to) as i8 - sq_x(mv.from) as i8;
            let bdy = sq_y(mv.to) as i8 - sq_y(mv.from) as i8;
            if bdx != 0 && bdy != 0 && bdx.abs() == bdy.abs() {
                let sx = if bdx > 0 { 1 } else { -1 };
                let sy = if bdy > 0 { 1 } else { -1 };
                let mut bx = sq_x(mv.from) as i8 + sx;
                let mut by = sq_y(mv.from) as i8 + sy;
                while bx != sq_x(mv.to) as i8 || by != sq_y(mv.to) as i8 {
                    if let Some(vs) = slot_of(p2, sq(bx, by)) {
                        let v = &p2.pieces[vs];
                        if v.player != player && v.key != PieceKey::King {
                            let already = p2.doomed.iter().any(|d| d.id == v.id);
                            if !already {
                                let vid = v.id;
                                p2.doomed.push(DoomedEntry { id: vid, by: player });
                            }
                        }
                    }
                    bx += sx; by += sy;
                }
            }
        }
    }

    // Promotion
    if mv.flags & MF_PROMO != 0 {
        if let Some(s) = mover_slot {
            let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
            let is_iron = k == PieceKey::IronPawn || sk == PieceKey::IronPawn;
            let is_gold = k == PieceKey::GoldenPawn;
            if !is_iron && !is_gold && !enemy_aristocrat(p2, opp) {
                let nk = if mv.promo != 0 { PieceKey::from_u16(mv.promo as u16) } else { PieceKey::Queen };
                let nk = match nk {
                    PieceKey::Queen | PieceKey::Rook | PieceKey::Bishop | PieceKey::Knight => nk,
                    _ => PieceKey::Queen,
                };
                p2.pieces[s].key = nk;
                p2.pieces[s].sub_key = nk;
            }
            if k == PieceKey::GoldenPawn && sq_y(mv.to) as i8 == last_rank
               && !enemy_aristocrat(p2, opp) {
                p2.gold_win = Some(player);
            }
        }
    }

    // HeroPawn auto-queen
    if mv.flags & MF_PROMO == 0 {
        if let Some(s) = mover_slot {
            let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
            if k == PieceKey::HeroPawn || sk == PieceKey::HeroPawn {
                if let Some(hk) = find_king(p2, opp) {
                    let atk = piece_attacks(p2, s, &occ(p2), true, true);
                    if atk.has(sq(p2.pieces[hk].x as i8, p2.pieces[hk].y as i8)) {
                        p2.pieces[s].key = PieceKey::Queen;
                        p2.pieces[s].sub_key = PieceKey::Queen;
                    }
                }
            }
        }
    }

    // Shrike first-step
    if mv.flags & MF_SHRIKE_FIRST != 0 {
        if let Some(s) = mover_slot {
            let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
            if (k == PieceKey::Shrike || sk == PieceKey::Shrike)
               && sq_y(mv.from) as i8 == sq_y(mv.to) as i8 - 2 * (if player == 0 { 1 } else { -1 }) {
                let d_: i8 = if player == 0 { 1 } else { -1 };
                let target_sq = sq(sq_x(mv.from) as i8, sq_y(mv.from) as i8 + d_);
                if let Some(vs2) = slot_of(p2, target_sq) {
                    if p2.pieces[vs2].player != player && p2.pieces[vs2].key != PieceKey::King {
                        kill_slot(p2, vs2);
                    }
                }
            }
        }
    }

    // Marauder kill
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if (k == PieceKey::Marauder || sk == PieceKey::Marauder)
           && vic_pre.as_ref().map_or(false, |v| v.player != player) {
            p2.pieces[s].kills = p2.pieces[s].kills.saturating_add(1);
        }
    }

    // Pilgrim odo
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if k == PieceKey::Pilgrim || sk == PieceKey::Pilgrim {
            let dist = (sq_x(mv.to) as i8 - sq_x(mv.from) as i8).abs()
                .max((sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs()) as u8;
            let new_odo = p2.pieces[s].odo.saturating_add(dist);
            p2.pieces[s].odo = new_odo;
            if new_odo >= 20 { p2.pieces[s].state_flags |= 1 << 8; }
        }
    }

    // electroKnight (Sandbox-verified 2026-11-03): the zap is deterministic
    // (movegen picks the first destructible neighbour clockwise from north
    // around the landing square). An AntiViolence victim is DESTROYED, it
    // does not block the lightning: the old wiki AV-suppression check is
    // removed. Iron victims stay immune (defence-in-depth; movegen already
    // filters them). A CHARGED capture consumes the charge even when the
    // zap kills nothing (e.g. only an Iron neighbour was adjacent).
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if k == PieceKey::ElectroKnight || sk == PieceKey::ElectroKnight {
            let was_charged = p2.pieces[s].echarged() || p2.pieces[s].quiet >= 3;
            let q = p2.pieces[s].quiet.saturating_add(1);
            p2.pieces[s].quiet = q;
            if q >= 3 { p2.pieces[s].state_flags |= 1 << 1; }
            if vic_pre.as_ref().map_or(false, |v| v.player != player) {
                if mv.flags & MF_ELECTRO != 0 {
                    let tx = sq_x(mv.special_a) as i8;
                    let ty = sq_y(mv.special_a) as i8;
                    if let Some(vs2) = slot_of(p2, sq(tx, ty)) {
                        let t = &p2.pieces[vs2];
                        let is_king = t.key == PieceKey::King || t.sub_key == PieceKey::King;
                        let is_iron = t.key == PieceKey::IronPawn || t.sub_key == PieceKey::IronPawn;
                        if t.player != player && !is_king && !is_iron {
                            kill_slot(p2, vs2);
                        }
                    }
                }
                if was_charged {
                    p2.pieces[s].quiet = 0;
                    p2.pieces[s].state_flags &= !(1 << 1);
                }
            }
        } else {
            for e in 0..p2.pieces.len() {
                if !p2.pieces[e].alive || p2.pieces[e].player != player { continue; }
                let ek = p2.pieces[e].key; let es = p2.pieces[e].sub_key;
                if ek == PieceKey::ElectroKnight || es == PieceKey::ElectroKnight {
                    p2.pieces[e].quiet = 0;
                    p2.pieces[e].state_flags &= !(1 << 1);
                }
            }
        }
    }

    // Bishopper
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if k == PieceKey::Bishopper || sk == PieceKey::Bishopper
           || k == PieceKey::BisHop || sk == PieceKey::BisHop {
            if mv.flags & MF_BISHOPPER_CROUCH != 0 {
                p2.pieces[s].state_flags |= 1 << 0;
            } else if vic_pre.as_ref().map_or(false, |v| v.key == PieceKey::BladeRunner || v.sub_key == PieceKey::BladeRunner) {
                p2.pieces[s].state_flags |= 1 << 0;
            } else {
                p2.pieces[s].state_flags &= !(1 << 0);
            }
        }
    }

    // Icicle thaw (Sandbox-verified 2026-11-03): playing the frozen
    // piece's stay-in-place move removes Frozen (and resets the frost
    // counter) at the cost of the whole turn. Afterwards normal moves
    // return. Without this, frozen was effectively permanent.
    if mv.flags & MF_ICICLE_THAW != 0 {
        if let Some(s) = mover_slot {
            p2.pieces[s].state_flags &= !(1 << 4);
            p2.pieces[s].frost = 0;
        }
    }

    // Banker
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if (k == PieceKey::Banker || sk == PieceKey::Banker)
           && vic_pre.as_ref().map_or(false, |v| is_pawn_like(v.key) && v.player != player) {
            let mut best_rank = -1i8;
            let mut best_slot: Option<usize> = None;
            for b in 0..p2.pieces.len() {
                if !p2.pieces[b].alive || p2.pieces[b].player != player { continue; }
                let bk = p2.pieces[b].key;
                if !is_pawn_like(bk) || bk == PieceKey::GoldenPawn { continue; }
                let r = if player == 0 { p2.pieces[b].y as i8 } else { 7 - p2.pieces[b].y as i8 };
                if r > best_rank { best_rank = r; best_slot = Some(b); }
            }
            if let Some(b) = best_slot {
                p2.pieces[b].key = PieceKey::GoldenPawn;
                p2.pieces[b].sub_key = PieceKey::GoldenPawn;
            }
        }
    }

    // Gunslinger standoff (Sandbox-verified 2026-11-03): the action
    // destroys ONLY the enemy. The Gunslinger SURVIVES (the old mutual-
    // destroy self-kill is removed). The move is offered only when the
    // standoff timer is armed (see movegen_special + update below).
    if mv.flags & MF_GUN_STANDOFF != 0 {
        if let Some(vs2) = slot_of(p2, mv.special_a) {
            if p2.pieces[vs2].player != player && p2.pieces[vs2].key != PieceKey::King {
                kill_slot(p2, vs2);
            }
        }
    }

    // Fish movedLast
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if k == PieceKey::Fish || sk == PieceKey::Fish {
            for f in 0..p2.pieces.len() {
                if !p2.pieces[f].alive || p2.pieces[f].player != player { continue; }
                let fk = p2.pieces[f].key; let fs = p2.pieces[f].sub_key;
                if fk == PieceKey::Fish || fs == PieceKey::Fish {
                    if f == s { p2.pieces[f].state_flags |= 1 << 3; }
                    else       { p2.pieces[f].state_flags &= !(1 << 3); }
                }
            }
        } else {
            // Fish movedLast expiry (2026-10-31): a non-Fish move by the
            // owner ends the bonus window for all friendly Fish.
            for f in 0..p2.pieces.len() {
                if !p2.pieces[f].alive || p2.pieces[f].player != player { continue; }
                let fk = p2.pieces[f].key; let fs = p2.pieces[f].sub_key;
                if fk == PieceKey::Fish || fs == PieceKey::Fish {
                    p2.pieces[f].state_flags &= !(1 << 3);
                }
            }
        }
    }

    // Fission reactor 5-cap explosion
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if (k == PieceKey::FissionReactor || sk == PieceKey::FissionReactor)
           && vic_pre.as_ref().map_or(false, |v| v.player != player) {
            let c = p2.pieces[s].caps.saturating_add(1);
            p2.pieces[s].caps = c;
            if c >= 5 {
                let cx = p2.pieces[s].x as i8;
                let cy = p2.pieces[s].y as i8;
                for &(dx, dy) in &DIRS_BISHOP {
                    let nx = cx + dx; let ny = cy + dy;
                    if !on_board(nx, ny) { continue; }
                    if let Some(vs2) = slot_of(p2, sq(nx, ny)) {
                        if p2.pieces[vs2].key != PieceKey::King { kill_slot(p2, vs2); }
                    }
                }
                kill_slot(p2, s);
            }
        }
    }

    let djinn_gone = mv.flags & MF_DJINN_DISSIPATE != 0;

    // Sandbox-verified Djinn reform (2026-11-03): a limbo Djinn reforms on
    // its home tile on the NEXT CAPTURE by either side. Two guards that
    // the old code lacked: (1) only limbo entries predating this move can
    // reform — a Djinn that dies ON this move (captured, or dissipated
    // via the early branch) must not revive in the same apply; (2) the
    // trigger is a capture (an enemy victim), not any death — standoff
    // kills, explosions and other non-capture deaths do not reform.
    let limbo_before = p2.limbo.len();

    if let Some(v) = &vic_pre {
        if v.key == PieceKey::Djinn || v.sub_key == PieceKey::Djinn {
            p2.limbo.push(LimboPiece {
                id: v.id, key: v.key, sub_key: v.sub_key, player: v.player,
                home_x: sq_x(mv.to) as i8, home_y: sq_y(mv.to) as i8,
            });
        }
    }

    let alive_now = alive_count(p2);
    let deaths_now = n0.saturating_sub(alive_now)
                   - if djinn_gone { 1 } else { 0 };
    let capture_happened = vic_pre.as_ref().map_or(false, |v| v.player != player);
    if capture_happened && deaths_now > 0 && limbo_before > 0 {
        for i in (0..limbo_before).rev() {
            let ld = p2.limbo[i].clone();
            if ld.home_x < 0 || ld.home_y < 0 { continue; }
            let hsq = sq(ld.home_x, ld.home_y);
            let blocked = slot_of(p2, hsq).map_or(false, |s| {
                let p = &p2.pieces[s];
                p.player == ld.player || p.key == PieceKey::King
            });
            if blocked { continue; }
            if let Some(vs2) = slot_of(p2, hsq) {
                if p2.pieces[vs2].player != ld.player && p2.pieces[vs2].key != PieceKey::King {
                    kill_slot(p2, vs2);
                }
            }
            let revived = Piece {
                id: ld.id, key: ld.key, sub_key: ld.sub_key, player: ld.player,
                x: ld.home_x as u8, y: ld.home_y as u8,
                home_x: ld.home_x, home_y: ld.home_y, alive: true,
                ..Default::default()
            };
            p2.pieces.push(revived);
            p2.at[hsq as usize] = p2.pieces.len() as u16;
            p2.limbo.remove(i);
            break;
        }
    }

    // hordeMother spawn
    if mv.flags & MF_HORDE_SPAWN != 0 {
        if let Some(s) = mover_slot {
            let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
            if k == PieceKey::HordeMother || sk == PieceKey::HordeMother {
                let hsx = sq_x(mv.special_a) as i8;
                let hsy = sq_y(mv.special_a) as i8;
                if on_board(hsx, hsy) && p2.at[sq(hsx, hsy) as usize] == 0 {
                    if p2.pieces[s].horde_id == 0 {
                        p2.horde_seq += 1;
                        p2.pieces[s].horde_id = p2.horde_seq;
                    }
                    let hid = p2.pieces[s].horde_id;
                    p2.id_seq += 1;
                    let new_id = p2.id_seq;
                    let hq = Piece {
                        id: new_id, key: PieceKey::Hordeling, sub_key: PieceKey::Hordeling,
                        player, x: hsx as u8, y: hsy as u8,
                        horde_id: hid, alive: true, ..Default::default()
                    };
                    p2.pieces.push(hq);
                    p2.at[sq(hsx, hsy) as usize] = p2.pieces.len() as u16;
                }
            }
        }
    }

    // horde cleanup
    if let Some(v) = &vic_pre {
        if v.horde_id != 0 {
            let hid = v.horde_id;
            let vid = v.id;
            for i in 0..p2.pieces.len() {
                if p2.pieces[i].alive && p2.pieces[i].horde_id == hid && p2.pieces[i].id != vid {
                    p2.pieces[i].alive = false;
                    if xy_on_board(p2.pieces[i].x, p2.pieces[i].y) { // v130
                        let s = sq(p2.pieces[i].x as i8, p2.pieces[i].y as i8);
                        p2.at[s as usize] = 0;
                    }
                }
            }
        }
    }

    // Recruit sac
    if mv.flags & MF_RECRUIT_SAC != 0 {
        if let Some(s) = mover_slot {
            let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
            if k == PieceKey::Recruit || sk == PieceKey::Recruit {
                let rx = sq_x(mv.special_a) as i8;
                let ry = sq_y(mv.special_a) as i8;
                if on_board(rx, ry) && p2.at[sq(rx, ry) as usize] == 0 {
                    kill_slot(p2, s);
                    p2.id_seq += 1;
                    let new_id = p2.id_seq;
                    let pawn = Piece {
                        id: new_id, key: PieceKey::Pawn, sub_key: PieceKey::Pawn,
                        player, x: rx as u8, y: ry as u8, alive: true, ..Default::default()
                    };
                    p2.pieces.push(pawn);
                    p2.at[sq(rx, ry) as usize] = p2.pieces.len() as u16;
                }
            }
        }
    }

    if mv.flags & MF_ROCKET_BLAST != 0 {
        if let Some(s) = mover_slot { p2.pieces[s].state_flags |= 1 << 5; }
    }

    // Icicle frost
    if let Some(s) = mover_slot {
        let k = p2.pieces[s].key; let sk = p2.pieces[s].sub_key;
        if (k == PieceKey::Icicle || sk == PieceKey::Icicle) && PROVISIONAL_ICICLE_FREEZE {
            let ix = p2.pieces[s].x as i8;
            let iy = p2.pieces[s].y as i8;
            for dx in -1i8..=1 {
                for dy in -1i8..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = ix + dx; let ny = iy + dy;
                    if !on_board(nx, ny) { continue; }
                    if let Some(fs) = slot_of(p2, sq(nx, ny)) {
                        if p2.pieces[fs].player != player && p2.pieces[fs].key != PieceKey::King {
                            let f = p2.pieces[fs].frost.saturating_add(1);
                            p2.pieces[fs].frost = f;
                            if f >= 2 { p2.pieces[fs].state_flags |= 1 << 4; }
                        }
                    }
                }
            }
        }
    }

    // Pilgrim resurrect (wiki Pilgrim): a charged Pilgrim move revives one
    // dead ally onto the Pilgrim's origin square (mv.from). The ally is
    // chosen by movegen (special_a = dead slot); the real game picks it
    // at random, and think averages the variants into the expected
    // value. The revived piece returns fresh: all counters and state
    // bits cleared, which includes the ElectroKnight charge reset the
    // wiki states explicitly (a resurrected EK returns with charge 0).
    // Pilgrim charge itself is NOT consumed (charged never clears, so
    // every charged move resurrects while dead allies remain). This
    // runs before the WA advance below, so a WA cannot steal the origin
    // square first; if the origin is already occupied by then (e.g. a
    // Djinn revived onto its home square, which happens to be the
    // origin), the resurrect is skipped rather than colliding.
    if mv.flags & MF_PILGRIM_RESURRECT != 0 {
        let origin_sq = mv.from;
        if slot_of(p2, origin_sq).is_none() {
            let slot = mv.special_a as usize;
            if slot < p2.pieces.len() {
                let dead = !p2.pieces[slot].alive && p2.pieces[slot].player == player;
                if dead {
                    p2.pieces[slot].alive = true;
                    p2.pieces[slot].x = sq_x(origin_sq);
                    p2.pieces[slot].y = sq_y(origin_sq);
                    p2.pieces[slot].quiet = 0;
                    p2.pieces[slot].odo = 0;
                    p2.pieces[slot].kills = 0;
                    p2.pieces[slot].caps = 0;
                    p2.pieces[slot].frost = 0;
                    p2.pieces[slot].encore = 0;
                    p2.pieces[slot].state_flags = 0;
                    p2.at[origin_sq as usize] = (slot + 1) as u16;
                    // Sandbox-verified (2026-11-03): the travel counter
                    // resets to 0/20 on the resurrecting move; the next
                    // resurrection needs 20 fresh squares of travel.
                    if let Some(ms) = slot_of(p2, mv.to) {
                        if p2.pieces[ms].key == PieceKey::Pilgrim
                            || p2.pieces[ms].sub_key == PieceKey::Pilgrim
                        {
                            p2.pieces[ms].odo = 0;
                            p2.pieces[ms].state_flags &= !(1 << 8);
                        }
                    }
                }
            }
        }
    }

    // War Automaton advance (wiki WarAutomaton page). After ANY death in
    // the action, every on-board WA advances one square forward, once.
    // Rules, all from the wiki: (a) the WA that made the capture itself
    // does NOT advance; (b) if two WAs would land on the same square,
    // the one belonging to the player who made the capture goes first
    // (the other then finds the square occupied and stays); (c) an
    // advance onto the final rank promotes immediately, but ONLY when
    // the triggering capture was made by a friendly piece (same side
    // as the WA) — an enemy-triggered advance does not promote;
    // (d) an advance onto an enemy King eliminates that King
    // (king_thrown, a win like Sumo's) — the advance never lands on
    // any other occupied square (blocked WAs stay). Check timing is
    // NOT handled here: legality is judged pre-advance by callers
    // using apply_full_no_wa (see above); this block always produces
    // the post-advance position for continuation.
    if deaths_now > 0 && !skip_wa {
        let mover_sq = if mv.flags & MF_DANCER_DOUBLE != 0 { mv.special_a } else { mv.to };
        let mut cands: Vec<usize> = Vec::new();
        for i in 0..p2.pieces.len() {
            if !p2.pieces[i].alive { continue; }
            if !xy_on_board(p2.pieces[i].x, p2.pieces[i].y) { continue; } // v130
            let k = p2.pieces[i].key; let sk = p2.pieces[i].sub_key;
            if k != PieceKey::WarAutomaton && sk != PieceKey::WarAutomaton { continue; }
            if sq(p2.pieces[i].x as i8, p2.pieces[i].y as i8) == mover_sq { continue; }
            cands.push(i);
        }
        // Capturer's side first (stable within each side, so index order
        // breaks any remaining same-square contest deterministically).
        cands.sort_by_key(|&i| if p2.pieces[i].player == player { 0 } else { 1 });
        for i in cands {
            if !p2.pieces[i].alive { continue; }
            if !xy_on_board(p2.pieces[i].x, p2.pieces[i].y) { continue; }
            let owner = p2.pieces[i].player;
            let a: i8 = if owner == 0 { 1 } else { -1 };
            let nx = p2.pieces[i].x as i8;
            let ny = p2.pieces[i].y as i8 + a;
            if !on_board(nx, ny) { continue; }
            let ahead = sq(nx, ny);
            match slot_of(p2, ahead) {
                None => {
                    let old_sq = sq(p2.pieces[i].x as i8, p2.pieces[i].y as i8);
                    p2.at[old_sq as usize] = 0;
                    p2.pieces[i].x = nx as u8;
                    p2.pieces[i].y = ny as u8;
                    p2.at[ahead as usize] = (i + 1) as u16;
                    // Promotion on a friendly-triggered advance only.
                    let final_rank: i8 = if owner == 0 { 7 } else { 0 };
                    if ny == final_rank && owner == player {
                        p2.pieces[i].key = PieceKey::Queen;
                        p2.pieces[i].sub_key = PieceKey::Queen;
                    }
                }
                Some(vs) => {
                    // Only an enemy King ahead is capturable by advance.
                    let v = &p2.pieces[vs];
                    if v.player != owner
                        && (v.key == PieceKey::King || v.sub_key == PieceKey::King)
                    {
                        let old_sq = sq(p2.pieces[i].x as i8, p2.pieces[i].y as i8);
                        kill_slot(p2, vs);
                        p2.at[old_sq as usize] = 0;
                        p2.pieces[i].x = nx as u8;
                        p2.pieces[i].y = ny as u8;
                        p2.at[ahead as usize] = (i + 1) as u16;
                        p2.king_thrown = Some(owner);
                    }
                    // Any other occupier blocks the advance: WA stays.
                }
            }
        }
    }

    // Gunslinger standoff timer (Sandbox-verified 2026-11-03). Runs last,
    // after every position mutation (including the WA advance above), so
    // threat_after is judged on the final position. The dissipate early
    // branch calls this separately (it returns before apply_triggers).
    update_gunslinger_armed(pos_pre, p2, player);
}

/// True if the gunslinger at `idx` shares a mutual diagonal threat with an
/// enemy piece: the first piece on one of its diagonals is an enemy, alive,
/// non-King, non-Iron, not AV-suppressed, not paralyzed, and attacks the
/// gunslinger's square back. Mirrors the candidate scan in movegen_special.
fn gunslinger_mutual_threat_at(pos: &Pos, idx: usize, occ: &[bool; 64]) -> bool {
    let p = &pos.pieces[idx];
    if !p.alive || !xy_on_board(p.x, p.y) { return false; }
    let x = p.x as i8; let y = p.y as i8;
    let here = sq(x, y);
    for &(dx, dy) in &DIRS_BISHOP {
        let mut nx = x + dx; let mut ny = y + dy;
        while on_board(nx, ny) {
            let t = sq(nx, ny);
            match slot_of(pos, t) {
                None => { nx += dx; ny += dy; }
                Some(vi) => {
                    let v = &pos.pieces[vi];
                    let v_iron = PROVISIONAL_IRON_INVULN
                        && (v.key == PieceKey::IronPawn || v.sub_key == PieceKey::IronPawn);
                    if v.alive
                        && v.player != p.player
                        && !is_king_target(v)
                        && !v_iron
                        && !av_suppressed(pos, vi)
                        && !is_paralyzed(pos, vi, occ)
                    {
                        let vatk = piece_attacks(pos, vi, occ, true, true);
                        if vatk.has(here) { return true; }
                    }
                    break;
                }
            }
        }
    }
    false
}

/// Sandbox standoff timer (2026-11-03): armed ⟺ the mutual threat survived
/// a full round. After each move:
///   armed_next = threat_after
///     && ((mover is the opponent && threat_before)
///         || (mover is the owner && armed_before && threat_before)).
/// Only the opponent's move can arm (threat intact before and after); the
/// owner's own move preserves an armed state but never creates one; any
/// break of the threat disarms. A gunslinger with no pre-move counterpart
/// (newly appeared) starts disarmed.
fn update_gunslinger_armed(pos_pre: &Pos, p2: &mut Pos, mover: u8) {
    let occ_pre = occ(pos_pre);
    let occ_post = occ(p2);
    for i in 0..p2.pieces.len() {
        if !p2.pieces[i].alive || !xy_on_board(p2.pieces[i].x, p2.pieces[i].y) { continue; }
        let k = p2.pieces[i].key; let sk = p2.pieces[i].sub_key;
        if k != PieceKey::Gunslinger && sk != PieceKey::Gunslinger { continue; }
        let owner = p2.pieces[i].player;
        let armed_before = p2.pieces[i].standoff_armed();
        let threat_after = gunslinger_mutual_threat_at(p2, i, &occ_post);
        let threat_before = pos_pre.pieces.iter().position(|q| q.id == p2.pieces[i].id)
            .map_or(false, |pi| gunslinger_mutual_threat_at(pos_pre, pi, &occ_pre));
        let armed_next = threat_after
            && ((mover != owner && threat_before)
                || (mover == owner && armed_before && threat_before));
        if armed_next { p2.pieces[i].state_flags |= 1 << 10; }
        else { p2.pieces[i].state_flags &= !(1 << 10); }
    }
}

#[cfg(test)]
mod side_to_move_tests {
    use super::apply_full;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, Move, PieceKey, MF_BISHOPPER_CROUCH, MF_KNIGHTMARE_OFF, MF_KNIGHTMARE_DROP, MF_DJINN_DISSIPATE, SQ_NONE};

    fn position_with_piece(key: PieceKey) -> Pos {
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.pieces.push(Piece {
            id: 1, key, sub_key: key, player: 0, x: 0, y: 0,
            alive: true, ..Default::default()
        });
        pos.at[sq(0, 0) as usize] = 1;
        pos
    }

    #[test]
    fn ordinary_child_flips_stm() {
        let pos = position_with_piece(PieceKey::Rook);
        let mv = Move { from: sq(0, 0), to: sq(0, 1), ..Default::default() };
        let child = apply_full(&pos, &mv, 0, 1).expect("ordinary move applies");
        assert_eq!(child.stm, 1);
    }

    #[test]
    fn early_return_special_child_flips_stm() {
        let pos = position_with_piece(PieceKey::Bishopper);
        let mv = Move { from: sq(0, 0), to: sq(0, 0), flags: MF_BISHOPPER_CROUCH, ..Default::default() };
        let child = apply_full(&pos, &mv, 0, 1).expect("special move applies");
        assert_eq!(child.stm, 1);
    }

    #[test]
    fn knightmare_off_early_return_flips_stm() {
        // Knightmare on b4 jumping off to the ring square (-1,2): the
        // off-move carries the 14x14 code in `to`, not SQ_NONE.
        use crate::types::off_code;
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.pieces.push(Piece {
            id: 1, key: PieceKey::Knightmare, sub_key: PieceKey::Knightmare, player: 0,
            x: 1, y: 3, alive: true, ..Default::default()
        });
        pos.at[sq(1, 3) as usize] = 1;
        let mv = Move { from: sq(1, 3), to: off_code(-1, 2), flags: MF_KNIGHTMARE_OFF, ..Default::default() };
        let child = apply_full(&pos, &mv, 0, 1).expect("Knightmare off applies");
        assert_eq!(child.stm, 1);
        assert_eq!(child.offboard.len(), 1);
        assert_eq!((child.offboard[0].ox, child.offboard[0].oy), (-1, 2));
    }

    #[test]
    fn knightmare_drop_early_return_flips_stm() {
        use crate::pos::OffboardPiece;
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.offboard.push(OffboardPiece {
            id: 7, key: PieceKey::Knightmare, sub_key: PieceKey::Knightmare,
            player: 0, off_since: 1, ox: -1, oy: 2,
        });
        let mv = Move {
            from: SQ_NONE, to: sq(1, 0), flags: MF_KNIGHTMARE_DROP,
            offboard_id: 7, ..Default::default()
        };
        let child = apply_full(&pos, &mv, 0, 1).expect("Knightmare drop applies");
        assert_eq!(child.stm, 1);
    }

    #[test]
    fn djinn_dissipate_early_return_flips_stm() {
        let pos = position_with_piece(PieceKey::Djinn);
        let mv = Move { from: sq(0, 0), to: sq(0, 0), flags: MF_DJINN_DISSIPATE, ..Default::default() };
        let child = apply_full(&pos, &mv, 0, 1).expect("Djinn dissipate applies");
        assert_eq!(child.stm, 1);
    }

    #[test]
    fn fish_moved_last_expires_on_non_fish_move() {
        // 2026-10-31 Fish fix: a non-Fish move by the owner must clear
        // friendly Fish movedLast, or search over-grants the bonus.
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.pieces.push(Piece {
            id: 1, key: PieceKey::Fish, sub_key: PieceKey::Fish, player: 0,
            x: 3, y: 3, alive: true, state_flags: 1 << 3, ..Default::default()
        });
        pos.pieces.push(Piece {
            id: 2, key: PieceKey::Pawn, sub_key: PieceKey::Pawn, player: 0,
            x: 4, y: 1, alive: true, ..Default::default()
        });
        pos.at[sq(3, 3) as usize] = 1;
        pos.at[sq(4, 1) as usize] = 2;
        let mv = Move { from: sq(4, 1), to: sq(4, 2), ..Default::default() };
        let child = apply_full(&pos, &mv, 0, 1).expect("pawn move applies");
        let fish = child.pieces.iter().find(|p| p.key == PieceKey::Fish).unwrap();
        assert_eq!(fish.state_flags & (1 << 3), 0, "non-Fish move must clear Fish movedLast");
    }

    #[test]
    fn fish_moved_last_set_on_fish_move() {
        // The set path must keep working: a Fish move sets its own bit.
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.pieces.push(Piece {
            id: 1, key: PieceKey::Fish, sub_key: PieceKey::Fish, player: 0,
            x: 3, y: 3, alive: true, ..Default::default()
        });
        pos.at[sq(3, 3) as usize] = 1;
        let mv = Move { from: sq(3, 3), to: sq(4, 5), ..Default::default() };
        let child = apply_full(&pos, &mv, 0, 1).expect("fish move applies");
        let fish = child.pieces.iter().find(|p| p.key == PieceKey::Fish).unwrap();
        assert_ne!(fish.state_flags & (1 << 3), 0, "Fish move must set its movedLast");
    }

    #[test]
    fn shrike_first_move_kills_intermediate_piece() {
        use crate::types::MF_SHRIKE_FIRST;
        // Shrike 2-step first move must kill the enemy on the intermediate
        // square (its path), not just the destination.
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.pieces.push(Piece {
            id: 1, key: PieceKey::Shrike, sub_key: PieceKey::Shrike, player: 0,
            x: 3, y: 1, alive: true, ..Default::default()
        });
        pos.pieces.push(Piece {
            id: 2, key: PieceKey::Pawn, sub_key: PieceKey::Pawn, player: 1,
            x: 3, y: 2, alive: true, ..Default::default()
        });
        pos.at[sq(3, 1) as usize] = 1;
        pos.at[sq(3, 2) as usize] = 2;
        let mv = Move {
            from: sq(3, 1), to: sq(3, 3), flags: MF_SHRIKE_FIRST, ..Default::default()
        };
        let child = apply_full(&pos, &mv, 0, 1).expect("shrike move applies");
        let victim = child.pieces.iter().find(|p| p.id == 2).unwrap();
        assert!(!victim.alive, "intermediate enemy must die on a Shrike 2-step");
        let shrike = child.pieces.iter().find(|p| p.id == 1).unwrap();
        assert_eq!((shrike.x, shrike.y), (3, 3));
    }
}

#[cfg(test)]
mod wiki_ek_tests {
    use super::apply_full;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, Move, PieceKey, MF_CAPTURE, MF_ELECTRO};

    // (key, player, x, y, flags, quiet)
    fn build_pos(spec: &[(PieceKey, u8, u8, u8, u16, u8)]) -> Pos {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        for (i, &(key, player, x, y, flags, quiet)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, state_flags: flags, quiet, ..Default::default()
            });
            pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        pos
    }

    fn ek_electro_move() -> Move {
        // EK b1 -> victim c3, lightning target d3.
        Move {
            from: sq(1, 0), to: sq(2, 2),
            flags: MF_CAPTURE | MF_ELECTRO, special_a: sq(3, 2),
            ..Default::default()
        }
    }

    #[test]
    fn electro_kills_target_without_av() {
        let pos = build_pos(&[
            (PieceKey::ElectroKnight, 0, 1, 0, 0, 3),
            (PieceKey::Pawn, 1, 2, 2, 0, 0),
            (PieceKey::Pawn, 1, 3, 2, 0, 0),
            (PieceKey::King, 0, 0, 7, 0, 0),
            (PieceKey::King, 1, 7, 7, 0, 0),
        ]);
        let child = apply_full(&pos, &ek_electro_move(), 0, 1).expect("ek move applies");
        assert_eq!(child.at[sq(3, 2) as usize], 0, "lightning target must die without AV");
        let ek = child.pieces.iter().find(|p| p.key == PieceKey::ElectroKnight).unwrap();
        assert_eq!((ek.x, ek.y), (2, 2));
        assert_eq!(ek.quiet, 0, "charge is consumed by the electro capture");
    }

    #[test]
    fn av_adjacent_but_not_first_candidate_does_not_block_zap() {
        // Sandbox-corrected (2026-11-03): the wiki "AV blocks lightning"
        // rule is refuted for an AV that IS the zap candidate (it dies).
        // This position has the AV on b3 adjacent to the landing square
        // c3 but NOT first in clockwise scan order (pawn d3 is E, AV is W),
        // so the zap victim is d3. Whether a real AV shields a NEIGHBOUR
        // was never Sandbox-tested and is UNVERIFIED; the zap-first model
        // below (zap fires despite an adjacent AV, as proven for the
        // candidate case) is the parsimonious reading, not a proven fact.
        // If a play-test later shows neighbour shielding, restore the block.
        let pos = build_pos(&[
            (PieceKey::ElectroKnight, 0, 1, 0, 0, 3),
            (PieceKey::Pawn, 1, 2, 2, 0, 0),
            (PieceKey::Pawn, 1, 3, 2, 0, 0),
            (PieceKey::AntiViolence, 1, 1, 2, 0, 0),
            (PieceKey::King, 0, 0, 7, 0, 0),
            (PieceKey::King, 1, 7, 7, 0, 0),
        ]);
        let child = apply_full(&pos, &ek_electro_move(), 0, 1).expect("ek move applies");
        assert_eq!(child.at[sq(3, 2) as usize], 0, "d3 zap victim must die");
        assert_ne!(child.at[sq(1, 2) as usize], 0, "non-candidate AV survives the zap");
        let ek = child.pieces.iter().find(|p| p.key == PieceKey::ElectroKnight).unwrap();
        assert_eq!((ek.x, ek.y), (2, 2), "the capture itself still happens");
        assert_eq!(ek.quiet, 0, "charge is still consumed");
    }

    #[test]
    fn iron_lightning_target_survives() {
        let pos = build_pos(&[
            (PieceKey::ElectroKnight, 0, 1, 0, 0, 3),
            (PieceKey::Pawn, 1, 2, 2, 0, 0),
            (PieceKey::IronPawn, 1, 3, 2, 0, 0),
            (PieceKey::King, 0, 0, 7, 0, 0),
            (PieceKey::King, 1, 7, 7, 0, 0),
        ]);
        let child = apply_full(&pos, &ek_electro_move(), 0, 1).expect("ek move applies");
        assert_ne!(child.at[sq(3, 2) as usize], 0, "Iron must survive lightning");
    }
}

#[cfg(test)]
mod wiki_sumo_tests {
    use super::apply_full;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, Move, PieceKey, MF_IS_PUSH};

    fn build_pos(spec: &[(PieceKey, u8, u8, u8)]) -> Pos {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        for (i, &(key, player, x, y)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, ..Default::default()
            });
            pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        pos
    }

    fn push_move(from: (i8, i8), to: (i8, i8), dir: (i8, i8)) -> Move {
        Move {
            from: sq(from.0, from.1), to: sq(to.0, to.1),
            flags: MF_IS_PUSH, push_dx: dir.0, push_dy: dir.1,
            ..Default::default()
        }
    }

    #[test]
    fn one_piece_goes_two_squares_wiki_example() {
        // Wiki example: d4 -> f4. Sumo b4 -> c4, victim d4 pushed to f4.
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 1, 3),
            (PieceKey::Pawn, 1, 3, 3),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &push_move((1, 3), (2, 3), (1, 0)), 0, 1)
            .expect("push applies");
        let sumo = child.pieces.iter().find(|p| p.key == PieceKey::SumoRook).unwrap();
        assert_eq!((sumo.x, sumo.y), (2, 3), "Sumo stops before the target");
        let vic = child.pieces.iter().find(|p| p.key == PieceKey::Pawn).unwrap();
        assert_eq!((vic.x, vic.y), (5, 3), "single piece goes two squares");
        assert!(vic.alive);
    }

    #[test]
    fn two_pieces_go_one_square_each() {
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 1, 3),
            (PieceKey::Pawn, 1, 3, 3),
            (PieceKey::Pawn, 1, 4, 3),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &push_move((1, 3), (2, 3), (1, 0)), 0, 1)
            .expect("push applies");
        let pawns: Vec<_> = child.pieces.iter().filter(|p| p.key == PieceKey::Pawn).collect();
        let mut spots: Vec<(u8, u8)> = pawns.iter().map(|p| (p.x, p.y)).collect();
        spots.sort();
        assert_eq!(spots, vec![(4, 3), (5, 3)], "both pieces move one square");
    }

    #[test]
    fn three_pieces_hold_ground_push_rejected() {
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 1, 3),
            (PieceKey::Pawn, 1, 3, 3),
            (PieceKey::Pawn, 1, 4, 3),
            (PieceKey::Pawn, 1, 5, 3),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        assert!(apply_full(&pos, &push_move((1, 3), (2, 3), (1, 0)), 0, 1).is_none(),
            "3-piece push must be rejected");
    }

    #[test]
    fn friendly_victim_pushes_own_king_rejected() {
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 1, 3),
            (PieceKey::Pawn, 0, 3, 3),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &push_move((1, 3), (2, 3), (1, 0)), 0, 1)
            .expect("friendly push applies");
        let vic = child.pieces.iter().find(|p| p.key == PieceKey::Pawn).unwrap();
        assert_eq!((vic.x, vic.y), (5, 3));
        let pos2 = build_pos(&[
            (PieceKey::SumoRook, 0, 1, 3),
            (PieceKey::King, 0, 3, 3),
            (PieceKey::King, 1, 7, 7),
        ]);
        assert!(apply_full(&pos2, &push_move((1, 3), (2, 3), (1, 0)), 0, 1).is_none(),
            "own King must never be pushed");
    }

    #[test]
    fn pushed_pawn_onto_final_rank_does_not_promote() {
        // Pawn f6 (3,5) pushed north two squares onto f8 (3,7): stays a Pawn.
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 3, 3),
            (PieceKey::Pawn, 1, 3, 5),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &push_move((3, 3), (3, 4), (0, 1)), 0, 1)
            .expect("push applies");
        let vic = child.pieces.iter().find(|p| p.key == PieceKey::Pawn).unwrap();
        assert_eq!((vic.x, vic.y), (3, 7));
        assert_eq!(vic.key, PieceKey::Pawn, "pushed pawn must not promote");
        assert!(vic.alive);
    }

    #[test]
    fn iron_and_knightmare_pushed_off_board_die() {
        for key in [PieceKey::IronPawn, PieceKey::Knightmare] {
            let pos = build_pos(&[
                (PieceKey::SumoRook, 0, 4, 3),
                (key, 1, 6, 3),
                (PieceKey::King, 0, 0, 7),
                (PieceKey::King, 1, 7, 7),
            ]);
            let child = apply_full(&pos, &push_move((4, 3), (5, 3), (1, 0)), 0, 1)
                .expect("push applies");
            let vic = child.pieces.iter().find(|p| p.key == key).unwrap();
            assert!(!vic.alive, "{key:?} pushed off-board must die");
            assert!(child.offboard.is_empty(), "Knightmare pushed off must be captured, not offboarded");
        }
    }

    #[test]
    fn enemy_king_pushed_but_not_thrown() {
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 1, 3),
            (PieceKey::King, 1, 3, 3),
            (PieceKey::King, 0, 0, 7),
        ]);
        let child = apply_full(&pos, &push_move((1, 3), (2, 3), (1, 0)), 0, 1)
            .expect("enemy King push applies");
        let ek = child.pieces.iter().find(|p| p.key == PieceKey::King && p.player == 1).unwrap();
        assert_eq!((ek.x, ek.y), (5, 3));
        assert!(ek.alive);
        assert_eq!(child.king_thrown, None, "King moved, not thrown");
    }
}

#[cfg(test)]
mod wiki_wa_tests {
    use super::apply_full;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, Move, PieceKey, MF_CAPTURE};
    use crate::veto::king_move_safe;

    fn build_pos(spec: &[(PieceKey, u8, u8, u8)]) -> Pos {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        for (i, &(key, player, x, y)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, ..Default::default()
            });
            pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        pos
    }

    fn cap_move(from: (i8, i8), to: (i8, i8)) -> Move {
        Move { from: sq(from.0, from.1), to: sq(to.0, to.1), flags: MF_CAPTURE, ..Default::default() }
    }

    fn wa_at(pos: &Pos, player: u8, x: u8, y: u8) -> bool {
        pos.pieces.iter().any(|p| p.alive
            && (p.key == PieceKey::WarAutomaton || p.sub_key == PieceKey::WarAutomaton)
            && p.player == player && p.x == x && p.y == y)
    }

    #[test]
    fn capturer_wa_does_not_advance_others_do() {
        let pos = build_pos(&[
            (PieceKey::WarAutomaton, 0, 3, 3),
            (PieceKey::Pawn, 1, 4, 4),
            (PieceKey::WarAutomaton, 0, 1, 1),
            (PieceKey::WarAutomaton, 1, 6, 6),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &cap_move((3, 3), (4, 4)), 0, 1).expect("wa capture applies");
        assert!(wa_at(&child, 0, 4, 4), "capturing WA stays on its capture square");
        assert!(!wa_at(&child, 0, 4, 5), "capturing WA must not advance further");
        assert!(wa_at(&child, 0, 1, 2), "other friendly WA advances");
        assert!(wa_at(&child, 1, 6, 5), "enemy WA advances too (any death triggers all)");
    }

    #[test]
    fn two_was_same_square_capturer_side_wins() {
        let pos = build_pos(&[
            (PieceKey::WarAutomaton, 0, 3, 3),
            (PieceKey::WarAutomaton, 1, 3, 5),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 1, 1, 2),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &cap_move((0, 1), (1, 2)), 0, 1).expect("pawn capture applies");
        assert!(wa_at(&child, 0, 3, 4), "capturer-side WA claims the square");
        assert!(wa_at(&child, 1, 3, 5), "other-side WA stays (square taken)");
    }

    #[test]
    fn friendly_trigger_promotes_enemy_trigger_does_not() {
        // Friendly: White WA (3,6) -> (3,7) on a White capture -> Queen.
        let pos = build_pos(&[
            (PieceKey::WarAutomaton, 0, 3, 6),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 1, 1, 2),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child = apply_full(&pos, &cap_move((0, 1), (1, 2)), 0, 1).expect("capture applies");
        let wa = child.pieces.iter().find(|p| p.x == 3 && p.y == 7).unwrap();
        assert_eq!(wa.key, PieceKey::Queen, "friendly-triggered advance promotes");
        assert_eq!(wa.sub_key, PieceKey::Queen);
        // Enemy: Black WA (3,1) -> (3,0) on a White capture -> stays WA.
        let pos2 = build_pos(&[
            (PieceKey::WarAutomaton, 1, 3, 1),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 1, 1, 2),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        let child2 = apply_full(&pos2, &cap_move((0, 1), (1, 2)), 0, 1).expect("capture applies");
        let wa2 = child2.pieces.iter().find(|p| p.x == 3 && p.y == 0).unwrap();
        assert_eq!(wa2.key, PieceKey::WarAutomaton, "enemy-triggered advance must not promote");
    }

    #[test]
    fn wa_advance_onto_enemy_king_eliminates_it() {
        let pos = build_pos(&[
            (PieceKey::WarAutomaton, 0, 3, 3),
            (PieceKey::King, 1, 3, 4),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 1, 1, 2),
            (PieceKey::King, 0, 0, 7),
        ]);
        let child = apply_full(&pos, &cap_move((0, 1), (1, 2)), 0, 1).expect("capture applies");
        assert!(wa_at(&child, 0, 3, 4), "WA moves onto the King square");
        let bk = child.pieces.iter().find(|p| p.key == PieceKey::King && p.player == 1).unwrap();
        assert!(!bk.alive, "King eliminated by WA advance");
        assert_eq!(child.king_thrown, Some(0));
    }

    #[test]
    fn king_safety_judged_pre_advance() {
        // Discovered: WA (5,4) blocks Rook (7,4) -> King (4,4). A capture
        // elsewhere advances WA to (5,5), exposing the King post-advance,
        // but the capture itself is legal (pre-advance safe).
        let pos = build_pos(&[
            (PieceKey::King, 0, 4, 4),
            (PieceKey::Rook, 1, 7, 4),
            (PieceKey::WarAutomaton, 0, 5, 4),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 1, 1, 2),
            (PieceKey::King, 1, 7, 7),
        ]);
        assert!(king_move_safe(&pos, &cap_move((0, 1), (1, 2)), 0, 1),
            "capture exposing King only via WA advance must be legal");
        // Block: King in check, WA (5,3) would advance to (5,4) and block,
        // but the capture does not resolve the check by itself -> illegal.
        let pos2 = build_pos(&[
            (PieceKey::King, 0, 4, 4),
            (PieceKey::Rook, 1, 7, 4),
            (PieceKey::WarAutomaton, 0, 5, 3),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 1, 1, 2),
            (PieceKey::King, 1, 7, 7),
        ]);
        assert!(!king_move_safe(&pos2, &cap_move((0, 1), (1, 2)), 0, 1),
            "WA block must not count as resolving check");
    }
}

#[cfg(test)]
mod wiki_pilgrim_tests {
    use super::apply_full;
    use crate::pos::{LimboPiece, Piece, Pos};
    use crate::types::{sq, Move, PieceKey, MF_CAPTURE, MF_PILGRIM_RESURRECT};

    fn build_pos(spec: &[(PieceKey, u8, u8, u8)]) -> Pos {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        for (i, &(key, player, x, y)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, ..Default::default()
            });
            pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        pos
    }

    #[test]
    fn resurrect_places_dead_ally_on_origin_fresh() {
        let mut pos = build_pos(&[
            (PieceKey::Pilgrim, 0, 3, 3),
            (PieceKey::Pawn, 0, 5, 5),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        pos.pieces[1].alive = false;
        pos.at[sq(5, 5) as usize] = 0;
        let mv = Move {
            from: sq(3, 3), to: sq(5, 5),
            flags: MF_PILGRIM_RESURRECT, special_a: 1,
            ..Default::default()
        };
        let child = apply_full(&pos, &mv, 0, 1).expect("pilgrim move applies");
        let pg = child.pieces.iter().find(|p| p.key == PieceKey::Pilgrim).unwrap();
        assert_eq!((pg.x, pg.y), (5, 5));
        let rev = &child.pieces[1];
        assert!(rev.alive, "dead ally must revive");
        assert_eq!((rev.x, rev.y), (3, 3), "revived piece appears on Pilgrim origin");
        assert_eq!(rev.state_flags, 0);
        assert_eq!(rev.quiet, 0);
        assert_eq!(rev.odo, 0);
    }

    #[test]
    fn resurrected_electroknight_returns_uncharged() {
        let mut pos = build_pos(&[
            (PieceKey::Pilgrim, 0, 3, 3),
            (PieceKey::ElectroKnight, 0, 5, 5),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        pos.pieces[1].alive = false;
        pos.pieces[1].quiet = 3;
        pos.pieces[1].state_flags = 1 << 1;
        pos.at[sq(5, 5) as usize] = 0;
        let mv = Move {
            from: sq(3, 3), to: sq(5, 5),
            flags: MF_PILGRIM_RESURRECT, special_a: 1,
            ..Default::default()
        };
        let child = apply_full(&pos, &mv, 0, 1).expect("pilgrim move applies");
        let rev = &child.pieces[1];
        assert!(rev.alive);
        assert_eq!(rev.quiet, 0, "resurrected EK returns with charge 0");
        assert_eq!(rev.state_flags & (1 << 1), 0, "echarged bit cleared");
    }

    #[test]
    fn djinn_revive_wins_origin_over_resurrect() {
        // Djinn home == Pilgrim origin (3,3). Pilgrim captures (death
        // triggers Djinn revive onto home first), so the resurrect finds
        // the origin occupied and skips instead of colliding.
        let mut pos = build_pos(&[
            (PieceKey::Pilgrim, 0, 3, 3),
            (PieceKey::Pawn, 1, 5, 5),
            (PieceKey::Pawn, 0, 2, 2),
            (PieceKey::Djinn, 0, 3, 3),
            (PieceKey::King, 0, 0, 7),
            (PieceKey::King, 1, 7, 7),
        ]);
        pos.pieces[2].alive = false;
        pos.at[sq(2, 2) as usize] = 0;
        pos.pieces[3].alive = false;
        pos.pieces[3].home_x = 3;
        pos.pieces[3].home_y = 3;
        pos.at[sq(3, 3) as usize] = 1; // Pilgrim occupies origin, not Djinn
        pos.limbo.push(LimboPiece {
            id: 4, key: PieceKey::Djinn, sub_key: PieceKey::Djinn,
            player: 0, home_x: 3, home_y: 3,
        });
        let mv = Move {
            from: sq(3, 3), to: sq(5, 5),
            flags: MF_CAPTURE | MF_PILGRIM_RESURRECT, special_a: 2,
            ..Default::default()
        };
        let child = apply_full(&pos, &mv, 0, 1).expect("pilgrim capture applies");
        // Djinn revive pushes a NEW alive entry (the old dead entry stays
        // dead); find the revived one rather than indexing the old slot.
        let dj = child.pieces.iter().find(|p| p.alive
            && p.key == PieceKey::Djinn && p.player == 0).expect("Djinn must revive");
        assert_eq!((dj.x, dj.y), (3, 3), "Djinn revives onto home/origin");
        assert!(!child.pieces[2].alive, "resurrect must skip an occupied origin");
    }
}
