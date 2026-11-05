use crate::pos::*;
use crate::types::*;
use crate::movegen::*;
use crate::attacks::*;

pub fn gen_special_moves(pos: &Pos, p_idx: usize, player: u8, opp: u8,
                         occ_local: &[bool; 64], out: &mut Vec<Move>) {
    let p = &pos.pieces[p_idx];
    let x = p.x as i8; let y = p.y as i8;
    use PieceKey::*;
    use crate::eval::piece_value;
    let (k, s) = resolve_piece_keys(pos, p);
    let is = |t: PieceKey| k == t || s == t;

    // Gunslinger: bishop+mutual_destroy_standoff
    // Sandbox-verified (2026-11-03): the standoff is NOT offered on a fresh
    // mutual threat. It arms only after the mutual threat survives a full
    // round (both sides move once with the threat intact); breaking the
    // threat disarms it. The armed bit is maintained by apply_triggers.
    // When armed, the action destroys ONLY the enemy; the Gunslinger
    // survives (see apply_triggers).
    if is(Gunslinger) && !av_suppressed(pos, p_idx) && p.standoff_armed() {
        let here = sq(x, y);
        let mut candidates: Vec<(usize, i32)> = Vec::new();
        for &(dx, dy) in &DIRS_BISHOP {
            let mut nx = x + dx; let mut ny = y + dy;
            while on_board(nx, ny) {
                let t = sq(nx, ny);
                if let Some(vi) = slot_of(pos, t) {
                    let v = &pos.pieces[vi];
                    let v_iron = PROVISIONAL_IRON_INVULN && (v.key == IronPawn || v.sub_key == IronPawn);
                    if v.alive
                        && v.player == opp
                        && !is_king_target(v)
                        && !v_iron
                        && !av_suppressed(pos, vi)
                        && !is_paralyzed(pos, vi, occ_local)
                    {
                        let vatk = piece_attacks(pos, vi, occ_local, true, true);
                        if vatk.has(here) {
                            candidates.push((vi, piece_value(v)));
                        }
                    }
                    break;
                }
                nx += dx; ny += dy;
            }
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        for (vi, _) in candidates {
            let v = &pos.pieces[vi];
            let mut mv = Move { from: here, to: here, ..Default::default() };
            mv.flags = MF_GUN_STANDOFF;
            mv.special_a = sq(v.x as i8, v.y as i8);
            out.push(mv);
        }
    }

    // 1. Rocketman: king+random_teleport_once
    // Normal 1-step King moves (quiet & capture) are generated in movegen.rs.
    // Once per game (!p.used()), Rocketman can teleport (blast off) to any empty square.
    if is(Rocketman) && !p.used() {
        let here = sq(x, y);
        for t in 0..64u8 {
            if t != here && pos.at[t as usize] == 0 {
                out.push(Move {
                    from: here,
                    to: t,
                    flags: MF_ROCKET_BLAST,
                    ..Default::default()
                });
            }
        }
    }

    // HordeMother: bishop+spawn_hordeling_on_capture
    if is(HordeMother) && !av_suppressed(pos, p_idx) {
        let atk = piece_attacks(pos, p_idx, occ_local, true, false);
        let mut bb = atk.0;
        while bb != 0 {
            let t = bb.trailing_zeros() as u8; bb &= bb - 1;
            let vi = pos.at[t as usize];
            if vi == 0 { continue; }
            let v = &pos.pieces[(vi - 1) as usize];
            if !v.alive || v.player != opp || is_king_target(v) { continue; }
            if PROVISIONAL_IRON_INVULN && (v.key == IronPawn || v.sub_key == IronPawn) { continue; }
            // Sandbox-verified (2026-11-03): the spawn choice is ANY free
            // square on ranks 2-7 (y 1..=6); ranks 1/8 and occupied squares
            // are excluded. Squares freed by this very move (the mother's
            // origin, vacated by the capture, and the victim's square)
            // count as free. One variant per eligible square; think ranks
            // them individually (same_move includes special_a, so no
            // dedup collapses them) and the best square wins.
            let mut spawned = false;
            for hy in 1i8..=6 {
                for hx in 0i8..=7 {
                    let hsq = sq(hx, hy);
                    let occ = pos.at[hsq as usize];
                    if occ != 0 && hsq != sq(x, y) && hsq != t { continue; }
                    let mut mv = Move { from: sq(x, y), to: t, ..Default::default() };
                    mv.flags = MF_CAPTURE | MF_HORDE_SPAWN;
                    mv.special_a = hsq;
                    out.push(mv);
                    spawned = true;
                }
            }
            if !spawned {
                out.push(Move { from: sq(x, y), to: t, flags: MF_CAPTURE, ..Default::default() });
            }
        }
    }

    // 3. Fish: knight+king_step_after_move
    // If Fish moved on the previous turn, it gains a 1-step King move (including captures).
    if is(Fish) && p.moved_last() {
        let here = sq(x, y);
        let suppressed = av_suppressed(pos, p_idx);
        for &(dx, dy) in &DIRS_KING {
            let nx = x + dx; let ny = y + dy;
            if !on_board(nx, ny) { continue; }
            let t = sq(nx, ny);
            let v_idx = pos.at[t as usize];
            if v_idx == 0 {
                out.push(Move { from: here, to: t, ..Default::default() });
            } else {
                let v = &pos.pieces[(v_idx - 1) as usize];
                let iron = v.key == IronPawn || v.sub_key == IronPawn;
                let v_king = is_king_target(v);
                if v.alive
                    && v.player == opp
                    && !v_king
                    && !(PROVISIONAL_IRON_INVULN && iron)
                    && !suppressed
                {
                    out.push(Move {
                        from: here,
                        to: t,
                        flags: MF_CAPTURE,
                        ..Default::default()
                    });
                }
            }
        }
    }

    // ElectroKnight: knight+charge_aoe_capture
    // Sandbox-verified (2026-11-03): the zap is DETERMINISTIC, not random.
    // A charged capture zaps exactly ONE additional enemy: the first
    // destructible piece scanning the LANDING square's 8 neighbours
    // clockwise from north (N, NE, E, SE, S, SW, W, NW). No player choice.
    // Iron pieces are immune and SKIPPED (not absorbing); kings are never
    // victims. An AntiViolence adjacent to the landing square is itself a
    // legal victim and is destroyed, it does not block the lightning
    // (whether AV shields a neighbouring victim is UNVERIFIED and is NOT
    // modelled). Exactly one EK move is emitted per capture target.
    if is(ElectroKnight) && (p.echarged() || p.quiet >= 3) && !av_suppressed(pos, p_idx) {
        let atk = piece_attacks(pos, p_idx, occ_local, true, false);
        let mut bb = atk.0;
        while bb != 0 {
            let t = bb.trailing_zeros() as u8; bb &= bb - 1;
            let vi = pos.at[t as usize];
            if vi == 0 { continue; }
            let v = &pos.pieces[(vi - 1) as usize];
            let v_iron = PROVISIONAL_IRON_INVULN && (v.key == IronPawn || v.sub_key == IronPawn);
            if !v.alive || v.player != opp || is_king_target(v) || v_iron { continue; }
            let tx = sq_x(t) as i8; let ty = sq_y(t) as i8;
            // Clockwise from north; first destructible neighbour wins.
            for &(ax, ay) in &[(0i8, 1i8), (1, 1), (1, 0), (1, -1),
                               (0, -1), (-1, -1), (-1, 0), (-1, 1)] {
                let ex = tx + ax; let ey = ty + ay;
                if !on_board(ex, ey) { continue; }
                if let Some(ei) = slot_of(pos, sq(ex, ey)) {
                    let e = &pos.pieces[ei];
                    let e_iron = PROVISIONAL_IRON_INVULN && (e.key == IronPawn || e.sub_key == IronPawn);
                    if e.alive && e.player == opp && !is_king_target(e) && !e_iron {
                        let mut mv = Move { from: sq(x, y), to: t, ..Default::default() };
                        mv.flags = MF_CAPTURE | MF_ELECTRO;
                        mv.special_a = sq(ex, ey);
                        out.push(mv);
                        break;
                    }
                }
            }
        }
    }

    // 2. Recruit: pawn+inherit_captured_piece_moves
    // When Recruit captures a piece, it gains that piece's movement abilities.
    if is(Recruit) {
        let here = sq(x, y);
        let last_rank: i8 = if player == 0 { 7 } else { 0 };
        let is_iron = k == IronPawn || s == IronPawn;
        let inh = recruit_inherited_key(pos, p);
        let inh_non_cap = is_non_capturer(inh) || inh == BladeRunner;
        let suppressed = av_suppressed(pos, p_idx);
        let inh_bb = recruit_inherited_targets(pos, p_idx, occ_local);
        let mut mask = inh_bb.0;
        while mask != 0 {
            let t = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            let ty = sq_y(t) as i8;
            let v_idx = pos.at[t as usize];
            let is_cap = if v_idx == 0 {
                if inh == PawnWithKnife { continue; }
                false
            } else {
                if inh_non_cap || inh == HordeMother || inh == SumoRook { continue; }
                let v = &pos.pieces[(v_idx - 1) as usize];
                let iron = v.key == IronPawn || v.sub_key == IronPawn;
                let v_king = is_king_target(v);
                if !v.alive
                    || v.player != opp
                    || v_king
                    || (PROVISIONAL_IRON_INVULN && iron)
                    || suppressed
                {
                    continue;
                }
                true
            };

            let mut mv = Move {
                from: here,
                to: t,
                flags: if is_cap { MF_CAPTURE } else { 0 },
                ..Default::default()
            };

            let can_promo = ty == last_rank
                && !is_iron
                && !blocked_by_enemy_aristocrat(pos, opp, if is_cap { Some(t) } else { Option::None });
            if can_promo {
                mv.flags |= MF_PROMO;
                for &pr_k in &[Queen, Rook, Bishop, Knight] {
                    mv.promo = pr_k as u8;
                    if !out.iter().any(|m| {
                        m.from == here && m.to == t && m.promo == mv.promo && m.flags == mv.flags
                    }) {
                        out.push(mv);
                    }
                }
            } else if !out.iter().any(|m| {
                m.from == here
                    && m.to == t
                    && m.promo == 0
                    && (m.flags & MF_CAPTURE) == (mv.flags & MF_CAPTURE)
            }) {
                out.push(mv);
            }
        }
    }

    // 4. Dancer: bishop+double_move_after_check
    // Triggers only after giving check (p.encore_pending()).
    // Per the rule the double move cannot capture on either leg: only quiet
    // legs are generated (a capturing double would be an illegal move).
    if is(Dancer) && p.encore_pending() {
        let origin = sq(x, y);
        let suppressed_origin = av_suppressed(pos, p_idx);
        let mut leg1: Vec<(i8, i8, bool)> = Vec::new();
        for &(dx, dy) in &DIRS_BISHOP {
            let mut nx = x + dx; let mut ny = y + dy;
            while on_board(nx, ny) {
                let t = sq(nx, ny);
                let v_idx = pos.at[t as usize];
                if v_idx == 0 {
                    leg1.push((nx, ny, false));
                } else {
                    // Dancer double cannot capture: a piece blocks leg 1
                    // without being captured.
                    break;
                }
                nx += dx; ny += dy;
            }
        }
        for &(l1x, l1y, leg1_cap) in &leg1 {
            let l1_sq = sq(l1x, l1y);
            // Check if the intermediate square l1_sq is adjacent to an uncaptured enemy AntiViolence.
            let mut suppressed_leg2 = false;
            for adx in -1i8..=1 {
                for ady in -1i8..=1 {
                    if adx == 0 && ady == 0 { continue; }
                    let ax = l1x + adx; let ay = l1y + ady;
                    if !on_board(ax, ay) { continue; }
                    let asq = sq(ax, ay);
                    if asq == l1_sq { continue; }
                    if let Some(qi) = slot_of(pos, asq) {
                        let q = &pos.pieces[qi];
                        if q.alive && q.player == opp && (q.key == AntiViolence || q.sub_key == AntiViolence) {
                            suppressed_leg2 = true;
                        }
                    }
                }
            }
            for &(dx, dy) in &DIRS_BISHOP {
                let mut nx = l1x + dx; let mut ny = l1y + dy;
                while on_board(nx, ny) {
                    let t = sq(nx, ny);
                    if t == origin {
                        // Dancer vacated origin on leg 1: cannot stop on origin, but can slide past it.
                        nx += dx; ny += dy;
                        continue;
                    }
                    let v_idx = pos.at[t as usize];
                    if v_idx == 0 {
                        let mut mv = Move { from: origin, to: l1_sq, ..Default::default() };
                        mv.flags = MF_DANCER_DOUBLE | if leg1_cap { MF_CAPTURE } else { 0 };
                        mv.special_a = t;
                        out.push(mv);
                    } else {
                        // Dancer double cannot capture: a piece blocks leg 2.
                        break;
                    }
                    nx += dx; ny += dy;
                }
            }
        }
    }

    if is(Djinn) {
        out.push(Move { from: sq(x, y), to: sq(x, y),
                        flags: MF_DJINN_DISSIPATE, ..Default::default() });
    }
    if is(Knightmare) {
        // Sandbox-verified (2026-11-03): jumping off the board is a normal
        // knight jump whose landing square lies in the 14x14 staging ring
        // (outside the 8x8, inside -3..=10). One off-move per such jump;
        // the landing square is carried as a 14x14 code in `to`. Landings
        // occupied by another offboard piece are excluded (untested in
        // the Sandbox; two pieces sharing an offboard square is not a
        // position the game can display, so it is treated as illegal).
        let here = sq(x, y);
        for &(dx, dy) in &LEAPS_KNIGHT {
            let nx = x + dx; let ny = y + dy;
            if !off_in_ring(nx, ny) { continue; }
            if pos.offboard.iter().any(|o| o.ox == nx && o.oy == ny) { continue; }
            out.push(Move { from: here, to: off_code(nx, ny),
                            flags: MF_KNIGHTMARE_OFF, ..Default::default() });
        }
    }

    // 5. Bishopper: bishop+jump_pieces
    // Pre-jump normal bishop moves are generated in movegen.rs; post-jump moves use MF_BISHOPPER_LEAP.
    if is(Bishopper) || is(BisHop) {
        let here = sq(x, y);
        if !p.crouched() {
            out.push(Move { from: here, to: here,
                            flags: MF_BISHOPPER_CROUCH, ..Default::default() });
        } else {
            // Crouched: Sandbox-verified (2026-11-03) the leap goes ONLY to
            // the square immediately beyond the screen (empty = quiet leap,
            // enemy = capture leap). No double leap: after offering that
            // one square, stop along the diagonal.
            let suppressed = av_suppressed(pos, p_idx);
            for &(dx, dy) in &DIRS_BISHOP {
                let mut nx = x + dx; let mut ny = y + dy;
                let mut jumped = false;
                while on_board(nx, ny) {
                    let t = sq(nx, ny);
                    let v_idx = pos.at[t as usize];
                    if !jumped {
                        if v_idx != 0 {
                            jumped = true;
                        }
                    } else {
                        if v_idx != 0 {
                            let v = &pos.pieces[(v_idx - 1) as usize];
                            let iron = v.key == IronPawn || v.sub_key == IronPawn;
                            let v_king = is_king_target(v);
                            if v.alive
                                && v.player == opp
                                && !v_king
                                && !(PROVISIONAL_IRON_INVULN && iron)
                                && !suppressed
                            {
                                out.push(Move {
                                    from: here,
                                    to: t,
                                    flags: MF_CAPTURE | MF_BISHOPPER_LEAP,
                                    ..Default::default()
                                });
                            }
                        } else {
                            out.push(Move {
                                from: here,
                                to: t,
                                flags: MF_BISHOPPER_LEAP,
                                ..Default::default()
                            });
                        }
                        // Adjacent-beyond only: never continue past it.
                        break;
                    }
                    nx += dx; ny += dy;
                }
            }
        }
    }
}
