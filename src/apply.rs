use crate::pos::*;
use crate::types::*;
use crate::journal::*;
use crate::eval::*;
use crate::movegen::piece_attacks;
use crate::attacks::av_suppressed;

pub fn is_unsafe_key(k: PieceKey) -> bool {
    use PieceKey::*;
    matches!(k, Knightmare|Djinn|FissionReactor|GoldenPawn|BladeRunner|HordeMother
              | Recruit|Rocketman|ElectroKnight|Icicle|Dancer|Bishopper|BisHop
              | WarAutomaton|Banker|Gunslinger|HeroPawn|EpeePawn|SumoRook|Shrike|King)
}

pub fn make_journaled(pos: &mut Pos, mv: &Move, player: u8, opp: u8, j: &mut Journal) -> bool {
    let specials = MF_KNIGHTMARE_OFF | MF_KNIGHTMARE_DROP | MF_DJINN_DISSIPATE
                 | MF_DANCER_DOUBLE | MF_HORDE_SPAWN | MF_RECRUIT_SAC
                 | MF_ROCKET_BLAST | MF_ELECTRO | MF_BISHOPPER_LEAP
                 | MF_BISHOPPER_CROUCH | MF_GUN_STANDOFF | MF_EP
                 | MF_SHRIKE_FIRST | MF_CASTLE_K | MF_CASTLE_Q | MF_IS_PUSH
                 | MF_PILGRIM_RESURRECT | MF_ICICLE_THAW;
    // MF_ICICLE_THAW is a stay-in-place move (from == to): journaling it
    // writes at[to] then at[from] = 0 on the same square (net at = 0)
    // while occ_after_journaled yields occ = true there, so any later
    // PhaseRook ray over that square indexes pieces[65535] and panics.
    // Like every other from == to special, thaw must fall back to
    // apply_full, which clears Frozen correctly.
    if mv.flags & specials != 0 { return false; }
    if !pos.doomed.is_empty() { return false; }
    if mv.from == SQ_NONE { return false; }

    let mover_slot = match slot_of(pos, mv.from) { Some(s) => s, None => return false };
    let mk = pos.pieces[mover_slot].key;
    let msk = pos.pieces[mover_slot].sub_key;
    if is_unsafe_key(mk) || is_unsafe_key(msk) { return false; }

    if let Some(v_slot) = slot_of(pos, mv.to) {
        let vk = pos.pieces[v_slot].key;
        let vs = pos.pieces[v_slot].sub_key;
        if is_unsafe_key(vk) || is_unsafe_key(vs) { return false; }
        // On ANY capture/death:
        // - Hordeling (horde_id != 0) shared death wipes the connected horde
        // - WarAutomaton auto-advances 1 square (wiki/Sandbox rule)
        // - Djinn in limbo revives onto home square
        // make_journaled does not track these secondary effects, so fall back to apply_full.
        if pos.pieces[v_slot].horde_id != 0
            || vk == PieceKey::Hordeling
            || vs == PieceKey::Hordeling
            || !pos.limbo.is_empty()
            || pos.pieces.iter().any(|p| {
                p.alive && (p.key == PieceKey::WarAutomaton || p.sub_key == PieceKey::WarAutomaton)
            })
        {
            return false;
        }
    }

    let old_ket = pos.ket_x2;
    let old_qc  = pos.queen_count;
    let sign_m  = if player == 0 { 1i32 } else { -1i32 };

    let old_mval = piece_value(&pos.pieces[mover_slot])
                 + pst_bonus(&pos.pieces[mover_slot], old_ket)
                 + acc_tax(&pos.pieces[mover_slot]);
    let was_q = is_queen_piece_by_value(&pos.pieces[mover_slot]);
    let mut edelta = 0i32;
    let mut new_qc = old_qc;

    let mut vic_captured: Option<usize> = None;
    if let Some(v_slot) = slot_of(pos, mv.to) {
        let vic_val = piece_value(&pos.pieces[v_slot])
                    + pst_bonus(&pos.pieces[v_slot], old_ket)
                    + acc_tax(&pos.pieces[v_slot]);
        let sign_v = if pos.pieces[v_slot].player == 0 { 1 } else { -1 };
        edelta -= sign_v * vic_val;
        if is_queen_piece_by_value(&pos.pieces[v_slot]) { new_qc = new_qc.saturating_sub(1); }
        vic_captured = Some(v_slot);
    }

    j.at_put(pos, mv.to, (mover_slot + 1) as u16);
    j.at_put(pos, mv.from, 0);

    if let Some(v_slot) = vic_captured {
        j.alive_put(pos, v_slot as u32, false);
        let vk = pos.pieces[v_slot].key;
        let vs = pos.pieces[v_slot].sub_key;
        let vplayer = pos.pieces[v_slot].player;
        j.v.push(JEntry::PosLastCap { player: vplayer, old: pos.last_cap[vplayer as usize] });
        pos.last_cap[vplayer as usize] = (vk, vs);
    }

    let msi = mover_slot as u32;
    j.u8_put(pos, msi, JField::X, sq_x(mv.to));
    j.u8_put(pos, msi, JField::Y, sq_y(mv.to));
    if !pos.pieces[mover_slot].moved() {
        j.u16_put(pos, msi, JField::StateFlags,
                  pos.pieces[mover_slot].state_flags | (1 << 2));
    }

    if mk == PieceKey::Pilgrim || msk == PieceKey::Pilgrim {
        let dist = (sq_x(mv.to) as i8 - sq_x(mv.from) as i8).abs()
            .max((sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs()) as u8;
        let new_odo = pos.pieces[mover_slot].odo.saturating_add(dist);
        j.u8_put(pos, msi, JField::Odo, new_odo);
        if new_odo >= 20 && !pos.pieces[mover_slot].pilgrim_charged() {
            j.u16_put(pos, msi, JField::StateFlags,
                      pos.pieces[mover_slot].state_flags | (1 << 8));
        }
    }

    // Lever-3: the two loops below were separate full scans over
    // pos.pieces (Fish movedLast update, then ElectroKnight quiet/charge
    // reset). They are fused into one scan with identical final state:
    // per piece, the Fish update runs before the ElectroKnight update,
    // exactly as before, and journal entries still capture each write's
    // pre-value, so unwind (which pops in reverse) restores exactly.
    // The Fish branch is unchanged: a Fish move sets movedLast only on
    // the mover, any other move by the owner clears it on all friendly
    // Fish (2026-10-31 expiry fix, preserved verbatim in effect).
    // The ElectroKnight branch reads only quiet/echarged, which the Fish
    // branch never touches (it writes only bit 3), except on a piece
    // that is both Fish and ElectroKnight, where the EK clear is now
    // computed from the post-Fish flags — the same value the old
    // sequential loops produced.
    let mover_is_fish = mk == PieceKey::Fish || msk == PieceKey::Fish;
    for fi in 0..pos.pieces.len() {
        if !pos.pieces[fi].alive { continue; }
        if pos.pieces[fi].player != player { continue; }
        let k2 = pos.pieces[fi].key;
        let s2 = pos.pieces[fi].sub_key;
        if k2 == PieceKey::Fish || s2 == PieceKey::Fish {
            let should = if mover_is_fish { (fi == mover_slot) as u16 } else { 0 };
            let cur = pos.pieces[fi].state_flags;
            let new = (cur & !(1u16 << 3)) | (should << 3);
            if new != cur { j.u16_put(pos, fi as u32, JField::StateFlags, new); }
        }
        if k2 == PieceKey::ElectroKnight || s2 == PieceKey::ElectroKnight {
            if pos.pieces[fi].quiet != 0 || pos.pieces[fi].echarged() {
                let flags = pos.pieces[fi].state_flags;
                let cleared = flags & !(1u16 << 1);
                j.u8_put (pos, fi as u32, JField::Quiet, 0);
                j.u16_put(pos, fi as u32, JField::StateFlags, cleared);
            }
        }
    }

    if vic_captured.is_some() && (mk == PieceKey::Marauder || msk == PieceKey::Marauder) {
        let nk = pos.pieces[mover_slot].kills.saturating_add(1);
        j.u8_put(pos, msi, JField::Kills, nk);
    }

    let mover_is_pawn = is_pawn_like(mk) || is_pawn_like(msk);
    let last_rank: i8 = if player == 0 { 7 } else { 0 };
    if mover_is_pawn && sq_y(mv.to) as i8 == last_rank
       && mk != PieceKey::IronPawn && msk != PieceKey::IronPawn
       && !enemy_aristocrat(pos, opp) {
        let new_key = if mv.promo != 0 { PieceKey::from_u16(mv.promo as u16) } else { PieceKey::Queen };
        let nk = match new_key {
            PieceKey::Queen | PieceKey::Rook | PieceKey::Bishop | PieceKey::Knight => new_key,
            _ => PieceKey::Queen,
        };
        j.u16_put(pos, msi, JField::Key, nk as u16);
        j.u16_put(pos, msi, JField::SubKey, nk as u16);
    }

    j.v.push(JEntry::PosEp { old: pos.ep });
    pos.ep = if mover_is_pawn
        && (sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs() == 2 {
        Some(EpInfo {
            x: sq_x(mv.from) as i8,
            y: (sq_y(mv.from) as i8 + sq_y(mv.to) as i8) / 2,
            pawn_x: sq_x(mv.to) as i8,
            pawn_y: sq_y(mv.to) as i8,
        })
    } else { None };

    j.v.push(JEntry::PosHalf { old: pos.halfmove });
    pos.halfmove = if mover_is_pawn || vic_captured.is_some() { 0 }
                   else { pos.halfmove.saturating_add(1) };

    j.v.push(JEntry::HistTrunc { old_len: pos.hist_hashes.len() as u32 });
    pos.hist_hashes.push(rep_hash(pos, opp));

    let new_mval = piece_value(&pos.pieces[mover_slot])
                 + pst_bonus(&pos.pieces[mover_slot], old_ket)
                 + acc_tax(&pos.pieces[mover_slot]);
    edelta += sign_m * (new_mval - old_mval);
    let is_q_now = is_queen_piece_by_value(&pos.pieces[mover_slot]);
    if !was_q && is_q_now { new_qc = new_qc.saturating_add(1); }
    if was_q && !is_q_now { new_qc = new_qc.saturating_sub(1); }

    let new_ket = if new_qc >= 2 { 0 } else if new_qc == 1 { 1 } else { 2 };
    j.v.push(JEntry::PosEval { old: pos.eval_sum });
    if new_ket != old_ket {
        j.v.push(JEntry::PosKet { old: old_ket });
        j.v.push(JEntry::PosQC  { old: old_qc  });
        pos.ket_x2 = new_ket;
        pos.queen_count = new_qc;
        pos.eval_sum = compute_full_eval_sum(pos, new_ket);
    } else {
        pos.eval_sum = pos.eval_sum.saturating_add(edelta);
    }
    j.v.push(JEntry::PosStm { old: pos.stm });
    pos.stm = opp;

    let _ = av_suppressed; // silence unused
    let _ = piece_attacks;
    true
}

#[cfg(test)]
mod side_to_move_tests {
    use super::make_journaled;
    use crate::journal::Journal;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, Move, PieceKey};

    #[test]
    fn journaled_move_flips_stm_and_unwind_restores_it() {
        let mut pos = Pos::default();
        pos.stm = 0;
        pos.pieces.push(Piece {
            id: 1, key: PieceKey::Rook, sub_key: PieceKey::Rook,
            player: 0, x: 0, y: 0, alive: true, ..Default::default()
        });
        pos.at[sq(0, 0) as usize] = 1;
        let mv = Move { from: sq(0, 0), to: sq(0, 1), ..Default::default() };
        let mut journal = Journal::default();
        let mark = journal.mark();

        assert!(make_journaled(&mut pos, &mv, 0, 1, &mut journal));
        assert_eq!(pos.stm, 1, "child position must record the opponent to move");
        journal.unwind(&mut pos, mark);
        assert_eq!(pos.stm, 0, "unwind must restore the parent side to move");
        assert_eq!(pos.pieces[0].y, 0, "unwind must restore the piece square");
    }

    #[test]
    fn journaled_capture_defers_when_wa_limbo_or_hordeling_present() {
        let mut base = Pos::default();
        base.pieces.push(Piece {
            id: 1, key: PieceKey::Rook, sub_key: PieceKey::Rook,
            player: 0, x: 0, y: 0, alive: true, ..Default::default()
        });
        base.pieces.push(Piece {
            id: 2, key: PieceKey::Pawn, sub_key: PieceKey::Pawn,
            player: 1, x: 0, y: 1, alive: true, ..Default::default()
        });
        base.at[sq(0, 0) as usize] = 1;
        base.at[sq(0, 1) as usize] = 2;
        let cap = Move { from: sq(0, 0), to: sq(0, 1), flags: crate::types::MF_CAPTURE, ..Default::default() };

        // 1. With WarAutomaton on board, capture must fall back to apply_full.
        let mut pos_wa = base.clone();
        pos_wa.pieces.push(Piece {
            id: 3, key: PieceKey::WarAutomaton, sub_key: PieceKey::WarAutomaton,
            player: 0, x: 3, y: 3, alive: true, ..Default::default()
        });
        pos_wa.at[sq(3, 3) as usize] = 3;
        let mut j = Journal::default();
        assert!(!make_journaled(&mut pos_wa, &cap, 0, 1, &mut j));

        // 2. With Djinn in limbo, capture must fall back to apply_full.
        let mut pos_limbo = base.clone();
        pos_limbo.limbo.push(crate::pos::LimboPiece {
            id: 4, key: PieceKey::Djinn, sub_key: PieceKey::Djinn,
            player: 0, home_x: 4, home_y: 4,
        });
        assert!(!make_journaled(&mut pos_limbo, &cap, 0, 1, &mut j));
    }
}
