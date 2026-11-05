use crate::pos::*;
use crate::types::*;
use crate::attacks::*;

// ── Fast register-accumulated ray & leap emitters ─────────────────────────
#[inline(always)]
fn on_board_fast(x: i8, y: i8) -> bool {
    (x as u8) < 8 && (y as u8) < 8
}

#[inline]
fn add_rays(out: &mut Bitboard, x: i8, y: i8, dirs: &[(i8, i8)], occ: &[bool; 64], max_range: i8) {
    let mut mask = out.0;
    if max_range <= 0 {
        for &(dx, dy) in dirs {
            let mut nx = x + dx;
            let mut ny = y + dy;
            while on_board_fast(nx, ny) {
                let s = sq(nx, ny);
                mask |= 1u64 << s;
                if occ[s as usize] {
                    break;
                }
                nx += dx;
                ny += dy;
            }
        }
    } else {
        for &(dx, dy) in dirs {
            let mut nx = x + dx;
            let mut ny = y + dy;
            let mut steps = 0i8;
            while on_board_fast(nx, ny) && steps < max_range {
                let s = sq(nx, ny);
                mask |= 1u64 << s;
                if occ[s as usize] {
                    break;
                }
                nx += dx;
                ny += dy;
                steps += 1;
            }
        }
    }
    out.0 = mask;
}

#[inline]
fn add_leaps(out: &mut Bitboard, x: i8, y: i8, leaps: &[(i8, i8)]) {
    let mut mask = out.0;
    for &(dx, dy) in leaps {
        let nx = x + dx;
        let ny = y + dy;
        if on_board_fast(nx, ny) {
            mask |= 1u64 << sq(nx, ny);
        }
    }
    out.0 = mask;
}

#[inline(always)]
fn add_king(out: &mut Bitboard, x: i8, y: i8) {
    add_leaps(out, x, y, &DIRS_KING);
}

#[inline(always)]
fn add_knight(out: &mut Bitboard, x: i8, y: i8) {
    add_leaps(out, x, y, &LEAPS_KNIGHT);
}

#[inline(always)]
fn add_camel(out: &mut Bitboard, x: i8, y: i8) {
    add_leaps(out, x, y, &LEAPS_CAMEL);
}

#[inline(always)]
fn add_pawn_diag(out: &mut Bitboard, x: i8, y: i8, player: u8) {
    let ny = y + if player == 0 { 1 } else { -1 };
    if (ny as u8) < 8 {
        let mut mask = out.0;
        if x > 0 {
            mask |= 1u64 << sq(x - 1, ny);
        }
        if x < 7 {
            mask |= 1u64 << sq(x + 1, ny);
        }
        out.0 = mask;
    }
}

#[inline]
fn add_queen(out: &mut Bitboard, x: i8, y: i8, occ: &[bool; 64]) {
    add_rays(out, x, y, &DIRS_ROOK, occ, 0);
    add_rays(out, x, y, &DIRS_BISHOP, occ, 0);
}

// ── Shared classification & resolution helpers ────────────────────────────
#[inline(always)]
pub fn is_pawn_key(k: PieceKey) -> bool {
    is_pawn_like(k) || k == PieceKey::Blueprint
}

#[inline(always)]
pub fn is_king_target(p: &Piece) -> bool {
    use PieceKey::*;
    (matches!(
        p.key,
        King | WarKing | MeditatingKing | PopeKing | IrreverentKing | Rocketman
    ) || matches!(
        p.sub_key,
        King | WarKing | MeditatingKing | PopeKing | IrreverentKing | Rocketman
    )) && p.key != Recruit
        && p.sub_key != Recruit
}

/// Resolves `Blueprint` (`pawn+copy_left_pawn`) and `Pinata` (`random_pixie_transform`)
/// into their effective `(key, sub_key)` pair.
#[inline]
pub fn resolve_piece_keys(pos: &Pos, p: &Piece) -> (PieceKey, PieceKey) {
    let mut pk = p.key;
    let mut sk = p.sub_key;

    if pk == PieceKey::Blueprint || sk == PieceKey::Blueprint {
        let other = if pk == PieceKey::Blueprint { sk } else { pk };
        if other != PieceKey::None && other != PieceKey::Blueprint && other != PieceKey::Pawn {
            return (other, other);
        }
        if !p.moved() && xy_on_board(p.x, p.y) {
            let mut cx = p.x as i8 - 1;
            let cy = p.y as i8;
            while cx >= 0 {
                if let Some(li) = slot_of(pos, sq(cx, cy)) {
                    let lp = &pos.pieces[li];
                    if lp.alive
                        && lp.player == p.player
                        && (is_pawn_key(lp.key) || is_pawn_key(lp.sub_key))
                    {
                        let l_other = if lp.key == PieceKey::Blueprint {
                            lp.sub_key
                        } else {
                            lp.key
                        };
                        if l_other != PieceKey::Blueprint && l_other != PieceKey::None {
                            return (l_other, lp.sub_key);
                        } else if lp.key == PieceKey::Blueprint
                            || lp.sub_key == PieceKey::Blueprint
                        {
                            cx -= 1;
                            continue;
                        }
                    }
                }
                break;
            }
        }
        if pk == PieceKey::Blueprint {
            pk = PieceKey::Pawn;
        }
        if sk == PieceKey::Blueprint {
            sk = PieceKey::Pawn;
        }
    }

    if pk == PieceKey::Pinata || sk == PieceKey::Pinata {
        let other = if pk == PieceKey::Pinata { sk } else { pk };
        if other != PieceKey::None && other != PieceKey::Pinata {
            return (other, other);
        }
    }

    (pk, sk)
}

/// Checks if an alive enemy `Aristocrat` survives the move (which blocks pawn promotion in `apply.rs`/`apply_full.rs`).
#[inline]
pub fn blocked_by_enemy_aristocrat(pos: &Pos, opp: u8, captured_sq: Option<u8>) -> bool {
    for p in &pos.pieces {
        if !p.alive || p.player != opp || !xy_on_board(p.x, p.y) {
            continue;
        }
        if let Some(csq) = captured_sq {
            if sq(p.x as i8, p.y as i8) == csq {
                continue;
            }
        }
        if p.key == PieceKey::Aristocrat || p.sub_key == PieceKey::Aristocrat {
            return true;
        }
    }
    false
}

// ── Family classification ─────────────────────────────────────────────────
#[inline(always)]
pub fn family_of(k: PieceKey) -> u8 {
    use PieceKey::*;
    match k {
        Pawn | PawnWithKnife | GoldenPawn | HeroPawn | EpeePawn
        | Shrike | IronPawn | Hordeling | TranslucentPawn | Ooze
        | Recruit | WarAutomaton | WarpJumper | Blueprint => 1,

        Knight | Camel | Knightmare | ElectroKnight | Fish | Banker
        | AntiViolence | Pinata => 2,

        Bishop | Bishopper | BisHop | BladeRunner | Bouncer
        | Aristocrat | Basilisk | Gunslinger | Pilgrim | Cardinal
        | Dancer | Djinn | HordeMother | Icicle => 3,

        Rook | SumoRook | PhaseRook | RookZero => 4,

        Queen | VoidQueen | NullQueen | WitchQueen | FissionReactor => 5,

        King | WarKing | MeditatingKing | PopeKing
        | IrreverentKing | Rocketman | Marauder => 6,

        _ => 0,
    }
}

#[inline(always)]
fn has_pixie_attack_override(k: PieceKey) -> bool {
    use PieceKey::*;
    matches!(
        k,
        PawnWithKnife
            | Shrike
            | PhaseRook
            | Rocketman
            | Pilgrim
            | Camel
            | Marauder
            | Cardinal
            | Bouncer
            | WarAutomaton
            | Bishopper
            | BisHop
            | Recruit
            | Fish
    )
}

// ── piece_attacks ─────────────────────────────────────────────────────────
pub fn piece_attacks(
    pos: &Pos,
    p_idx: usize,
    occ: &[bool; 64],
    no_drift: bool,
    strict: bool,
) -> Bitboard {
    let p = &pos.pieces[p_idx];
    let mut bb = Bitboard(0);

    if PROVISIONAL_ICICLE_FREEZE && p.frozen() {
        return bb;
    }
    let (pk, sk) = resolve_piece_keys(pos, p);
    if is_non_capturer(pk) || is_non_capturer(sk) {
        return bb;
    }

    let x = p.x as i8;
    let y = p.y as i8;
    let player = p.player;

    // Fast path for Marauder: its variable kill-range replaces standard abilities/family rays.
    if pk == PieceKey::Marauder || sk == PieceKey::Marauder {
        let range = 1 + 2 * (p.kills as i8);
        add_rays(&mut bb, x, y, &DIRS_ROOK, occ, range);
        add_rays(&mut bb, x, y, &DIRS_BISHOP, occ, range);
        if pk != PieceKey::Marauder || sk != PieceKey::Marauder {
            apply_pixie_attack_overrides(
                pos, p_idx, p, pk, sk, x, y, player, occ, no_drift, strict, &mut bb, true,
            );
        }
        return bb;
    }

    // Fast path for pure Camel (`leaper_1_3`): replaces Knight leaps with (1, 3) / (3, 1) leaps.
    if (pk == PieceKey::Camel || sk == PieceKey::Camel)
        && pk != PieceKey::Recruit
        && sk != PieceKey::Recruit
    {
        add_camel(&mut bb, x, y);
        if pk != PieceKey::Camel || sk != PieceKey::Camel {
            apply_pixie_attack_overrides(
                pos, p_idx, p, pk, sk, x, y, player, occ, no_drift, strict, &mut bb, false,
            );
        }
        return bb;
    }

    let n_ab = p.n_abilities as usize;
    for a in 0..n_ab {
        let key = p.abilities[a].key;
        let range = p.abilities[a].range as i8;
        match key {
            AB_LEAP12 => add_knight(&mut bb, x, y),
            AB_PAWN_CAPTURE => add_pawn_diag(&mut bb, x, y, player),
            AB_RIDE_CARDINAL => add_rays(&mut bb, x, y, &DIRS_ROOK, occ, range),
            AB_RIDE_DIAGONAL => add_rays(&mut bb, x, y, &DIRS_BISHOP, occ, range),
            AB_RIDE_PRINCIPAL => add_queen(&mut bb, x, y, occ),
            AB_CARDINAL_RETREAT => {
                let d: i8 = if player == 0 { -1 } else { 1 };
                if on_board_fast(x, y + d) && !occ[sq(x, y + d) as usize] {
                    bb.set(sq(x, y + d));
                }
            }
            _ => {}
        }
    }

    if bb.is_empty() {
        let fam = {
            let f = family_of(pk);
            if f != 0 && !(pk == PieceKey::Pinata && family_of(sk) != 0) {
                f
            } else {
                let fs = family_of(sk);
                if fs != 0 { fs } else { f }
            }
        };
        match fam {
            1 => add_pawn_diag(&mut bb, x, y, player),
            2 => add_knight(&mut bb, x, y),
            3 => add_rays(&mut bb, x, y, &DIRS_BISHOP, occ, 0),
            4 => add_rays(&mut bb, x, y, &DIRS_ROOK, occ, 0),
            5 => add_queen(&mut bb, x, y, occ),
            6 => add_king(&mut bb, x, y),
            _ => {
                if !strict {
                    add_queen(&mut bb, x, y, occ);
                }
            }
        }
    }

    // Fast skip for standard pieces and Pixie pieces without custom attack pattern overrides.
    if has_pixie_attack_override(pk) || has_pixie_attack_override(sk) {
        apply_pixie_attack_overrides(
            pos, p_idx, p, pk, sk, x, y, player, occ, no_drift, strict, &mut bb, false,
        );
    }

    bb
}

#[inline]
fn apply_pixie_attack_overrides(
    pos: &Pos,
    p_idx: usize,
    p: &Piece,
    pk: PieceKey,
    sk: PieceKey,
    x: i8,
    y: i8,
    player: u8,
    occ: &[bool; 64],
    no_drift: bool,
    strict: bool,
    bb: &mut Bitboard,
    skip_marauder: bool,
) {
    use PieceKey::*;
    let is = |t: PieceKey| pk == t || sk == t;

    if is(PawnWithKnife) {
        add_pawn_diag(bb, x, y, player);
        let dirx: i8 = if x < 4 { 1 } else { -1 };
        let diry: i8 = if player == 0 { 1 } else { -1 };
        let qx = x + 2 * dirx;
        let qy = y + 2 * diry;
        if on_board_fast(qx, qy) && !occ[sq(x + dirx, y + diry) as usize] {
            bb.set(sq(qx, qy));
        }
    }
    if is(Shrike) && !p.moved() {
        let sd: i8 = if player == 0 { 1 } else { -1 };
        if on_board_fast(x, y + sd) {
            let s1 = sq(x, y + sd);
            bb.set(s1);
            if on_board_fast(x, y + 2 * sd) {
                let s1_blocked = if occ[s1 as usize] {
                    let v1_idx = pos.at[s1 as usize];
                    if v1_idx != 0 {
                        let v1 = &pos.pieces[(v1_idx - 1) as usize];
                        v1.player == player
                            || is_king_target(v1)
                            || (PROVISIONAL_IRON_INVULN
                                && (v1.key == IronPawn || v1.sub_key == IronPawn))
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !s1_blocked {
                    bb.set(sq(x, y + 2 * sd));
                }
            }
        }
    }
    if is(PhaseRook) {
        phase_rook_into(bb, p, pos, occ);
    }
    if is(Rocketman) {
        add_king(bb, x, y);
    }
    // NOTE (lTLi6n, 2026-10-27): a moved_last Fish's bonus 1-step King
    // moves are MOVES, not attacks. The server accepted White Pha1->a4
    // at turn 44 (Fish on c2 "attacking" d1) and Kd1->d2x at turn 46
    // (Fish on e1 "attacking" d2), so the bonus step neither gives check
    // nor defends squares. Counting it here fabricated checks and a
    // phantom mate (White 0 legal moves in the engine's model vs 30
    // with the flag cleared). The bonus MOVES themselves still come
    // from movegen_special.rs; this function only feeds attacks.
    if is(Recruit) {
        add_pawn_diag(bb, x, y, player);
        bb.0 |= recruit_inherited_targets(pos, p_idx, occ).0;
    }
    if is(Pilgrim) {
        add_rays(bb, x, y, &DIRS_BISHOP, occ, 0);
        if p.odo >= 20 || p.pilgrim_charged() {
            add_rays(bb, x, y, &DIRS_ROOK, occ, 0);
        }
    }
    if is(Camel) {
        if !is(Recruit) {
            *bb = Bitboard(0);
        }
        add_camel(bb, x, y);
    }
    if !skip_marauder && is(Marauder) {
        *bb = Bitboard(0);
        let range = 1 + 2 * (p.kills as i8);
        add_rays(bb, x, y, &DIRS_ROOK, occ, range);
        add_rays(bb, x, y, &DIRS_BISHOP, occ, range);
    }
    if is(Cardinal) {
        add_rays(bb, x, y, &DIRS_BISHOP, occ, 0);
        let cy = y - if player == 0 { 1 } else { -1 };
        if on_board_fast(x, cy) && !occ[sq(x, cy) as usize] {
            bb.set(sq(x, cy));
        }
    }
    if is(Bouncer) {
        bouncer_into(bb, x, y, occ);
    }
    if !no_drift && !strict && is(WarAutomaton) {
        let wy = y + if player == 0 { 1 } else { -1 };
        if on_board_fast(x, wy) {
            let mut db = Bitboard(0);
            if is(Marauder) {
                let range = 1 + 2 * (p.kills as i8);
                add_rays(&mut db, x, wy, &DIRS_ROOK, occ, range);
                add_rays(&mut db, x, wy, &DIRS_BISHOP, occ, range);
            } else if is(Camel) {
                add_camel(&mut db, x, wy);
            } else {
                let fam = {
                    let f = family_of(pk);
                    if f != 0 { f } else { family_of(sk) }
                };
                match fam {
                    1 => add_pawn_diag(&mut db, x, wy, player),
                    2 => add_knight(&mut db, x, wy),
                    3 => add_rays(&mut db, x, wy, &DIRS_BISHOP, occ, 0),
                    4 => add_rays(&mut db, x, wy, &DIRS_ROOK, occ, 0),
                    5 => add_queen(&mut db, x, wy, occ),
                    6 => add_king(&mut db, x, wy),
                    _ => {}
                }
            }
            bb.0 |= db.0;
        }
    }
    if is(Bishopper) || is(BisHop) {
        add_rays(bb, x, y, &DIRS_BISHOP, occ, 0);
        if p.crouched() {
            bishopper_leap_into(bb, x, y, occ);
        }
    }
}

#[inline]
fn phase_rook_into(bb: &mut Bitboard, p: &Piece, pos: &Pos, occ: &[bool; 64]) {
    let x = p.x as i8;
    let y = p.y as i8;
    let player = p.player;
    let mut mask = bb.0;
    for &(dx, dy) in &DIRS_ROOK {
        let mut nx = x + dx;
        let mut ny = y + dy;
        // Server/JS rule (see evidence/phaserook-server-cases.md): a PhaseRook
        // ray may pass through AT MOST ONE friendly piece; the second friendly
        // stops the ray. Friendly squares are never generated (no landing on
        // a friend). v137 removed this guard ("pass through any number"),
        // which the server rejects (3ktEV7 a8->f8/g8, KZtmMe a1->h1) and the
        // JS candidate list omits. NOTE: the v137 ability flavor name
        // (`rook+pass_through_allies`) suggests unlimited was intentional in
        // v137's design, but the server is the legality authority in real
        // games, and the app's own JS model implements at-most-one — so the
        // engine follows the server here. The discrepancy is documented,
        // not silently resolved in favor of either side's flavor text.
        let mut phased_friend = false;
        while on_board_fast(nx, ny) {
            let s = sq(nx, ny);
            if !occ[s as usize] {
                mask |= 1u64 << s;
            } else {
                let v = pos.at[s as usize];
                if v == 0 || (v as usize) > pos.pieces.len() {
                    break;
                }
                let owner = pos.pieces[(v - 1) as usize].player;
                if owner == player {
                    if phased_friend {
                        break;
                    }
                    phased_friend = true;
                } else {
                    // The server candidate list in game tVmhFr rejected
                    // PhaseRook b1->b6 when the ray first crossed friendly
                    // b2 and then reached the enemy Queen. Keep empty squares
                    // beyond one phased friend reachable, but do not generate
                    // an enemy capture beyond that friend: the server is the
                    // legality authority for the WASM root move.
                    if !phased_friend {
                        mask |= 1u64 << s;
                    }
                    break;
                }
            }
            nx += dx;
            ny += dy;
        }
    }
    // Phantom-occupancy cleanup: a stale occ bit (no piece on that square)
    // can already be in the mask, added by the generic ability ray before
    // this function ran. Clear any such square so a phantom never counts
    // as an attack. No-op on in-sync boards (occ[s] always has a piece).
    for s in 0..64usize {
        let v = pos.at[s];
        if occ[s] && (v == 0 || (v as usize) > pos.pieces.len()) {
            mask &= !(1u64 << s);
        }
    }
    bb.0 = mask;
}

#[inline]
fn bouncer_into(bb: &mut Bitboard, x: i8, y: i8, occ: &[bool; 64]) {
    let mut mask = bb.0;
    for &(dx0, dy0) in &DIRS_BISHOP {
        let mut dx = dx0;
        let mut dy = dy0;
        let mut cx = x + dx;
        let mut cy = y + dy;
        let mut lx = x;
        let mut ly = y;
        let mut bounced = false;
        loop {
            if !on_board_fast(cx, cy) {
                if bounced || (lx == x && ly == y) {
                    break;
                }
                if cx < 0 || cx > 7 {
                    dx = -dx;
                }
                if cy < 0 || cy > 7 {
                    dy = -dy;
                }
                bounced = true;
                cx = lx + dx;
                cy = ly + dy;
                continue;
            }
            if cx == x && cy == y {
                break;
            }
            let s = sq(cx, cy);
            mask |= 1u64 << s;
            if occ[s as usize] {
                break;
            }
            lx = cx;
            ly = cy;
            cx += dx;
            cy += dy;
        }
    }
    bb.0 = mask;
}

#[inline]
fn bishopper_leap_into(bb: &mut Bitboard, x: i8, y: i8, occ: &[bool; 64]) {
    let mut mask = bb.0;
    for &(dx, dy) in &DIRS_BISHOP {
        let mut nx = x + dx;
        let mut ny = y + dy;
        let mut jumped = false;
        while on_board_fast(nx, ny) {
            let s = sq(nx, ny);
            mask |= 1u64 << s;
            if jumped {
                if occ[s as usize] {
                    break;
                }
            } else if occ[s as usize] {
                jumped = true;
            }
            nx += dx;
            ny += dy;
        }
    }
    bb.0 = mask;
}

#[inline]
pub fn bishopper_post_jump_mask(x: i8, y: i8, occ: &[bool; 64]) -> Bitboard {
    let mut mask = 0u64;
    for &(dx, dy) in &DIRS_BISHOP {
        let mut nx = x + dx;
        let mut ny = y + dy;
        let mut jumped = false;
        while on_board_fast(nx, ny) {
            let s = sq(nx, ny);
            if jumped {
                mask |= 1u64 << s;
                if occ[s as usize] {
                    break;
                }
            } else if occ[s as usize] {
                jumped = true;
            }
            nx += dx;
            ny += dy;
        }
    }
    Bitboard(mask)
}

#[inline]
pub fn recruit_inherited_key(pos: &Pos, p: &Piece) -> PieceKey {
    use PieceKey::*;
    let pk = p.key;
    let sk = p.sub_key;
    let direct = if pk == Recruit { sk } else { pk };
    if direct != None && direct != Recruit && direct != Pawn {
        return direct;
    }
    if p.moved() {
        let opp = (1 - (p.player & 1)) as usize;
        let (lc_k, lc_s) = pos.last_cap[opp];
        if lc_k != None && lc_k != Recruit && lc_k != Pawn {
            return lc_k;
        }
        if lc_s != None && lc_s != Recruit && lc_s != Pawn {
            return lc_s;
        }
        let me = (p.player & 1) as usize;
        let (mc_k, mc_s) = pos.last_cap[me];
        if mc_k != None && mc_k != Recruit && mc_k != Pawn {
            return mc_k;
        }
        if mc_s != None && mc_s != Recruit && mc_s != Pawn {
            return mc_s;
        }
    }
    None
}

pub fn recruit_inherited_targets(pos: &Pos, p_idx: usize, occ_local: &[bool; 64]) -> Bitboard {
    use PieceKey::*;
    let p = &pos.pieces[p_idx];
    let x = p.x as i8;
    let y = p.y as i8;
    let player = p.player;
    let inh = recruit_inherited_key(pos, p);
    let mut bb = Bitboard(0);

    let n_ab = p.n_abilities as usize;
    for a in 0..n_ab {
        let key = p.abilities[a].key;
        let range = p.abilities[a].range as i8;
        match key {
            AB_LEAP12 => add_knight(&mut bb, x, y),
            AB_RIDE_CARDINAL => add_rays(&mut bb, x, y, &DIRS_ROOK, occ_local, range),
            AB_RIDE_DIAGONAL => add_rays(&mut bb, x, y, &DIRS_BISHOP, occ_local, range),
            AB_RIDE_PRINCIPAL => add_queen(&mut bb, x, y, occ_local),
            AB_CARDINAL_RETREAT => {
                let d: i8 = if player == 0 { -1 } else { 1 };
                if on_board_fast(x, y + d) && !occ_local[sq(x, y + d) as usize] {
                    bb.set(sq(x, y + d));
                }
            }
            _ => {}
        }
    }

    if inh != PieceKey::None && inh != Recruit && inh != Pawn {
        if inh == Marauder {
            let range = 1 + 2 * (p.kills as i8);
            add_rays(&mut bb, x, y, &DIRS_ROOK, occ_local, range);
            add_rays(&mut bb, x, y, &DIRS_BISHOP, occ_local, range);
        } else if inh == Camel {
            add_camel(&mut bb, x, y);
        } else if inh == PhaseRook {
            phase_rook_into(&mut bb, p, pos, occ_local);
        } else if inh == Bouncer {
            bouncer_into(&mut bb, x, y, occ_local);
        } else if inh == BladeRunner {
            bb.0 |= blade_runner_dests(p, pos, occ_local).0;
        } else if is_non_capturer(inh) {
            let mut dummy = p.clone();
            dummy.key = inh;
            bb.0 |= non_capturer_moves(&dummy, occ_local).0;
        } else {
            match family_of(inh) {
                2 => add_knight(&mut bb, x, y),
                3 => add_rays(&mut bb, x, y, &DIRS_BISHOP, occ_local, 0),
                4 => add_rays(&mut bb, x, y, &DIRS_ROOK, occ_local, 0),
                5 => add_queen(&mut bb, x, y, occ_local),
                6 => add_king(&mut bb, x, y),
                _ => {}
            }
            if inh == Cardinal {
                let cy = y - if player == 0 { 1 } else { -1 };
                if on_board_fast(x, cy) && !occ_local[sq(x, cy) as usize] {
                    bb.set(sq(x, cy));
                }
            }
            if inh == Pilgrim && (p.odo >= 20 || p.pilgrim_charged()) {
                add_rays(&mut bb, x, y, &DIRS_ROOK, occ_local, 0);
            }
            if inh == Fish && p.moved_last() {
                add_king(&mut bb, x, y);
            }
            if (inh == Bishopper || inh == BisHop) && p.crouched() {
                bishopper_leap_into(&mut bb, x, y, occ_local);
            }
            if inh == PawnWithKnife {
                let dirx: i8 = if x < 4 { 1 } else { -1 };
                let diry: i8 = if player == 0 { 1 } else { -1 };
                let qx = x + 2 * dirx;
                let qy = y + 2 * diry;
                if on_board_fast(qx, qy) && !occ_local[sq(x + dirx, y + diry) as usize] {
                    bb.set(sq(qx, qy));
                }
            }
        }
    }

    bb
}

// ── non_capturer_moves ────────────────────────────────────────────────────
pub fn non_capturer_moves(p: &Piece, occ: &[bool; 64]) -> Bitboard {
    use PieceKey::*;
    let kind: u8 = match p.key {
        AntiViolence => 2,
        Basilisk => 3,
        IronPawn => 1,
        Icicle => 3,
        _ => match p.sub_key {
            AntiViolence => 2,
            Basilisk => 3,
            IronPawn => 1,
            Icicle => 3,
            _ => return Bitboard(0),
        },
    };
    let x = p.x as i8;
    let y = p.y as i8;
    let mut mask = 0u64;
    match kind {
        1 => {
            let ny = y + if p.player == 0 { 1 } else { -1 };
            if on_board_fast(x, ny) {
                let s = sq(x, ny);
                if !occ[s as usize] {
                    mask |= 1u64 << s;
                }
            }
        }
        2 => {
            for &(dx, dy) in &LEAPS_KNIGHT {
                let nx = x + dx;
                let ny = y + dy;
                if on_board_fast(nx, ny) {
                    let s = sq(nx, ny);
                    if !occ[s as usize] {
                        mask |= 1u64 << s;
                    }
                }
            }
        }
        3 => {
            for &(dx, dy) in &DIRS_BISHOP {
                let mut nx = x + dx;
                let mut ny = y + dy;
                while on_board_fast(nx, ny) {
                    let s = sq(nx, ny);
                    if occ[s as usize] {
                        break;
                    }
                    mask |= 1u64 << s;
                    nx += dx;
                    ny += dy;
                }
            }
        }
        _ => {}
    }
    Bitboard(mask)
}

// ── Sumo helpers ──────────────────────────────────────────────────────────
#[inline]
// ── gen_moves ─────────────────────────────────────────────────────────────
pub fn gen_moves(pos: &Pos, player: u8, stage: Option<u8>) -> Vec<Move> {
    let mut out = gen_moves_pseudo(pos, player, stage);
    // AV fix — filter AV moves that lift suppression and expose own king.
    // Fast skip if the side to move has no alive AntiViolence piece.
    let has_av = pos.pieces.iter().any(|p| {
        p.alive
            && p.player == player
            && xy_on_board(p.x, p.y)
            && (p.key == PieceKey::AntiViolence || p.sub_key == PieceKey::AntiViolence)
    });
    if has_av {
        filter_av_illegal(pos, &mut out, player);
    }
    out
}

pub fn gen_moves_pseudo(pos: &Pos, player: u8, stage: Option<u8>) -> Vec<Move> {
    let occ_local = occ(pos);
    let opp = 1 - player;
    let last_rank: i8 = if player == 0 { 7 } else { 0 };
    let mut out: Vec<Move> = Vec::with_capacity(64);
    // When stage is None, stage quiet moves separately so captures/promotions/specials
    // are always emitted first for immediate tactical cutoff efficiency.
    let mut quiets: Vec<Move> = if stage.is_none() {
        Vec::with_capacity(48)
    } else {
        Vec::new()
    };

    let mut has_basilisk = false;
    for q in pos.pieces.iter() {
        if !xy_on_board(q.x, q.y) {
            continue;
        } // v130
        if q.alive
            && q.player == opp
            && (q.key == PieceKey::Basilisk || q.sub_key == PieceKey::Basilisk)
        {
            has_basilisk = true;
            break;
        }
    }

    let n_pieces = pos.pieces.len();
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) {
            continue;
        } // v130
        if !p.alive || p.player != player {
            continue;
        }
        if PROVISIONAL_ICICLE_FREEZE && p.frozen() {
            // Sandbox-verified (2026-11-03): a Frozen piece's ONLY legal
            // move is stay-in-place, which thaws it at the cost of the
            // whole turn (afterwards normal moves return). No movement,
            // no captures, no abilities. The old code gave frozen pieces
            // ZERO moves, which reads as stalemate in search.
            if stage != Some(0) {
                let here = sq(p.x as i8, p.y as i8);
                let mv = Move { from: here, to: here, flags: MF_ICICLE_THAW, ..Default::default() };
                if stage.is_none() {
                    quiets.push(mv);
                } else {
                    out.push(mv);
                }
            }
            continue;
        }
        if has_basilisk && is_paralyzed(pos, i, &occ_local) {
            continue;
        }

        let x = p.x as i8;
        let y = p.y as i8;
        let (pk, sk) = resolve_piece_keys(pos, p);
        let is_pawn = is_pawn_key(pk) || is_pawn_key(sk);
        let is_iron = pk == PieceKey::IronPawn || sk == PieceKey::IronPawn;
        let is_shrike = pk == PieceKey::Shrike || sk == PieceKey::Shrike;

        if is_pawn {
            if stage.is_none() {
                gen_pawn_moves(pos, i, pk, sk, player, opp, &occ_local, Some(0), &mut out);
                gen_pawn_moves(pos, i, pk, sk, player, opp, &occ_local, Some(1), &mut quiets);
            } else {
                gen_pawn_moves(pos, i, pk, sk, player, opp, &occ_local, stage, &mut out);
            }
        }

        let is_non_cap = is_non_capturer(pk)
            || is_non_capturer(sk)
            || pk == PieceKey::Icicle
            || sk == PieceKey::Icicle;
        let is_br = pk == PieceKey::BladeRunner || sk == PieceKey::BladeRunner;

        // Avoid wasted piece_attacks computation when overridden by BladeRunner or non-capturer.
        let atk = if is_br {
            blade_runner_dests(p, pos, &occ_local)
        } else if is_non_cap {
            let mut resolved_p = p.clone();
            resolved_p.key = pk;
            resolved_p.sub_key = sk;
            let mut nc = non_capturer_moves(&resolved_p, &occ_local);
            if pk == PieceKey::PhaseRook || sk == PieceKey::PhaseRook {
                phase_rook_into(&mut nc, p, pos, &occ_local);
            }
            nc
        } else {
            piece_attacks(pos, i, &occ_local, true, false)
        };

        let is_horde = pk == PieceKey::HordeMother || sk == PieceKey::HordeMother;
        let is_sumo = pk == PieceKey::SumoRook || sk == PieceKey::SumoRook;
        let is_king_like = (pk == PieceKey::King
            || sk == PieceKey::King
            || pk == PieceKey::Rocketman
            || sk == PieceKey::Rocketman)
            && pk != PieceKey::Recruit
            && sk != PieceKey::Recruit;
        let is_knife = pk == PieceKey::PawnWithKnife || sk == PieceKey::PawnWithKnife;
        let is_recruit = pk == PieceKey::Recruit || sk == PieceKey::Recruit;
        let is_fish = pk == PieceKey::Fish || sk == PieceKey::Fish;
        let recruit_inh_bb = if is_recruit && stage == Some(1) {
            recruit_inherited_targets(pos, i, &occ_local)
        } else {
            Bitboard(0)
        };
        let is_crouched_bishopper = (pk == PieceKey::Bishopper
            || sk == PieceKey::Bishopper
            || pk == PieceKey::BisHop
            || sk == PieceKey::BisHop)
            && p.crouched();
        let post_jump_bb = if is_crouched_bishopper {
            bishopper_post_jump_mask(x, y, &occ_local)
        } else {
            Bitboard(0)
        };

        // Hoist AntiViolence suppression check once per piece instead of per target square.
        let mut p_av_checked = false;
        let mut p_av_sup = false;

        let from_sq = sq(x, y);
        let mut bb = atk.0;
        while bb != 0 {
            let s = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            // Post-jump Bishopper squares are emitted with MF_BISHOPPER_LEAP in movegen_special.rs;
            // pre-jump normal Bishop moves are emitted here in movegen.rs.
            if is_crouched_bishopper && post_jump_bb.has(s) {
                continue;
            }
            let v_idx = pos.at[s as usize];

            // Fast stage-0 filter before coordinate & piece lookups.
            if stage == Some(0) && v_idx == 0 {
                if !is_pawn || is_iron || (sq_y(s) as i8) != last_rank {
                    continue;
                }
            }
            if is_pawn && v_idx == 0 {
                if !is_recruit || !recruit_inh_bb.has(s) {
                    continue;
                }
            }

            let tx = sq_x(s) as i8;
            let ty = sq_y(s) as i8;

            if is_king_like && (tx - x).abs().max((ty - y).abs()) > 1 {
                continue;
            }
            // Fish's 1-step King moves after moving are emitted in movegen_special.rs.
            if is_fish && !is_recruit && (tx - x).abs().max((ty - y).abs()) <= 1 {
                continue;
            }
            if is_shrike && tx == x {
                continue;
            }
            if is_knife && (tx - x).abs() == 2 && (ty - y).abs() == 2 {
                let kcx = if tx > x { 1 } else { -1 };
                let kdy = if ty > y { 1 } else { -1 };
                if pos.at[sq(x + kcx, y + kdy) as usize] != 0 {
                    continue;
                }
                // Knife's +1 diagonal range works only TOWARD the centre:
                // the destination must be nearer the board centre (3.5, 3.5)
                // than the origin (doubled coords avoid floats).
                let (fx, fy) = (2 * x as i32 - 7, 2 * y as i32 - 7);
                let (gx, gy) = (2 * tx as i32 - 7, 2 * ty as i32 - 7);
                if gx * gx + gy * gy >= fx * fx + fy * fy {
                    continue;
                }
            }

            let vic = if v_idx != 0 {
                Some(&pos.pieces[(v_idx - 1) as usize])
            } else {
                None
            };

            if let Some(v) = vic {
                if is_non_cap || is_br || !v.alive || v.player == player {
                    continue;
                }
                let v_king = is_king_target(v);
                if v_king && !is_sumo {
                    continue;
                }
                if PROVISIONAL_IRON_INVULN
                    && (v.key == PieceKey::IronPawn || v.sub_key == PieceKey::IronPawn)
                {
                    continue;
                }
                if v.player == opp {
                    if is_horde {
                        continue;
                    }
                    if !p_av_checked {
                        p_av_sup = av_suppressed(pos, i);
                        p_av_checked = true;
                    }
                    if p_av_sup {
                        continue;
                    }
                }
            }

            let can_promo = is_pawn
                && ty == last_rank
                && !is_iron
                && !blocked_by_enemy_aristocrat(pos, opp, if vic.is_some() { Some(s) } else { None });
            let tactical = vic.is_some() || can_promo;
            if let Some(st) = stage {
                if st == 0 && !tactical {
                    continue;
                }
                if st == 1 && tactical {
                    continue;
                }
            }

            let mut mv = Move {
                from: from_sq,
                to: s,
                ..Default::default()
            };
            // SumoRook (`rook+knockback_no_capture`) pushes enemy pieces via MF_IS_PUSH instead of MF_CAPTURE.
            if vic.is_some() && !is_sumo {
                mv.flags |= MF_CAPTURE;
            }

            if is_sumo {
                // Wiki ram (pixiewiki Sumo Rook): Sumo never captures and
                // never moves onto a piece. A push is a move to the empty
                // square immediately BEFORE the first piece on its rook
                // ray (V-1), shoving the consecutive line starting at V.
                // The gap rule (at least one empty square between Sumo
                // and V) holds by construction: V-1 exists as a ray
                // square only when V is not adjacent to the Sumo.
                // AV does not block knockback, so no suppression check
                // here (unlike captures); friendly pieces push too,
                // except the mover's own King, which is never pushed.
                if vic.is_some() {
                    continue;
                }
                let rdx = (tx - x).signum();
                let rdy = (ty - y).signum();
                let nx = tx + rdx;
                let ny = ty + rdy;
                if on_board_fast(nx, ny) && slot_of(pos, sq(nx, ny)).is_some() {
                    let mut line_len = 0usize;
                    let mut own_king_in_line = false;
                    let mut lx = nx;
                    let mut ly = ny;
                    while on_board_fast(lx, ly) {
                        if let Some(li) = slot_of(pos, sq(lx, ly)) {
                            let lp = &pos.pieces[li];
                            if lp.player == player && is_king_target(lp) {
                                own_king_in_line = true;
                            }
                            line_len += 1;
                            lx += rdx;
                            ly += rdy;
                        } else {
                            break;
                        }
                    }
                    let mut pushable = !own_king_in_line && (line_len == 1 || line_len == 2);
                    if pushable && line_len == 1 {
                        // Lone piece needs room for 2 squares: if V+2 is
                        // occupied by a non-line piece (gap behind), the
                        // 2-square shove cannot land, so stay quiet.
                        let dx2 = nx + 2 * rdx;
                        let dy2 = ny + 2 * rdy;
                        if on_board_fast(dx2, dy2) && pos.at[sq(dx2, dy2) as usize] != 0 {
                            pushable = false;
                        }
                    }
                    if pushable {
                        mv.flags |= MF_IS_PUSH;
                        mv.push_dx = rdx;
                        mv.push_dy = rdy;
                    }
                }
            }

            let target_vec = if stage.is_none() && !tactical {
                &mut quiets
            } else {
                &mut out
            };

            if can_promo {
                mv.flags |= MF_PROMO;
                for &pr_k in &[
                    PieceKey::Queen,
                    PieceKey::Rook,
                    PieceKey::Bishop,
                    PieceKey::Knight,
                ] {
                    mv.promo = pr_k as u8;
                    target_vec.push(mv);
                }
            } else {
                target_vec.push(mv);
            }
        }

        if stage.is_none() || stage == Some(0) {
            crate::movegen_special::gen_special_moves(pos, i, player, opp, &occ_local, &mut out);
        }
    }

    // Knightmare offboard moves (Sandbox-verified 2026-11-03): an offboard
    // Knightmare moves as a normal knight across the 14x14 staging board.
    // A knight jump landing on an 8x8 square is a DROP (capture if an
    // enemy stands there, quiet if empty; own-occupied squares are never
    // offered). A jump landing back in the staging ring is a HOP to a new
    // offboard square. There is NO own-half restriction (a drop deep in
    // the enemy half is legal), every hop/drop costs a turn, and there is
    // no one-use limit. This REPLACES the old own-half, empty-only drop
    // list, which contradicted all three Sandbox verdicts.
    if stage.is_none() || stage == Some(1) {
        let drop_out = if stage.is_none() {
            &mut quiets
        } else {
            &mut out
        };
        for ob in &pos.offboard {
            if ob.player != player {
                continue;
            }
            if ob.key != PieceKey::Knightmare && ob.sub_key != PieceKey::Knightmare {
                continue;
            }
            for &(dx, dy) in &LEAPS_KNIGHT {
                let lx = ob.ox + dx;
                let ly = ob.oy + dy;
                if lx >= 0 && lx < 8 && ly >= 0 && ly < 8 {
                    let s = sq(lx, ly);
                    match pos.at[s as usize] {
                        0 => {
                            drop_out.push(Move {
                                from: SQ_NONE,
                                to: s,
                                flags: MF_KNIGHTMARE_DROP,
                                offboard_id: ob.id,
                                ..Default::default()
                            });
                        }
                        v => {
                            let tgt = &pos.pieces[(v - 1) as usize];
                            if tgt.alive
                                && tgt.player != player
                                && tgt.key != PieceKey::King
                                && tgt.sub_key != PieceKey::King
                            {
                                drop_out.push(Move {
                                    from: SQ_NONE,
                                    to: s,
                                    flags: MF_KNIGHTMARE_DROP | MF_CAPTURE,
                                    offboard_id: ob.id,
                                    ..Default::default()
                                });
                            }
                            // Own-occupied: never offered.
                        }
                    }
                } else if off_in_ring(lx, ly) {
                    if pos.offboard.iter().any(|o| o.ox == lx && o.oy == ly) {
                        continue;
                    }
                    drop_out.push(Move {
                        from: SQ_NONE,
                        to: off_code(lx, ly),
                        flags: MF_KNIGHTMARE_HOP,
                        offboard_id: ob.id,
                        ..Default::default()
                    });
                }
            }
        }
    }

    if stage.is_none() || stage == Some(1) {
        let castle_out = if stage.is_none() {
            &mut quiets
        } else {
            &mut out
        };
        crate::movegen_castle::castle_moves(pos, player, opp, &occ_local, castle_out);
    }

    if stage.is_none() && !quiets.is_empty() {
        out.append(&mut quiets);
    }
    expand_pilgrim_resurrects(pos, player, &mut out);
    out
}

/// Wiki Pilgrim resurrect (pixiewiki Pilgrim): a CHARGED Pilgrim move
/// resurrects one random dead ally onto the Pilgrim's origin square.
/// The target is random in the real game, so movegen emits one move per
/// eligible dead ally (same from/to, different special_a = dead slot)
/// and think averages them by (from, to) into the expected value — the
/// same pattern as ElectroKnight lightning. With no dead allies the
/// move stays plain (no flag). Resurrect is automatic, not optional,
/// so the plain move is NOT also emitted when dead allies exist.
/// Eligible: dead (alive=false), same side, not a King, not in limbo
/// (a limbo Djinn revives on its own) and not offboard (a Knightmare
/// offboard is not dead). Charged = pilgrim_charged bit or odo >= 20,
/// OR the move itself taking odo to >= 20: Sandbox-verified (2026-11-03)
/// the resurrection fires ON the move that reaches 20 cumulative squares,
/// not only on later moves by an already-charged Pilgrim.
fn expand_pilgrim_resurrects(pos: &Pos, player: u8, out: &mut Vec<Move>) {
    // Fast exit: avoid 2 heap allocations + full move-list drain on every node
    // when `player` has no alive Pilgrim capable of reaching odo >= 20 this move
    // (max Chebyshev travel on an 8x8 board is 7 squares, so odo must be >= 13).
    let has_ready_pilgrim = pos.pieces.iter().any(|p| {
        p.alive
            && p.player == player
            && xy_on_board(p.x, p.y)
            && (p.key == PieceKey::Pilgrim || p.sub_key == PieceKey::Pilgrim)
            && (p.pilgrim_charged() || p.odo >= 13)
    });
    if !has_ready_pilgrim {
        return;
    }

    let dead: Vec<usize> = pos
        .pieces
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            !d.alive
                && d.player == player
                && d.key != PieceKey::King
                && d.sub_key != PieceKey::King
                && !pos.limbo.iter().any(|l| l.id == d.id)
                && !pos.offboard.iter().any(|o| o.id == d.id)
        })
        .map(|(i, _)| i)
        .filter(|&i| i < 256)
        .collect();
    if dead.is_empty() {
        return;
    }
    let mut expanded: Vec<Move> = Vec::with_capacity(out.len());
    for mv in out.drain(..) {
        let mover = if mv.from < 64 {
            slot_of(pos, mv.from).map(|s| &pos.pieces[s])
        } else {
            None
        };
        let is_pilgrim = mover.map_or(false, |p| {
            p.key == PieceKey::Pilgrim || p.sub_key == PieceKey::Pilgrim
        });
        let charged = mover.map_or(false, |p| {
            if p.pilgrim_charged() || p.odo >= 20 {
                return true;
            }
            // Threshold-crossing move: odo before the move plus this
            // move's travel (Chebyshev distance, same as apply_full's
            // odo update) reaching 20 resurrects on this very move.
            if mv.from < 64 && mv.to < 64 {
                let dist = (sq_y(mv.to) as i8 - sq_y(mv.from) as i8)
                    .abs()
                    .max((sq_x(mv.to) as i8 - sq_x(mv.from) as i8).abs())
                    as u8;
                return p.odo.saturating_add(dist) >= 20;
            }
            false
        });
        if is_pilgrim && charged {
            for &slot in dead.iter() {
                let mut rm = mv;
                rm.flags |= MF_PILGRIM_RESURRECT;
                rm.special_a = slot as u8;
                expanded.push(rm);
            }
        } else {
            expanded.push(mv);
        }
    }
    *out = expanded;
}

fn filter_av_illegal(pos: &Pos, moves: &mut Vec<Move>, player: u8) {
    let opp = 1 - player;
    let mut i = 0;
    while i < moves.len() {
        let mv = moves[i];
        let is_av = if mv.from != SQ_NONE {
            slot_of(pos, mv.from).map_or(false, |s| {
                let p = &pos.pieces[s];
                p.key == PieceKey::AntiViolence || p.sub_key == PieceKey::AntiViolence
            })
        } else {
            false
        };
        if is_av {
            let bad = match crate::apply_full::apply_full(pos, &mv, player, opp) {
                Some(p2) => king_attacked(&p2, player, opp, None),
                None => true,
            };
            if bad {
                moves.remove(i);
                continue;
            }
        }
        i += 1;
    }
}

/// `BladeRunner` (`bishop+pass_through_delayed_kill`):
/// Slides along diagonals to empty squares, passing through enemy non-King,
/// non-invulnerable pieces (which `apply_full.rs` marks as `doomed` for delayed kill),
/// while stopping at allied pieces, enemy Kings, or invulnerable `IronPawn`s.
#[inline]
fn blade_runner_dests(p: &Piece, pos: &Pos, occ_local: &[bool; 64]) -> Bitboard {
    let mut mask = 0u64;
    let x = p.x as i8;
    let y = p.y as i8;
    let p_idx = slot_of(pos, sq(x, y));
    let suppressed = p_idx.map_or(false, |idx| av_suppressed(pos, idx));
    for &(dx, dy) in &DIRS_BISHOP {
        let mut nx = x + dx;
        let mut ny = y + dy;
        while on_board_fast(nx, ny) {
            let s = sq(nx, ny);
            if occ_local[s as usize] {
                if suppressed {
                    break;
                }
                let v_idx = pos.at[s as usize];
                if v_idx == 0 {
                    break;
                }
                let v = &pos.pieces[(v_idx - 1) as usize];
                let v_iron = PROVISIONAL_IRON_INVULN
                    && (v.key == PieceKey::IronPawn || v.sub_key == PieceKey::IronPawn);
                if !v.alive || v.player == p.player || is_king_target(v) || v_iron {
                    break;
                }
                // Pass through enemy non-King, non-invulnerable piece without stopping on `s`.
                nx += dx;
                ny += dy;
                continue;
            }
            mask |= 1u64 << s;
            nx += dx;
            ny += dy;
        }
    }
    Bitboard(mask)
}

fn gen_pawn_moves(
    pos: &Pos,
    p_idx: usize,
    pk: PieceKey,
    sk: PieceKey,
    player: u8,
    opp: u8,
    occ_local: &[bool; 64],
    stage: Option<u8>,
    out: &mut Vec<Move>,
) {
    let p = &pos.pieces[p_idx];
    let x = p.x as i8;
    let y = p.y as i8;
    let d: i8 = if player == 0 { 1 } else { -1 };
    let start_rank: i8 = if player == 0 { 1 } else { 6 };
    let last_rank: i8 = if player == 0 { 7 } else { 0 };
    let is_iron = pk == PieceKey::IronPawn || sk == PieceKey::IronPawn;
    let is_shrike = pk == PieceKey::Shrike || sk == PieceKey::Shrike;
    let is_warp = pk == PieceKey::WarpJumper || sk == PieceKey::WarpJumper;
    let is_epee = pk == PieceKey::EpeePawn || sk == PieceKey::EpeePawn;
    let from_sq = sq(x, y);
    let aristocrat_blocks_promo = blocked_by_enemy_aristocrat(pos, opp, None);

    let ny = y + d;
    if on_board_fast(x, ny) {
        let to = sq(x, ny);
        if pos.at[to as usize] == 0 {
            let promo_push = ny == last_rank && !is_iron && !aristocrat_blocks_promo;
            let emit = match stage {
                None => true,
                Some(0) => promo_push,
                Some(1) => !promo_push,
                _ => false,
            };
            if emit {
                let mut mv = Move {
                    from: from_sq,
                    to,
                    ..Default::default()
                };
                if promo_push {
                    mv.flags |= MF_PROMO;
                    for &pr_k in &[
                        PieceKey::Queen,
                        PieceKey::Rook,
                        PieceKey::Bishop,
                        PieceKey::Knight,
                    ] {
                        mv.promo = pr_k as u8;
                        out.push(mv);
                    }
                } else {
                    out.push(mv);
                }
            }
        }
    }

    if y == start_rank {
        let ny2 = y + 2 * d;
        if on_board_fast(x, ny2) {
            let mid = sq(x, y + d);
            let to = sq(x, ny2);
            if pos.at[mid as usize] == 0
                && pos.at[to as usize] == 0
                && (stage.is_none() || stage == Some(1))
            {
                let mut mv = Move {
                    from: from_sq,
                    to,
                    ..Default::default()
                };
                if is_shrike {
                    mv.flags |= MF_SHRIKE_FIRST;
                }
                out.push(mv);
            }
        }
    }

    // Shrike (`pawn+capture_forward_2_first_move`):
    // On its first move, can advance 2 squares forward capturing an enemy piece on s1 and/or s2,
    // provided neither s1 nor s2 is blocked by a friendly piece, King, or invulnerable IronPawn.
    if is_shrike
        && (y == start_rank || !p.moved())
        && (stage.is_none() || stage == Some(0))
        && !av_suppressed(pos, p_idx)
    {
        let ny2 = y + 2 * d;
        if on_board_fast(x, ny2) && on_board_fast(x, y + d) {
            let s1 = sq(x, y + d);
            let s2 = sq(x, ny2);
            let t1 = pos.at[s1 as usize];
            let t2 = pos.at[s2 as usize];
            let t1_enemy = if t1 != 0 {
                let v1 = &pos.pieces[(t1 - 1) as usize];
                let v1_iron = PROVISIONAL_IRON_INVULN
                    && (v1.key == PieceKey::IronPawn || v1.sub_key == PieceKey::IronPawn);
                v1.alive && v1.player == opp && !is_king_target(v1) && !v1_iron
            } else {
                false
            };
            let t2_enemy = if t2 != 0 {
                let v2 = &pos.pieces[(t2 - 1) as usize];
                let v2_iron = PROVISIONAL_IRON_INVULN
                    && (v2.key == PieceKey::IronPawn || v2.sub_key == PieceKey::IronPawn);
                v2.alive && v2.player == opp && !is_king_target(v2) && !v2_iron
            } else {
                false
            };
            if (t1 == 0 || t1_enemy) && (t2 == 0 || t2_enemy) && (t1_enemy || t2_enemy) {
                let mut mv = Move {
                    from: from_sq,
                    to: s2,
                    ..Default::default()
                };
                mv.flags |= MF_SHRIKE_FIRST | MF_CAPTURE;
                out.push(mv);
            }
        }
    }

    // WarpJumper (`pawn+jump_through_pawns`):
    // Can jump over a blocking pawn (friendly or enemy) directly in front of it on any rank
    // to land on the empty square two ranks ahead (promoting if landing on `last_rank`).
    if is_warp && on_board_fast(x, ny) {
        let t = pos.at[sq(x, ny) as usize];
        let ny2 = y + 2 * d;
        if t != 0 && on_board_fast(x, ny2) && pos.at[sq(x, ny2) as usize] == 0 {
            let blocker = &pos.pieces[(t - 1) as usize];
            if blocker.alive && (is_pawn_key(blocker.key) || is_pawn_key(blocker.sub_key)) {
                let to = sq(x, ny2);
                let promo_jump = ny2 == last_rank && !is_iron && !aristocrat_blocks_promo;
                let emit = match stage {
                    None => true,
                    Some(0) => promo_jump,
                    Some(1) => !promo_jump,
                    _ => false,
                };
                if emit {
                    let mut mv = Move {
                        from: from_sq,
                        to,
                        ..Default::default()
                    };
                    if promo_jump {
                        mv.flags |= MF_PROMO;
                        for &pr_k in &[
                            PieceKey::Queen,
                            PieceKey::Rook,
                            PieceKey::Bishop,
                            PieceKey::Knight,
                        ] {
                            mv.promo = pr_k as u8;
                            out.push(mv);
                        }
                    } else {
                        out.push(mv);
                    }
                }
            }
        }
    }

    // En passant (standard adjacent EP + EpeePawn `pawn+global_en_passant`)
    if let Some(ep) = &pos.ep {
        if !is_iron
            && (stage.is_none() || stage == Some(0))
            && !av_suppressed(pos, p_idx)
        {
            let ep_x = ep.x as i8;
            let ep_y = ep.y as i8;
            if on_board_fast(ep_x, ep_y) && on_board_fast(ep.pawn_x, ep.pawn_y) {
                let ep_to = sq(ep_x, ep_y);
                let victim_sq = sq(ep.pawn_x, ep.pawn_y);
                let v_idx = pos.at[victim_sq as usize];
                if v_idx != 0 && pos.at[ep_to as usize] == 0 {
                    let v = &pos.pieces[(v_idx - 1) as usize];
                    let v_iron = PROVISIONAL_IRON_INVULN
                        && (v.key == PieceKey::IronPawn || v.sub_key == PieceKey::IronPawn);
                    if v.alive
                        && v.player == opp
                        && (is_pawn_key(v.key) || is_pawn_key(v.sub_key))
                        && !v_iron
                    {
                        let is_std_ep = (x - ep_x).abs() == 1 && y + d == ep_y;
                        if is_std_ep {
                            let mut mv = Move {
                                from: from_sq,
                                to: ep_to,
                                ep_victim: victim_sq,
                                ..Default::default()
                            };
                            mv.flags |= MF_EP | MF_CAPTURE;
                            out.push(mv);
                        } else if is_epee {
                            let mut mv = Move {
                                from: from_sq,
                                to: ep_to,
                                ep_victim: victim_sq,
                                ..Default::default()
                            };
                            mv.flags |= MF_EP | MF_CAPTURE | MF_EPEE_TELEPORT;
                            out.push(mv);
                        }
                    }
                }
            }
        }
    }
    let _ = occ_local;
}
// ── PhaseRook regression tests ─────────────────────────────────────────────
// First test module ever added to this crate (previously zero #[test]s).
// Rule under test: a PhaseRook ray may pass through AT MOST ONE friendly
// piece; the second friendly stops the ray; friendly squares are never
// generated. Direct enemy captures stop the ray normally. The tVmhFr log
// repeatedly shows b1->b6 (through friendly b2 to capture the Queen) missing
// from the server candidate list, so capture-after-phasing is now suppressed.
// CAVEAT: those turns also show model drift (missed/phantom pieces), so this
// last rule is a conservative compatibility hypothesis, not yet confirmed by
// a clean, fully synchronized replay.
#[cfg(test)]
mod phaserook_regression_tests {
    use super::phase_rook_into;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, Bitboard, PieceKey};

    /// Build a position from (key, player, x, y) pieces plus its occupancy
    /// array, with `pos.at` consistent with `pos.pieces` (slot index + 1).
    fn build_pos(spec: &[(PieceKey, u8, u8, u8)]) -> (Pos, [bool; 64]) {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        let mut occ = [false; 64];
        for (i, &(key, player, x, y)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32,
                key,
                sub_key: PieceKey::None,
                player,
                x,
                y,
                ..Default::default()
            });
            let s = sq(x as i8, y as i8) as usize;
            pos.at[s] = (i + 1) as u16;
            occ[s] = true;
        }
        (pos, occ)
    }

    fn rook_targets(pos: &Pos, occ: &[bool; 64]) -> Bitboard {
        let p = pos
            .pieces
            .iter()
            .find(|p| p.key == PieceKey::PhaseRook)
            .expect("test position must contain a PhaseRook");
        let mut bb = Bitboard(0);
        phase_rook_into(&mut bb, p, pos, occ);
        bb
    }

    #[test]
    fn empty_board_full_rays() {
        let (pos, occ) = build_pos(&[(PieceKey::PhaseRook, 0, 0, 0)]);
        let bb = rook_targets(&pos, &occ);
        // a-file a2..a8 + rank b1..h1 = 14 squares.
        for y in 1..8 {
            assert!(bb.has(sq(0, y)), "a{} must be generated", y + 1);
        }
        for x in 1..8 {
            assert!(bb.has(sq(x, 0)), "{}1 must be generated", (b'a' + x as u8) as char);
        }
        assert_eq!(bb.0.count_ones(), 14, "empty-board rook must have 14 targets");
        assert!(!bb.has(sq(0, 0)), "origin square must not be generated");
    }

    #[test]
    fn one_friend_passed() {
        // KZtmMe A1 shape: Pha1 with friendly pawn on a2.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 0),
            (PieceKey::Pawn, 0, 0, 1),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(!bb.has(sq(0, 1)), "friendly a2 must never be generated");
        for y in 2..8 {
            assert!(bb.has(sq(0, y)), "a{} must be generated past one friend", y + 1);
        }
    }

    #[test]
    fn two_friends_stop_ray() {
        // THE v137 regression test: v137 generates a5..a8 here; fixed code stops.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 0),
            (PieceKey::Pawn, 0, 0, 1),
            (PieceKey::Pawn, 0, 0, 3),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(bb.has(sq(0, 2)), "a3 between the two friends must be generated");
        assert!(!bb.has(sq(0, 1)), "friendly a2 must never be generated");
        assert!(!bb.has(sq(0, 3)), "friendly a4 must never be generated");
        for y in 4..8 {
            assert!(!bb.has(sq(0, y)), "a{} past the second friend must stop", y + 1);
        }
    }

    #[test]
    fn four_friends_shipped_case() {
        // 3ktEV7 R3 shape: Black Pha8 with friends b8,c8,d8,e8.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 1, 0, 7),
            (PieceKey::Knight, 1, 1, 7),
            (PieceKey::Pilgrim, 1, 2, 7),
            (PieceKey::Queen, 1, 3, 7),
            (PieceKey::King, 1, 4, 7),
        ]);
        let bb = rook_targets(&pos, &occ);
        for x in 1..8 {
            assert!(!bb.has(sq(x, 7)), "rank-8 square {} must not be generated", x);
        }
        for y in 0..7 {
            assert!(bb.has(sq(0, y)), "a{} south must be generated", y + 1);
        }
    }

    #[test]
    fn friendly_destination_never_generated() {
        // 3ktEV7 R1/R2 core: Pha8, friend on c8, b8 empty.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 1, 0, 7),
            (PieceKey::Pilgrim, 1, 2, 7),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(bb.has(sq(1, 7)), "empty b8 before the friend must be generated");
        assert!(!bb.has(sq(2, 7)), "friendly c8 must never be generated");
        for x in 3..8 {
            assert!(bb.has(sq(x, 7)), "square past exactly one friend must be generated");
        }
    }

    #[test]
    fn enemy_capture_stops_ray() {
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 0),
            (PieceKey::Pawn, 1, 0, 4),
        ]);
        let bb = rook_targets(&pos, &occ);
        for y in 1..4 {
            assert!(bb.has(sq(0, y)), "a{} before the enemy must be generated", y + 1);
        }
        assert!(bb.has(sq(0, 4)), "enemy a5 must be generated as a capture");
        for y in 5..8 {
            assert!(!bb.has(sq(0, y)), "a{} past the capture must stop", y + 1);
        }
    }

    #[test]
    fn capture_beyond_one_friendly_phased_piece_is_not_generated() {
        // Reproduce the move shape rejected by the server in tVmhFr:
        // PhaseRook b1, friendly pawn b2, enemy Queen b6.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 1, 0),
            (PieceKey::Pawn, 0, 1, 1),
            (PieceKey::Queen, 1, 1, 5),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(!bb.has(sq(1, 1)), "friendly b2 must not be generated");
        assert!(bb.has(sq(1, 2)), "empty b3 past one friend remains reachable");
        assert!(bb.has(sq(1, 4)), "empty b5 remains reachable");
        assert!(!bb.has(sq(1, 5)), "enemy Queen b6 beyond phased friend must not be capturable");
    }

    #[test]
    fn enemy_on_first_square() {
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 0),
            (PieceKey::Pawn, 1, 0, 1),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(bb.has(sq(0, 1)), "adjacent enemy a2 must be capturable");
        for y in 2..8 {
            assert!(!bb.has(sq(0, y)), "a{} past the adjacent capture must stop", y + 1);
        }
    }

    #[test]
    fn per_ray_budget_resets() {
        // d4 rook; north has two friends (pass d5, stop d6), south one
        // friend (pass d2, reach d1), east an enemy (capture e4, stop),
        // west empty (full ray). Budgets are per ray, not per move.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 3, 3),
            (PieceKey::Pawn, 0, 3, 4),
            (PieceKey::Pawn, 0, 3, 5),
            (PieceKey::Pawn, 0, 3, 1),
            (PieceKey::Pawn, 1, 4, 3),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(!bb.has(sq(3, 4)) && !bb.has(sq(3, 5)), "north friends never generated");
        assert!(!bb.has(sq(3, 6)) && !bb.has(sq(3, 7)), "north past second friend stops");
        assert!(!bb.has(sq(3, 1)), "south friend never generated");
        assert!(bb.has(sq(3, 0)), "south past one friend (d1) must be generated");
        assert!(bb.has(sq(4, 3)), "east enemy e4 must be capturable");
        assert!(!bb.has(sq(5, 3)), "east past the capture must stop");
        for x in 0..3 {
            assert!(bb.has(sq(x, 3)), "west ray must be full");
        }
    }

    #[test]
    fn corner_edges_no_wraparound() {
        let (pos, occ) = build_pos(&[(PieceKey::PhaseRook, 0, 7, 7)]);
        let bb = rook_targets(&pos, &occ);
        assert_eq!(bb.0.count_ones(), 14, "corner rook must have exactly 14 targets");
        assert!(!bb.has(sq(0, 0)), "a1 must not leak in via wraparound");
        assert!(!bb.has(sq(7, 7)), "origin h8 must not be generated");
    }

    #[test]
    fn server_accepted_slides() {
        // 3ktEV7 A5: Pha1 + friendly knight b1 -> c1 generated, b1 not.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 0),
            (PieceKey::Knight, 0, 1, 0),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(!bb.has(sq(1, 0)), "friendly b1 must never be generated");
        assert!(bb.has(sq(2, 0)), "c1 past one friend must be generated");
        // KZtmMe A3: Pha4 + friendly pawn a2 -> a1 generated, a2 not.
        // Fresh position per case: generation is stateless across boards.
        let (pos, occ) = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 3),
            (PieceKey::Pawn, 0, 0, 1),
        ]);
        let bb = rook_targets(&pos, &occ);
        assert!(!bb.has(sq(0, 1)), "friendly a2 must never be generated");
        assert!(bb.has(sq(0, 0)), "a1 past one friend must be generated");
        assert!(bb.has(sq(0, 2)), "a3 must be generated");
    }
}

// ── Pixie power fix tests (2026-10-31 audit batch) ────────────────────────
#[cfg(test)]
mod pixie_power_fix_tests {
    use super::gen_moves;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, PieceKey, MF_CAPTURE, MF_DANCER_DOUBLE, MF_IS_PUSH, MF_PILGRIM_RESURRECT};

    fn build_pos(spec: &[(PieceKey, u8, u8, u8, u16)]) -> Pos {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        for (i, &(key, player, x, y, flags)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, state_flags: flags, ..Default::default()
            });
            pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        pos
    }

    #[test]
    fn dancer_double_never_captures() {
        // Dancer with encore pending, enemy on a bishop ray: doubles must
        // exist (quiet legs) but none may capture.
        let pos = build_pos(&[
            (PieceKey::Dancer, 0, 3, 3, 1 << 7),
            (PieceKey::Pawn, 1, 5, 5, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 0, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        let doubles: Vec<_> = moves
            .iter()
            .filter(|m| m.flags & MF_DANCER_DOUBLE != 0)
            .collect();
        assert!(!doubles.is_empty(), "a quiet Dancer double must exist");
        for d in doubles {
            assert_eq!(
                d.flags & MF_CAPTURE, 0,
                "Dancer double must not capture ({}->{})", d.from, d.to
            );
        }
    }

    #[test]
    fn knife_two_diagonal_only_toward_centre() {
        // Knife at (1,1) capturing to (3,3) moves toward the centre: allowed.
        // Knife at (5,5) capturing to (7,7) moves away: forbidden.
        let pos = build_pos(&[
            (PieceKey::PawnWithKnife, 0, 1, 1, 0),
            (PieceKey::Pawn, 1, 3, 3, 0),
            (PieceKey::PawnWithKnife, 0, 5, 5, 0),
            (PieceKey::Pawn, 1, 7, 7, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 0, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        assert!(
            moves.iter().any(|m| m.from == sq(1, 1) && m.to == sq(3, 3)),
            "knife 2-diagonal toward centre must be generated"
        );
        assert!(
            !moves.iter().any(|m| m.from == sq(5, 5) && m.to == sq(7, 7)),
            "knife 2-diagonal away from centre must NOT be generated"
        );
    }

    #[test]
    fn sumo_ram_push_to_before_target() {
        // Sumo a4, enemy pawn d4, gap b4/c4: the ram is a4 -> c4 with a
        // push east, never a move onto d4; b4 stays a quiet rook move.
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 0, 3, 0),
            (PieceKey::Pawn, 1, 3, 3, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        let push = moves.iter().find(|m| m.from == sq(0, 3) && m.to == sq(2, 3));
        let push = push.expect("ram to c4 must exist");
        assert_ne!(push.flags & MF_IS_PUSH, 0, "c4 move must be flagged as a push");
        assert_eq!((push.push_dx, push.push_dy), (1, 0));
        assert!(moves.iter().all(|m| !(m.from == sq(0, 3) && m.to == sq(3, 3))),
            "Sumo must never move onto the target square");
        let quiet = moves.iter().find(|m| m.from == sq(0, 3) && m.to == sq(1, 3))
            .expect("b4 must stay a quiet rook move");
        assert_eq!(quiet.flags & MF_IS_PUSH, 0);
    }

    #[test]
    fn sumo_adjacent_target_cannot_be_pushed() {
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 0, 3, 0),
            (PieceKey::Pawn, 1, 1, 3, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        assert!(moves.iter().all(|m| !(m.from == sq(0, 3) && m.to == sq(1, 3))),
            "adjacent target: no move onto it, and no ram from the origin");
        assert!(moves.iter().all(|m| m.flags & MF_IS_PUSH == 0),
            "no push moves at all with an adjacent target");
    }

    #[test]
    fn sumo_three_piece_line_holds_ground() {
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 0, 3, 0),
            (PieceKey::Pawn, 1, 3, 3, 0),
            (PieceKey::Pawn, 1, 4, 3, 0),
            (PieceKey::Pawn, 1, 5, 3, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        let to_c4 = moves.iter().find(|m| m.from == sq(0, 3) && m.to == sq(2, 3))
            .expect("c4 must still be a legal rook square");
        assert_eq!(to_c4.flags & MF_IS_PUSH, 0, "3-piece line: no push, quiet only");
    }

    #[test]
    fn sumo_friendly_victim_pushes_but_own_king_does_not() {
        // Friendly pawn line pushes.
        let pos = build_pos(&[
            (PieceKey::SumoRook, 0, 0, 3, 0),
            (PieceKey::Pawn, 0, 3, 3, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        assert!(moves.iter().any(|m| m.from == sq(0, 3) && m.to == sq(2, 3)
            && m.flags & MF_IS_PUSH != 0), "friendly non-King victim must push");
        // Own King in the line: quiet only.
        let pos2 = build_pos(&[
            (PieceKey::SumoRook, 0, 0, 3, 0),
            (PieceKey::King, 0, 3, 3, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        let moves2 = gen_moves(&pos2, 0, None);
        assert!(moves2.iter().all(|m| !(m.from == sq(0, 3) && m.to == sq(2, 3)
            && m.flags & MF_IS_PUSH != 0)), "own King must never be pushed");
    }

    #[test]
    fn pilgrim_charged_move_expands_per_dead_ally() {
        let mut pos = build_pos(&[
            (PieceKey::Pilgrim, 0, 3, 3, 1 << 8),
            (PieceKey::Pawn, 0, 5, 5, 0),
            (PieceKey::Knight, 0, 2, 2, 0),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        // Kill the two allies (dead pieces leave the board: at[] cleared).
        for (kx, ky) in [(5i8, 5i8), (2, 2)] {
            let s = sq(kx, ky);
            pos.at[s as usize] = 0;
        }
        for p in pos.pieces.iter_mut() {
            if (p.key == PieceKey::Pawn || p.key == PieceKey::Knight) && p.player == 0 {
                p.alive = false;
            }
        }
        let moves = gen_moves(&pos, 0, None);
        let to_f5: Vec<_> = moves.iter()
            .filter(|m| m.from == sq(3, 3) && m.to == sq(5, 5))
            .collect();
        assert_eq!(to_f5.len(), 2, "one Pilgrim move per dead ally, no plain move");
        assert!(to_f5.iter().all(|m| m.flags & MF_PILGRIM_RESURRECT != 0));
        assert_ne!(to_f5[0].special_a, to_f5[1].special_a, "variants differ by resurrected ally");
    }

    #[test]
    fn pilgrim_without_dead_allies_stays_plain() {
        let pos = build_pos(&[
            (PieceKey::Pilgrim, 0, 3, 3, 1 << 8),
            (PieceKey::King, 0, 0, 7, 0),
            (PieceKey::King, 1, 7, 7, 0),
        ]);
        let moves = gen_moves(&pos, 0, None);
        let to_f5: Vec<_> = moves.iter()
            .filter(|m| m.from == sq(3, 3) && m.to == sq(5, 5))
            .collect();
        assert_eq!(to_f5.len(), 1);
        assert_eq!(to_f5[0].flags & MF_PILGRIM_RESURRECT, 0);
    }

    #[test]
    fn phase_rook_mismatched_occ_does_not_panic() {
        let pos = build_pos(&[
            (PieceKey::PhaseRook, 0, 0, 0, 0),
            (PieceKey::King, 0, 4, 0, 0),
            (PieceKey::King, 1, 4, 7, 0),
        ]);
        let mut occ = [false; 64];
        occ[sq(0, 0) as usize] = true;
        // Simulate stale occupancy on a2 with at[a2] == 0: must break ray cleanly without panic.
        occ[sq(0, 1) as usize] = true;
        let bb = super::piece_attacks(&pos, 0, &occ, true, false);
        assert!(!bb.has(sq(0, 1)), "phantom occupied square must stop PhaseRook ray without panicking");
    }
}
