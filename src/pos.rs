use crate::types::*;

pub const MAX_ABILITIES: usize = 8;
pub const MAX_HIST: usize = 1024;

#[derive(Clone, Copy, Default)]
pub struct Ability {
    pub key: u8,
    pub range: u8,
}

#[derive(Clone, Default)]
pub struct Piece {
    pub id: u32,
    pub key: PieceKey,
    pub sub_key: PieceKey,
    pub player: u8,
    pub x: u8,
    pub y: u8,
    pub state_flags: u16,
    pub odo: u8,
    pub kills: u8,
    pub caps: u8,
    pub quiet: u8,
    pub frost: u8,
    pub encore: u8,
    pub horde_id: u32,
    pub home_x: i8,
    pub home_y: i8,
    pub n_abilities: u8,
    pub abilities: [Ability; MAX_ABILITIES],
    pub alive: bool,
}

impl Piece {
    #[inline] pub fn crouched(&self)        -> bool { self.state_flags & (1 << 0) != 0 }
    #[inline] pub fn echarged(&self)        -> bool { self.state_flags & (1 << 1) != 0 }
    #[inline] pub fn moved(&self)           -> bool { self.state_flags & (1 << 2) != 0 }
    #[inline] pub fn moved_last(&self)      -> bool { self.state_flags & (1 << 3) != 0 }
    #[inline] pub fn frozen(&self)          -> bool { self.state_flags & (1 << 4) != 0 }
    #[inline] pub fn used(&self)            -> bool { self.state_flags & (1 << 5) != 0 }
    #[inline] pub fn dissipated(&self)      -> bool { self.state_flags & (1 << 6) != 0 }
    #[inline] pub fn encore_pending(&self)  -> bool { self.state_flags & (1 << 7) != 0 }
    #[inline] pub fn pilgrim_charged(&self) -> bool { self.state_flags & (1 << 8) != 0 }
    #[inline] pub fn charged(&self)         -> bool { self.state_flags & (1 << 9) != 0 }
    // Sandbox-verified Gunslinger standoff timer (2026-11-03): set once a
    // mutual threat has survived a full round (see apply_triggers).
    #[inline] pub fn standoff_armed(&self)  -> bool { self.state_flags & (1 << 10) != 0 }
}

#[derive(Clone, Copy, Default)]
pub struct EpInfo {
    pub x: i8,
    pub y: i8,
    pub pawn_x: i8,
    pub pawn_y: i8,
}

#[derive(Clone, Copy, Default)]
pub struct DoomedEntry {
    pub id: u32,
    pub by: u8,
}

#[derive(Clone, Default)]
pub struct LimboPiece {
    pub id: u32,
    pub key: PieceKey,
    pub sub_key: PieceKey,
    pub player: u8,
    pub home_x: i8,
    pub home_y: i8,
}

#[derive(Clone, Default)]
pub struct OffboardPiece {
    pub id: u32,
    pub key: PieceKey,
    pub sub_key: PieceKey,
    pub player: u8,
    pub off_since: u8,
    // 14x14 staging coords (-3..=10). Legacy buffers decode these as 0,0;
    // no such buffer exists in this workflow, and movegen never trusts a
    // decoded offboard square without a knight-reachability check.
    pub ox: i8,
    pub oy: i8,
}

#[derive(Clone)]
pub struct Pos {
    pub at: [u16; 64],          // slot index + 1; 0 = empty
    pub pieces: Vec<Piece>,

    pub limbo: Vec<LimboPiece>,
    pub offboard: Vec<OffboardPiece>,
    pub doomed: Vec<DoomedEntry>,

    pub hist_hashes: Vec<u32>,
    pub ep: Option<EpInfo>,
    pub halfmove: u16,

    pub eval_sum: i32,
    pub ket_x2: u8,             // 0=endgame, 1=half, 2=midgame
    pub queen_count: u8,

    pub king_thrown: Option<u8>,
    pub gold_win: Option<u8>,

    pub last_cap: [(PieceKey, PieceKey); 2],
    pub horde_seq: u32,
    pub id_seq: u32,

    pub stm: u8,                // side to move
}

#[inline]
pub fn slot_of(pos: &Pos, sq: u8) -> Option<usize> {
    if sq >= 64 { return None; } // v130: guard garbage squares
    let s = pos.at[sq as usize];
    if s == 0 { None } else { Some((s - 1) as usize) }
}

#[inline]
pub fn piece_at(pos: &Pos, sq: u8) -> Option<&Piece> {
    slot_of(pos, sq).map(|i| &pos.pieces[i])
}

#[inline]
pub fn put_piece_at(pos: &mut Pos, slot: usize, s: u8) {
    pos.pieces[slot].x = sq_x(s) as u8;
    pos.pieces[slot].y = sq_y(s) as u8;
    pos.at[s as usize] = (slot + 1) as u16;
}

#[inline]
pub fn clear_square(pos: &mut Pos, s: u8) {
    pos.at[s as usize] = 0;
}

pub fn occ(pos: &Pos) -> [bool; 64] {
    let mut o = [false; 64];
    for s in 0..64usize {
        let v = pos.at[s];
        if v != 0 && pos.pieces[(v - 1) as usize].alive {
            o[s] = true;
        }
    }
    o
}

/// Lever-3: occupancy after a journaled move, derived from the parent
/// occupancy instead of a fresh 64-square rescan. Exact on the journaled
/// path ONLY: `make_journaled` moves one alive piece from `mv.from` to
/// `mv.to` and changes nothing else `occ()` can observe (no specials,
/// no spawns, no doomed pieces; a victim sits on `mv.to` itself and
/// that square ends occupied by the mover). Call sites use this only
/// where `make_journaled` returned true; the clone-based `apply_full`
/// branches keep computing `occ()` fresh.
#[inline]
pub fn occ_after_journaled(parent_occ: &[bool; 64], mv: &Move) -> [bool; 64] {
    let mut o = *parent_occ;
    if mv.from < 64 {
        o[mv.from as usize] = false;
    }
    if mv.to < 64 {
        o[mv.to as usize] = true;
    }
    o
}

pub fn alive_count(pos: &Pos) -> usize {
    pos.pieces.iter().filter(|p| p.alive).count()
}

// is_paralyzed: basilisk ray
pub fn is_paralyzed(pos: &Pos, piece_idx: usize, occ: &[bool; 64]) -> bool {
    let p = &pos.pieces[piece_idx];
    let px = p.x as i8;
    let py = p.y as i8;

    for (qi, q) in pos.pieces.iter().enumerate() {
            if !xy_on_board(q.x, q.y) { continue; } // v130
        if !q.alive || q.player == p.player { continue; }
        if q.key != PieceKey::Basilisk && q.sub_key != PieceKey::Basilisk { continue; }
        if PROVISIONAL_ICICLE_FREEZE && q.frozen() { continue; }
        let _ = qi;

        let dx = px - q.x as i8;
        let dy = py - q.y as i8;
        if dx == 0 || dx.abs() != dy.abs() { continue; }

        let sx = if dx > 0 { 1 } else { -1 };
        let sy = if dy > 0 { 1 } else { -1 };
        let mut bx = q.x as i8 + sx;
        let mut by = q.y as i8 + sy;
        let mut blocked = false;
        while bx != px || by != py {
            if bx < 0 || bx > 7 || by < 0 || by > 7 { break; }  // v131: guard Basilisk ray
            if occ[sq(bx, by) as usize] { blocked = true; break; }
            bx += sx; by += sy;
        }
        if !blocked { return true; }
    }
    false
}
impl Default for Pos {
    fn default() -> Self {
        Pos {
            at: [0u16; 64],
            pieces: Vec::new(),
            limbo: Vec::new(),
            offboard: Vec::new(),
            doomed: Vec::new(),
            hist_hashes: Vec::new(),
            ep: None,
            halfmove: 0,
            eval_sum: 0,
            ket_x2: 2,
            queen_count: 0,
            king_thrown: None,
            gold_win: None,
            last_cap: [(crate::types::PieceKey::None, crate::types::PieceKey::None); 2],
            horde_seq: 0,
            id_seq: 0,
            stm: 0,
        }
    }
}
