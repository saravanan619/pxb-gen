use crate::pos::*;
use crate::types::*;

#[derive(Clone, Copy)]
pub enum JField {
    X, Y, Odo, Kills, Caps, Quiet, Frost, Encore,
    StateFlags, HordeId, Key, SubKey, HomeX, HomeY,
}

#[derive(Clone)]
pub enum JEntry {
    PieceU8  { i: u32, f: JField, old: u8 },
    PieceU16 { i: u32, f: JField, old: u16 },
    PieceU32 { i: u32, f: JField, old: u32 },
    PieceI8  { i: u32, f: JField, old: i8 },
    PieceBool{ i: u32, old: bool },
    AtU16    { sq: u8, old: u16 },
    PiecesTrunc  { old_len: u32 },
    LimboTrunc   { old_len: u32 },
    OffboardTrunc{ old_len: u32 },
    DoomedTrunc  { old_len: u32 },
    HistTrunc    { old_len: u32 },
    PosEp        { old: Option<EpInfo> },
    PosHalf      { old: u16 },
    PosStm       { old: u8 },
    PosEval      { old: i32 },
    PosKet       { old: u8 },
    PosQC        { old: u8 },
    PosKT        { old: Option<u8> },
    PosGW        { old: Option<u8> },
    PosLastCap   { player: u8, old: (PieceKey, PieceKey) },
}

#[derive(Default, Clone)]
pub struct Journal {
    pub v: Vec<JEntry>,
}

impl Journal {
    #[inline] pub fn mark(&self) -> usize { self.v.len() }

    pub fn unwind(&mut self, pos: &mut Pos, mark: usize) {
        while self.v.len() > mark {
            let e = self.v.pop().unwrap();
            match e {
                JEntry::PieceU8 { i, f, old }  => set_piece_u8 (pos, i, f, old),
                JEntry::PieceU16{ i, f, old }  => set_piece_u16(pos, i, f, old),
                JEntry::PieceU32{ i, f, old }  => set_piece_u32(pos, i, f, old),
                JEntry::PieceI8 { i, f, old }  => set_piece_i8 (pos, i, f, old),
                JEntry::PieceBool{ i, old }    => pos.pieces[i as usize].alive = old,
                JEntry::AtU16   { sq, old }    => pos.at[sq as usize] = old,
                JEntry::PiecesTrunc  { old_len } => pos.pieces.truncate(old_len as usize),
                JEntry::LimboTrunc   { old_len } => pos.limbo.truncate(old_len as usize),
                JEntry::OffboardTrunc{ old_len } => pos.offboard.truncate(old_len as usize),
                JEntry::DoomedTrunc  { old_len } => pos.doomed.truncate(old_len as usize),
                JEntry::HistTrunc    { old_len } => pos.hist_hashes.truncate(old_len as usize),
                JEntry::PosEp   { old }        => pos.ep = old,
                JEntry::PosHalf { old }        => pos.halfmove = old,
                JEntry::PosStm { old }         => pos.stm = old,
                JEntry::PosEval { old }        => pos.eval_sum = old,
                JEntry::PosKet  { old }        => pos.ket_x2 = old,
                JEntry::PosQC   { old }        => pos.queen_count = old,
                JEntry::PosKT   { old }        => pos.king_thrown = old,
                JEntry::PosGW   { old }        => pos.gold_win = old,
                JEntry::PosLastCap { player, old } => pos.last_cap[player as usize] = old,
            }
        }
    }

    #[inline]
    pub fn at_put(&mut self, pos: &mut Pos, s: u8, new: u16) {
        self.v.push(JEntry::AtU16 { sq: s, old: pos.at[s as usize] });
        pos.at[s as usize] = new;
    }

    #[inline]
    pub fn u8_put(&mut self, pos: &mut Pos, i: u32, f: JField, new: u8) {
        let old = get_piece_u8(pos, i, f);
        self.v.push(JEntry::PieceU8 { i, f, old });
        set_piece_u8(pos, i, f, new);
    }

    #[inline]
    pub fn u16_put(&mut self, pos: &mut Pos, i: u32, f: JField, new: u16) {
        let old = get_piece_u16(pos, i, f);
        self.v.push(JEntry::PieceU16 { i, f, old });
        set_piece_u16(pos, i, f, new);
    }

    #[inline]
    pub fn u32_put(&mut self, pos: &mut Pos, i: u32, f: JField, new: u32) {
        let old = get_piece_u32(pos, i, f);
        self.v.push(JEntry::PieceU32 { i, f, old });
        set_piece_u32(pos, i, f, new);
    }

    #[inline]
    pub fn alive_put(&mut self, pos: &mut Pos, i: u32, new: bool) {
        self.v.push(JEntry::PieceBool { i, old: pos.pieces[i as usize].alive });
        pos.pieces[i as usize].alive = new;
    }
}

#[inline]
pub fn get_piece_u8(pos: &Pos, i: u32, f: JField) -> u8 {
    let p = &pos.pieces[i as usize];
    match f {
        JField::X => p.x,
        JField::Y => p.y,
        JField::Odo => p.odo,
        JField::Kills => p.kills,
        JField::Caps => p.caps,
        JField::Quiet => p.quiet,
        JField::Frost => p.frost,
        JField::Encore => p.encore,
        _ => 0,
    }
}
#[inline]
pub fn set_piece_u8(pos: &mut Pos, i: u32, f: JField, v: u8) {
    let p = &mut pos.pieces[i as usize];
    match f {
        JField::X => p.x = v,
        JField::Y => p.y = v,
        JField::Odo => p.odo = v,
        JField::Kills => p.kills = v,
        JField::Caps => p.caps = v,
        JField::Quiet => p.quiet = v,
        JField::Frost => p.frost = v,
        JField::Encore => p.encore = v,
        _ => {}
    }
}
#[inline]
pub fn get_piece_u16(pos: &Pos, i: u32, f: JField) -> u16 {
    let p = &pos.pieces[i as usize];
    match f {
        JField::StateFlags => p.state_flags,
        JField::Key => p.key as u16,
        JField::SubKey => p.sub_key as u16,
        _ => 0,
    }
}
#[inline]
pub fn set_piece_u16(pos: &mut Pos, i: u32, f: JField, v: u16) {
    let p = &mut pos.pieces[i as usize];
    match f {
        JField::StateFlags => p.state_flags = v,
        JField::Key => p.key = PieceKey::from_u16(v),
        JField::SubKey => p.sub_key = PieceKey::from_u16(v),
        _ => {}
    }
}
#[inline]
pub fn get_piece_u32(pos: &Pos, i: u32, f: JField) -> u32 {
    let p = &pos.pieces[i as usize];
    match f {
        JField::HordeId => p.horde_id,
        _ => 0,
    }
}
#[inline]
pub fn set_piece_u32(pos: &mut Pos, i: u32, f: JField, v: u32) {
    let p = &mut pos.pieces[i as usize];
    match f {
        JField::HordeId => p.horde_id = v,
        _ => {}
    }
}
#[inline]
pub fn get_piece_i8(pos: &Pos, i: u32, f: JField) -> i8 {
    let p = &pos.pieces[i as usize];
    match f {
        JField::HomeX => p.home_x,
        JField::HomeY => p.home_y,
        _ => 0,
    }
}
#[inline]
pub fn set_piece_i8(pos: &mut Pos, i: u32, f: JField, v: i8) {
    let p = &mut pos.pieces[i as usize];
    match f {
        JField::HomeX => p.home_x = v,
        JField::HomeY => p.home_y = v,
        _ => {}
    }
}
