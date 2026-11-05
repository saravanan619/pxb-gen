use core::mem::size_of;

// ─── PieceKey ─────────────────────────────────────────────────────────────
// Discriminants are stable and append-only. Never renumber.

#[repr(u16)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PieceKey {
    #[default] None = 0,
    Pawn = 1, Knight = 2, Bishop = 3, Rook = 4, Queen = 5, King = 6,

    // Pawn-likes
    PawnWithKnife = 7, GoldenPawn = 8, HeroPawn = 9, EpeePawn = 10,
    Shrike = 11, IronPawn = 12, Hordeling = 13, TranslucentPawn = 14,
    Ooze = 15, Recruit = 16, WarAutomaton = 17, WarpJumper = 18,

    // Pixie pieces
    AntiViolence = 19, Aristocrat = 20, Banker = 21, Basilisk = 22,
    Bishopper = 23, BisHop = 24, BladeRunner = 25, Bouncer = 26,
    Camel = 27, Cardinal = 28, Dancer = 29, Djinn = 30,
    ElectroKnight = 31, Fish = 32, FissionReactor = 33, Gunslinger = 34,
    HordeMother = 35, Icicle = 36, Knightmare = 37, Marauder = 38,
    PhaseRook = 39, Pilgrim = 40, Rocketman = 41, SumoRook = 42,

    // Extended
    ExileBishop = 43, HordeKnight = 44, CloudKnight = 45, RookZero = 46,
    Pinata = 47, Blueprint = 48, Wraith = 49, VoidQueen = 50,
    NullQueen = 51, WitchQueen = 52, WarKing = 53, MeditatingKing = 54,
    PopeKing = 55, IrreverentKing = 56, Monument = 57, Mushroom = 58,
    LavaForge = 59, Timelord = 60, Druidess = 61, Dryad = 62,
    Duplicator = 63, Puppet = 64, Samurai = 65, Uni = 66, Dummy = 67,
}

impl PieceKey {
    #[inline]
    pub fn from_u16(v: u16) -> Self {
        use PieceKey::*;
        match v {
            1=>Pawn, 2=>Knight, 3=>Bishop, 4=>Rook, 5=>Queen, 6=>King,
            7=>PawnWithKnife, 8=>GoldenPawn, 9=>HeroPawn, 10=>EpeePawn,
            11=>Shrike, 12=>IronPawn, 13=>Hordeling, 14=>TranslucentPawn,
            15=>Ooze, 16=>Recruit, 17=>WarAutomaton, 18=>WarpJumper,
            19=>AntiViolence, 20=>Aristocrat, 21=>Banker, 22=>Basilisk,
            23=>Bishopper, 24=>BisHop, 25=>BladeRunner, 26=>Bouncer,
            27=>Camel, 28=>Cardinal, 29=>Dancer, 30=>Djinn,
            31=>ElectroKnight, 32=>Fish, 33=>FissionReactor, 34=>Gunslinger,
            35=>HordeMother, 36=>Icicle, 37=>Knightmare, 38=>Marauder,
            39=>PhaseRook, 40=>Pilgrim, 41=>Rocketman, 42=>SumoRook,
            43=>ExileBishop, 44=>HordeKnight, 45=>CloudKnight, 46=>RookZero,
            47=>Pinata, 48=>Blueprint, 49=>Wraith, 50=>VoidQueen,
            51=>NullQueen, 52=>WitchQueen, 53=>WarKing, 54=>MeditatingKing,
            55=>PopeKing, 56=>IrreverentKing, 57=>Monument, 58=>Mushroom,
            59=>LavaForge, 60=>Timelord, 61=>Druidess, 62=>Dryad,
            63=>Duplicator, 64=>Puppet, 65=>Samurai, 66=>Uni, 67=>Dummy,
            _=>None,
        }
    }
}

#[inline]
pub fn is_pawn_like(k: PieceKey) -> bool {
    use PieceKey::*;
    matches!(k,
        Pawn | PawnWithKnife | GoldenPawn | HeroPawn | EpeePawn
        | Shrike | IronPawn | Hordeling | TranslucentPawn | Ooze
        | Recruit | WarAutomaton | WarpJumper)
}

#[inline]
pub fn is_non_capturer(k: PieceKey) -> bool {
    use PieceKey::*;
    matches!(k, AntiViolence | IronPawn | Basilisk | Icicle)
}

// ─── Bitboard ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    #[inline] pub fn has(self, s: u8)     -> bool { if s >= 64 { return false; } (self.0 >> s) & 1 != 0 }
    #[inline] pub fn set(&mut self, s: u8)       { if s < 64 { self.0 |= 1u64 << s; } }
    #[inline] pub fn clear(&mut self, s: u8)     { if s < 64 { self.0 &= !(1u64 << s); } }
    #[inline] pub fn is_empty(self)       -> bool { self.0 == 0 }
    #[inline] pub fn pop_lsb(&mut self)   -> u8 {
        let s = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        s
    }
}

#[inline] pub fn on_board(x: i8, y: i8) -> bool { (x as u32) < 8 && (y as u32) < 8 }
#[inline] pub fn sq(x: i8, y: i8) -> u8         { ((y as u8) << 3) | (x as u8) }
#[inline] pub fn sq_x(s: u8) -> u8              { s & 7 }
#[inline] pub fn sq_y(s: u8) -> u8              { s >> 3 }
// v130: i8-correct checked square. (x as u8) >= 8 rejects -1 (255), 8, etc.
#[inline] pub fn sq_checked(x: i8, y: i8) -> Option<u8> {
    if (x as u8) >= 8 || (y as u8) >= 8 { None } else { Some(((y as u8) << 3) | (x as u8)) }
}
// v130: piece-coord validity for u8-stored coords. Behavior-neutral for 0..7.
#[inline] pub fn xy_on_board(x: u8, y: u8) -> bool { x < 8 && y < 8 }

pub const SQ_NONE: u8 = 0xFF;

// ─── Move flags ───────────────────────────────────────────────────────────

pub const MF_CAPTURE: u32         = 1 << 0;
pub const MF_EP: u32              = 1 << 1;
pub const MF_PROMO: u32           = 1 << 2;
pub const MF_CASTLE_K: u32        = 1 << 3;
pub const MF_CASTLE_Q: u32        = 1 << 4;
pub const MF_IS_PUSH: u32         = 1 << 5;
pub const MF_ROCKET_BLAST: u32    = 1 << 6;
pub const MF_HORDE_SPAWN: u32     = 1 << 7;
pub const MF_ELECTRO: u32         = 1 << 8;
pub const MF_GUN_STANDOFF: u32    = 1 << 9;
pub const MF_RECRUIT_SAC: u32     = 1 << 10;
pub const MF_DANCER_DOUBLE: u32   = 1 << 11;
pub const MF_DJINN_DISSIPATE: u32 = 1 << 12;
pub const MF_KNIGHTMARE_OFF: u32  = 1 << 13;
pub const MF_KNIGHTMARE_DROP: u32 = 1 << 14;
pub const MF_KNIGHTMARE_HOP: u32  = 1 << 20;
pub const MF_ICICLE_THAW: u32     = 1 << 21;
pub const MF_BISHOPPER_CROUCH: u32= 1 << 15;
pub const MF_BISHOPPER_LEAP: u32  = 1 << 16;
pub const MF_SHRIKE_FIRST: u32    = 1 << 17;
pub const MF_EPEE_TELEPORT: u32   = 1 << 18;
pub const MF_PILGRIM_RESURRECT: u32 = 1 << 19;

// ─── Move ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Move {
    pub flags: u32,
    pub offboard_id: u32,
    pub from: u8,
    pub to: u8,
    pub promo: u8,
    pub ep_victim: u8,
    pub special_a: u8,
    pub special_b: u8,
    pub push_dx: i8,
    pub push_dy: i8,
}

const _: () = assert!(size_of::<Move>() == 16);

impl Move {
    #[inline] pub fn has(&self, f: u32) -> bool { self.flags & f != 0 }
}

// ─── Knightmare 14x14 staging board (Sandbox-verified 2026-11-03) ─────────
// Offboard squares use coords -3..=10 on both axes (the 8x8 board outlined
// in the middle). A Move's `to` field carries the 14x14 code (0..=195)
// for MF_KNIGHTMARE_OFF / MF_KNIGHTMARE_HOP moves; drop moves land on
// 8x8 squares and carry the plain 8x8 square in `to` instead.
#[inline] pub fn off_code(ox: i8, oy: i8) -> u8 { ((oy as i16 + 3) * 14 + (ox as i16 + 3)) as u8 }
#[inline] pub fn off_xy(code: u8) -> (i8, i8) { ((code % 14) as i8 - 3, (code / 14) as i8 - 3) }
#[inline] pub fn off_on_stage(ox: i8, oy: i8) -> bool { ox >= -3 && ox <= 10 && oy >= -3 && oy <= 10 }
#[inline] pub fn off_in_ring(ox: i8, oy: i8) -> bool {
    off_on_stage(ox, oy) && !(ox >= 0 && ox < 8 && oy >= 0 && oy < 8)
}

pub const DIRS_ROOK:   [(i8,i8); 4] = [(1,0),(-1,0),(0,1),(0,-1)];
pub const DIRS_BISHOP: [(i8,i8); 4] = [(1,1),(1,-1),(-1,1),(-1,-1)];
pub const DIRS_KING:   [(i8,i8); 8] = [
    (1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)
];
pub const LEAPS_KNIGHT: [(i8,i8); 8] = [
    (1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)
];
pub const LEAPS_CAMEL: [(i8,i8); 8] = [
    (3,1),(3,-1),(-3,1),(-3,-1),(1,3),(1,-3),(-1,3),(-1,-3)
];

// ─── Provisional flags ────────────────────────────────────────────────────

pub const PROVISIONAL_IRON_INVULN:      bool = true;
pub const PROVISIONAL_ICICLE_FREEZE:    bool = true;
pub const PROVISIONAL_FROZEN_PERMANENT: bool = true;

// ─── Ability key (u8 wire encoding) ───────────────────────────────────────

pub const AB_NONE: u8             = 0;
pub const AB_LEAP12: u8           = 1;
pub const AB_PAWN_CAPTURE: u8     = 2;
pub const AB_RIDE_CARDINAL: u8    = 3;
pub const AB_RIDE_DIAGONAL: u8    = 4;
pub const AB_RIDE_PRINCIPAL: u8   = 5;
pub const AB_CAPTURE_THEN_MOVE: u8= 6;
pub const AB_CARDINAL_RETREAT: u8 = 7;
pub const AB_PAWN_SINGLE: u8      = 8;
pub const AB_PAWN_DOUBLE: u8      = 9;
pub const AB_KING_CASTLE: u8      = 10;
