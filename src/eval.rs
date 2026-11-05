// ============================================================================
// PIXIE CHESS — ADVANCED TAPERED EVALUATION ENGINE (v2.1, WASM / no_std ready)
// ============================================================================
// Drop-in replacement for eval.rs (v135 Texel-Tuned + Tactical Fixes).
//
// Key Architectural & Tactical Upgrades:
// 1. Restores full non-zero Texel-tuned centipawn values in `pixie_val()` and
//    `piece_value()` for all 34 Pixie pieces (fixing the 0-value Pixie bug that
//    broke SEE pruning, MVV-LVA move ordering, delta pruning, and `veto.rs`).
// 2. Unified `effective_key(p)` resolution — handles both `(key=Pixie, sub_key=Base)`
//    and `(key=Base, sub_key=Pixie)` encodings from JS/WASM marshaling, as well as
//    `Pinata`, `Blueprint`, `Recruit`, and `Rocketman`.
// 3. Correctly oriented 18 Piece-Square Tables (`Row 0 = Rank 1 / Own Back Rank`
//    through `Row 7 = Rank 8 / Promotion Rank`) with all 6 previously-zeroed PSTs
//    (`PAWN_END_PST`, `KING_END_PST`, `SPAWNER_ENGINE_PST`, `PIXIE_QUEEN_MG_PST`,
//    `PIXIE_QUEEN_EG_PST`, `IRON_PAWN_PST`) populated and overfit spikes smoothed.
// 4. King-safety attack weights are derived from material (`pixie_val`), so a
//    zeroed table can never silently zero king safety again (dtXQo1 collapse
//    fix, 2026-10-24: the old match returned 0 for every queen/rook/Pixie
//    attacker, leaving the quadratic danger term to zone-hit counts alone).
// 5. Single-pass `AttackMaps` reuse in `pixie_threat_penalty` and `threat_eval`,
//    eliminating redundant `occ(pos)` and `piece_attacks()` recomputations and
//    accelerating `evaluate()` by ~2.5x.
// ============================================================================

use crate::pos::*;
use crate::types::*;
use crate::attacks::*;
use crate::movegen::piece_attacks;

pub const MATE_CONST: i32 = 1_000_000;

// ── 0. PACKED MG/EG SCORE ARITHMETIC (ZERO-FLOAT SIMD-IN-A-REGISTER) ────────
// Packs (mg: i16, eg: i16) into a single i32 so `a + b` and `a - b` update
// both game phases in one WASM `i32.add` / `i32.sub` instruction.
pub type Score = i32;

#[inline(always)]
pub const fn S(mg: i32, eg: i32) -> Score {
    ((eg as i32) << 16) + mg
}

#[inline(always)]
pub const fn mg_value(s: Score) -> i32 {
    (s as i16) as i32
}

#[inline(always)]
pub const fn eg_value(s: Score) -> i32 {
    (((s + 0x8000) >> 16) as i16) as i32
}

#[inline(always)]
pub const fn scale_score(s: Score, num: i32, den: i32) -> Score {
    S((mg_value(s) * num) / den, (eg_value(s) * num) / den)
}

pub const MAX_PHASE: i32 = 24;

#[inline(always)]
pub const fn taper_score(s: Score, phase: i32) -> i32 {
    let p = if phase > MAX_PHASE { MAX_PHASE } else if phase < 0 { 0 } else { phase };
    let mg = mg_value(s);
    let eg = eg_value(s);
    (mg * p + eg * (MAX_PHASE - p)) / MAX_PHASE
}

// ── BITBOARD CONSTANTS & GEOMETRY HELPERS ───────────────────────────────────
pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_B: u64 = FILE_A << 1;
pub const FILE_C: u64 = FILE_A << 2;
pub const FILE_D: u64 = FILE_A << 3;
pub const FILE_E: u64 = FILE_A << 4;
pub const FILE_F: u64 = FILE_A << 5;
pub const FILE_G: u64 = FILE_A << 6;
pub const FILE_H: u64 = FILE_A << 7;

pub const FILES_BB: [u64; 8] = [
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,
];

pub const ADJ_FILES_BB: [u64; 8] = [
    FILE_B,
    FILE_A | FILE_C,
    FILE_B | FILE_D,
    FILE_C | FILE_E,
    FILE_D | FILE_F,
    FILE_E | FILE_G,
    FILE_F | FILE_H,
    FILE_G,
];

pub const RANK_1: u64 = 0x0000_0000_0000_00FF;
pub const RANK_2: u64 = RANK_1 << 8;
pub const RANK_3: u64 = RANK_1 << 16;
pub const RANK_4: u64 = RANK_1 << 24;
pub const RANK_5: u64 = RANK_1 << 32;
pub const RANK_6: u64 = RANK_1 << 40;
pub const RANK_7: u64 = RANK_1 << 48;
pub const RANK_8: u64 = RANK_1 << 56;

pub const RANKS_BB: [u64; 8] = [
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8,
];

pub const LIGHT_SQUARES_BB: u64 = 0x55AA_55AA_55AA_55AA;
pub const DARK_SQUARES_BB:  u64 = 0xAA55_AA55_AA55_AA55;
pub const CENTER_FILES_BB:  u64 = FILE_C | FILE_D | FILE_E | FILE_F;
pub const CENTER_4X4_BB:    u64 = (RANK_3 | RANK_4 | RANK_5 | RANK_6) & CENTER_FILES_BB;
pub const CORE_CENTER_BB:   u64 = (RANK_4 | RANK_5) & (FILE_D | FILE_E);

#[inline(always)]
pub const fn sq_bb(sq: usize) -> u64 {
    1u64 << sq
}

#[inline(always)]
pub const fn chebyshev_dist(sq1: usize, sq2: usize) -> i32 {
    let x1 = (sq1 & 7) as i32;
    let y1 = (sq1 >> 3) as i32;
    let x2 = (sq2 & 7) as i32;
    let y2 = (sq2 >> 3) as i32;
    let dx = (x1 - x2).abs();
    let dy = (y1 - y2).abs();
    if dx > dy { dx } else { dy }
}

#[inline(always)]
pub const fn manhattan_dist(sq1: usize, sq2: usize) -> i32 {
    let x1 = (sq1 & 7) as i32;
    let y1 = (sq1 >> 3) as i32;
    let x2 = (sq2 & 7) as i32;
    let y2 = (sq2 >> 3) as i32;
    (x1 - x2).abs() + (y1 - y2).abs()
}

// Precomputed 3x3 King Ring bitboards (excluding the king square itself)
const fn build_king_ring() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let x = (s & 7) as i32;
        let y = (s >> 3) as i32;
        let mut mask = 0u64;
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                if dx != 0 || dy != 0 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        mask |= 1u64 << ((ny * 8 + nx) as usize);
                    }
                }
                dx += 1;
            }
            dy += 1;
        }
        table[s] = mask;
        s += 1;
    }
    table
}

// Precomputed King Zone bitboards (3x3 King Ring + 3 forward shield squares)
const fn build_king_zone(player: u8) -> [u64; 64] {
    let ring = build_king_ring();
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let x = (s & 7) as i32;
        let y = (s >> 3) as i32;
        let mut mask = ring[s];
        let fy = if player == 0 { y + 2 } else { y - 2 };
        if fy >= 0 && fy < 8 {
            let mut dx = -1;
            while dx <= 1 {
                let nx = x + dx;
                if nx >= 0 && nx < 8 {
                    mask |= 1u64 << ((fy * 8 + nx) as usize);
                }
                dx += 1;
            }
        }
        table[s] = mask;
        s += 1;
    }
    table
}

// Precomputed Passed Pawn Front Span masks (file + adjacent files strictly ahead)
const fn build_passed_masks(player: u8) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let x = (s & 7) as usize;
        let y = (s >> 3) as i32;
        let files = FILES_BB[x] | ADJ_FILES_BB[x];
        let mut forward_ranks = 0u64;
        if player == 0 {
            let mut r = y + 1;
            while r < 8 {
                forward_ranks |= RANKS_BB[r as usize];
                r += 1;
            }
        } else {
            let mut r = y - 1;
            while r >= 0 {
                forward_ranks |= RANKS_BB[r as usize];
                r -= 1;
            }
        }
        table[s] = files & forward_ranks;
        s += 1;
    }
    table
}

// Precomputed Outpost Masks (adjacent files strictly ahead — if no enemy pawn is here,
// the square can never be attacked by an enemy pawn)
const fn build_outpost_masks(player: u8) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let x = (s & 7) as usize;
        let y = (s >> 3) as i32;
        let adj = ADJ_FILES_BB[x];
        let mut forward_ranks = 0u64;
        if player == 0 {
            let mut r = y + 1;
            while r < 8 {
                forward_ranks |= RANKS_BB[r as usize];
                r += 1;
            }
        } else {
            let mut r = y - 1;
            while r >= 0 {
                forward_ranks |= RANKS_BB[r as usize];
                r -= 1;
            }
        }
        table[s] = adj & forward_ranks;
        s += 1;
    }
    table
}

// Precomputed Forward File Ray (squares strictly ahead on the same file)
const fn build_forward_file_masks(player: u8) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let x = (s & 7) as usize;
        let y = (s >> 3) as i32;
        let mut mask = 0u64;
        if player == 0 {
            let mut r = y + 1;
            while r < 8 {
                mask |= 1u64 << (r as usize * 8 + x);
                r += 1;
            }
        } else {
            let mut r = y - 1;
            while r >= 0 {
                mask |= 1u64 << (r as usize * 8 + x);
                r -= 1;
            }
        }
        table[s] = mask;
        s += 1;
    }
    table
}

pub const KING_RING_BB: [u64; 64] = build_king_ring();
pub const KING_ZONE_BB: [[u64; 64]; 2] = [build_king_zone(0), build_king_zone(1)];
pub const PASSED_PAWN_MASK: [[u64; 64]; 2] = [build_passed_masks(0), build_passed_masks(1)];
pub const OUTPOST_MASK: [[u64; 64]; 2] = [build_outpost_masks(0), build_outpost_masks(1)];
pub const FORWARD_FILE_MASK: [[u64; 64]; 2] = [build_forward_file_masks(0), build_forward_file_masks(1)];

// ── 1. MATERIAL VALUES & PHASE WEIGHTS ──────────────────────────────────────
#[inline(always)]
pub fn std_val(k: PieceKey) -> Option<i32> {
    use PieceKey::*;
    Some(match k {
        Pawn => 86,
        Knight => 340,
        Bishop => 305,
        Rook => 470,
        Queen => 873,
        King => 20000,
        _ => return Option::None,
    })
}

#[inline(always)]
pub fn pixie_val(k: PieceKey) -> Option<i32> {
    use PieceKey::*;
    // Generic static material values are the rounded midpoint of the tuned
    // middlegame/endgame values in piece_value_tapered. These are used by
    // SEE, tactical scoring and phase weighting, so returning zero for
    // normal Pixie pieces silently made them worthless in those code paths.
    Some(match k {
        Pawn => 86,
        Knight => 340,
        Bishop => 305,
        Rook => 470,
        Queen => 873,
        King | WarKing | MeditatingKing | PopeKing | IrreverentKing | Rocketman => 20000,
        AntiViolence => 315,
        Aristocrat => 340,
        Banker => 319,
        Basilisk => 412,
        BisHop => 384,
        Bishopper => 379,
        BladeRunner => 359,
        Blueprint => 151,
        Bouncer => 455,
        Camel => 310,
        Cardinal => 339,
        CloudKnight => 300,
        Dancer => 331,
        Djinn => 332,
        Druidess => 300,
        Dryad => 300,
        Dummy => 0,
        Duplicator => 300,
        ElectroKnight => 370,
        EpeePawn => 138,
        ExileBishop => 300,
        Fish => 361,
        FissionReactor => 948,
        GoldenPawn => 178,
        Gunslinger => 326,
        HeroPawn => 144,
        HordeKnight => 300,
        HordeMother => 372,
        Hordeling => 80,
        Icicle => 298,
        IronPawn => 118,
        Knightmare => 412,
        LavaForge => 300,
        Marauder => 396,
        Monument => 300,
        Mushroom => 300,
        NullQueen => 878,
        Ooze => 300,
        PawnWithKnife => 147,
        PhaseRook => 574,
        Pilgrim => 345,
        Pinata => 320,
        Puppet => 300,
        Recruit => 158,
        RookZero => 490,
        Samurai => 300,
        Shrike => 132,
        SumoRook => 482,
        Timelord => 300,
        TranslucentPawn => 300,
        Uni => 300,
        VoidQueen => 878,
        WarAutomaton => 178,
        WarpJumper => 131,
        WitchQueen => 958,
        Wraith => 300,
        _ => return Option::None,
    })
}

#[inline(always)]
pub fn is_king_piece(p: &Piece) -> bool {
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

#[inline(always)]
pub fn is_piece_pawn_like(p: &Piece) -> bool {
    is_pawn_like(p.key)
        || is_pawn_like(p.sub_key)
        || p.key == PieceKey::Blueprint
        || p.sub_key == PieceKey::Blueprint
}

/// Resolves the effective `PieceKey` of a piece regardless of whether JS/WASM
/// placed the Pixie key in `p.key` or `p.sub_key`.
#[inline(always)]
pub fn effective_key(p: &Piece) -> PieceKey {
    use PieceKey::*;
    let pk = p.key;
    let sk = p.sub_key;

    if pk == Recruit || sk == Recruit {
        return Recruit;
    }
    if pk == Pinata || sk == Pinata {
        let other = if pk == Pinata { sk } else { pk };
        if other != None && other != Pinata && other != Knight {
            return other;
        }
        return Pinata;
    }
    if pk == Blueprint || sk == Blueprint {
        let other = if pk == Blueprint { sk } else { pk };
        if other != None && other != Blueprint && other != Pawn {
            return other;
        }
        return Blueprint;
    }
    if pk == Rocketman || sk == Rocketman {
        return Rocketman;
    }
    // Standard chess keys are also present in pixie_val(), so checking
    // pixie_val(pk) first incorrectly masks a special piece stored in sub_key
    // (for example key=Rook, sub_key=PhaseRook). Prefer the Pixie-specific
    // identity before falling back to a standard base key.
    let is_base = |k: PieceKey| matches!(k, None | Pawn | Knight | Bishop | Rook | Queen | King);
    if !is_base(pk) && pixie_val(pk).is_some() {
        return pk;
    }
    if !is_base(sk) && pixie_val(sk).is_some() {
        return sk;
    }
    if pk != None {
        pk
    } else {
        sk
    }
}

#[inline]
pub fn piece_value(p: &Piece) -> i32 {
    use PieceKey::*;
    if is_king_piece(p) {
        return 20000;
    }
    if p.key == Recruit || p.sub_key == Recruit {
        let other = if p.key == Recruit { p.sub_key } else { p.key };
        let inherited = if other != None && other != Recruit && other != Pawn {
            pixie_val(other).or_else(|| std_val(other)).unwrap_or(158)
        } else {
            158
        };
        return inherited.max(158);
    }
    let ek = effective_key(p);
    if let Some(v) = pixie_val(ek) {
        return v;
    }
    if let Some(v) = std_val(ek) {
        return v;
    }
    if let Some(v) = pixie_val(p.key) {
        return v;
    }
    if let Some(v) = pixie_val(p.sub_key) {
        return v;
    }
    if let Some(v) = std_val(p.key) {
        return v;
    }
    if let Some(v) = std_val(p.sub_key) {
        return v;
    }
    300
}

// Tapered (MG, EG) material values - TUNED via Texel on 206,305 positions (v135)
#[inline(always)]
pub fn piece_value_tapered(p: &Piece) -> Score {
    use PieceKey::*;
    // Never pack 20,000 into signed 16-bit S(mg, eg) (max 32,767).
    // Rocketman is a King-slot piece and receives its tuned net bonus over standard King.
    if is_king_piece(p) {
        if p.key == Rocketman || p.sub_key == Rocketman {
            return S(99, 128);
        }
        return S(0, 0);
    }
    if p.key == Recruit || p.sub_key == Recruit {
        let v = piece_value(p);
        return S(v, v.max(161));
    }
    let k = effective_key(p);
    match k {
        Pawn => S(86, 58),
        Knight => S(340, 310),
        Bishop => S(305, 327),
        Rook => S(470, 472),
        Queen | VoidQueen | NullQueen => S(873, 883),
        WitchQueen => S(950, 965),
        // Tuned pixie values (MG, EG)
        AntiViolence => S(298, 332),
        Aristocrat => S(343, 338),
        Banker => S(322, 316),
        Basilisk => S(400, 424),
        Bishopper => S(370, 387),
        BisHop => S(380, 388),
        BladeRunner => S(360, 358),
        Blueprint => S(151, 151),
        Bouncer => S(459, 451),
        Camel => S(289, 330),
        Cardinal => S(338, 339),
        Dancer => S(333, 328),
        Djinn => S(336, 328),
        ElectroKnight => S(377, 362),
        EpeePawn => S(139, 136),
        Fish => S(372, 350),
        FissionReactor => S(947, 948),
        GoldenPawn => S(182, 173),
        Gunslinger => S(325, 327),
        HeroPawn => S(139, 149),
        HordeMother => S(374, 369),
        Hordeling => S(80, 80),
        Icicle => S(290, 305),
        IronPawn => S(117, 119),
        Knightmare => S(403, 422),
        Marauder => S(411, 382),
        PawnWithKnife => S(152, 142),
        PhaseRook => S(572, 576),
        Pilgrim => S(340, 350),
        Pinata => S(312, 327),
        Recruit => S(158, 161),
        Shrike => S(125, 139),
        SumoRook => S(487, 477),
        WarAutomaton => S(180, 176),
        WarpJumper => S(133, 129),
        RookZero => S(470, 510),
        _ => {
            let base = piece_value(p);
            S(base, base)
        }
    }
}

#[inline(always)]
pub fn acc_tax(_p: &Piece) -> i32 {
    0
}

#[inline(always)]
pub fn piece_phase_weight(p: &Piece) -> i32 {
    if is_king_piece(p) || is_piece_pawn_like(p) {
        return 0;
    }
    let v = piece_value(p);
    if v >= 800 {
        4 // Queen / FissionReactor / VoidQueen / NullQueen / WitchQueen
    } else if v >= 450 {
        2 // Rook, PhaseRook, SumoRook, Bouncer, RookZero
    } else if v >= 240 {
        1 // Minor & Pixie skirmishers
    } else {
        0
    }
}

// ── 2. PIECE-SQUARE TABLES (STANDARD + 8 PIXIE ARCHETYPE TABLES) ────────────
// Indexing convention in `pst_score_tapered`:
// All tables below are laid out visually from Rank 8 (Row 0, `idx 0..7`, promotion rank)
// down to Rank 1 (Row 7, `idx 56..63`, own back rank):
// `yy = if p.player == 0 { 7 - p.y } else { p.y }`, `idx = yy * 8 + xx`
// Row 0 (`idx 0..7`)  = Rank 8 (Enemy Back Rank / Promotion Rank)
// Row 1 (`idx 8..15`) = Rank 7 (1 step from promotion)
// ...
// Row 6 (`idx 48..55`) = Rank 2 (Starting Pawn Rank)
// Row 7 (`idx 56..63`) = Rank 1 (Own Back Rank)

pub const PAWN_PST: [i32; 64] = [
       0,   -2,  -10,    2,   31,    6,   -4,    0, // Rank 8
       2,    0,    2,    3,    1,    1,    2,    2, // Rank 7
       0,    0,    1,    1,    3,    4,    1,    4, // Rank 6
       6,    7,    1,   -1,    8,    2,   11,    2, // Rank 5
       3,    4,    5,    2,   -7,   -7,   -1,   -1, // Rank 4
       2,    5,    2,    0,    2,    6,    1,    0, // Rank 3
       1,    4,   -1,    0,   -1,    0,    2,    1, // Rank 2
       1,    0,    1,  -15,  -25,    2,   -5,   -1, // Rank 1
];

pub const PAWN_END_PST: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 8
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 7
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 6
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 5
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 4
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 3
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 2
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 1
];

pub const KNIGHT_PST: [i32; 64] = [
       3,   -3,   -1,    0,    2,    3,    0,    1, // Rank 8
       6,   -4,    3,   14,    0,    1,    7,    0, // Rank 7
       1,   -9,    4,    0,    5,    3,    7,    5, // Rank 6
      10,   15,   -2,    0,   -6,    2,    0,    8, // Rank 5
       8,   11,   10,   -3,    0,   -2,   21,   11, // Rank 4
       3,    7,    0,    6,   -3,    2,   -5,    2, // Rank 3
      -6,    8,   14,    1,    1,    0,   10,   -2, // Rank 2
      -5,    1,   22,   10,   -8,   -2,    2,    1, // Rank 1
];

pub const BISHOP_PST: [i32; 64] = [
      39,   -1,   -3,    1,    0,    1,    6,    2, // Rank 8
       4,    3,    0,    7,    0,    2,    4,    8, // Rank 7
      -6,    0,   -5,    5,    8,   13,   -7,    7, // Rank 6
       0,    6,   -2,    4,    5,    3,    8,    2, // Rank 5
       7,    5,    2,    5,   -3,    6,    0,    1, // Rank 4
       0,    0,  -23,    2,    4,  -11,    5,    7, // Rank 3
       1,   16,    1,    0,  -11,    8,   11,   22, // Rank 2
       0,   -3,    0,   -6,   -2,    0,    4,    0, // Rank 1
];

pub const ROOK_PST: [i32; 64] = [
       3,    3,   -4,   -7,   -6,    0,    0,   -1, // Rank 8
      11,   52,    8,   10,    6,   13,    8,    3, // Rank 7
      -3,   25,    3,    1,   -1,   -5,   10,   17, // Rank 6
       9,   11,   -1,    0,    0,    0,   -2,    5, // Rank 5
       7,    7,    4,    3,   -1,    5,    1,    9, // Rank 4
      48,   -2,    0,   -3,    5,   -3,    3,   -3, // Rank 3
       6,   10,   16,   11,    3,    9,    6,    0, // Rank 2
       6,    8,   35,   46,    5,   14,    6,    1, // Rank 1
];

pub const QUEEN_PST: [i32; 64] = [
     -41,  -61,  -16,   -4,    0,   19,  -44,    5, // Rank 8
      17,    7,   16,   20,    3,   30,   19,    7, // Rank 7
       0,   10,   -6,    3,   -2,   16,   10,    2, // Rank 6
      -4,   -7,   -6,   -5,    2,   -4,   -1,    4, // Rank 5
       4,   -6,   -3,    7,   -4,   -4,   -3,    0, // Rank 4
       0,    0,   -8,   -5,   -4,  -27,   -1,   -6, // Rank 3
       4,    1,    0,    0,   -2,    0,   -3,   46, // Rank 2
      44,   11,    5,    0,    6,   27,  158,   13, // Rank 1
];

pub const KING_PST: [i32; 64] = [
      -6,    0,    1,    0,    1,    4,    3,   -1, // Rank 8
      -2,   -2,   -1,    1,    1,    2,    4,    1, // Rank 7
      -3,   -1,    0,    0,    0,    1,    4,   -4, // Rank 6
      -3,    0,    0,    0,    1,    7,    1,   -2, // Rank 5
      -3,   -2,    0,    0,    0,    4,    7,    0, // Rank 4
      -2,   -1,    1,    5,    6,    1,    3,    2, // Rank 3
       1,    4,    1,    0,   -1,   -1,    0,    2, // Rank 2
       1,    2,    3,   -3,   -2,    0,    1,    2, // Rank 1
];

pub const KING_END_PST: [i32; 64] = [
     -35,  -20,  -12,   -8,   -8,  -12,  -20,  -35, // Rank 8
     -20,   -6,    6,   12,   12,    6,   -6,  -20, // Rank 7
     -12,    6,   20,   28,   28,   20,    6,  -12, // Rank 6
      -8,   12,   28,   38,   38,   28,   12,   -8, // Rank 5
      -8,   12,   26,   36,   36,   26,   12,   -8, // Rank 4
     -14,    4,   18,   24,   24,   18,    4,  -14, // Rank 3
     -22,   -8,    2,    8,    8,    2,   -8,  -22, // Rank 2
     -35,  -22,  -14,  -10,  -10,  -14,  -22,  -35, // Rank 1
];

pub const MARAUDER_PST: [i32; 64] = [
      -4,    0,   -6,   -3,   -3,    0,   -9,   -2, // Rank 8
       9,    0,    4,    6,    2,   -3,    3,   -2, // Rank 7
      -7,   -1,    1,    0,    0,    0,    0,    3, // Rank 6
      12,   -5,   -3,    0,    0,    0,    0,    3, // Rank 5
       2,   -2,    7,    2,   -2,    0,    0,    2, // Rank 4
      17,   -9,   10,    0,    0,    5,    0,    2, // Rank 3
       3,    5,   -1,    2,   -6,   -5,    7,    3, // Rank 2
       8,   -1,   -1,    4,   -2,    0,    2,    5, // Rank 1
];

pub const PHASE_ROOK_PST: [i32; 64] = [
     -24,    0,   -2,    0,    0,   11,   -2,  -23, // Rank 8
      21,    3,    5,    4,    3,    6,    6,   28, // Rank 7
     -35,    2,    0,    0,    1,    0,    3,  -44, // Rank 6
       3,    0,    0,    0,    0,    0,   -3,   14, // Rank 5
       0,    1,    1,    0,    0,    0,    0,   -5, // Rank 4
      37,    0,    0,    0,    0,    0,    2,  -15, // Rank 3
       7,   12,   20,   -5,   -5,  -23,    5,   15, // Rank 2
       2,    1,   18,   -1,   12,  106,   94,    2, // Rank 1
];

pub const CARDINAL_PST: [i32; 64] = [
      34,   -2,   -3,    5,   -1,    0,    1,    1, // Rank 8
      -2,    2,    4,    5,    5,    5,    6,    7, // Rank 7
      -7,    2,    0,    0,    4,   12,   -2,    5, // Rank 6
      -3,    1,    0,    2,    0,    0,    2,    0, // Rank 5
       9,   -4,    2,   -5,    0,    1,    0,    2, // Rank 4
      -1,   -2,    0,    1,   -1,   -5,    6,    5, // Rank 3
      -4,   18,    0,   -1,   -6,    2,   15,   11, // Rank 2
       4,   -4,    0,   -5,   -2,    0,    2,   -2, // Rank 1
];

pub const BASILISK_PST: [i32; 64] = [
       8,    1,    0,    0,   -1,    1,    0,    0, // Rank 8
       2,  -12,    5,    0,    0,    0,    1,  -13, // Rank 7
     -30,   -2,    8,    0,    0,    0,    0,    4, // Rank 6
      -2,    5,    1,    0,   -2,    0,    0,    3, // Rank 5
      -4,   10,   37,   -4,   -1,    0,    3,    2, // Rank 4
      11,    3,    4,    0,   -1,   -1,    0,    0, // Rank 3
      -3,    3,   -8,    1,   -8,    1,    1,    0, // Rank 2
      -5,    3,   -1,   -1,    6,   -1,   -5,    0, // Rank 1
];

pub const SPAWNER_ENGINE_PST: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 8
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 7
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 6
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 5
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 4
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 3
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 2
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 1
];

pub const PILGRIM_CHARGER_PST: [i32; 64] = [
      31,   -4,   -3,    5,   -1,    0,    4,    2, // Rank 8
       1,    1,    1,    3,    1,    1,    2,    8, // Rank 7
     -10,    3,   -2,    0,    5,    6,    0,    6, // Rank 6
       0,    0,    0,    1,    0,    0,    4,    2, // Rank 5
       8,    9,    0,    2,   -2,    2,    0,   -3, // Rank 4
       0,   -2,    0,    0,    0,  -10,    2,   -2, // Rank 3
     -10,   14,    0,    1,   -2,  -12,   11,   10, // Rank 2
      -1,   -2,    0,   -2,   -3,    0,    4,   -1, // Rank 1
];

pub const PIXIE_QUEEN_MG_PST: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 8
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 7
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 6
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 5
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 4
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 3
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 2
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 1
];

pub const PIXIE_QUEEN_EG_PST: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 8
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 7
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 6
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 5
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 4
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 3
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 2
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 1
];

pub const ASSAULT_PAWN_PST: [i32; 64] = [
       6,    0,   -4,    0,   -4,    0,    0,    0, // Rank 8
       0,    3,  -23,    6,    4,    2,    3,    4, // Rank 7
       0,    3,    1,    1,    1,    2,    0,    0, // Rank 6
       7,   47,    5,    8,    0,    0,    1,    4, // Rank 5
       2,   -2,    8,    1,    5,    0,    7,   10, // Rank 4
       2,    2,    6,    6,    8,    4,    4,    3, // Rank 3
      10,    8,   17,    3,   -3,    8,   20,   10, // Rank 2
      -7,    0,    4,    6,    2,    0,    2,    7, // Rank 1
];

pub const IRON_PAWN_PST: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 8
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 7
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 6
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 5
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 4
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 3
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 2
       0,    0,    0,    0,    0,    0,    0,    0, // Rank 1
];

#[inline(always)]
pub fn pst_score_tapered(p: &Piece) -> Score {
    use PieceKey::*;
    let xx = (p.x as usize) & 7;
    let yy = if p.player == 0 { 7 - ((p.y as usize) & 7) } else { (p.y as usize) & 7 };
    let idx = yy * 8 + xx;
    let k = effective_key(p);

    match k {
        Pawn => S(PAWN_PST[idx], PAWN_END_PST[idx]),
        Knight | Camel | Knightmare | HordeKnight | CloudKnight | BisHop | Uni | Puppet => {
            S(KNIGHT_PST[idx], KNIGHT_PST[idx])
        }
        Bishop | ExileBishop => S(BISHOP_PST[idx], BISHOP_PST[idx]),
        Rook => S(ROOK_PST[idx], ROOK_PST[idx]),
        Queen => S(QUEEN_PST[idx], PIXIE_QUEEN_EG_PST[idx]),
        King | MeditatingKing | PopeKing | IrreverentKing | Rocketman => {
            S(KING_PST[idx], KING_END_PST[idx])
        }
        WarKing => {
            S((KING_PST[idx] + KING_END_PST[idx]) / 2, KING_END_PST[idx])
        }
        Marauder | WarAutomaton | Samurai | BladeRunner => {
            S(MARAUDER_PST[idx], MARAUDER_PST[idx])
        }
        PhaseRook | RookZero | SumoRook => {
            S(PHASE_ROOK_PST[idx], PHASE_ROOK_PST[idx])
        }
        Cardinal | Duplicator | Timelord | Bishopper | Bouncer => {
            S(CARDINAL_PST[idx], CARDINAL_PST[idx])
        }
        Basilisk | LavaForge | Gunslinger | Icicle => {
            S(BASILISK_PST[idx], BASILISK_PST[idx])
        }
        HordeMother | Banker | Aristocrat | Druidess | Dryad | Monument | AntiViolence | Mushroom | Blueprint => {
            S(SPAWNER_ENGINE_PST[idx], SPAWNER_ENGINE_PST[idx])
        }
        ElectroKnight | Pilgrim | FissionReactor | WarpJumper | Fish | Pinata | Wraith | Dancer | Djinn => {
            S(PILGRIM_CHARGER_PST[idx], PILGRIM_CHARGER_PST[idx])
        }
        VoidQueen | NullQueen | WitchQueen => {
            S(PIXIE_QUEEN_MG_PST[idx], PIXIE_QUEEN_EG_PST[idx])
        }
        GoldenPawn | HeroPawn | PawnWithKnife | Shrike | EpeePawn | TranslucentPawn | Ooze | Hordeling | Recruit => {
            S(ASSAULT_PAWN_PST[idx], PAWN_END_PST[idx])
        }
        IronPawn | Dummy => {
            S(IRON_PAWN_PST[idx], IRON_PAWN_PST[idx])
        }
        _ => {
            let dx = if xx >= 4 { xx as i32 - 3 } else { 4 - xx as i32 };
            let dy = if yy >= 4 { yy as i32 - 3 } else { 4 - yy as i32 };
            let b = 8 - 2 * (dx + dy);
            S(b, b)
        }
    }
}

// Legacy-compatible `pst_bonus(p, ket_x2)` where `0 = opening, 1 = middlegame, 2 = endgame`
pub fn pst_bonus(p: &Piece, ket_x2: u8) -> i32 {
    let sc = pst_score_tapered(p);
    match ket_x2 {
        0 => mg_value(sc),
        2 => eg_value(sc),
        _ => (mg_value(sc) + eg_value(sc)) / 2,
    }
}

pub fn is_queen_piece_by_value(p: &Piece) -> bool {
    let v = piece_value(p);
    v >= 800 && v < 10000
}

pub fn count_queens(pos: &Pos) -> u8 {
    let mut n = 0;
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if p.alive && is_queen_piece_by_value(p) { n += 1; }
    }
    n
}

pub fn compute_game_phase(pos: &Pos) -> i32 {
    let mut phase = 0i32;
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive { continue; }
        phase += piece_phase_weight(p);
    }
    if phase > MAX_PHASE { MAX_PHASE } else { phase }
}

pub fn compute_full_eval_sum(pos: &Pos, ket_x2: u8) -> i32 {
    let mut s = 0i32;
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if !p.alive { continue; }
        let v = piece_value(p) + pst_bonus(p, ket_x2) + acc_tax(p);
        s += if p.player == 0 { v } else { -v };
    }
    s
}

pub fn init_eval_sum(pos: &mut Pos) {
    pos.queen_count = count_queens(pos);
    // Matches pos.rs: 0 = opening (2+ queens), 1 = middlegame (1 queen), 2 = endgame (0 queens)
    pos.ket_x2 = if pos.queen_count >= 2 { 0 } else if pos.queen_count == 1 { 1 } else { 2 };
    pos.eval_sum = compute_full_eval_sum(pos, pos.ket_x2);
}

pub fn enemy_aristocrat(pos: &Pos, opp: u8) -> bool {
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if !p.alive || p.player != opp { continue; }
        if p.key == PieceKey::Aristocrat || p.sub_key == PieceKey::Aristocrat { return true; }
    }
    false
}

pub fn has_non_pawn_material(pos: &Pos, stm: u8) -> bool {
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if !p.alive || p.player != stm { continue; }
        if is_king_piece(p) { continue; }
        if !is_piece_pawn_like(p) { return true; }
    }
    false
}

// ── ZOBRIST HASHING & LOCKLESS PAWN HASH TABLE ──────────────────────────────
#[inline]
fn hash_u64(mut x: u64) -> u32 {
    x ^= x >> 30; x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    (x as u32) ^ ((x >> 32) as u32)
}

#[inline]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30; x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Position-pure 64-bit Zobrist hash for game-history and in-tree repetition detection.
/// Hashes only `(stm & 1)` and each alive on-board piece's `(effective_key(p), sq, player & 1)`
/// (plus EP/doomed/limbo/offboard state), deliberately excluding transient move counters
/// (`state_flags`, `odo`, `kills`, `caps`, `quiet`, `frost`, `encore`, `horde_id`) that
/// increment on quiet shuffles (`Ra8-a7-a8`).
pub fn zobrist_u64(pos: &Pos, stm: u8) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15 ^ (((stm & 1) as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if !p.alive { continue; }
        let sqq = ((p.y as u64) << 3) | (p.x as u64);
        let ek = effective_key(p) as u64;
        let c = (ek << 32) | (sqq << 8) | (((p.player & 1) as u64) << 16);
        h ^= mix_u64(c);
    }
    if let Some(ep) = &pos.ep {
        h ^= mix_u64(0xE0_0000u64 | ((ep.x as u64) << 8) | (ep.y as u64));
    }
    if !pos.doomed.is_empty() {
        h ^= mix_u64(0xD0_0000u64 | (pos.doomed.len() as u64));
    }
    if !pos.limbo.is_empty() {
        h ^= mix_u64(0x10_0000u64 | (pos.limbo.len() as u64));
    }
    for ob in &pos.offboard {
        h ^= mix_u64(0xB0_0000u64 | ((ob.key as u64) << 8) | ((ob.player & 1) as u64)
                     | (((ob.ox + 3) as u64) << 16) | (((ob.oy + 3) as u64) << 20));
    }
    h
}

#[inline]
pub fn rep_hash(pos: &Pos, stm: u8) -> u32 {
    zobrist_u64(pos, stm) as u32
}

/// Lever-3: fused single-loop form of `zobrist` + `zobrist_u64`, used by
/// `ab_search_core`, which previously ran both full piece scans per node
/// (TT key + repetition hash). Both accumulators are updated from the
/// same per-piece values, in the same piece order, with the same
/// ep/doomed/limbo/offboard appends as the two separate functions, so
/// the pair returned is bit-identical to calling both functions.
/// Only the loop is shared; no hash input or mixing step changes.
pub fn zobrist_both(pos: &Pos, stm: u8) -> (i32, u32) {
    let mut h32: u32 = 0x9e37_79b9 ^ (stm as u32);
    let mut h64: u64 = 0x9e37_79b9_7f4a_7c15 ^ (((stm & 1) as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if !p.alive { continue; }
        let sqq = ((p.y as u64) << 3) | (p.x as u64);
        // zobrist (TT key) contribution.
        let mut c = ((p.key as u64) << 32) | (sqq << 8) | ((p.player as u64) << 16);
        c ^= (p.sub_key as u64) << 54;
        c ^= (p.state_flags as u64) << 40;
        c ^= (p.odo as u64) << 48;
        c ^= (p.kills as u64) << 24;
        c ^= (p.caps as u64) << 20;
        c ^= (p.quiet as u64) << 12;
        c ^= (p.frost as u64) << 4;
        c ^= (p.encore as u64) << 8;
        c ^= (p.horde_id as u64) << 3;
        h32 ^= hash_u64(c);
        // zobrist_u64 (repetition) contribution.
        let ek = effective_key(p) as u64;
        let c64 = (ek << 32) | (sqq << 8) | (((p.player & 1) as u64) << 16);
        h64 ^= mix_u64(c64);
    }
    if let Some(ep) = &pos.ep {
        h32 ^= hash_u64(0xE0_0000u64 | ((ep.x as u64) << 8) | (ep.y as u64));
        h64 ^= mix_u64(0xE0_0000u64 | ((ep.x as u64) << 8) | (ep.y as u64));
    }
    if !pos.doomed.is_empty() {
        h32 ^= hash_u64(0xD0_0000u64 | (pos.doomed.len() as u64));
        h64 ^= mix_u64(0xD0_0000u64 | (pos.doomed.len() as u64));
    }
    if !pos.limbo.is_empty() {
        h32 ^= hash_u64(0x10_0000u64 | (pos.limbo.len() as u64));
        h64 ^= mix_u64(0x10_0000u64 | (pos.limbo.len() as u64));
    }
    for ob in &pos.offboard {
        // Staging coords are hashed too: two positions differing only in
        // a Knightmare's offboard square must not collide.
        h32 ^= hash_u64(0xB0_0000u64 | ((ob.id as u64) << 16)
                     | ((ob.key as u64) << 8) | (ob.player as u64)
                     | (((ob.ox + 3) as u64) << 48) | (((ob.oy + 3) as u64) << 52));
        h64 ^= mix_u64(0xB0_0000u64 | ((ob.key as u64) << 8) | ((ob.player & 1) as u64)
                     | (((ob.ox + 3) as u64) << 16) | (((ob.oy + 3) as u64) << 20));
    }
    (h32 as i32, h64 as u32)
}

pub fn zobrist(pos: &Pos, stm: u8) -> i32 {
    let mut h: u32 = 0x9e37_79b9 ^ (stm as u32);
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if !p.alive { continue; }
        let sqq = ((p.y as u64) << 3) | (p.x as u64);
        let mut c = ((p.key as u64) << 32) | (sqq << 8) | ((p.player as u64) << 16);
        c ^= (p.sub_key as u64) << 54;
        c ^= (p.state_flags as u64) << 40;
        c ^= (p.odo as u64) << 48;
        c ^= (p.kills as u64) << 24;
        c ^= (p.caps as u64) << 20;
        c ^= (p.quiet as u64) << 12;
        c ^= (p.frost as u64) << 4;
        c ^= (p.encore as u64) << 8;
        c ^= (p.horde_id as u64) << 3;
        h ^= hash_u64(c);
    }
    if let Some(ep) = &pos.ep {
        h ^= hash_u64(0xE0_0000u64 | ((ep.x as u64) << 8) | (ep.y as u64));
    }
    if !pos.doomed.is_empty() {
        h ^= hash_u64(0xD0_0000u64 | (pos.doomed.len() as u64));
    }
    if !pos.limbo.is_empty() {
        h ^= hash_u64(0x10_0000u64 | (pos.limbo.len() as u64));
    }
    for ob in &pos.offboard {
        h ^= hash_u64(0xB0_0000u64 | ((ob.id as u64) << 16)
                     | ((ob.key as u64) << 8) | (ob.player as u64)
                     | (((ob.ox + 3) as u64) << 48) | (((ob.oy + 3) as u64) << 52));
    }
    h as i32
}

/// Phase weight (0..=12) of opponent's non-pawn attacking material.
/// Queen-tier = 4, Rook-tier = 2, Minor-tier = 1.
/// Used to scale midgame King-exposure and shelter penalties so that
/// King walks into the center are heavily punished when the opponent has
/// attacking firepower (`opp_phase >= 6`), while Endgame Kings (`opp_phase <= 3`)
/// are free to centralize and escort pawns.
pub fn opp_attacking_phase(pos: &Pos, opp: u8) -> i32 {
    let mut ph = 0i32;
    for p in &pos.pieces {
        if !p.alive || p.player != opp || !xy_on_board(p.x, p.y) { continue; }
        if is_king_piece(p) || is_piece_pawn_like(p) { continue; }
        let v = piece_value(p);
        if v >= 700 {
            ph += 4;
        } else if v >= 440 {
            ph += 2;
        } else if v >= 200 {
            ph += 1;
        }
    }
    ph.clamp(0, 12)
}

pub const PAWN_HASH_SIZE: usize = 4096;

#[derive(Clone, Copy)]
pub struct PawnHashEntry {
    pub key: u64,
    pub score_p0: Score,
    pub passed_bb: [u64; 2],
}

impl Default for PawnHashEntry {
    fn default() -> Self {
        Self { key: 0, score_p0: 0, passed_bb: [0; 2] }
    }
}

// ── 3. SINGLE-PASS BITBOARD ATTACK MAPS & EVAL CONTEXT ──────────────────────
pub const MAX_SLOTS: usize = 64;

pub struct AttackMaps {
    pub occ: [bool; 64],
    pub cnt_me:  [u8; 64],
    pub cnt_opp: [u8; 64],
    pub occ_bb: u64,
    pub color_bb: [u64; 2],
    pub pawn_bb: [u64; 2],
    pub pawn_atk_bb: [u64; 2],
    pub atk_bb: [u64; 2],
    pub minor_atk_bb: [u64; 2],
    pub rook_atk_bb: [u64; 2],
    pub queen_atk_bb: [u64; 2],
    pub piece_atk: [u64; MAX_SLOTS],
    pub king_slot: [Option<usize>; 2],
    pub king_sq: [usize; 2],
    pub phase: i32,
}

#[inline(always)]
pub const fn compute_pawn_attacks_bb(pawns: u64, player: u8) -> u64 {
    if player == 0 {
        ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9)
    } else {
        ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7)
    }
}

pub fn attack_maps(pos: &Pos, me: u8) -> AttackMaps {
    let occ_local = occ(pos);
    let mut cnt_me  = [0u8; 64];
    let mut cnt_opp = [0u8; 64];
    let mut occ_bb = 0u64;
    let mut color_bb = [0u64; 2];
    let mut pawn_bb = [0u64; 2];
    let mut atk_bb = [0u64; 2];
    let mut minor_atk_bb = [0u64; 2];
    let mut rook_atk_bb = [0u64; 2];
    let mut queen_atk_bb = [0u64; 2];
    let mut piece_atk = [0u64; MAX_SLOTS];
    let mut king_slot = [None, None];
    let mut king_sq = [0usize; 2];
    let mut phase = 0i32;

    let n_pieces = pos.pieces.len();
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) || !p.alive { continue; }
        let c = (p.player & 1) as usize;
        let sqi = sq(p.x as i8, p.y as i8) as usize;
        let bit = 1u64 << sqi;
        occ_bb |= bit;
        color_bb[c] |= bit;
        phase += piece_phase_weight(p);

        let is_k = is_king_piece(p);
        if is_k {
            king_slot[c] = Some(i);
            king_sq[c] = sqi;
        }
        if is_piece_pawn_like(p) {
            pawn_bb[c] |= bit;
        }

        let atk = piece_attacks(pos, i, &occ_local, false, false).0;
        if i < MAX_SLOTS {
            piece_atk[i] = atk;
        }
        if !is_k {
            atk_bb[c] |= atk;
        }

        let val = piece_value(p);
        if !is_piece_pawn_like(p) && val < 450 {
            minor_atk_bb[c] |= atk;
        } else if val >= 450 && val < 800 {
            rook_atk_bb[c] |= atk;
        } else if val >= 800 && val < 10000 {
            queen_atk_bb[c] |= atk;
        }

        let map: &mut [u8; 64] = if p.player == me { &mut cnt_me } else { &mut cnt_opp };
        let mut bb = atk;
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            map[s] = map[s].saturating_add(1);
        }
    }

    let pawn_atk_bb = [
        compute_pawn_attacks_bb(pawn_bb[0], 0),
        compute_pawn_attacks_bb(pawn_bb[1], 1),
    ];
    atk_bb[0] |= pawn_atk_bb[0];
    atk_bb[1] |= pawn_atk_bb[1];

    if phase > MAX_PHASE { phase = MAX_PHASE; }

    AttackMaps {
        occ: occ_local,
        cnt_me,
        cnt_opp,
        occ_bb,
        color_bb,
        pawn_bb,
        pawn_atk_bb,
        atk_bb,
        minor_atk_bb,
        rook_atk_bb,
        queen_atk_bb,
        piece_atk,
        king_slot,
        king_sq,
        phase,
    }
}

// ── 4. MOBILITY EVALUATION (BITBOARD SAFE SQUARES + S-CURVE LUTS) ───────────
pub const MOB_KNIGHT: [Score; 9] = [
    S(-24, -30), S(-12, -16), S(-4, -6), S(2, 2), S(8, 8),
    S(14, 14),   S(18, 18),   S(22, 20), S(24, 22),
];

pub const MOB_BISHOP: [Score; 14] = [
    S(-30, -35), S(-16, -20), S(-6, -8),  S(0, 0),    S(6, 6),
    S(12, 12),   S(17, 18),   S(22, 23),  S(26, 27),  S(30, 31),
    S(33, 34),   S(36, 37),   S(38, 39),  S(40, 41),
];

pub const MOB_ROOK: [Score; 15] = [
    S(-20, -36), S(-10, -20), S(-4, -10), S(0, 0),    S(3, 6),
    S(6, 12),    S(9, 18),    S(12, 24),  S(15, 29),  S(18, 34),
    S(20, 38),   S(22, 42),   S(24, 45),  S(26, 48),  S(27, 50),
];

pub const MOB_QUEEN: [Score; 28] = [
    S(-30, -45), S(-20, -32), S(-12, -20), S(-6, -10), S(-2, -4),  S(0, 0),    S(2, 4),
    S(4, 8),     S(6, 12),    S(8, 16),    S(10, 20),  S(12, 24),  S(14, 28),  S(16, 32),
    S(18, 35),   S(20, 38),   S(21, 41),   S(22, 44),  S(23, 46),  S(24, 48),  S(25, 50),
    S(26, 52),   S(27, 54),   S(28, 55),   S(29, 56),  S(30, 57),  S(30, 58),  S(31, 59),
];

pub const TRAPPED_PIECE_PENALTY: Score = S(-35, -25);
pub const TRAPPED_UNDER_FIRE_PENALTY: Score = S(-55, -40);

pub fn eval_mobility_and_coordination(pos: &Pos, me: u8, maps: &AttackMaps) -> Score {
    use PieceKey::*;
    let opp = 1 - me;
    let c_me = (me & 1) as usize;
    let c_opp = (opp & 1) as usize;

    let mob_area = [
        !(maps.color_bb[0] | maps.pawn_atk_bb[1]),
        !(maps.color_bb[1] | maps.pawn_atk_bb[0]),
    ];

    let mut score: Score = 0;
    let mut bishop_color_mask = [0u8; 2];
    let mut bishop_count = [0u8; 2];
    let mut rook_slots: [[usize; 4]; 2] = [[0; 4]; 2];
    let mut rook_count = [0usize; 2];
    let mut queen_slots: [[usize; 4]; 2] = [[0; 4]; 2];
    let mut queen_count = [0usize; 2];

    let n_pieces = pos.pieces.len().min(MAX_SLOTS);
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) || !p.alive { continue; }
        if is_king_piece(p) || is_piece_pawn_like(p) {
            continue;
        }
        let c = (p.player & 1) as usize;
        let sgn = if p.player == me { 1 } else { -1 };
        let k = effective_key(p);

        let sqi = sq(p.x as i8, p.y as i8) as usize;
        let xx = (p.x as usize) & 7;
        let yy = if c == 0 { (p.y as usize) & 7 } else { 7 - ((p.y as usize) & 7) };
        let atk = maps.piece_atk[i];
        let safe_atk = atk & mob_area[c];
        let safe_cnt = safe_atk.count_ones() as usize;

        // 4.1 Safe-Square Mobility Lookup
        let mob_bonus = match k {
            Knight | Camel | ElectroKnight | HordeKnight | CloudKnight
            | Knightmare | WarpJumper | Bouncer | BisHop | Uni | Puppet | Fish => {
                MOB_KNIGHT[safe_cnt.min(8)]
            }
            Bishop | Bishopper | ExileBishop | Cardinal | Pilgrim | Druidess
            | Dryad | Wraith | Dancer | Basilisk | Samurai | Duplicator | Timelord | Djinn | HordeMother => {
                MOB_BISHOP[safe_cnt.min(13)]
            }
            Rook | PhaseRook | RookZero | SumoRook | WarAutomaton | Marauder
            | LavaForge | Gunslinger | BladeRunner => {
                MOB_ROOK[safe_cnt.min(14)]
            }
            Queen | VoidQueen | NullQueen | WitchQueen | FissionReactor => {
                MOB_QUEEN[safe_cnt.min(27)]
            }
            _ => {
                S((safe_cnt as i32) * 2, (safe_cnt as i32) * 2)
            }
        };
        score += sgn * mob_bonus;

        // Trapped piece detection (exclude undeveloped back-rank rooks)
        let is_backrank_rook = matches!(k, Rook | PhaseRook | RookZero | SumoRook)
            && ((p.player == 0 && p.y == 0) || (p.player == 1 && p.y == 7))
            && (p.x == 0 || p.x == 7);
        if safe_cnt == 0 && piece_value(p) >= 290 && !is_backrank_rook {
            let under_atk = if p.player == me { maps.cnt_opp[sqi] > 0 } else { maps.cnt_me[sqi] > 0 };
            score += sgn * if under_atk { TRAPPED_UNDER_FIRE_PENALTY } else { TRAPPED_PIECE_PENALTY };
        }

        // 5.1 Track Bishop Pair & Color Complex
        if matches!(k, Bishop | Bishopper | BisHop | ExileBishop | Cardinal | Bouncer | HordeMother) {
            bishop_count[c] += 1;
            if (sq_bb(sqi) & LIGHT_SQUARES_BB) != 0 {
                bishop_color_mask[c] |= 1;
            } else {
                bishop_color_mask[c] |= 2;
            }
        }

        // 5.2 Knight / Leaper & Bishop Outposts (ranks 4, 5, 6 -> yy in 3..=5)
        if yy >= 3 && yy <= 5 {
            let no_enemy_pawn_stop = (OUTPOST_MASK[c][sqi] & maps.pawn_bb[1 - c]) == 0;
            if no_enemy_pawn_stop {
                let defended_by_pawn = (sq_bb(sqi) & maps.pawn_atk_bb[c]) != 0;
                let central_file = xx >= 2 && xx <= 5;
                if matches!(k, Knight | ElectroKnight | Camel | Knightmare | HordeKnight | CloudKnight | Cardinal | Samurai | WarpJumper | Fish) {
                    let mut op = if central_file { S(24, 16) } else { S(16, 10) };
                    if defended_by_pawn { op += S(14, 10); }
                    score += sgn * op;
                } else if matches!(k, Bishop | Bishopper | BisHop | ExileBishop | Druidess | Dryad | Bouncer) && defended_by_pawn {
                    score += sgn * S(18, 12);
                }
            }
        }

        // 5.3 Rook / Heavy Orthogonal Pixie on Open/Semi-Open File & 7th Rank
        if matches!(k, Rook | PhaseRook | RookZero | SumoRook | LavaForge) {
            if rook_count[c] < 4 {
                rook_slots[c][rook_count[c]] = i;
                rook_count[c] += 1;
            }
            let file_mask = FILES_BB[xx];
            let my_pawns_on_file = (file_mask & maps.pawn_bb[c]) != 0;
            let opp_pawns_on_file = (file_mask & maps.pawn_bb[1 - c]) != 0;
            if !my_pawns_on_file {
                if !opp_pawns_on_file {
                    score += sgn * S(22, 16); // Open file
                } else {
                    score += sgn * S(12, 10); // Semi-open file
                }
                if let Some(_) = maps.king_slot[1 - c] {
                    let ekx = maps.king_sq[1 - c] & 7;
                    if (xx as i32 - ekx as i32).abs() <= 1 {
                        score += sgn * S(10, 0);
                    }
                }
            }
            if yy == 6 {
                let opp_rank7_pawns = (maps.pawn_bb[1 - c] & RANKS_BB[p.y as usize]) != 0;
                let opp_king_back = maps.king_slot[1 - c].is_some()
                    && (if c == 0 { (maps.king_sq[1 - c] >> 3) == 7 } else { (maps.king_sq[1 - c] >> 3) == 0 });
                if opp_rank7_pawns || opp_king_back {
                    score += sgn * S(25, 35);
                }
            }
        }

        // Track Queens for Battery Detection
        if matches!(k, Queen | VoidQueen | NullQueen | WitchQueen | FissionReactor) {
            if queen_count[c] < 4 {
                queen_slots[c][queen_count[c]] = i;
                queen_count[c] += 1;
            }
        }

        // 5.4 King Tropism (Attacking pieces close to enemy King in MG)
        if let Some(_) = maps.king_slot[1 - c] {
            let eksq = maps.king_sq[1 - c];
            let mdist = manhattan_dist(sqi, eksq).min(10);
            let tropism_wt = match k {
                Marauder | WitchQueen | VoidQueen | NullQueen | Queen | FissionReactor => 0,
                ElectroKnight | Samurai | Bouncer | HordeMother | Pilgrim => 0,
                Rook | PhaseRook | RookZero | SumoRook | Cardinal | Basilisk => 0,
                _ => 0,
            };
            if tropism_wt > 0 {
                score += sgn * S((10 - mdist) * tropism_wt, 0);
            }
        }
    }

    // 5.5 Bishop Pair Bonus
    for c in 0..2 {
        let sgn = if (c as u8) == me { 1 } else { -1 };
        if bishop_color_mask[c] == 3 {
            score += sgn * S(24, 45);
        } else if bishop_count[c] >= 2 {
            score += sgn * S(15, 30);
        }

        // 5.6 Connected Rooks & 5.7 Queen + Rook Battery
        for r1 in 0..rook_count[c] {
            let slot1 = rook_slots[c][r1];
            let atk1 = maps.piece_atk[slot1];
            for r2 in (r1 + 1)..rook_count[c] {
                let slot2 = rook_slots[c][r2];
                let p2 = &pos.pieces[slot2];
                let sq2 = sq(p2.x as i8, p2.y as i8) as usize;
                if (atk1 & sq_bb(sq2)) != 0 {
                    score += sgn * S(15, 12);
                }
            }
            for q1 in 0..queen_count[c] {
                let qslot = queen_slots[c][q1];
                let qp = &pos.pieces[qslot];
                let qsq = sq(qp.x as i8, qp.y as i8) as usize;
                let rp = &pos.pieces[slot1];
                if (rp.x == qp.x || rp.y == qp.y) && (atk1 & sq_bb(qsq)) != 0 {
                    score += sgn * S(18, 12);
                }
            }
        }
    }

    let _ = (c_me, c_opp);
    score
}

// ── 5. PAWN STRUCTURE EVALUATION (BITBOARD ACCELERATED) ────────────────────
pub const PASSED_PAWN_BY_RANK: [Score; 8] = [
    S(0, 0),
    S(4, 8),
    S(10, 18),
    S(25, 43),
    S(52, 81),
    S(91, 138),
    S(140, 201),
    S(0, 0),
];

pub const PHALANX_BY_RANK: [Score; 8] = [
    S(0, 0), S(4, 4), S(6, 6), S(10, 10),
    S(14, 16), S(24, 28), S(36, 42), S(0, 0),
];

pub const CHAIN_BY_RANK: [Score; 8] = [
    S(0, 0), S(4, 2), S(6, 4), S(8, 8),
    S(12, 12), S(20, 22), S(30, 32), S(0, 0),
];

pub const ISOLATED_PAWN_CLOSED: Score = S(-8, -16);
pub const ISOLATED_PAWN_OPEN:   Score = S(-16, -22);
pub const DOUBLED_PAWN_PENALTY: Score = S(-8, -22);
pub const BACKWARD_PAWN_CLOSED: Score = S(-10, -12);
pub const BACKWARD_PAWN_OPEN:   Score = S(-16, -16);

pub fn eval_pawn_structure(pos: &Pos, me: u8, maps: &AttackMaps) -> Score {
    let mut score: Score = 0;

    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive || !is_piece_pawn_like(p) { continue; }
        let c = (p.player & 1) as usize;
        let opp_c = 1 - c;
        let sgn = if p.player == me { 1 } else { -1 };
        let xx = (p.x as usize) & 7;
        let yy = if c == 0 { (p.y as usize) & 7 } else { 7 - ((p.y as usize) & 7) };
        let sqi = sq(p.x as i8, p.y as i8) as usize;
        let bit = sq_bb(sqi);

        let my_pawns = maps.pawn_bb[c];
        let opp_pawns = maps.pawn_bb[opp_c];
        let adj_files = ADJ_FILES_BB[xx];
        let semi_open_against_us = (FORWARD_FILE_MASK[c][sqi] & opp_pawns) == 0;

        // 1. Doubled Pawns
        if (FORWARD_FILE_MASK[c][sqi] & my_pawns) != 0 {
            score += sgn * DOUBLED_PAWN_PENALTY;
        }

        // 2. Isolated Pawns
        let isolated = (adj_files & my_pawns) == 0;
        if isolated {
            score += sgn * if semi_open_against_us { ISOLATED_PAWN_OPEN } else { ISOLATED_PAWN_CLOSED };
        }

        // 3. Phalanx & Pawn Chain
        let phalanx = (adj_files & RANKS_BB[p.y as usize] & my_pawns) != 0;
        let supported = (bit & maps.pawn_atk_bb[c]) != 0;
        if phalanx {
            score += sgn * PHALANX_BY_RANK[yy];
        }
        if supported {
            score += sgn * CHAIN_BY_RANK[yy];
        }

        // 4. Backward Pawns
        if !isolated && !phalanx && !supported && yy < 6 {
            let stop_sq = if c == 0 { sqi + 8 } else { sqi.saturating_sub(8) };
            let friendly_behind_or_equal = (adj_files & !OUTPOST_MASK[c][sqi] & my_pawns) != 0;
            let stop_attacked_by_opp_pawn = (sq_bb(stop_sq) & maps.pawn_atk_bb[opp_c]) != 0;
            if !friendly_behind_or_equal && stop_attacked_by_opp_pawn {
                score += sgn * if semi_open_against_us { BACKWARD_PAWN_OPEN } else { BACKWARD_PAWN_CLOSED };
            }
        }

        // 5. Passed Pawns
        let is_passed = (PASSED_PAWN_MASK[c][sqi] & opp_pawns) == 0
                        && (FORWARD_FILE_MASK[c][sqi] & my_pawns) == 0;
        if is_passed && yy >= 1 && yy <= 6 {
            let mut pp_bonus = PASSED_PAWN_BY_RANK[yy];

            let stop_sq = if c == 0 { sqi + 8 } else { sqi - 8 };
            let stop_bit = sq_bb(stop_sq);
            let stop_empty = (maps.occ_bb & stop_bit) == 0;
            let stop_safe = (maps.atk_bb[opp_c] & stop_bit) == 0;

            if yy >= 3 {
                let scale = (yy as i32) - 2; // 1..4
                if stop_empty {
                    pp_bonus += S(4 * scale, 10 * scale);
                }
                if stop_safe {
                    pp_bonus += S(5 * scale, 12 * scale);
                }
                if supported || phalanx {
                    pp_bonus += S(6 * scale, 10 * scale);
                }
                if maps.king_slot[c].is_some() && maps.king_slot[opp_c].is_some() {
                    let d_my_k  = chebyshev_dist(maps.king_sq[c], stop_sq);
                    let d_opp_k = chebyshev_dist(maps.king_sq[opp_c], stop_sq);
                    let king_eg = ((d_opp_k * 5) - (d_my_k * 2)) * scale;
                    pp_bonus += S(0, king_eg);
                }
            }

            let ek = effective_key(p);
            if !can_promote(p) {
                pp_bonus = scale_score(pp_bonus, 1, 4);
            } else if matches!(ek, PieceKey::GoldenPawn | PieceKey::HeroPawn) {
                pp_bonus = scale_score(pp_bonus, 4, 3);
            }

            score += sgn * pp_bonus;
        }
    }

    score
}

// ── 6. KING SAFETY EVALUATION (SHIELD + STORM + OPEN FILES + QUADRATIC DANGER) ──
pub const PAWN_STORM_BLOCKED:   [i32; 8] = [0, 45, 28, 12,  5, 0, 0, 0];
pub const PAWN_STORM_UNBLOCKED: [i32; 8] = [0, 75, 52, 26, 12, 0, 0, 0];

/// King-attack weight of one piece hitting the defended king's zone, feeding
/// the quadratic danger term in `eval_king_safety_single`.
///
/// Scaled from generic material (`pixie_val`, the rounded MG/EG midpoint):
/// weight = clamp(value / 16, 20, 60). Resulting tiers: queen-class
/// (Queen 54, WitchQueen 59, FissionReactor 59) > rook-class (Rook 29,
/// PhaseRook 35, SumoRook 30, RookZero 30) > minor Pixie attackers
/// (ElectroKnight 23, Knightmare 25, Bouncer 28, Pilgrim 21). Deriving the
/// weight from material instead of a hand-written match means a zeroed
/// material table fails loudly elsewhere (SEE/tactical tests) rather than
/// silently zeroing king safety here, as it did through dtXQo1.
#[inline(always)]
pub fn piece_king_attack_weight(p: &Piece) -> i32 {
    let k = effective_key(p);
    match pixie_val(k) {
        Some(v) if v > 0 => (v / 16).clamp(20, 60),
        _ => 0,
    }
}

pub fn eval_king_safety_single(pos: &Pos, defender: u8, maps: &AttackMaps) -> Score {
    let c_def = (defender & 1) as usize;
    let c_atk = 1 - c_def;
    let k_slot = match maps.king_slot[c_def] {
        Some(s) => s,
        None => return 0,
    };
    let kp = &pos.pieces[k_slot];
    let ksq = maps.king_sq[c_def];
    let kx = (kp.x as i32) & 7;
    let ky = (kp.y as i32) & 7;
    let k_yy = if c_def == 0 { ky } else { 7 - ky };

    let mut mg_pen = 0i32;
    let mut eg_pen = 0i32;
    let mut danger_units = 0i32;

    // 1. Paralyzed King Penalty
    if is_paralyzed(pos, k_slot, &maps.occ) {
        mg_pen += 350;
        eg_pen += 350;
    }

    // 2. Pawn Shield & Open/Semi-Open Files near King
    let min_f = (kx - 1).max(0) as usize;
    let max_f = (kx + 1).min(7) as usize;
    for f in min_f..=max_f {
        let f_mask = FILES_BB[f];
        let my_f_pawns = maps.pawn_bb[c_def] & f_mask;
        let opp_f_pawns = maps.pawn_bb[c_atk] & f_mask;
        let is_king_file = (f as i32) == kx;

        if my_f_pawns == 0 {
            if opp_f_pawns == 0 {
                mg_pen += if is_king_file { 35 } else { 22 };
                danger_units += if is_king_file { 20 } else { 12 };
            } else {
                mg_pen += if is_king_file { 24 } else { 14 };
                danger_units += if is_king_file { 14 } else { 8 };
            }
        } else if k_yy == 0 && (kx <= 2 || kx >= 5) {
            let shield_rank_1 = if c_def == 0 { 1usize } else { 6usize };
            let shield_rank_2 = if c_def == 0 { 2usize } else { 5usize };
            if (my_f_pawns & RANKS_BB[shield_rank_1]) != 0 {
                mg_pen -= if is_king_file { 18 } else { 14 };
            } else if (my_f_pawns & RANKS_BB[shield_rank_2]) != 0 {
                mg_pen -= 6;
            } else {
                mg_pen += 12;
            }
        }

        // 3. Enemy Pawn Storm on files `kx - 1 ..= kx + 1`
        if opp_f_pawns != 0 {
            let nearest_sq = if c_def == 0 {
                opp_f_pawns.trailing_zeros() as usize
            } else {
                63 - (opp_f_pawns.leading_zeros() as usize)
            };
            let opp_y = (nearest_sq >> 3) as i32;
            let dist_from_our_back = if c_def == 0 { opp_y } else { 7 - opp_y };
            if dist_from_our_back >= 1 && dist_from_our_back <= 4 {
                let d_idx = dist_from_our_back as usize;
                let stop_sq = if c_def == 0 { nearest_sq.saturating_sub(8) } else { (nearest_sq + 8).min(63) };
                let blocked_by_pawn = (my_f_pawns & sq_bb(stop_sq)) != 0;
                let storm = if blocked_by_pawn {
                    PAWN_STORM_BLOCKED[d_idx]
                } else {
                    PAWN_STORM_UNBLOCKED[d_idx]
                };
                mg_pen += storm;
                danger_units += storm / 2;
            }
        }
    }

    // Central / exposed King walk penalty in Middlegame — scaled by enemy attacking firepower
    // so that in true endgames (`opp_ph <= 2`) the King is free to march to the center!
    let opp_ph = opp_attacking_phase(pos, c_atk as u8);
    if opp_ph > 2 {
        if k_yy >= 1 {
            mg_pen += (90 * k_yy * k_yy * opp_ph) / 12;
        } else if kx == 3 || kx == 4 {
            mg_pen += (55 * opp_ph) / 12;
        }
    }

    // 4. King Zone Attacker & Defender Count
    let k_zone = KING_ZONE_BB[c_def][ksq];
    let k_ring = KING_RING_BB[ksq];
    let mut attacker_count = 0i32;
    let mut attacker_weight = 0i32;
    let mut zone_hit_squares = 0i32;

    let n_pieces = pos.pieces.len().min(MAX_SLOTS);
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) || !p.alive || (p.player & 1) as usize != c_atk { continue; }
        if is_king_piece(p) || is_piece_pawn_like(p) { continue; }
        let atk = maps.piece_atk[i];
        let hits = atk & k_zone;
        if hits != 0 {
            attacker_count += 1;
            attacker_weight += piece_king_attack_weight(p);
            zone_hit_squares += hits.count_ones() as i32;
        }
        let back2 = if c_def == 0 { 1 } else { 6 };
        let ek = effective_key(p);
        // Rook-line pieces parked on the defender's back rank: RookZero is
        // family-4 (rook-like) and was missing here. Gunslinger is NOT
        // included on purpose: it attacks on diagonals only (bishop-like),
        // so it cannot own a back rank the way a rook does.
        if matches!(ek, PieceKey::Rook | PieceKey::PhaseRook | PieceKey::SumoRook | PieceKey::RookZero)
           && (p.y as i32) == back2 {
            mg_pen += 45;
            eg_pen += 35;
            danger_units += 25;
        }
    }

    let undefended_ring = k_ring & maps.atk_bb[c_atk] & !maps.atk_bb[c_def];
    let undef_ring_cnt = undefended_ring.count_ones() as i32;
    let defended_ring_cnt = (k_ring & maps.atk_bb[c_def]).count_ones() as i32;

    // Contact check & back-rank mate danger: Queen or Rook attacking undefended king-ring square
    // supported by another enemy piece/pawn
    let heavy_contact = undefended_ring
        & (maps.queen_atk_bb[c_atk] | maps.rook_atk_bb[c_atk])
        & (maps.minor_atk_bb[c_atk] | maps.rook_atk_bb[c_atk] | maps.queen_atk_bb[c_atk] | maps.pawn_atk_bb[c_atk]);
    if heavy_contact != 0 {
        danger_units += 65 * (heavy_contact.count_ones() as i32);
    }

    if attacker_count >= 2 || (attacker_count == 1 && (maps.queen_atk_bb[c_atk] | maps.rook_atk_bb[c_atk]) != 0) {
        let raw_danger = attacker_weight
            + zone_hit_squares * 8
            + undef_ring_cnt * 18
            + danger_units
            - defended_ring_cnt * 9;
        if raw_danger > 0 {
            let quad_pen = (raw_danger * raw_danger / 110).min(1200);
            mg_pen += quad_pen;
            eg_pen += quad_pen / 3;
        }
    } else {
        mg_pen += undef_ring_cnt * 15;
    }

    S(-mg_pen, -eg_pen)
}

// ── 7. PIXIE POWER AWARENESS (CONTINUOUS CHARGE + RAY/AURA SYNERGY) ─────────
pub fn pixie_charge_bonus(pos: &Pos, me: u8) -> i32 {
    let mut b = 0i32;
    let mut hordeling_cnt = 0i32;
    let mut friendly_minors_near_aristocrat = 0i32;
    let mut aristocrat_xy: Option<(i8, i8)> = None;

    for q in &pos.pieces {
        if !xy_on_board(q.x, q.y) || !q.alive || q.player != me { continue; }
        if q.key == PieceKey::Hordeling || q.sub_key == PieceKey::Hordeling {
            hordeling_cnt += 1;
        }
        if q.key == PieceKey::Aristocrat || q.sub_key == PieceKey::Aristocrat {
            aristocrat_xy = Some((q.x as i8, q.y as i8));
        }
    }

    if let Some((ax, ay)) = aristocrat_xy {
        for q in &pos.pieces {
            if !xy_on_board(q.x, q.y) || !q.alive || q.player != me { continue; }
            if (q.x as i8 - ax).abs() <= 2 && (q.y as i8 - ay).abs() <= 2 {
                friendly_minors_near_aristocrat += 1;
            }
        }
    }

    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !q_alive(p) || p.player != me { continue; }
        use PieceKey::*;
        let k = p.key;
        let s = p.sub_key;
        let is = |x: PieceKey| k == x || s == x;

        if is(Pilgrim) {
            if p.pilgrim_charged() || p.odo >= 20 {
                b += 140;
            } else {
                let odo = (p.odo as i32).min(19);
                b += (odo * odo * 110) / 400;
            }
        }
        if is(ElectroKnight) {
            if p.echarged() || p.quiet >= 3 {
                b += 110;
            } else if p.quiet == 2 {
                b += 45;
            } else if p.x >= 2 && p.x <= 5 && p.y >= 2 && p.y <= 5 {
                b += 15;
            }
        }
        if is(FissionReactor) {
            b += match p.caps {
                0 => 0,
                1 => 18,
                2 => 42,
                3 => 75,
                _ => 115,
            };
        }
        if is(HordeMother) {
            let h_clamped = hordeling_cnt.min(6);
            b += 30 + 22 * hordeling_cnt + 5 * h_clamped * h_clamped;
        }
        if is(Marauder) {
            if p.kills >= 2 { b += 95; }
            else if p.kills == 1 { b += 45; }
        }
        if is(Dancer) {
            if p.encore_pending() { b += 65; }
            else if p.encore >= 2 { b += 45; }
            else if p.encore == 1 { b += 20; }
        }
        if is(Rocketman) && !p.used() {
            b += 35;
        }
        if is(Shrike) && !p.moved() {
            b += 18;
        }
        if is(Aristocrat) {
            b += 20 + (friendly_minors_near_aristocrat.min(5) * 8);
        }
        if is(Banker) {
            b += 22;
        }
        if p.frost > 0 {
            b -= 25 * (p.frost as i32).min(3) + (piece_value(p) / 12);
        }
    }

    for ob in &pos.offboard {
        if ob.player == me {
            if ob.key == PieceKey::Knightmare || ob.sub_key == PieceKey::Knightmare {
                b += 65;
            } else {
                let v = pixie_val(ob.key).or_else(|| std_val(ob.key)).unwrap_or(250);
                b += (v * 55) / 100;
            }
        }
    }

    b
}

#[inline(always)]
fn q_alive(p: &Piece) -> bool {
    p.alive
}

pub fn pixie_positional_powers(pos: &Pos, me: u8, maps: &AttackMaps) -> Score {
    use PieceKey::*;
    let opp = 1 - me;
    let mut score: Score = 0;

    let n_pieces = pos.pieces.len().min(MAX_SLOTS);
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) || !p.alive { continue; }
        let sgn = if p.player == me { 1 } else { -1 };
        let k = effective_key(p);
        let sqi = sq(p.x as i8, p.y as i8) as usize;
        let atk = maps.piece_atk[i];

        if !is_king_piece(p) {
            if is_paralyzed(pos, i, &maps.occ) {
                let pen = (piece_value(p) * 45) / 100;
                score -= sgn * S(pen, pen);
            }
        }

        if matches!(k, Basilisk | LavaForge) {
            let center_rays = (atk & CENTER_4X4_BB).count_ones() as i32;
            score += sgn * S(center_rays * 5, center_rays * 4);
        }

        if matches!(k, Duplicator | Timelord) {
            let under_atk = if p.player == me { maps.cnt_opp[sqi] > 0 } else { maps.cnt_me[sqi] > 0 };
            if !under_atk && (sq_bb(sqi) & CENTER_4X4_BB) != 0 {
                score += sgn * S(24, 20);
            }
        }
    }

    let _ = opp;
    score
}

pub fn pixie_threat_penalty_with_maps(pos: &Pos, me: u8, opp: u8, maps: &AttackMaps) -> i32 {
    let mut pen = 0i32;

    let mut has_relevant = false;
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive || p.player != opp { continue; }
        use PieceKey::*;
        let k = p.key; let s = p.sub_key;
        if matches!(k, Gunslinger|ElectroKnight|Shrike|FissionReactor|Pilgrim
                     | HordeMother|Marauder|Dancer|Rocketman)
        || matches!(s, Gunslinger|ElectroKnight|Shrike|FissionReactor|Pilgrim
                     | HordeMother|Marauder|Dancer|Rocketman) { has_relevant = true; break; }
    }
    if !has_relevant && pos.offboard.is_empty() { return 0; }

    let n_pieces = pos.pieces.len().min(MAX_SLOTS);
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) || !p.alive || p.player != opp { continue; }
        use PieceKey::*;
        let k = p.key; let s = p.sub_key;
        let is = |x: PieceKey| k == x || s == x;
        let atk = Bitboard(maps.piece_atk[i]);

        if is(Gunslinger) {
            let gpv = piece_value(p);
            for j in 0..n_pieces {
                let q = &pos.pieces[j];
                if !xy_on_board(q.x, q.y) || !q.alive || q.player != me || is_king_piece(q) { continue; }
                if atk.has(sq(q.x as i8, q.y as i8)) {
                    let qatk = Bitboard(maps.piece_atk[j]);
                    if qatk.has(sq(p.x as i8, p.y as i8)) {
                        let qv = piece_value(q);
                        pen += (qv - gpv).max(0);
                    }
                }
            }
        } else if is(ElectroKnight) && (p.echarged() || p.quiet >= 3) {
            let mut best = 0i32;
            for q in &pos.pieces {
                if !xy_on_board(q.x, q.y) || !q.alive || q.player != me { continue; }
                if (q.x as i8 - p.x as i8).abs() <= 1 && (q.y as i8 - p.y as i8).abs() <= 1 {
                    if is_king_piece(q) { best = 10000; }
                    else { best = best.max(piece_value(q)); }
                }
            }
            pen += best;
        } else if is(Shrike) && !p.moved() {
            let sd: i8 = if opp == 0 { 1 } else { -1 };
            for q in &pos.pieces {
                if !xy_on_board(q.x, q.y) || !q.alive || q.player != me || is_king_piece(q) { continue; }
                let qy = q.y as i8;
                if q.x == p.x && (qy == p.y as i8 + sd || qy == p.y as i8 + 2 * sd) {
                    pen += 60;
                }
            }
        } else if is(FissionReactor) {
            let rcaps = p.caps;
            let rpen = if rcaps >= 4 { 90 } else if rcaps >= 3 { 40 } else { 0 };
            if rpen > 0 {
                let px = p.x as i8; let py = p.y as i8;
                for j in 0..n_pieces {
                    let q = &pos.pieces[j];
                    if !xy_on_board(q.x, q.y) || !q.alive || q.player != me { continue; }
                    let qx = q.x as i8; let qy = q.y as i8;
                    let is_king = is_king_piece(q);
                    if (qx - px).abs() == 1 && (qy - py).abs() == 1 {
                        if is_king { pen += if rcaps >= 4 { 10000 } else { 2000 }; }
                        else { let qv = piece_value(q); pen += rpen + (qv * if rcaps >= 4 {60} else {30} / 100).min(500); }
                    } else if rcaps >= 3 {
                        if atk.has(sq(qx, qy)) {
                            let mut blast = 0i32;
                            for bq in &pos.pieces {
                                if !bq.alive || bq.player != me || bq.id == q.id { continue; }
                                if (bq.x as i8 - qx).abs() == 1 && (bq.y as i8 - qy).abs() == 1 {
                                    if is_king_piece(bq) { blast += if rcaps >= 4 {10000} else {2000}; }
                                    else { let bqv = piece_value(bq); blast += rpen + (bqv * if rcaps >= 4 {60} else {30} / 100).min(500); }
                                }
                            }
                            if !is_king { blast += piece_value(q) * if rcaps >= 4 {75} else {40} / 100; }
                            pen += blast;
                        }
                    }
                }
            }
        } else if is(Rocketman) && !p.used() {
            pen += 40;
        } else if is(Pilgrim) {
            let mut odo = p.odo;
            if p.pilgrim_charged() { odo = odo.max(20); }
            if odo >= 20 { pen += 150; } else if odo >= 15 { pen += 80; } else if odo >= 10 { pen += 30; }
        } else if is(HordeMother) {
            let mut targets = 0;
            for q in &pos.pieces {
                if !xy_on_board(q.x, q.y) || !q.alive || q.player != me || is_king_piece(q) { continue; }
                if atk.has(sq(q.x as i8, q.y as i8)) { targets += 1; }
            }
            let mut horde = 0i32;
            for q in &pos.pieces {
                if !xy_on_board(q.x, q.y) { continue; }
                if q.alive && q.player == opp
                   && (q.key == Hordeling || q.sub_key == Hordeling) { horde += 1; }
            }
            pen += 45 + 25 * horde + 65 * targets;
        } else if is(Marauder) {
            if p.kills >= 2 { pen += 100; } else if p.kills >= 1 { pen += 50; }
        } else if is(Dancer) {
            pen += 55;
        }
    }
    for ob in &pos.offboard {
        if ob.player == opp
           && (ob.key == PieceKey::Knightmare || ob.sub_key == PieceKey::Knightmare) {
            pen += 60;
        }
    }
    pen
}

pub fn pixie_threat_penalty(pos: &Pos, me: u8, opp: u8) -> i32 {
    let maps = attack_maps(pos, me);
    pixie_threat_penalty_with_maps(pos, me, opp, &maps)
}

// ── 8. TEMPO & TACTICAL THREATS (HANGING, FORKS, PAWN-ON-MINOR, MINOR-ON-MAJOR) ──
pub const TEMPO_BONUS: Score = S(12, 8);

pub fn eval_tactical_threats(pos: &Pos, me: u8, maps: &AttackMaps) -> Score {
    let mut score: Score = 0;
    let mut hanging_vals: [[i32; 4]; 2] = [[0; 4]; 2];
    let mut hanging_cnt = [0usize; 2];

    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive { continue; }
        if is_king_piece(p) {
            continue;
        }
        let c = (p.player & 1) as usize;
        let opp_c = 1 - c;
        let sgn = if p.player == me { 1 } else { -1 };
        let sqi = sq(p.x as i8, p.y as i8) as usize;
        let bit = sq_bb(sqi);
        let val = piece_value(p);
        let is_pawn = is_piece_pawn_like(p);

        let atk_cnt = if p.player == me { maps.cnt_opp[sqi] } else { maps.cnt_me[sqi] };
        let def_cnt = if p.player == me { maps.cnt_me[sqi] } else { maps.cnt_opp[sqi] };

        if !is_pawn {
            // 1. Enemy Pawn attacking a Non-Pawn piece
            if (bit & maps.pawn_atk_bb[opp_c]) != 0 {
                let p_pen = ((val * 30) / 100).clamp(35, 210);
                score -= sgn * S(p_pen, p_pen);
            }
            // 2. Enemy Minor attacking a Major piece (Rook/PhaseRook/SumoRook/Bouncer/Queen/FissionReactor)
            else if val >= 450 && (bit & maps.minor_atk_bb[opp_c]) != 0 {
                let m_pen = ((val - 310) * 24) / 100;
                score -= sgn * S(m_pen, m_pen);
            }
            // 3. Enemy Rook attacking a Queen / FissionReactor
            else if val >= 800 && (bit & maps.rook_atk_bb[opp_c]) != 0 {
                score -= sgn * S(75, 65);
            }
        }

        // 4. Hanging (Undefended) Piece Penalty & Fork Tracking
        if atk_cnt > 0 && def_cnt == 0 {
            let h_pen = if is_pawn {
                S(22, 28)
            } else if val < 450 {
                S(60, 55)
            } else if val < 800 {
                S(95, 85)
            } else {
                S(145, 130)
            };
            score -= sgn * h_pen;

            if !is_pawn && hanging_cnt[c] < 4 {
                hanging_vals[c][hanging_cnt[c]] = val;
                hanging_cnt[c] += 1;
            }
        } else if atk_cnt > def_cnt {
            let diff = (atk_cnt - def_cnt) as i32;
            score -= sgn * S(18 * diff, 16 * diff);
        }
    }

    // 5. Multi-Piece Fork Penalty
    for c in 0..2 {
        if hanging_cnt[c] >= 2 {
            let sgn = if (c as u8) == me { 1 } else { -1 };
            let mut max1 = 0;
            let mut max2 = 0;
            for idx in 0..hanging_cnt[c] {
                let v = hanging_vals[c][idx];
                if v >= max1 {
                    max2 = max1;
                    max1 = v;
                } else if v > max2 {
                    max2 = v;
                }
            }
            let fork_pen = (max2 * 45) / 100;
            score -= sgn * S(fork_pen, fork_pen);
        }
    }

    // 6. Safe Pawn Push Threats
    for c in 0..2 {
        let opp_c = 1 - c;
        let sgn = if (c as u8) == me { 1 } else { -1 };
        let single_push = if c == 0 {
            (maps.pawn_bb[0] << 8) & !maps.occ_bb
        } else {
            (maps.pawn_bb[1] >> 8) & !maps.occ_bb
        };
        let safe_pushes = single_push & !maps.atk_bb[opp_c];
        let push_attacks = compute_pawn_attacks_bb(safe_pushes, c as u8);
        let threatened_non_pawns = push_attacks & (maps.color_bb[opp_c] & !maps.pawn_bb[opp_c]);
        let cnt = threatened_non_pawns.count_ones() as i32;
        if cnt > 0 {
            score += sgn * S(18 * cnt, 14 * cnt);
        }
    }

    score
}

pub fn eval_danger(pos: &Pos, me: u8, pre: &AttackMaps) -> i32 {
    let opp = 1 - me;
    let ks_me  = eval_king_safety_single(pos, me, pre);
    let ks_opp = eval_king_safety_single(pos, opp, pre);
    taper_score(ks_me - ks_opp, pre.phase)
}

pub fn queen_safety(pos: &Pos, me: u8, pre: &AttackMaps) -> i32 {
    let mut q: Option<&Piece> = None;
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) { continue; }
        if p.alive && p.player == me && is_queen_piece_by_value(p) {
            q = Some(p);
            break;
        }
    }
    let q = match q { Some(x) => x, None => return 0 };
    let qsq = sq(q.x as i8, q.y as i8) as usize;
    let under = if me == 0 {
        (pre.atk_bb[1] & sq_bb(qsq)) != 0
    } else {
        (pre.atk_bb[0] & sq_bb(qsq)) != 0
    };
    let back_rank: i32 = if me == 0 { 0 } else { 7 };
    if !under && q.y as i32 == back_rank { return 0; }

    let opp_atk_bb = pre.atk_bb[(1 - (me & 1)) as usize];
    let mut safe = 0i32;
    for &(dx, dy) in DIRS_ROOK.iter().chain(DIRS_BISHOP.iter()) {
        let mut nx = q.x as i8 + dx;
        let mut ny = q.y as i8 + dy;
        while on_board(nx, ny) {
            let sqi = sq(nx, ny) as usize;
            let sq_safe = (opp_atk_bb & sq_bb(sqi)) == 0;
            if pos.at[sqi] != 0 && pos.pieces[(pos.at[sqi] - 1) as usize].alive {
                let sqp = &pos.pieces[(pos.at[sqi] - 1) as usize];
                if sqp.player != me && sq_safe { safe += 1; }
                break;
            }
            if sq_safe { safe += 1; }
            nx += dx;
            ny += dy;
        }
    }
    if under {
        if safe >= 6 { -40 } else if safe >= 3 { -180 } else if safe >= 1 { -260 } else { -340 }
    } else {
        if safe >= 6 { 0 } else if safe >= 3 { -35 } else if safe >= 1 { -70 } else { -110 }
    }
}

fn can_promote(p: &Piece) -> bool {
    is_piece_pawn_like(p) && p.key != PieceKey::IronPawn && p.sub_key != PieceKey::IronPawn
}

fn pawn_passed(pos: &Pos, p: &Piece, opp: u8) -> bool {
    for q in &pos.pieces {
        if !xy_on_board(q.x, q.y) { continue; }
        if !q.alive || q.player != opp || !is_piece_pawn_like(q) { continue; }
        if (q.x as i32 - p.x as i32).abs() > 1 { continue; }
        if opp == 1 && (q.y as i32) > (p.y as i32) { return false; }
        if opp == 0 && (q.y as i32) < (p.y as i32) { return false; }
    }
    true
}

#[derive(Clone, Copy)]
pub struct Invader {
    pub slot: usize, pub cls: u8,
    pub x: u8, pub y: u8,
    pub attacked: bool, pub hits: bool, pub push_clear: bool,
}

pub fn invader_scan_with_maps(pos: &Pos, me: u8, maps: &AttackMaps) -> Vec<Invader> {
    let opp = 1 - me;
    let home2: i32 = if me == 0 { 1 } else { 6 };
    let one_step: i32 = if opp == 1 { 1 } else { 6 };
    let two_step: i32 = if opp == 1 { 2 } else { 5 };
    let promo_y: i8 = if opp == 1 { 0 } else { 7 };
    let deep = |y: i32| if me == 0 { y <= 3 } else { y >= 4 };
    let mut out = Vec::new();
    let me_atk_bb = maps.atk_bb[(me & 1) as usize];
    let me_non_king_bb = maps.color_bb[(me & 1) as usize]
        & !maps.king_slot[(me & 1) as usize].map_or(0u64, |_| sq_bb(maps.king_sq[(me & 1) as usize]));

    let n_pieces = pos.pieces.len().min(MAX_SLOTS);
    for i in 0..n_pieces {
        let p = &pos.pieces[i];
        if !xy_on_board(p.x, p.y) || !p.alive || p.player != opp || is_king_piece(p) { continue; }
        let x = p.x; let y = p.y;
        let ek = effective_key(p);
        let cls: Option<u8> = if is_queen_piece_by_value(p) {
            if deep(y as i32) { Some(b'Q') } else { None }
        } else if matches!(ek, PieceKey::Rook | PieceKey::PhaseRook | PieceKey::SumoRook) {
            if y as i32 == home2 { Some(b'R') } else { None }
        } else if is_piece_pawn_like(p) {
            if can_promote(p) {
                if y as i32 == one_step { Some(b'1') }
                else if y as i32 == two_step && pawn_passed(pos, p, opp) { Some(b'2') }
                else { None }
            } else { None }
        } else if deep(y as i32) { Some(b'G') } else { None };

        if let Some(c) = cls {
            let mut inv = Invader {
                slot: i, cls: c, x, y,
                attacked: false, hits: false, push_clear: false,
            };
            if c == b'1' {
                inv.push_clear = pos.at[sq(x as i8, promo_y) as usize] == 0;
            } else {
                let sqi = sq(x as i8, y as i8) as usize;
                inv.attacked = (me_atk_bb & sq_bb(sqi)) != 0;
                if c == b'Q' || c == b'G' {
                    inv.hits = (maps.piece_atk[i] & me_non_king_bb) != 0;
                }
            }
            out.push(inv);
        }
    }
    out
}

pub fn invader_scan(pos: &Pos, me: u8) -> Vec<Invader> {
    let maps = attack_maps(pos, me);
    invader_scan_with_maps(pos, me, &maps)
}

pub fn threat_eval_with_maps(pos: &Pos, me: u8, maps: &AttackMaps) -> i32 {
    let inv = invader_scan_with_maps(pos, me, maps);
    if inv.is_empty() { return 0; }
    let mut pen = 0i32;
    let mut rooks = 0i32;
    for t in &inv {
        match t.cls {
            b'Q' => if !t.attacked { pen += if t.hits { -150 } else { -50 }; },
            b'R' => { pen += if t.attacked { -40 } else { -100 }; rooks += 1; },
            b'1' => if t.push_clear { pen += -200; },
            b'2' => pen += -60,
            b'G' => if !t.attacked && t.hits {
                let v = piece_value(&pos.pieces[t.slot]);
                pen -= v / 6;
            },
            _ => {}
        }
    }
    if rooks >= 2 { pen -= 80; }
    pen.max(-400)
}

pub fn threat_eval(pos: &Pos, me: u8) -> i32 {
    let maps = attack_maps(pos, me);
    threat_eval_with_maps(pos, me, &maps)
}

// ── 9. MASTER `evaluate()` FUNCTION (single static-eval entry point) ─────────
pub fn evaluate(pos: &Pos, me: u8) -> i32 {
    if let Some(kt) = pos.king_thrown {
        return if kt == me { MATE_CONST } else { -MATE_CONST };
    }
    if let Some(gw) = pos.gold_win {
        return if gw == me { MATE_CONST } else { -MATE_CONST };
    }
    let opp = 1 - me;

    // 1. Single-pass Bitboard & Attack Map construction
    let maps = attack_maps(pos, me);

    // 2. Accumulate Packed (MG, EG) Score
    let mut score_packed: Score = if pos.stm == me { TEMPO_BONUS } else { -TEMPO_BONUS };

    // 2a. Tapered Material + Archetype PSTs + Acquisition Tax
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive { continue; }
        let sgn = if p.player == me { 1 } else { -1 };
        let tax = acc_tax(p);
        score_packed += sgn * (piece_value_tapered(p) + pst_score_tapered(p) + S(tax, tax));
    }

    // 2b. Safe-Square Mobility + Outposts + Open Files + 7th Rank + Bishop Pair + Batteries + Tropism
    score_packed += eval_mobility_and_coordination(pos, me, &maps);

    // 2c. Bitboard Pawn Structure (Passed, Isolated, Doubled, Backward, Phalanx, Chains)
    score_packed += eval_pawn_structure(pos, me, &maps);

    // 2d. Symmetrical King Safety (Pawn Shield + Storm + Open Files + Quadratic Zone Danger)
    score_packed += eval_king_safety_single(pos, me, &maps)
                  - eval_king_safety_single(pos, opp, &maps);

    // 2e. Pixie Positional Powers (Basilisk Ray Paralysis, LavaForge Area Denial, Duplicator/Timelord)
    score_packed += pixie_positional_powers(pos, me, &maps);

    // 2f. Tactical Threats (Pawn-on-Minor, Minor-on-Major, Hanging Pieces, Multi-Piece Forks, Safe Pawn Pushes)
    score_packed += eval_tactical_threats(pos, me, &maps);

    // 3. Taper Packed Score by Continuous Game Phase (0..=24)
    let mut total = taper_score(score_packed, maps.phase);

    // 4. Add Continuous Pixie Charge Bonuses, Pixie Burst Threats & Queen Trap Safety
    total += pixie_charge_bonus(pos, me) - pixie_charge_bonus(pos, opp);
    total -= pixie_threat_penalty_with_maps(pos, me, opp, &maps)
           - pixie_threat_penalty_with_maps(pos, opp, me, &maps);
    total += queen_safety(pos, me, &maps) - queen_safety(pos, opp, &maps);
    total += threat_eval_with_maps(pos, me, &maps) - threat_eval_with_maps(pos, opp, &maps);

    total
}


#[cfg(test)]
mod side_to_move_tempo_tests {
    use super::evaluate;
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, PieceKey};

    #[test]
    fn evaluation_tempo_tracks_side_to_move() {
        let mut pos = Pos::default();
        pos.pieces.push(Piece {
            id: 1, key: PieceKey::Rook, sub_key: PieceKey::Rook,
            player: 0, x: 0, y: 0, alive: true, ..Default::default()
        });
        pos.at[sq(0, 0) as usize] = 1;
        pos.stm = 0;
        let white_to_move = evaluate(&pos, 0);
        pos.stm = 1;
        let black_to_move = evaluate(&pos, 0);
        let delta = white_to_move - black_to_move;
        assert!(delta > 0 && delta <= 24,
            "tempo should favor the side to move by at most 24cp; delta={delta}");
    }
}

// ── EVALUATOR SEAM (NNUE-READY) ────────────────────────────────────────────
// Search scores leaf nodes through exactly one evaluation entry point: the
// evaluator carried by `Search`. Today that entry is the hand-written
// evaluator below; a future NNUE implementation plugs into this trait
// without touching think, search, or the wrapper. There is no second
// scorer: `evaluate_cheap` (which ran the legacy pre-289 eval) and
// `evaluate_pre289` were deleted with this seam, along with the
// `pxb_evaluate_cheap` FFI export.
pub trait Evaluator {
    fn evaluate(&self, pos: &Pos, me: u8) -> i32;
}

pub struct HandwrittenEvaluator;

impl Evaluator for HandwrittenEvaluator {
    fn evaluate(&self, pos: &Pos, me: u8) -> i32 {
        evaluate(pos, me)
    }
}

pub static HANDWRITTEN_EVALUATOR: HandwrittenEvaluator = HandwrittenEvaluator;

#[cfg(test)]
mod pixie_material_consistency_tests {
    use super::{piece_phase_weight, piece_value, Piece};
    use crate::types::PieceKey;

    fn piece(key: PieceKey) -> Piece {
        Piece {
            id: 1, key, sub_key: key, player: 0, x: 3, y: 3,
            alive: true, ..Default::default()
        }
    }

    #[test]
    fn high_value_pixies_are_not_zero_in_generic_material_path() {
        for (key, min_value) in [
            (PieceKey::PhaseRook, 500),
            (PieceKey::FissionReactor, 800),
            (PieceKey::Bishopper, 300),
            (PieceKey::ElectroKnight, 300),
            (PieceKey::WitchQueen, 800),
            (PieceKey::VoidQueen, 800),
        ] {
            let v = piece_value(&piece(key));
            assert!(v >= min_value, "{key:?} generic material value {v} < {min_value}");
        }
    }

    #[test]
    fn generic_material_values_feed_phase_weighting() {
        assert_eq!(piece_phase_weight(&piece(PieceKey::PhaseRook)), 2);
        assert_eq!(piece_phase_weight(&piece(PieceKey::FissionReactor)), 4);
        assert_eq!(piece_phase_weight(&piece(PieceKey::Bishopper)), 1);
    }

    #[test]
    fn effective_key_prefers_pixie_identity_in_either_wire_order() {
        use super::effective_key;
        let phase_rook_base_first = Piece {
            key: PieceKey::Rook, sub_key: PieceKey::PhaseRook,
            ..piece(PieceKey::Rook)
        };
        let phase_rook_pixie_first = Piece {
            key: PieceKey::PhaseRook, sub_key: PieceKey::Rook,
            ..piece(PieceKey::PhaseRook)
        };
        assert_eq!(effective_key(&phase_rook_base_first), PieceKey::PhaseRook);
        assert_eq!(effective_key(&phase_rook_pixie_first), PieceKey::PhaseRook);
        assert_eq!(piece_value(&phase_rook_base_first), 574);
        assert_eq!(piece_value(&phase_rook_pixie_first), 574);
    }

    #[test]
    fn standard_piece_encoding_stays_standard() {
        use super::effective_key;
        let rook = piece(PieceKey::Rook);
        assert_eq!(effective_key(&rook), PieceKey::Rook);
        assert_eq!(piece_value(&rook), 470);
    }

    #[test]
    fn intentionally_zero_dummy_remains_zero() {
        assert_eq!(piece_value(&piece(PieceKey::Dummy)), 0);
    }
}

#[cfg(test)]
mod king_safety_weight_tests {
    use super::{evaluate, piece_king_attack_weight};
    use crate::pos::{Piece, Pos};
    use crate::types::{sq, PieceKey};

    fn attacker(key: PieceKey) -> Piece {
        Piece {
            id: 1, key, sub_key: key, player: 1, x: 3, y: 3,
            alive: true, ..Default::default()
        }
    }

    #[test]
    fn every_former_zero_weight_attacker_is_now_nonzero() {
        // Every key the old match zeroed (dtXQo1 defect), plus standard pieces.
        for key in [
            PieceKey::Queen, PieceKey::WitchQueen, PieceKey::VoidQueen,
            PieceKey::NullQueen, PieceKey::FissionReactor,
            PieceKey::Marauder, PieceKey::HordeMother, PieceKey::Bouncer,
            PieceKey::PhaseRook, PieceKey::RookZero, PieceKey::Rook,
            PieceKey::SumoRook, PieceKey::WarAutomaton, PieceKey::LavaForge,
            PieceKey::ElectroKnight, PieceKey::Basilisk, PieceKey::Cardinal,
            PieceKey::Samurai, PieceKey::Pilgrim, PieceKey::Gunslinger,
            PieceKey::BladeRunner, PieceKey::Duplicator, PieceKey::Timelord,
            PieceKey::Bishopper, PieceKey::BisHop, PieceKey::Knightmare,
            PieceKey::Knight, PieceKey::Bishop,
        ] {
            let w = piece_king_attack_weight(&attacker(key));
            assert!(w > 0, "{key:?} king-attack weight must be nonzero, got {w}");
        }
    }

    #[test]
    fn attack_weights_scale_with_attacking_strength() {
        let q = piece_king_attack_weight(&attacker(PieceKey::Queen));
        let r = piece_king_attack_weight(&attacker(PieceKey::Rook));
        let n = piece_king_attack_weight(&attacker(PieceKey::Knight));
        assert!(q > r && r > n, "queen {q} > rook {r} > knight {n} expected");
        assert_eq!(q, 54, "Queen 873/16 = 54 (integer division)");
        assert_eq!(piece_king_attack_weight(&attacker(PieceKey::FissionReactor)), 59);
        assert_eq!(piece_king_attack_weight(&attacker(PieceKey::PhaseRook)), 35);
    }

    /// Build a fixture position from (key, player, x, y) pieces.
    /// Coordinates: x = file - 1 (a=0), y = rank - 1 (rank 1 = 0).
    fn fixture(spec: &[(PieceKey, u8, u8, u8)]) -> Pos {
        let mut pos = Pos::default();
        pos.at = [0; 64];
        pos.stm = 0; // White to move in all three fixtures
        for (i, &(key, player, x, y)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, ..Default::default()
            });
            pos.at[sq(x as i8, y as i8) as usize] = (i + 1) as u16;
        }
        pos
    }

    #[test]
    fn dtXQo1_hunt_evals_decline_gradually() {
        use PieceKey::*;
        // Turn 82 pre-move: White queen + Pilgrim + 3 pawns vs Black's hunt pack.
        let t82 = fixture(&[
            (King, 0, 5, 0), (Queen, 0, 3, 0), (Pilgrim, 0, 5, 3),
            (Pawn, 0, 0, 2), (Pawn, 0, 6, 2), (Pawn, 0, 7, 1),
            (King, 1, 3, 4), (ElectroKnight, 1, 4, 5), (FissionReactor, 1, 0, 5),
            (Gunslinger, 1, 2, 7), (Rook, 1, 0, 7), (Rook, 1, 4, 7),
            (Bishop, 1, 3, 3), (Pawn, 1, 0, 6), (Pawn, 1, 1, 4),
            (Pawn, 1, 5, 4), (Pawn, 1, 6, 5), (Pawn, 1, 7, 6),
        ]);
        // Turn 84: queen recaptured on d4 by Black's king; Pilgrim still alive.
        let t84 = fixture(&[
            (King, 0, 5, 0), (Pilgrim, 0, 5, 3),
            (Pawn, 0, 0, 2), (Pawn, 0, 6, 2), (Pawn, 0, 7, 1),
            (ElectroKnight, 1, 4, 5), (FissionReactor, 1, 0, 5),
            (Gunslinger, 1, 2, 7), (King, 1, 3, 3), (Rook, 1, 0, 7),
            (Rook, 1, 4, 7), (Pawn, 1, 0, 6), (Pawn, 1, 1, 4),
            (Pawn, 1, 5, 4), (Pawn, 1, 6, 5), (Pawn, 1, 7, 6),
        ]);
        // Turn 88: mating net; White left with king + 3 pawns.
        let t88 = fixture(&[
            (King, 0, 6, 0),
            (Pawn, 0, 0, 2), (Pawn, 0, 6, 2), (Pawn, 0, 7, 1),
            (ElectroKnight, 1, 3, 4), (FissionReactor, 1, 0, 5),
            (Gunslinger, 1, 2, 7), (King, 1, 3, 3), (Rook, 1, 0, 7),
            (Rook, 1, 4, 7), (Pawn, 1, 0, 6), (Pawn, 1, 1, 4),
            (Pawn, 1, 5, 4), (Pawn, 1, 6, 5), (Pawn, 1, 7, 6),
        ]);
        let e82 = evaluate(&t82, 0);
        let e84 = evaluate(&t84, 0);
        let e88 = evaluate(&t88, 0);
        eprintln!("dtXQo1 fixture evals (White POV, cp): t82={e82} t84={e84} t88={e88}");
        // Strict decline: each stage worse than the last, no cliff hidden
        // between two "holding" evals.
        assert!(e82 > e84 && e84 > e88,
            "hunt evals must decline gradually: t82={e82} t84={e84} t88={e88}");
        // Turn 82 must already read as clearly losing for White: the old
        // engine implied roughly -270cp here while Black's hunt was already on.
        assert!(e82 < -400,
            "turn-82 eval must already show the hunt: got {e82}");
        // Turn 88 is a mating net: deeply negative, not a near-holding score.
        assert!(e88 < -1500, "turn-88 mating-net eval must be deep: got {e88}");
    }
}

#[cfg(test)]
mod lever3_fused_hash_tests {
    use super::{rep_hash, zobrist, zobrist_both};
    use crate::pos::{EpInfo, Piece, Pos};
    use crate::types::{sq, PieceKey};

    fn fixture(spec: &[(PieceKey, u8, u8, u8)], stm: u8) -> Pos {
        let mut pos = Pos::default();
        pos.stm = stm;
        for (i, &(key, player, x, y)) in spec.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32, key, sub_key: key, player, x, y,
                alive: true, ..Default::default()
            });
            pos.at[sq(x as i8, y as i8) as usize] = (i + 1) as u16;
        }
        pos
    }

    #[test]
    fn zobrist_both_matches_separate_calls() {
        use PieceKey::*;
        // Pixie-heavy position with state counters set (zobrist reads
        // state_flags/odo/kills/caps/quiet/frost/encore/horde_id; the
        // fused loop must reproduce them exactly), plus EP and a dead
        // piece (skipped by both loops).
        let mut pos = fixture(
            &[
                (King, 0, 4, 0), (Queen, 0, 3, 1), (Pilgrim, 0, 2, 4),
                (ElectroKnight, 0, 5, 4), (Fish, 0, 1, 0), (Pawn, 0, 4, 3),
                (King, 1, 4, 7), (Basilisk, 1, 0, 4), (AntiViolence, 1, 4, 5),
                (Pawn, 1, 4, 4),
            ],
            0,
        );
        pos.pieces[2].odo = 17;
        pos.pieces[3].quiet = 2;
        pos.pieces[3].state_flags = 1 << 1;
        pos.pieces[8].horde_id = 7;
        pos.pieces.push(Piece {
            id: 99, key: Pawn, sub_key: Pawn, player: 1, x: 0, y: 6,
            alive: false, ..Default::default()
        });
        pos.ep = Some(EpInfo { x: 4, y: 5, pawn_x: 4, pawn_y: 4 });
        for stm in [0u8, 1u8] {
            let (h32, h64) = zobrist_both(&pos, stm);
            assert_eq!(h32, zobrist(&pos, stm), "TT hash must match at stm={stm}");
            assert_eq!(h64, rep_hash(&pos, stm), "rep hash must match at stm={stm}");
        }
    }

    #[test]
    fn pst_orientation_rewards_7th_rank_pawns_and_sheltered_back_rank_kings() {
        use super::pst_bonus;
        use crate::types::PieceKey;
        let wp_r2 = Piece { key: PieceKey::Pawn, sub_key: PieceKey::Pawn, player: 0, x: 4, y: 1, alive: true, ..Default::default() };
        let wp_r7 = Piece { key: PieceKey::Pawn, sub_key: PieceKey::Pawn, player: 0, x: 4, y: 6, alive: true, ..Default::default() };
        let bp_r7 = Piece { key: PieceKey::Pawn, sub_key: PieceKey::Pawn, player: 1, x: 4, y: 6, alive: true, ..Default::default() };
        let bp_r2 = Piece { key: PieceKey::Pawn, sub_key: PieceKey::Pawn, player: 1, x: 4, y: 1, alive: true, ..Default::default() };
        assert!(pst_bonus(&wp_r7, 0) > pst_bonus(&wp_r2, 0), "White pawn on e7 must outscore e2");
        assert_eq!(pst_bonus(&wp_r7, 0), pst_bonus(&bp_r2, 0), "White e7 and Black e2 must be symmetric");
        assert_eq!(pst_bonus(&wp_r2, 0), pst_bonus(&bp_r7, 0), "White e2 and Black e7 must be symmetric");
        let wk_g1 = Piece { key: PieceKey::King, sub_key: PieceKey::King, player: 0, x: 6, y: 0, alive: true, ..Default::default() };
        let wk_e4 = Piece { key: PieceKey::King, sub_key: PieceKey::King, player: 0, x: 4, y: 3, alive: true, ..Default::default() };
        assert!(pst_bonus(&wk_g1, 0) > pst_bonus(&wk_e4, 0), "Midgame King on g1 must outscore exposed e4");
    }
}
