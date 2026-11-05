//! `pxb-selfplay` — High-Speed Native Rust Self-Play Binary for Pixie Chess
//!
//! Generates millions of `<fen> | <result>` (or `<fen> <result>`) training positions
//! for Texel evaluation tuning using the full `pxb-engine` move generator, triggers,
//! evaluation function, and alpha-beta search.
//!
//! Build:
//!   cargo build --release --bin pxb-selfplay
//!
//! Example Usage:
//!   ./target/release/pxb-selfplay --games 100000 --depth 3 --output training_data.txt --threads 4

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use pxb_engine::apply_full::apply_full;
use pxb_engine::attacks::{find_king, king_attacked};
use pxb_engine::eval::{init_eval_sum, rep_hash};
use pxb_engine::movegen::{gen_moves, is_pawn_key};
use pxb_engine::pos::{alive_count, slot_of, EpInfo, Piece, Pos};
use pxb_engine::search::{abSearch, rep_count, Search, MATE};
use pxb_engine::types::*;
use pxb_engine::veto::{veto_score, VETO_THRESHOLD};

// Fallback symbol export in case `src/lib.rs` still declares an unguarded
// `extern "C" { fn pxb_now_ms() -> f64; }` when compiling for native targets.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn pxb_now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

// ─── Fast Deterministic PRNG (SplitMix64 + Xoshiro256++) ─────────────────

#[derive(Clone)]
struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    #[inline]
    fn from_seed(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    fn gen_range(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }

    #[inline]
    fn gen_i32_inclusive(&mut self, lo: i32, hi: i32) -> i32 {
        if lo >= hi {
            return lo;
        }
        let span = (hi - lo + 1) as u64;
        lo + ((self.next_u64() % span) as i32)
    }

    #[inline]
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}

// ─── Pixie Catalog & Slot Definitions (All 34 Official Pixies) ───────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    P,
    N,
    B,
    R,
    Q,
    K,
}

#[derive(Clone, Copy)]
struct PixieSpec {
    key: PieceKey,
    slot: Slot,
    code_2w: &'static str,
    code_2b: &'static str,
    code_1w: &'static str,
    code_1b: &'static str,
}

const ALL_34_PIXIES: [PixieSpec; 34] = [
    // Queen slot (1)
    PixieSpec { key: PieceKey::FissionReactor, slot: Slot::Q, code_2w: "Fr", code_2b: "fr", code_1w: "F", code_1b: "f" },
    // King slot (1)
    PixieSpec { key: PieceKey::Rocketman,      slot: Slot::K, code_2w: "Rm", code_2b: "rm", code_1w: "O", code_1b: "o" },
    // Rook slots (2)
    PixieSpec { key: PieceKey::PhaseRook,      slot: Slot::R, code_2w: "Pr", code_2b: "pr", code_1w: "H", code_1b: "h" },
    PixieSpec { key: PieceKey::SumoRook,       slot: Slot::R, code_2w: "Sr", code_2b: "sr", code_1w: "Y", code_1b: "y" },
    // Knight slots (7)
    PixieSpec { key: PieceKey::AntiViolence,   slot: Slot::N, code_2w: "Av", code_2b: "av", code_1w: "A", code_1b: "a" },
    PixieSpec { key: PieceKey::Banker,         slot: Slot::N, code_2w: "Bk", code_2b: "bk", code_1w: "X", code_1b: "x" },
    PixieSpec { key: PieceKey::Camel,          slot: Slot::N, code_2w: "Ca", code_2b: "ca", code_1w: "C", code_1b: "c" },
    PixieSpec { key: PieceKey::ElectroKnight,  slot: Slot::N, code_2w: "Ek", code_2b: "ek", code_1w: "E", code_1b: "e" },
    PixieSpec { key: PieceKey::Fish,           slot: Slot::N, code_2w: "Fi", code_2b: "fi", code_1w: "S", code_1b: "s" },
    PixieSpec { key: PieceKey::Knightmare,     slot: Slot::N, code_2w: "Km", code_2b: "km", code_1w: "I", code_1b: "i" },
    PixieSpec { key: PieceKey::Pinata,         slot: Slot::N, code_2w: "Pi", code_2b: "pi", code_1w: "T", code_1b: "t" },
    // Bishop slots (13)
    PixieSpec { key: PieceKey::Aristocrat,     slot: Slot::B, code_2w: "Ar", code_2b: "ar", code_1w: "À", code_1b: "à" },
    PixieSpec { key: PieceKey::Basilisk,       slot: Slot::B, code_2w: "Ba", code_2b: "ba", code_1w: "L", code_1b: "l" },
    PixieSpec { key: PieceKey::BisHop,         slot: Slot::B, code_2w: "Bh", code_2b: "bh", code_1w: "Ë", code_1b: "ë" },
    PixieSpec { key: PieceKey::BladeRunner,    slot: Slot::B, code_2w: "Br", code_2b: "br", code_1w: "Ü", code_1b: "ü" },
    PixieSpec { key: PieceKey::Bouncer,        slot: Slot::B, code_2w: "Bo", code_2b: "bo", code_1w: "U", code_1b: "u" },
    PixieSpec { key: PieceKey::Cardinal,       slot: Slot::B, code_2w: "Cd", code_2b: "cd", code_1w: "D", code_1b: "d" },
    PixieSpec { key: PieceKey::Dancer,         slot: Slot::B, code_2w: "Da", code_2b: "da", code_1w: "Ð", code_1b: "ð" },
    PixieSpec { key: PieceKey::Djinn,          slot: Slot::B, code_2w: "Dj", code_2b: "dj", code_1w: "J", code_1b: "j" },
    PixieSpec { key: PieceKey::Gunslinger,     slot: Slot::B, code_2w: "Gs", code_2b: "gs", code_1w: "Ğ", code_1b: "ğ" },
    PixieSpec { key: PieceKey::HordeMother,    slot: Slot::B, code_2w: "Hm", code_2b: "hm", code_1w: "Ö", code_1b: "ö" },
    PixieSpec { key: PieceKey::Icicle,         slot: Slot::B, code_2w: "Ic", code_2b: "ic", code_1w: "Ì", code_1b: "ì" },
    PixieSpec { key: PieceKey::Marauder,       slot: Slot::B, code_2w: "Ma", code_2b: "ma", code_1w: "M", code_1b: "m" },
    PixieSpec { key: PieceKey::Pilgrim,        slot: Slot::B, code_2w: "Pg", code_2b: "pg", code_1w: "G", code_1b: "g" },
    // Pawn slots (10)
    PixieSpec { key: PieceKey::Blueprint,      slot: Slot::P, code_2w: "Bp", code_2b: "bp", code_1w: "V", code_1b: "v" },
    PixieSpec { key: PieceKey::EpeePawn,       slot: Slot::P, code_2w: "Ep", code_2b: "ep", code_1w: "É", code_1b: "é" },
    PixieSpec { key: PieceKey::GoldenPawn,     slot: Slot::P, code_2w: "Gp", code_2b: "gp", code_1w: "Å", code_1b: "å" },
    PixieSpec { key: PieceKey::HeroPawn,       slot: Slot::P, code_2w: "Hp", code_2b: "hp", code_1w: "Ê", code_1b: "ê" },
    PixieSpec { key: PieceKey::IronPawn,       slot: Slot::P, code_2w: "Ip", code_2b: "ip", code_1w: "Î", code_1b: "î" },
    PixieSpec { key: PieceKey::PawnWithKnife,  slot: Slot::P, code_2w: "Pk", code_2b: "pk", code_1w: "Z", code_1b: "z" },
    PixieSpec { key: PieceKey::Recruit,        slot: Slot::P, code_2w: "Rc", code_2b: "rc", code_1w: "Ř", code_1b: "ř" },
    PixieSpec { key: PieceKey::Shrike,         slot: Slot::P, code_2w: "Sh", code_2b: "sh", code_1w: "Š", code_1b: "š" },
    PixieSpec { key: PieceKey::WarAutomaton,   slot: Slot::P, code_2w: "Wa", code_2b: "wa", code_1w: "W", code_1b: "w" },
    PixieSpec { key: PieceKey::WarpJumper,     slot: Slot::P, code_2w: "Wj", code_2b: "wj", code_1w: "Ž", code_1b: "ž" },
];

// Real-game distribution of pixie counts per side in the 3..=6 range:
// ~14% 3 pixies, ~17% 4 pixies, ~8% 5 pixies, ~61% 6 pixies.
const PIXIE_COUNT_DIST: [usize; 16] = [3, 3, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];

// ─── Board Initialization & Loadout Sampling ─────────────────────────────

fn sample_side_keys(rng: &mut FastRng, min_pixies: usize, max_pixies: usize) -> [PieceKey; 16] {
    use PieceKey::*;
    // Indices 0..8: back rank (a..h)
    // Indices 8..16: pawn rank (a..h)
    let mut keys = [
        Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook,
        Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn,
    ];

    let target_count = if min_pixies == max_pixies {
        min_pixies
    } else {
        let sampled = PIXIE_COUNT_DIST[rng.gen_range(PIXIE_COUNT_DIST.len())];
        sampled.clamp(min_pixies, max_pixies)
    };

    let mut r_slots = [0usize, 7];
    let mut n_slots = [1usize, 6];
    let mut b_slots = [2usize, 5];
    let q_slots = [3usize];
    let k_slots = [4usize];
    let mut p_slots = [8usize, 9, 10, 11, 12, 13, 14, 15];

    rng.shuffle(&mut r_slots);
    rng.shuffle(&mut n_slots);
    rng.shuffle(&mut b_slots);
    rng.shuffle(&mut p_slots);

    let mut r_len = r_slots.len();
    let mut n_len = n_slots.len();
    let mut b_len = b_slots.len();
    let mut q_len = q_slots.len();
    let mut k_len = k_slots.len();
    let mut p_len = p_slots.len();

    let mut pool: [usize; 34] = core::array::from_fn(|i| i);
    rng.shuffle(&mut pool);

    let mut placed = 0usize;
    for &spec_idx in &pool {
        if placed >= target_count {
            break;
        }
        let spec = ALL_34_PIXIES[spec_idx];
        let slot_idx_opt = match spec.slot {
            Slot::R if r_len > 0 => { r_len -= 1; Some(r_slots[r_len]) }
            Slot::N if n_len > 0 => { n_len -= 1; Some(n_slots[n_len]) }
            Slot::B if b_len > 0 => { b_len -= 1; Some(b_slots[b_len]) }
            Slot::Q if q_len > 0 => { q_len -= 1; Some(q_slots[q_len]) }
            Slot::K if k_len > 0 => { k_len -= 1; Some(k_slots[k_len]) }
            Slot::P if p_len > 0 => { p_len -= 1; Some(p_slots[p_len]) }
            _ => Option::None,
        };
        if let Some(slot_idx) = slot_idx_opt {
            keys[slot_idx] = spec.key;
            placed += 1;
        }
    }

    keys
}

fn setup_random_game(rng: &mut FastRng, min_pixies: usize, max_pixies: usize) -> Pos {
    use PieceKey::*;
    let mut pos = Pos::default();
    pos.pieces.reserve(36);

    // Non-King, non-Pinata pixies that Pinata can transform into
    const PINATA_POOL: [PieceKey; 32] = [
        FissionReactor, PhaseRook, SumoRook,
        AntiViolence, Banker, Camel, ElectroKnight, Fish, Knightmare,
        Aristocrat, Basilisk, BisHop, BladeRunner, Bouncer, Cardinal,
        Dancer, Djinn, Gunslinger, HordeMother, Icicle, Marauder, Pilgrim,
        Blueprint, EpeePawn, GoldenPawn, HeroPawn, IronPawn, PawnWithKnife,
        Recruit, Shrike, WarAutomaton, WarpJumper,
    ];

    let mut id_counter = 0u32;

    for player in 0..2u8 {
        let side_keys = sample_side_keys(rng, min_pixies, max_pixies);
        let back_y: u8 = if player == 0 { 0 } else { 7 };
        let pawn_y: u8 = if player == 0 { 1 } else { 6 };

        // Resolve Blueprint left-to-right on the pawn rank (indices 8..16 -> files 0..8)
        let mut resolved_sub = [None; 16];
        for i in 0..16 {
            let k = side_keys[i];
            if k == Rocketman {
                // Rocketman occupies the King slot: sub_key = King so find_king() & eval locate it
                resolved_sub[i] = King;
            } else if k == Recruit {
                resolved_sub[i] = Pawn;
            } else if k == Pinata {
                let target = PINATA_POOL[rng.gen_range(PINATA_POOL.len())];
                resolved_sub[i] = target;
            } else if k == Blueprint {
                if i > 8 {
                    let left_k = side_keys[i - 1];
                    let left_eff = if left_k == Blueprint {
                        resolved_sub[i - 1]
                    } else {
                        left_k
                    };
                    resolved_sub[i] = left_eff;
                } else {
                    resolved_sub[i] = Pawn;
                }
            } else {
                resolved_sub[i] = k;
            }
        }

        for i in 0..16 {
            id_counter += 1;
            let file_x = (i % 8) as u8;
            let rank_y = if i < 8 { back_y } else { pawn_y };
            let k = side_keys[i];
            let sk = resolved_sub[i];

            let piece = Piece {
                id: id_counter,
                key: k,
                sub_key: sk,
                player,
                x: file_x,
                y: rank_y,
                home_x: file_x as i8,
                home_y: rank_y as i8,
                alive: true,
                ..Default::default()
            };
            let s = sq(file_x as i8, rank_y as i8);
            pos.pieces.push(piece);
            pos.at[s as usize] = pos.pieces.len() as u16;
        }
    }

    pos.id_seq = id_counter;
    pos.stm = 0;
    init_eval_sum(&mut pos);
    let h = rep_hash(&pos, 0);
    pos.hist_hashes.push(h);
    pos
}

// ─── Custom FEN Serialization ────────────────────────────────────────────

#[inline]
fn piece_fen_token(p: &Piece, one_char_mode: bool) -> &'static str {
    use PieceKey::*;
    let w = p.player == 0;
    let k = if p.key == Blueprint && p.sub_key != None && p.sub_key != Blueprint && p.sub_key != Pawn {
        p.sub_key
    } else if p.key != None {
        p.key
    } else {
        p.sub_key
    };

    match k {
        Pawn | Hordeling | TranslucentPawn | Ooze | Dummy => if w { "P" } else { "p" },
        Knight | HordeKnight | CloudKnight | Uni | Puppet => if w { "N" } else { "n" },
        Bishop | ExileBishop | Druidess | Dryad | Duplicator | Wraith | Monument | Mushroom | LavaForge | Timelord => if w { "B" } else { "b" },
        Rook | RookZero | Samurai => if w { "R" } else { "r" },
        Queen | VoidQueen | NullQueen | WitchQueen => if w { "Q" } else { "q" },
        King | WarKing | MeditatingKing | PopeKing | IrreverentKing => if w { "K" } else { "k" },
        BisHop => {
            if one_char_mode {
                if w { "Ë" } else { "ë" }
            } else {
                if w { "Bh" } else { "bh" }
            }
        }
        _ => {
            for spec in &ALL_34_PIXIES {
                if spec.key == k {
                    return if one_char_mode {
                        if w { spec.code_1w } else { spec.code_1b }
                    } else {
                        if w { spec.code_2w } else { spec.code_2b }
                    };
                }
            }
            if w { "P" } else { "p" }
        }
    }
}

fn write_pos_fen(pos: &Pos, ply: usize, one_char_mode: bool, out: &mut String) {
    for r in (0..8i8).rev() {
        let mut empty = 0u8;
        for f in 0..8i8 {
            let s = sq(f, r);
            if let Some(slot) = slot_of(pos, s) {
                let p = &pos.pieces[slot];
                if p.alive {
                    if empty > 0 {
                        out.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    out.push_str(piece_fen_token(p, one_char_mode));
                } else {
                    empty += 1;
                }
            } else {
                empty += 1;
            }
        }
        if empty > 0 {
            out.push((b'0' + empty) as char);
        }
        if r > 0 {
            out.push('/');
        }
    }

    out.push(' ');
    out.push(if pos.stm == 0 { 'w' } else { 'b' });
    out.push_str(" - - ");

    // Write halfmove and fullmove integers without format! allocation overhead
    let mut itoa_buf = [0u8; 16];
    let hm_str = format_u32(pos.halfmove as u32, &mut itoa_buf);
    out.push_str(hm_str);
    out.push(' ');
    let fm_str = format_u32((ply as u32) / 2 + 1, &mut itoa_buf);
    out.push_str(fm_str);
}

#[inline]
fn format_u32(mut n: u32, buf: &mut [u8; 16]) -> &str {
    if n == 0 {
        return "0";
    }
    let mut i = 16;
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    unsafe { core::str::from_utf8_unchecked(&buf[i..16]) }
}

// ─── Position Advancement & Search Move Selection ────────────────────────

fn advance_position(pos: &Pos, mv: &Move, stm: u8, opp: u8) -> Option<Pos> {
    let pre_alive = alive_count(pos);
    let mover_pre = if mv.from != SQ_NONE {
        slot_of(pos, mv.from).map(|s| &pos.pieces[s])
    } else {
        Option::None
    };
    let mover_is_pawn = mover_pre.map_or(false, |p| is_pawn_key(p.key) || is_pawn_key(p.sub_key));
    let mover_is_recruit = mover_pre.map_or(false, |p| {
        p.key == PieceKey::Recruit || p.sub_key == PieceKey::Recruit
    });

    let cap_sq = if mv.flags & MF_EP != 0 { mv.ep_victim } else { mv.to };
    let captured_key = if cap_sq < 64 {
        slot_of(pos, cap_sq).and_then(|vs| {
            let v = &pos.pieces[vs];
            if v.alive && v.player == opp {
                Some(if v.key != PieceKey::None && v.key != PieceKey::Pinata && v.key != PieceKey::Blueprint {
                    v.key
                } else {
                    v.sub_key
                })
            } else {
                Option::None
            }
        })
    } else {
        Option::None
    };

    let mut p2 = apply_full(pos, mv, stm, opp)?;

    // Ensure Recruit permanently stores the captured piece's PieceKey in `sub_key`
    if mover_is_recruit {
        if let Some(ck) = captured_key {
            if ck != PieceKey::None && ck != PieceKey::King && mv.to < 64 {
                if let Some(ms) = slot_of(&p2, mv.to) {
                    if p2.pieces[ms].key == PieceKey::Recruit {
                        p2.pieces[ms].sub_key = ck;
                    }
                }
            }
        }
    }

    // Update en-passant target square after double pawn push
    if mover_is_pawn
        && mv.from < 64
        && mv.to < 64
        && (sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs() == 2
        && (mv.flags & MF_CAPTURE == 0)
    {
        p2.ep = Some(EpInfo {
            x: sq_x(mv.from) as i8,
            y: (sq_y(mv.from) as i8 + sq_y(mv.to) as i8) / 2,
            pawn_x: sq_x(mv.to) as i8,
            pawn_y: sq_y(mv.to) as i8,
        });
    } else {
        p2.ep = Option::None;
    }

    // Update halfmove clock
    let post_alive = alive_count(&p2);
    if mover_is_pawn || (mv.flags & MF_CAPTURE != 0) || post_alive < pre_alive {
        p2.halfmove = 0;
    } else {
        p2.halfmove = pos.halfmove.saturating_add(1);
    }

    p2.stm = opp;
    // Replace apply_full's pre-Recruit/EP rep_hash with the final post-move rep_hash
    p2.hist_hashes.pop();
    p2.hist_hashes.push(rep_hash(&p2, opp));
    Some(p2)
}

#[inline]
fn has_only_kings(pos: &Pos) -> bool {
    if !pos.limbo.is_empty() || !pos.offboard.is_empty() {
        return false;
    }
    for p in &pos.pieces {
        if !p.alive || !xy_on_board(p.x, p.y) {
            continue;
        }
        let is_k = matches!(
            p.key,
            PieceKey::King
                | PieceKey::Rocketman
                | PieceKey::WarKing
                | PieceKey::MeditatingKing
                | PieceKey::PopeKing
                | PieceKey::IrreverentKing
        ) || matches!(
            p.sub_key,
            PieceKey::King
                | PieceKey::Rocketman
                | PieceKey::WarKing
                | PieceKey::MeditatingKing
                | PieceKey::PopeKing
                | PieceKey::IrreverentKing
        );
        if !is_k {
            return false;
        }
    }
    true
}

fn select_move(
    search: &mut Search,
    pos: &mut Pos,
    stm: u8,
    opp: u8,
    depth: i32,
    max_nodes: u64,
    ply: usize,
    jitter: i32,
    use_veto: bool,
    rng: &mut FastRng,
) -> (Option<Move>, i32) {
    // Honest depth on every move (Argon speedup 2): no hybrid schedule — the
    // user asked for depth 7, so target_depth = depth and the node cap is the
    // per-move cap passed in (use --max-nodes 1500000 for uncapped, or 15000
    // to cap only extreme tactical outliers).
    let target_depth = depth.max(1);
    let move_node_cap = max_nodes;

    search.t0_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0); // FIX: t0 was 0, so elapsed (=epoch ms) always exceeded any budget and the soft cutoff aborted every depth past 1. NOTE: Argon's pxb_now_ms() is an extern FFI fn, not callable from this bin crate — SystemTime kept.
    search.budget_ms = 1_000_000_000; // max_nodes binds now, not time (was 15s safety valve)
    search.nodes = 0;
    search.max_nodes = move_node_cap;
    search.fast_qsearch = true;
    search.cluster_worker = true;
    search.stop = false;
    search.best_root = Option::None;
    search.best_root_score = 0;
    search.no_tt = target_depth <= 2;
    search.no_pvs = false;
    search.null_ply = -1;
    search.killers = [[Option::None; 2]; 64];
    search.journal.v.clear();
    search.root_scored.clear();
    search.tt.last_root_scored.clear();
    search.tt.score_drop = 0;
    search.init_root_accumulator(pos);

    // PROBE: track the deepest iterative-deepening iteration that completed
    // without the search stopping (node/time cap). If the cap cut depth d,
    // reached stays at d-1, which is the honest depth for this move.
    let mut reached: i32 = 0;
    for d in 1..=target_depth {
        // Always allow d == 1 (~120 nodes) to finish all root moves so
        // s.tt.last_root_scored is populated before any mid-depth abort on d >= 2.
        search.max_nodes = if d == 1 { 1_500_000 } else { move_node_cap };
        search.best_root = Option::None;
        search.root_scored.clear();
        let _ = abSearch(
            search,
            pos,
            d,
            -1_000_000_000,
            1_000_000_000,
            stm,
            opp,
            stm,
            0,
            0,
            Option::None,
        );
        if search.stop {
            break;
        }
        reached = d;
        if search.nodes >= move_node_cap {
            break;
        }
    }

    if search.root_scored.is_empty() {
        // Fallback: check if any legal move exists
        let candidates = gen_moves(pos, stm, Option::None);
        for m in candidates {
            if let Some(p2) = apply_full(pos, &m, stm, opp) {
                if !king_attacked(&p2, stm, opp, Option::None) {
                    return (Some(m), reached);
                }
            }
        }
        return (Option::None, reached);
    }

    // Add opening root jitter for game diversity (matching pixie_ai.py)
    let eff_jitter = if ply < 12 { jitter } else { jitter.min(2) };
    if eff_jitter > 0 {
        for &mut (_, ref mut sc) in search.root_scored.iter_mut() {
            if sc.abs() < MATE - 10_000 {
                *sc = sc.saturating_add(rng.gen_i32_inclusive(-eff_jitter, eff_jitter));
            }
        }
    }

    search.root_scored.sort_by(|a, b| b.1.cmp(&a.1));

    if use_veto {
        let mut best_mv: Option<Move> = Option::None;
        let mut best_eff = i32::MIN;
        for &(rm, rscore) in search.root_scored.iter() {
            // Since root_scored is sorted descending and veto_score <= 25,
            // stop checking once rscore + 25 cannot beat best_eff.
            if best_eff > i32::MIN && rscore.saturating_add(25) <= best_eff {
                break;
            }
            let vs = veto_score(pos, &rm, stm, opp);
            if vs < VETO_THRESHOLD {
                continue;
            }
            let eff = rscore.saturating_add(vs);
            if eff > best_eff {
                best_eff = eff;
                best_mv = Some(rm);
            }
        }
        if best_mv.is_some() {
            return (best_mv, reached);
        }
    }

    (Some(search.root_scored[0].0), reached)
}

// ─── Single Self-Play Game Runner ────────────────────────────────────────

struct GameOutcome {
    result: f32, // 1.0 = White win, 0.5 = Draw, 0.0 = Black win
    positions_count: usize,
    nodes_searched: u64,
    depth_hist: [u32; 13], // PROBE: moves by honest reached depth (index = depth)
}

fn play_single_game(
    game_idx: usize,
    cfg: &Config,
    search: &mut Search,
    fens_scratch: &mut Vec<String>,
    out_buf: &mut String,
) -> GameOutcome {
    let seed = cfg
        .seed
        .wrapping_add((game_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = FastRng::from_seed(seed);

    // Periodically clear TT every 64 games to avoid stale hash collisions across long runs
    // without paying the 56 MB memset cost on every move.
    if game_idx % 64 == 0 && cfg.depth > 2 {
        search.tt.clear();
    }
    search.hist.fill(0);
    search.fast_qsearch = true;
    search.cluster_worker = true;

    let mut pos = setup_random_game(&mut rng, cfg.min_pixies, cfg.max_pixies);
    fens_scratch.clear();

    let mut total_nodes = 0u64;
    let mut depth_hist = [0u32; 13];
    let mut result: Option<f32> = Option::None;

    for ply in 0..cfg.max_moves {
        if let Some(kt) = pos.king_thrown {
            result = Some(if kt == 0 { 1.0 } else { 0.0 });
            break;
        }
        if let Some(gw) = pos.gold_win {
            result = Some(if gw == 0 { 1.0 } else { 0.0 });
            break;
        }

        let stm = pos.stm;
        let opp = 1 - stm;

        if find_king(&pos, stm).is_none() {
            result = Some(if opp == 0 { 1.0 } else { 0.0 });
            break;
        }
        if find_king(&pos, opp).is_none() {
            result = Some(if stm == 0 { 1.0 } else { 0.0 });
            break;
        }

        if pos.halfmove >= 100 || has_only_kings(&pos) {
            result = Some(0.5);
            break;
        }

        let h = *pos.hist_hashes.last().unwrap_or(&0);
        if ply > 0 && rep_count(&pos, stm, h) >= 3 {
            result = Some(0.5);
            break;
        }

        let (mv_opt, reached) = select_move(
            search,
            &mut pos,
            stm,
            opp,
            cfg.depth,
            cfg.max_nodes,
            ply,
            cfg.jitter,
            cfg.use_veto,
            &mut rng,
        );
        total_nodes += search.nodes;
        depth_hist[(reached as usize).min(12)] += 1;

        let mv = match mv_opt {
            Some(m) => m,
            Option::None => {
                if king_attacked(&pos, stm, opp, Option::None) {
                    result = Some(if opp == 0 { 1.0 } else { 0.0 });
                } else {
                    result = Some(0.5);
                }
                break;
            }
        };

        pos = match advance_position(&pos, &mv, stm, opp) {
            Some(next_pos) => next_pos,
            Option::None => {
                result = Some(0.5);
                break;
            }
        };

        let mut fen_line = String::with_capacity(96);
        write_pos_fen(&pos, ply + 1, cfg.one_char_fen, &mut fen_line);
        fens_scratch.push(fen_line);
    }

    let final_res = result.unwrap_or_else(|| {
        if let Some(kt) = pos.king_thrown {
            if kt == 0 { 1.0 } else { 0.0 }
        } else if let Some(gw) = pos.gold_win {
            if gw == 0 { 1.0 } else { 0.0 }
        } else {
            0.5
        }
    });

    let res_str = if final_res > 0.75 {
        "1.0"
    } else if final_res < 0.25 {
        "0.0"
    } else {
        "0.5"
    };

    for fen in fens_scratch.iter() {
        out_buf.push_str(fen);
        out_buf.push_str(&cfg.separator);
        out_buf.push_str(res_str);
        out_buf.push('\n');
    }

    GameOutcome {
        result: final_res,
        positions_count: fens_scratch.len(),
        nodes_searched: total_nodes,
        depth_hist,
    }
}

// ─── CLI Configuration & Multi-Threaded Runner ───────────────────────────

#[derive(Clone)]
struct Config {
    games: usize,
    depth: i32,
    max_nodes: u64,
    output: String,
    threads: usize,
    max_moves: usize,
    seed: u64,
    jitter: i32,
    min_pixies: usize,
    max_pixies: usize,
    separator: String,
    one_char_fen: bool,
    use_veto: bool,
    no_resume: bool,
}

impl Default for Config {
    fn default() -> Self {
        let cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            games: 1000,
            depth: 6,
            max_nodes: 2_500,
            output: String::from("training_data.txt"),
            threads: cores,
            max_moves: 200,
            seed: 42,
            jitter: 10,
            min_pixies: 3,
            max_pixies: 6,
            separator: String::from(" | "),
            one_char_fen: false,
            use_veto: false,
            no_resume: false,
        }
    }
}

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {prog} [OPTIONS]\n\n\
         Options:\n  \
           --games <N>         Number of self-play games to generate (default: 1000)\n  \
           --depth <D>         Search depth per move (default: 6)\n  \
           --max-nodes <N>     Hard in-search node cap per move (default: 2500)\n  \
           --output <FILE>     Output file path (default: training_data.txt)\n  \
           --threads <T>       Number of parallel worker threads (default: num cores)\n  \
           --max-moves <M>     Maximum plies per game before draw (default: 200)\n  \
           --seed <S>          Base random seed (default: 42)\n  \
           --jitter <J>        Opening root score jitter in cp (default: 10)\n  \
           --min-pixies <N>    Minimum pixies per side (default: 3)\n  \
           --max-pixies <N>    Maximum pixies per side (default: 6)\n  \
           --sep <STR>         Separator between FEN and result (default: \" | \")\n  \
           --no-pipe           Use single space \"<fen> <result>\" separator\n  \
           --fen-mode <MODE>   '2char' (default, e.g. Fr/Rm) or '1char' (unified 1-char)\n  \
           --veto              Enable root veto post-pass in self-play\n  \
           --no-resume         Overwrite output file instead of resuming\n  \
           -h, --help          Show this help message"
    );
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config::default();
    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(|s| s.as_str()).unwrap_or("pxb-selfplay");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage(prog);
                std::process::exit(0);
            }
            "--games" | "-g" => {
                i += 1;
                cfg.games = args
                    .get(i)
                    .ok_or("--games requires a value")?
                    .parse()
                    .map_err(|_| "Invalid --games value")?;
            }
            "--depth" | "-d" => {
                i += 1;
                cfg.depth = args
                    .get(i)
                    .ok_or("--depth requires a value")?
                    .parse()
                    .map_err(|_| "Invalid --depth value")?;
            }
            "--max-nodes" | "-m" => {
                i += 1;
                cfg.max_nodes = args
                    .get(i)
                    .ok_or("--max-nodes requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid --max-nodes value")?
                    .max(100);
            }
            "--output" | "-o" => {
                i += 1;
                cfg.output = args.get(i).ok_or("--output requires a path")?.clone();
            }
            "--threads" | "--workers" | "-t" => {
                i += 1;
                cfg.threads = args
                    .get(i)
                    .ok_or("--threads requires a value")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid --threads value")?
                    .max(1);
            }
            "--max-moves" => {
                i += 1;
                cfg.max_moves = args
                    .get(i)
                    .ok_or("--max-moves requires a value")?
                    .parse()
                    .map_err(|_| "Invalid --max-moves value")?;
            }
            "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .ok_or("--seed requires a value")?
                    .parse()
                    .map_err(|_| "Invalid --seed value")?;
            }
            "--jitter" => {
                i += 1;
                cfg.jitter = args
                    .get(i)
                    .ok_or("--jitter requires a value")?
                    .parse()
                    .map_err(|_| "Invalid --jitter value")?;
            }
            "--min-pixies" => {
                i += 1;
                cfg.min_pixies = args
                    .get(i)
                    .ok_or("--min-pixies requires a value")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid --min-pixies value")?
                    .clamp(0, 16);
            }
            "--max-pixies" => {
                i += 1;
                cfg.max_pixies = args
                    .get(i)
                    .ok_or("--max-pixies requires a value")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid --max-pixies value")?
                    .clamp(0, 16);
            }
            "--sep" => {
                i += 1;
                cfg.separator = args.get(i).ok_or("--sep requires a string")?.clone();
            }
            "--no-pipe" => {
                cfg.separator = String::from(" ");
            }
            "--fen-mode" => {
                i += 1;
                let mode = args.get(i).ok_or("--fen-mode requires '2char' or '1char'")?;
                cfg.one_char_fen = mode == "1char" || mode == "1";
            }
            "--veto" => {
                cfg.use_veto = true;
            }
            "--no-resume" => {
                cfg.no_resume = true;
            }
            other => {
                return Err(format!("Unknown argument: {other}"));
            }
        }
        i += 1;
    }

    if cfg.min_pixies > cfg.max_pixies {
        core::mem::swap(&mut cfg.min_pixies, &mut cfg.max_pixies);
    }
    Ok(cfg)
}

fn count_existing_progress(output_path: &str, meta_path: &str) -> (usize, usize) {
    let p = Path::new(output_path);
    if !p.exists() {
        return (0, 0);
    }
    let lines_cnt = match File::open(p) {
        Ok(f) => BufReader::new(f).lines().count(),
        Err(_) => 0,
    };
    if let Ok(s) = fs::read_to_string(meta_path) {
        if let Ok(g_done) = s.trim().parse::<usize>() {
            return (g_done, lines_cnt);
        }
    }
    (lines_cnt / 45, lines_cnt)
}

struct SharedWriter {
    writer: BufWriter<File>,
    games_done: usize,
    total_positions: usize,
    total_nodes: u64,
    w_wins: usize,
    draws: usize,
    b_wins: usize,
    depth_hist: [u32; 13], // PROBE: aggregate moves by honest reached depth
}

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ERROR] {e}");
            print_usage("pxb-selfplay");
            std::process::exit(1);
        }
    };

    let meta_path = format!("{}.progress", cfg.output);
    let (start_game, initial_positions) = if cfg.no_resume {
        let _ = fs::remove_file(&cfg.output);
        let _ = fs::remove_file(&meta_path);
        (0, 0)
    } else {
        count_existing_progress(&cfg.output, &meta_path)
    };

    if start_game >= cfg.games {
        println!(
            "[Resume] Target of {} games already reached ({} games, {} positions).",
            cfg.games, start_game, initial_positions
        );
        return;
    }

    if let Some(parent) = Path::new(&cfg.output).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let out_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.output)
        .expect("Failed to open output file for writing");

    let remaining = cfg.games - start_game;
    let num_threads = cfg.threads.min(remaining).max(1);

    println!(
        "Starting Pixie Chess Rust Self-Play: target={} games (resuming from {}, {} remaining) | \
         depth={} | threads={} | pixies={}..={} | output={}",
        cfg.games,
        start_game,
        remaining,
        cfg.depth,
        num_threads,
        cfg.min_pixies,
        cfg.max_pixies,
        cfg.output
    );

    let t0 = Instant::now();
    let next_game = AtomicUsize::new(start_game);
    let shared = Mutex::new(SharedWriter {
        writer: BufWriter::with_capacity(256 * 1024, out_file),
        games_done: start_game,
        total_positions: initial_positions,
        total_nodes: 0,
        w_wins: 0,
        draws: 0,
        b_wins: 0,
        depth_hist: [0u32; 13],
    });

    // Batch size per thread lock acquisition to minimize mutex contention
    let batch_size = if remaining >= num_threads * 16 { 4usize } else { 1usize };

    thread::scope(|s| {
        for _tid in 0..num_threads {
            let next_game_ref = &next_game;
            let shared_ref = &shared;
            let cfg_ref = &cfg;
            let meta_ref = &meta_path;

            s.spawn(move || {
                let mut search = Search::new();
                let mut fens_scratch: Vec<String> = Vec::with_capacity(256);
                let mut local_buf = String::with_capacity(32 * 1024);

                loop {
                    let g_start = next_game_ref.fetch_add(batch_size, Ordering::Relaxed);
                    if g_start >= cfg_ref.games {
                        break;
                    }
                    let g_end = (g_start + batch_size).min(cfg_ref.games);

                    local_buf.clear();
                    let mut batch_games = 0usize;
                    let mut batch_pos = 0usize;
                    let mut batch_nodes = 0u64;
                    let mut batch_w = 0usize;
                    let mut batch_d = 0usize;
                    let mut batch_b = 0usize;
                    let mut batch_hist = [0u32; 13];

                    for g_idx in g_start..g_end {
                        let outcome = play_single_game(
                            g_idx,
                            cfg_ref,
                            &mut search,
                            &mut fens_scratch,
                            &mut local_buf,
                        );
                        batch_games += 1;
                        batch_pos += outcome.positions_count;
                        batch_nodes += outcome.nodes_searched;
                        for i in 0..13 {
                            batch_hist[i] += outcome.depth_hist[i];
                        }
                        if outcome.result > 0.75 {
                            batch_w += 1;
                        } else if outcome.result < 0.25 {
                            batch_b += 1;
                        } else {
                            batch_d += 1;
                        }
                    }

                    let mut guard = shared_ref.lock().unwrap();
                    guard
                        .writer
                        .write_all(local_buf.as_bytes())
                        .expect("Failed to write positions to output file");
                    guard.games_done += batch_games;
                    guard.total_positions += batch_pos;
                    guard.total_nodes += batch_nodes;
                    guard.w_wins += batch_w;
                    guard.draws += batch_d;
                    guard.b_wins += batch_b;
                    for i in 0..13 {
                        guard.depth_hist[i] += batch_hist[i];
                    }

                    let done = guard.games_done;
                    let completed_now = done - start_game;
                    let log_interval = if cfg_ref.games >= 1000 { 50 } else { 10 };

                    if completed_now % log_interval < batch_games || done == cfg_ref.games {
                        let _ = guard.writer.flush();
                        let _ = fs::write(meta_ref, done.to_string());
                        let elapsed = t0.elapsed().as_secs_f64().max(1e-6);
                        let gpm_total = (completed_now as f64 / elapsed) * 60.0;
                        let gpm_per_core = gpm_total / (num_threads as f64);
                        let pos_per_sec = (guard.total_positions - initial_positions) as f64 / elapsed;
                        println!(
                            "[{}/{} games] positions={} ({:.0} pos/s) | speed={:.1} games/min ({:.1}/core) | W:{} D:{} B:{}",
                            done,
                            cfg_ref.games,
                            guard.total_positions,
                            pos_per_sec,
                            gpm_total,
                            gpm_per_core,
                            guard.w_wins,
                            guard.draws,
                            guard.b_wins,
                        );
                    }
                }
            });
        }
    });

    let mut final_guard = shared.lock().unwrap();
    let _ = final_guard.writer.flush();
    let _ = fs::write(&meta_path, final_guard.games_done.to_string());

    let total_secs = t0.elapsed().as_secs_f64().max(1e-6);
    let gpm = (remaining as f64 / total_secs) * 60.0;
    let gpm_core = gpm / (num_threads as f64);
    println!(
        "\nCompleted {} games ({} total positions, {} nodes) in {:.2}s \
         ({:.1} games/min total, {:.1} games/min/core). Saved to {}",
        final_guard.games_done,
        final_guard.total_positions,
        final_guard.total_nodes,
        total_secs,
        gpm,
        gpm_core,
        cfg.output
    );
    println!(
        "PROBE depth histogram (moves by honest reached depth): {:?}",
        final_guard.depth_hist
    );
}