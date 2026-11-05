// NNUE evaluator — Phase 2 wiring (SPEC.md v1, docs/nnue-feature-schema.md).
//
// Architecture: 5120 -> 256 (ReLU) -> 1 (sigmoid = P(side-to-move wins)).
// File format: W1 (5120x256, input-major) + B1 (256) + W2 (256) + B2 (1),
// f32 little-endian, NO header, exactly 5,244,932 bytes. The loader enforces
// that exact size plus finite weights; the model file carries no header, so
// there is no embedded schema version or table hash to check — the byte size
// is the enforceable proxy, and any change to the table below invalidates
// all existing weights (version bump + retrain, never silent reuse).
//
// Feature layout (SPEC §4): ((plane * 40) + piece_type) * 64 + square',
// plane 0 = side-to-move's own pieces, plane 1 = opponent's pieces,
// square' = square when stm is white, 63 - square when stm is black.
// Squares use the engine scheme sq = (y << 3) | x, y = rank - 1, which is
// identical to the trainer's python-chess convention (a1 = 0 .. h8 = 63).
//
// Identity resolution: a board piece's feature row is its EFFECTIVE key
// (crate::eval::effective_key), the same resolution the hand-written
// evaluator uses, so a PhaseRook stored base-first (key=Rook,
// sub_key=PhaseRook) still maps to Pr, and Pinata/Blueprint carriers stay
// transparent. Pieces with no training code (see piece_type_index)
// contribute NO NNUE input — they are skipped, never mapped to a wrong
// trained feature. The hand-written evaluator still sees them, so a
// position's NNUE view can be sparser than its hand-written view; that
// asymmetry is deliberate and documented, not a silent substitution.
//
// Only on-board, alive pieces get features. Limbo/offboard/doomed-list
// entries are not board pieces; doomed pieces still standing on the board
// DO get features (they are on the board, like the static evaluator sees).
// Special states (phased, carried, charged, ...) are NOT encoded: the
// schema has no state features.
//
// Scale: the network outputs P(stm wins); evaluate() must return
// centipawns from the caller's perspective, so the probability is mapped
// through cp = round(400 * log10(p / (1 - p))), p clamped to
// [0.001, 0.999] (=> cp in [-1200, 1200]), then negated when the caller
// is not the side to move. Mate/gold-win terminal states bypass the net
// and return MATE_CONST, exactly like the hand-written evaluator.

use crate::eval::{effective_key, Evaluator, MATE_CONST};
use crate::pos::*;
use crate::types::*;

pub const NNUE_NUM_TYPES: usize = 40;
pub const NNUE_INPUT_SIZE: usize = 5120;
pub const NNUE_HIDDEN_SIZE: usize = 256;
pub const NNUE_MODEL_BYTES: usize = 5_244_932;

/// Reconciled inventory table (SPEC §3 order). Each row is (PieceKey value,
/// trainer code). Row order is part of the schema: reordering invalidates
/// weights. Verified against the engine PieceKey enum VALUES in types.rs,
/// not names alone, and against the trainer's CODES list in
/// train/src/bin/pretokenize.rs (same order, same codes).
///
/// Caveats carried from training (SPEC §3, starred rows): Pg/Pi and Ca/Cd
/// are assigned by distinctive-letter logic only (pilgrim is the only
/// P-piece containing 'g'; camel contains no 'd', cardinal does). The
/// training features are indices, so training is unaffected by these
/// names; the engine side is verified here by PieceKey identity (Pilgrim
/// = 40 -> Pg, Pinata = 47 -> Pi, Camel = 27 -> Ca, Cardinal = 28 -> Cd).
/// Game-log codes use first-two-letters and DIFFER from data codes
/// (log Ph -> data Pr, Pa -> Pk, He -> Hp, El -> Ek, Su -> Sr, Kn -> Km,
/// Pi -> Pg or Pi, Bi -> NO data code: the bishopper gap).
pub const NNUE_TABLE: [(PieceKey, &str); NNUE_NUM_TYPES] = [
    (PieceKey::Aristocrat, "Ar"),     // 0
    (PieceKey::AntiViolence, "Av"),   // 1
    (PieceKey::Basilisk, "Ba"),       // 2
    (PieceKey::BisHop, "Bh"),         // 3  bis-hop (NOT Bishopper)
    (PieceKey::Banker, "Bk"),         // 4
    (PieceKey::Bouncer, "Bo"),        // 5
    (PieceKey::Blueprint, "Bp"),      // 6  extended key 48, but TRAINED
    (PieceKey::BladeRunner, "Br"),    // 7
    (PieceKey::Camel, "Ca"),          // 8  (*) distinctive-letter assignment
    (PieceKey::Cardinal, "Cd"),       // 9  (*) distinctive-letter assignment
    (PieceKey::Dancer, "Da"),         // 10
    (PieceKey::Djinn, "Dj"),          // 11
    (PieceKey::ElectroKnight, "Ek"),  // 12
    (PieceKey::EpeePawn, "Ep"),       // 13
    (PieceKey::Fish, "Fi"),           // 14
    (PieceKey::FissionReactor, "Fr"), // 15
    (PieceKey::GoldenPawn, "Gp"),     // 16
    (PieceKey::Gunslinger, "Gs"),     // 17
    (PieceKey::HordeMother, "Hm"),    // 18
    (PieceKey::HeroPawn, "Hp"),       // 19
    (PieceKey::Icicle, "Ic"),         // 20
    (PieceKey::IronPawn, "Ip"),       // 21
    (PieceKey::Knightmare, "Km"),     // 22
    (PieceKey::Marauder, "Ma"),       // 23
    (PieceKey::Pilgrim, "Pg"),        // 24 (*) distinctive-letter assignment
    (PieceKey::Pinata, "Pi"),         // 25 (*) extended key 47, but TRAINED
    (PieceKey::PawnWithKnife, "Pk"),  // 26
    (PieceKey::PhaseRook, "Pr"),      // 27
    (PieceKey::Recruit, "Rc"),        // 28
    (PieceKey::Rocketman, "Rm"),      // 29 (king-slot piece in hand-written
                                      // eval, but its own Rm row here)
    (PieceKey::Shrike, "Sh"),         // 30
    (PieceKey::SumoRook, "Sr"),       // 31
    (PieceKey::WarAutomaton, "Wa"),   // 32
    (PieceKey::WarpJumper, "Wj"),     // 33
    (PieceKey::Pawn, "P"),            // 34
    (PieceKey::Rook, "R"),            // 35
    (PieceKey::Knight, "N"),          // 36
    (PieceKey::Bishop, "B"),          // 37
    (PieceKey::Queen, "Q"),           // 38
    (PieceKey::King, "K"),            // 39
];

/// Map a resolved (effective) PieceKey to its feature-type index.
/// Returns None for every piece with no training code: Bishopper (23),
/// Hordeling (13), TranslucentPawn (14), Ooze (15), all king variants
/// (WarKing 53, MeditatingKing 54, PopeKing 55, IrreverentKing 56), all
/// queen variants (VoidQueen 50, NullQueen 51, WitchQueen 52), RookZero
/// (46), and every other extended key (43-45, 49, 57-67) EXCEPT Blueprint
/// (48) and Pinata (47), which ARE trained rows above. None also covers
/// PieceKey::None. Callers must skip None, never substitute a neighbour.
pub fn piece_type_index(key: PieceKey) -> Option<usize> {
    use PieceKey::*;
    Some(match key {
        Aristocrat => 0,
        AntiViolence => 1,
        Basilisk => 2,
        BisHop => 3,
        Banker => 4,
        Bouncer => 5,
        Blueprint => 6,
        BladeRunner => 7,
        Camel => 8,
        Cardinal => 9,
        Dancer => 10,
        Djinn => 11,
        ElectroKnight => 12,
        EpeePawn => 13,
        Fish => 14,
        FissionReactor => 15,
        GoldenPawn => 16,
        Gunslinger => 17,
        HordeMother => 18,
        HeroPawn => 19,
        Icicle => 20,
        IronPawn => 21,
        Knightmare => 22,
        Marauder => 23,
        Pilgrim => 24,
        Pinata => 25,
        PawnWithKnife => 26,
        PhaseRook => 27,
        Recruit => 28,
        Rocketman => 29,
        Shrike => 30,
        SumoRook => 31,
        WarAutomaton => 32,
        WarpJumper => 33,
        Pawn => 34,
        Rook => 35,
        Knight => 36,
        Bishop => 37,
        Queen => 38,
        King => 39,
        _ => return Option::None,
    })
}

/// Active NNUE features for a position, plus how many on-board alive pieces
/// were skipped for having no training code. Indices are sorted ascending
/// (matching the trainer pre-tokenizer's sorted, deduplicated records).
/// The side to move is read from `pos.stm`, NOT from any evaluator caller:
/// the network predicts P(side-to-move wins) for the position as stored.
pub struct FeatureSet {
    pub indices: Vec<u16>,
    pub skipped: usize,
}

pub fn pos_features(pos: &Pos) -> FeatureSet {
    let stm = pos.stm & 1;
    let mut indices: Vec<u16> = Vec::new();
    let mut skipped = 0usize;
    for p in &pos.pieces {
        if !xy_on_board(p.x, p.y) || !p.alive {
            continue;
        }
        let ek = effective_key(p);
        let t = match piece_type_index(ek) {
            Some(t) => t,
            None => {
                skipped += 1;
                continue;
            }
        };
        let sq = sq(p.x as i8, p.y as i8) as usize;
        let sqp = if stm == 0 { sq } else { 63 - sq };
        let plane = if p.player == stm { 0 } else { 1 };
        indices.push(((plane * NNUE_NUM_TYPES + t) * 64 + sqp) as u16);
    }
    indices.sort_unstable();
    indices.dedup();
    FeatureSet { indices, skipped }
}

// ── Weights ────────────────────────────────────────────────────────────────

pub struct NnueWeights {
    pub w1: Vec<f32>,
    pub b1: [f32; NNUE_HIDDEN_SIZE],
    pub w2: [f32; NNUE_HIDDEN_SIZE],
    pub b2: f32,
}

fn read_f32(bytes: &[u8], off: &mut usize) -> f32 {
    let v = f32::from_le_bytes([
        bytes[*off],
        bytes[*off + 1],
        bytes[*off + 2],
        bytes[*off + 3],
    ]);
    *off += 4;
    v
}

/// Parse + validate a model blob. Rejects (returns None) on any wrong size
/// or any non-finite weight. Parsing into a fresh struct first means a
/// rejected load can never corrupt previously loaded weights.
pub fn parse_weights(bytes: &[u8]) -> Option<NnueWeights> {
    if bytes.len() != NNUE_MODEL_BYTES {
        return None;
    }
    let mut off = 0usize;
    let mut w1 = Vec::with_capacity(NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE);
    for _ in 0..(NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE) {
        w1.push(read_f32(bytes, &mut off));
    }
    let mut b1 = [0.0f32; NNUE_HIDDEN_SIZE];
    for v in b1.iter_mut() {
        *v = read_f32(bytes, &mut off);
    }
    let mut w2 = [0.0f32; NNUE_HIDDEN_SIZE];
    for v in w2.iter_mut() {
        *v = read_f32(bytes, &mut off);
    }
    let b2 = read_f32(bytes, &mut off);
    debug_assert_eq!(off, NNUE_MODEL_BYTES);
    if !w1.iter().chain(b1.iter()).chain(w2.iter()).chain([&b2]).all(|v| v.is_finite()) {
        return None;
    }
    Some(NnueWeights { w1, b1, w2, b2 })
}

/// Sparse forward pass: only active features touch W1 (same order as the
/// trainer validator: sorted active indices, B1 first, ReLU, then W2 + B2).
/// Out-of-range indices are ignored defensively (the converter never emits
/// them); an empty feature set yields the bias-only prior, never a crash.
pub fn forward(weights: &NnueWeights, features: &[u16]) -> f32 {
    let mut hidden = [0.0f64; NNUE_HIDDEN_SIZE];
    for h in 0..NNUE_HIDDEN_SIZE {
        hidden[h] = weights.b1[h] as f64;
    }
    for &f in features {
        let fi = f as usize;
        if fi >= NNUE_INPUT_SIZE {
            continue;
        }
        let base = fi * NNUE_HIDDEN_SIZE;
        for h in 0..NNUE_HIDDEN_SIZE {
            hidden[h] += weights.w1[base + h] as f64;
        }
    }
    let mut out = weights.b2;
    for h in 0..NNUE_HIDDEN_SIZE {
        let v = hidden[h] as f32;
        if v > 0.0 {
            out += weights.w2[h] * v;
        }
    }
    1.0 / (1.0 + (-out).exp())
}

/// Win-probability -> centipawns (see module docs). Pure function, so the
/// probe can reproduce it exactly in JS and compare within rounding.
pub fn prob_to_cp(p: f32) -> i32 {
    let pc = p.clamp(0.001, 0.999);
    let cp = 400.0 * (pc / (1.0 - pc)).log10();
    (cp.round() as i32).clamp(-1200, 1200)
}

/// Total eval from a clamped net score plus state/material extras.
/// 2026-10-31 mirage fix: NEVER flat-clamp the total at ±1200. The net
/// score alone is clamped to ±1200 by prob_to_cp, but the extras (unmapped
/// material, king exposure, aura, pixie-state delta) carry real material
/// information: clamping the sum at 1200 made a position that hangs a Queen
/// score identically to one that keeps it whenever the raw total exceeded
/// 1200 (the +12.00 mirage in HBSOD3, ds6CQR, sWKTin). The ±20000 ceiling
/// is a safety net only: no realistic total approaches it (a few thousand
/// at most), and it sits far below MATE_CONST (1,000,000) and the veto
/// magnitudes (50,000+), so no mate or veto logic can mistake an eval for
/// a mate or a veto.
#[inline]
pub fn eval_total_cp(base_cp: i32, extra: i32) -> i32 {
    (base_cp + extra).clamp(-20000, 20000)
}

// ── Loaded-weight store + evaluator dispatch ───────────────────────────────
// The store is deliberately load-only in production: there is no unload
// API, so a think can never lose its weights mid-search. Tests use the
// cfg(test)-only clear helper below under a lock.

// ── Unified Brain: Compile-Time Embedded Default Weights + Runtime Override ─
// `EMBEDDED_MODEL_BYTES` bakes `assets/pxb-nnue-phase2.bin` (5,244,932 bytes)
// directly into the engine binary via `include_bytes!`. On first evaluation or
// search, `active_weights()` lazily parses and installs the embedded model
// into `WEIGHTS` with zero external file/fetch dependency. Calling
// `pxb_nnue_load` at runtime still overrides `WEIGHTS` if desired, and test
// suites can call `nnue_clear_for_tests()` to exercise the hand-written fallback.
pub static EMBEDDED_MODEL_BYTES: &[u8; NNUE_MODEL_BYTES] =
    include_bytes!("../../assets/pxb-nnue-phase2.bin");

// FIX (generator copy, 2026-11-05): the old `static mut WEIGHTS` was lazily
// initialized with NO synchronization. Two selfplay workers calling the NNUE
// eval for the first time concurrently both parsed and wrote the global at
// once -> data race -> segfault ~70% of multi-worker runs, 0.1s after start.
// OnceLock parses exactly once; concurrent callers block until it is ready.
#[cfg(not(test))]
static WEIGHTS: std::sync::OnceLock<Option<NnueWeights>> = std::sync::OnceLock::new();

#[cfg(not(test))]
#[inline]
fn active_weights() -> Option<&'static NnueWeights> {
    WEIGHTS
        .get_or_init(|| parse_weights(EMBEDDED_MODEL_BYTES))
        .as_ref()
}

#[cfg(not(test))]
pub(crate) fn with_weights<R>(f: impl FnOnce(&NnueWeights) -> R) -> Option<R> {
    active_weights().map(f)
}

#[cfg(not(test))]
pub fn nnue_load(bytes: &[u8]) -> bool {
    // Generator-only note: a runtime override is honored only if the weights
    // were never initialized (OnceLock is set-once). The selfplay generator
    // never calls this; it always uses the embedded model.
    match parse_weights(bytes) {
        Some(w) => WEIGHTS.set(Some(w)).is_ok(),
        None => false,
    }
}

#[cfg(not(test))]
pub fn nnue_is_loaded() -> bool {
    active_weights().is_some()
}

// Under `cargo test`, isolate weight overrides per-thread so `nnue_tests`
// (which exercises unloaded fallback and zero-weight loads) never races with
// concurrent search/think tests running on other threads.
#[cfg(test)]
enum TestWeightSlot {
    Embedded,
    Unloaded,
    Custom(Box<NnueWeights>),
}

#[cfg(test)]
static EMBEDDED_PARSED: std::sync::OnceLock<NnueWeights> = std::sync::OnceLock::new();

#[cfg(test)]
std::thread_local! {
    static TEST_WEIGHT_SLOT: std::cell::RefCell<TestWeightSlot> =
        const { std::cell::RefCell::new(TestWeightSlot::Embedded) };
}

#[cfg(test)]
fn embedded_weights_ref() -> &'static NnueWeights {
    EMBEDDED_PARSED.get_or_init(|| {
        parse_weights(EMBEDDED_MODEL_BYTES).expect("embedded NNUE weights must be valid")
    })
}

#[cfg(test)]
pub(crate) fn with_weights<R>(f: impl FnOnce(&NnueWeights) -> R) -> Option<R> {
    TEST_WEIGHT_SLOT.with(|slot| match &*slot.borrow() {
        TestWeightSlot::Embedded => Some(f(embedded_weights_ref())),
        TestWeightSlot::Unloaded => None,
        TestWeightSlot::Custom(w) => Some(f(w)),
    })
}

#[cfg(test)]
pub fn nnue_load(bytes: &[u8]) -> bool {
    match parse_weights(bytes) {
        Some(w) => {
            TEST_WEIGHT_SLOT.with(|slot| {
                *slot.borrow_mut() = TestWeightSlot::Custom(Box::new(w));
            });
            true
        }
        None => false,
    }
}

#[cfg(test)]
pub fn nnue_is_loaded() -> bool {
    TEST_WEIGHT_SLOT.with(|slot| !matches!(&*slot.borrow(), TestWeightSlot::Unloaded))
}

/// P(stm wins) for a position, or None when no weights are loaded.
pub fn nnue_prob(pos: &Pos) -> Option<f32> {
    with_weights(|w| forward(w, &pos_features(pos).indices))
}

/// Rank 6 Fix: Additive material correction (in centipawns from `me`'s perspective)
/// for alive, on-board pieces whose effective `PieceKey` is outside the 40-type
/// NNUE v1 schema (`piece_type_index(k).is_none()`).
///
/// - Keeps the 5120-feature (`2 * 40 * 64`) binary schema 100% untouched.
/// - Returns `0` on all positions containing only the 40 mapped piece types
///   (preserving `0cp` diff on all trained fixtures).
/// - Skips King-flagged pieces (`is_king_piece(p)`) such as `WarKing`, `MeditatingKing`,
///   `PopeKing`, and `IrreverentKing` so royal variants do not inject +20,000cp,
///   while valuing unmapped non-royal pieces (`Bishopper = 379`, `Ooze = 300`,
///   `RookZero = 490`, `ExileBishop = 330`, `Samurai = 400`, etc.) via `eval::piece_value(p)`.
#[inline]
pub fn unmapped_material_delta_cp(pos: &Pos, me: u8) -> i32 {
    let mut delta = 0i32;
    for p in &pos.pieces {
        if !p.alive || !xy_on_board(p.x, p.y) {
            continue;
        }
        let k = effective_key(p);
        if piece_type_index(k).is_none() && !crate::eval::is_king_piece(p) {
            let v = crate::eval::piece_value(p);
            if p.player == me {
                delta += v;
            } else {
                delta -= v;
            }
        }
    }
    delta
}

/// Rank 4 Fix: Phase-scaled King exposure correction (in centipawns from `me`'s
/// perspective) layered onto the NNUE leaf evaluation so deeper search plies
/// and opponent King-walk lines agree with the root shelter veto.
/// Only active when the opposing side still has midgame attacking firepower
/// (`opp_attacking_phase >= 6`) and a King has walked out of its back rank (`ky >= 1`).
#[inline]
pub fn king_exposure_nnue_delta(pos: &Pos, me: u8) -> i32 {
    let mut delta = 0i32;
    for p in &pos.pieces {
        if !p.alive || !xy_on_board(p.x, p.y) || !crate::eval::is_king_piece(p) {
            continue;
        }
        let c = p.player;
        let opp = 1 - c;
        let ky = if c == 0 { p.y as i32 } else { 7 - (p.y as i32) };
        if ky >= 1 {
            let opp_ph = crate::eval::opp_attacking_phase(pos, opp);
            if opp_ph >= 6 {
                let kx = p.x as i32;
                let pen = ((40 * ky * ky + if (2..=5).contains(&kx) { 25 * ky } else { 0 }) * opp_ph) / 12;
                if c == me {
                    delta -= pen;
                } else {
                    delta += pen;
                }
            }
        }
    }
    delta
}

/// Tactical aura correction for `AntiViolence` suppression and `HordeMother`
/// line-of-sight threats on high-value pieces during NNUE leaf evaluation.
#[inline]
pub fn pixie_aura_nnue_delta(pos: &Pos, me: u8) -> i32 {
    let mut delta = 0i32;
    for p in &pos.pieces {
        if !p.alive || !xy_on_board(p.x, p.y) {
            continue;
        }
        let ek = effective_key(p);
        if ek == PieceKey::AntiViolence {
            // Reward AntiViolence for suppressing adjacent enemy capturers
            let px = p.x as i8;
            let py = p.y as i8;
            let mut aura = 0i32;
            for q in &pos.pieces {
                if !q.alive || q.player == p.player || !xy_on_board(q.x, q.y) {
                    continue;
                }
                if (q.x as i8 - px).abs() <= 1 && (q.y as i8 - py).abs() <= 1 {
                    let qv = crate::eval::piece_value(q);
                    if qv >= 450 && !crate::eval::is_king_piece(q) {
                        aura += 35;
                    } else if qv >= 250 && !crate::eval::is_king_piece(q) {
                        aura += 18;
                    }
                }
            }
            if p.player == me {
                delta += aura.min(70);
            } else {
                delta -= aura.min(70);
            }
        }
    }
    delta
}

/// Pixie-state NNUE delta (2026-10-31 audit).
///
/// The 5120-feature schema encodes (side, type, square) only: per-piece
/// state (Marauder kills, Pilgrim odo, Reactor caps, ElectroKnight quiet,
/// Dancer encore, frost, limbo/offboard) is invisible to the net, and the
/// hand evaluator's state terms only run when weights are NOT loaded.
/// This delta prices that state on top of the net's output, from `me`'s
/// perspective, antisymmetrically: delta(pos, 1-me) == -delta(pos, me).
///
/// Calibration (live net, 2026-10-31 probe, test piece on d4, kings e1/e8):
/// Recalibrated 2026-10-31 for v3 weights (was 325/503 on the rolled-back
/// 106k net): net(Queen) - net(Marauder) = 203; net(Queen) - net(Pilgrim)
/// = 240, measured by pixie_state_calibration_probe on v3. Only the two
/// gap-derived terms (Marauder ramp, Pilgrim charged/ramp) were rescaled;
/// all other terms are state-only (the net is blind to those states on
/// both nets) and are unchanged. Marauder kills scale by range/7 (3/5/7
/// tiles); Pilgrim odo ramps quadratically to the full gap at charge.
/// All other terms are policy
/// numbers, deliberately conservative: the search already sees these
/// mechanics as moves, so the delta prices only the static/positional
/// residue the search cannot reach at shallow depth.
#[inline]
pub fn pixie_state_nnue_delta(pos: &Pos, me: u8) -> i32 {
    let me = me & 1;
    // Sign of a term owned by `player`: + when it benefits `me`.
    let sgn = |player: u8| -> i32 { if player == me { 1 } else { -1 } };
    let mut delta = 0i32;

    // Occupancy lookup through pos.at, guarded against stale indices.
    let slot_at = |x: i8, y: i8| -> Option<usize> {
        if !on_board(x, y) {
            return None;
        }
        let v = pos.at[sq(x, y) as usize];
        if v == 0 || (v as usize) > pos.pieces.len() {
            return None;
        }
        let idx = (v - 1) as usize;
        if pos.pieces[idx].alive {
            Some(idx)
        } else {
            None
        }
    };
    // Value of a piece as a threat victim of `owner`: enemy, non-king only.
    let victim_value = |idx: usize, owner: u8| -> Option<i32> {
        let q = &pos.pieces[idx];
        if q.player == owner || crate::eval::is_king_piece(q) {
            return None;
        }
        Some(crate::eval::piece_value(q))
    };

    // Friendly hordeling counts (HordeMother synergy, mirrored from hand eval).
    let mut hordelings = [0i32; 2];
    for p in &pos.pieces {
        if !p.alive || !xy_on_board(p.x, p.y) {
            continue;
        }
        if p.key == PieceKey::Hordeling || p.sub_key == PieceKey::Hordeling {
            hordelings[(p.player & 1) as usize] += 1;
        }
    }

    // Ray-threat jobs: (piece index, kind) with kind 0 = Marauder rays,
    // 1 = charged-Pilgrim rays, 2 = charged-ElectroKnight targets.
    let mut ray_jobs: Vec<(usize, u8)> = Vec::new();
    // Lazily built occupancy for the AntiViolence arm (square_attacked
    // needs it); built only when an AV is actually on the board.
    let mut occ_cache: Option<[bool; 64]> = None;

    for (i, p) in pos.pieces.iter().enumerate() {
        if !p.alive || !xy_on_board(p.x, p.y) {
            continue;
        }
        let ek = effective_key(p);
        let s = sgn(p.player);
        match ek {
            PieceKey::Marauder => {
                // Range 1 + 2*kills (3/5/7 tiles) vs queen rays: scale the
                // calibrated 203 gap (v3) by range/7: 87/145/203 exactly.
                let v = match p.kills {
                    0 => 0,
                    1 => 87,
                    2 => 145,
                    _ => 203,
                };
                delta += s * v;
                if p.kills >= 1 {
                    ray_jobs.push((i, 0));
                }
            }
            PieceKey::Pilgrim => {
                if p.pilgrim_charged() || p.odo >= 20 {
                    // Charged Pilgrim moves as a queen: full calibrated gap (240 on v3).
                    delta += s * 240;
                    ray_jobs.push((i, 1));
                } else {
                    // Quadratic ramp toward the gap (hand eval's shape,
                    // rescaled to the calibrated gap) + flat approaching
                    // threat, mirroring the hand 30/75 pens.
                    let o = p.odo as i32;
                    delta += s * ((o * o * 240) / 400);
                    if p.odo >= 15 {
                        delta += s * 75;
                    } else if p.odo >= 10 {
                        delta += s * 30;
                    }
                }
            }
            PieceKey::FissionReactor => {
                // Suicide discount: the reactor dies on its 5th capture.
                let disc = match p.caps {
                    0..=2 => 0,
                    3 => 25,
                    _ => 100,
                };
                delta -= s * disc;
                if p.caps >= 3 {
                    // Blast threat: diagonal neighbours die when it pops.
                    // One capture away (caps 4) is more dangerous than two.
                    let frac = if p.caps >= 4 { 40 } else { 20 };
                    let mut sum = 0i32;
                    for (dx, dy) in [(1i8, 1i8), (1, -1), (-1, 1), (-1, -1)] {
                        if let Some(vi) = slot_at(p.x as i8 + dx, p.y as i8 + dy) {
                            if let Some(v) = victim_value(vi, p.player) {
                                sum += v;
                            }
                        }
                    }
                    let pen = ((sum * frac) / 100).min(400);
                    delta += s * pen;
                }
            }
            PieceKey::ElectroKnight => {
                let charged = p.echarged() || p.quiet >= 3;
                if charged {
                    delta += s * 150;
                    ray_jobs.push((i, 2));
                } else if p.quiet == 2 {
                    delta += s * 50;
                }
            }
            PieceKey::Dancer => {
                if p.encore_pending() {
                    delta += s * 150;
                    delta += s * 75; // diffuse double-move threat
                } else if p.encore >= 2 {
                    delta += s * 60;
                } else if p.encore == 1 {
                    delta += s * 25;
                }
            }
            PieceKey::HordeMother => {
                let n = hordelings[(p.player & 1) as usize];
                let m = n.min(6);
                delta += s * (30 + 22 * n + 5 * m * m);
            }
            PieceKey::Rocketman => {
                if !p.used() {
                    delta += s * 50; // once-per-game teleport option
                }
            }
            PieceKey::Bishopper | PieceKey::BisHop => {
                if p.crouched() {
                    delta += s * 75; // leap-capture stance
                }
            }
            PieceKey::AntiViolence => {
                // Suppression protection (2026-10-31, AV-undervalued fix):
                // the aura credits AV for adjacent ENEMIES, but not for
                // shielding attacked FRIENDLIES, so the bot priced AV like
                // a Knight (~315) and let opponents happily trade one for
                // it. Precise version: OR the RAW attack bitboards of all
                // enemies adjacent to this AV (suppressed enemies cannot
                // capture, so square_attacked would hide them; raw
                // piece_attacks shows what they WOULD take). A friendly
                // adjacent to AV whose square is in that union is being
                // shielded: +25 each, King shielding fills the +60 cap
                // at once. Antisymmetric via s, like every arm here.
                let ax = p.x as i8;
                let ay = p.y as i8;
                let occ_ref = occ_cache.get_or_insert_with(|| occ(pos));
                let mut sup_u64 = 0u64;
                for (ei, q) in pos.pieces.iter().enumerate() {
                    if !q.alive || q.player == p.player || !xy_on_board(q.x, q.y) {
                        continue;
                    }
                    if (q.x as i8 - ax).abs() <= 1 && (q.y as i8 - ay).abs() <= 1 {
                        sup_u64 |= crate::movegen::piece_attacks(pos, ei, occ_ref, true, false).0;
                    }
                }
                if sup_u64 != 0 {
                    let mut prot = 0i32;
                    for q in &pos.pieces {
                        if !q.alive || q.player != p.player || !xy_on_board(q.x, q.y) {
                            continue;
                        }
                        if q.x == p.x && q.y == p.y {
                            continue;
                        }
                        if (q.x as i8 - ax).abs() <= 1 && (q.y as i8 - ay).abs() <= 1 {
                            let qsq = (q.y << 3) | q.x;
                            if (sup_u64 & (1u64 << qsq)) != 0 {
                                if crate::eval::is_king_piece(q) {
                                    prot = 60;
                                    break;
                                }
                                prot += 25;
                                if prot >= 60 {
                                    prot = 60;
                                    break;
                                }
                            }
                        }
                    }
                    delta += s * prot;
                }
            }
            _ => {}
        }
        // Frozen pieces lose their attacks but keep full net value.
        if p.frost > 0 {
            let pen = 25 * (p.frost as i32).min(3) + crate::eval::piece_value(p) / 12;
            delta -= s * pen;
        }
    }

    // Limbo (Djinn only in practice): a captured Djinn revives on its home
    // square at the next death, killing an enemy non-king standing there.
    for l in &pos.limbo {
        if l.key == PieceKey::Djinn || l.sub_key == PieceKey::Djinn {
            let s = sgn(l.player);
            delta += s * 150; // pending revive
            if let Some(vi) = slot_at(l.home_x, l.home_y) {
                let q = &pos.pieces[vi];
                if q.player != l.player && !crate::eval::is_king_piece(q) {
                    let pen = (crate::eval::piece_value(q) * 30 / 100).min(250);
                    delta += s * pen;
                }
            }
        }
    }

    // Offboard is Knightmare-only by construction: a reserve that can drop
    // anywhere, mirrored from the hand evaluator's +65.
    for o in &pos.offboard {
        if o.key == PieceKey::Knightmare || o.sub_key == PieceKey::Knightmare {
            delta += sgn(o.player) * 65;
        }
    }

    if !ray_jobs.is_empty() {
        let occ = occ(pos);
        for (idx, kind) in ray_jobs {
            let p = &pos.pieces[idx];
            let owner = p.player;
            let s = sgn(owner);
            if kind == 2 {
                // ElectroKnight splash: knight-target enemy piece with an
                // orthogonal enemy neighbour (the second AoE victim).
                let targets = crate::movegen::piece_attacks(pos, idx, &occ, true, false);
                let mut best_splash = 0i32;
                let mut bb = targets.0;
                while bb != 0 {
                    let t = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    if let Some(ti) = slot_at(sq_x(t) as i8, sq_y(t) as i8) {
                        let tp = &pos.pieces[ti];
                        if tp.player == owner || crate::eval::is_king_piece(tp) {
                            continue;
                        }
                        for (dx, dy) in [(1i8, 0i8), (-1, 0), (0, 1), (0, -1)] {
                            if let Some(si) =
                                slot_at(sq_x(t) as i8 + dx, sq_y(t) as i8 + dy)
                            {
                                if let Some(v) = victim_value(si, owner) {
                                    if v > best_splash {
                                        best_splash = v;
                                    }
                                }
                            }
                        }
                    }
                }
                let pen = (best_splash * 25 / 100).min(200);
                delta += s * pen;
            } else {
                // Marauder / charged-Pilgrim rays: most valuable victim.
                // One capture per move, so one victim, not the whole ray.
                let rays = crate::movegen::piece_attacks(pos, idx, &occ, true, false);
                let mut best = 0i32;
                let mut bb = rays.0;
                while bb != 0 {
                    let t = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    if let Some(vi) = slot_at(sq_x(t) as i8, sq_y(t) as i8) {
                        if let Some(v) = victim_value(vi, owner) {
                            if v > best {
                                best = v;
                            }
                        }
                    }
                }
                let pen = (best * 20 / 100).min(200);
                delta += s * pen;
            }
        }
    }

    delta
}

/// Evaluate `pos` from `me`'s perspective using the active NNUE weights and a
/// pre-computed `NnueAccumulator` (`O(NNUE_HIDDEN_SIZE)` = 256 multiply-adds, zero heap alloc).
#[inline]
pub fn nnue_evaluate_with_acc(
    pos: &Pos,
    acc: &crate::accumulator::NnueAccumulator,
    me: u8,
) -> Option<i32> {
    let me = me & 1;
    if let Some(kt) = pos.king_thrown {
        return Some(if kt == me { MATE_CONST } else { -MATE_CONST });
    }
    if let Some(gw) = pos.gold_win {
        return Some(if gw == me { MATE_CONST } else { -MATE_CONST });
    }
    let p = with_weights(|w| acc.forward(w))?;
    let stm = acc.stm & 1;
    let cp = prob_to_cp(p);
    let base_cp = if me == stm { cp } else { -cp };
    let extra = unmapped_material_delta_cp(pos, me)
        + king_exposure_nnue_delta(pos, me)
        + pixie_aura_nnue_delta(pos, me)
        + pixie_state_nnue_delta(pos, me);
    Some(eval_total_cp(base_cp, extra))
}

/// NNUE score from the caller's perspective, or None when unloaded.
/// Terminal states match the hand-written evaluator exactly.
pub fn nnue_evaluate(pos: &Pos, me: u8) -> Option<i32> {
    let me = me & 1;
    if let Some(kt) = pos.king_thrown {
        return Some(if kt == me { MATE_CONST } else { -MATE_CONST });
    }
    if let Some(gw) = pos.gold_win {
        return Some(if gw == me { MATE_CONST } else { -MATE_CONST });
    }
    let p = nnue_prob(pos)?;
    let stm = pos.stm & 1;
    let cp = prob_to_cp(p);
    let base_cp = if me == stm { cp } else { -cp };
    let extra = unmapped_material_delta_cp(pos, me)
        + king_exposure_nnue_delta(pos, me)
        + pixie_aura_nnue_delta(pos, me)
        + pixie_state_nnue_delta(pos, me);
    Some(eval_total_cp(base_cp, extra))
}

/// Single dispatch point: NNUE when weights are loaded, otherwise EXACTLY
/// the hand-written evaluator (same function, same result — proven by the
/// no-weights dispatch test below and by WASM probes against the revamp
/// binary). Every evaluator call site goes through this or through a
/// Search whose evaluator is one of the two statics below.
pub fn evaluate_dispatched(pos: &Pos, me: u8) -> i32 {
    if let Some(v) = nnue_evaluate(pos, me) {
        return v;
    }
    crate::eval::evaluate(pos, me)
}

/// Dispatching evaluator: checks the loaded flag on every call. This is
/// Search's default evaluator; run_ranked_root_search pins one evaluator
/// per think (see think.rs) so a load completing mid-think cannot flip
/// the scorer inside a single search.
pub struct DispatchingEvaluator;

impl Evaluator for DispatchingEvaluator {
    fn evaluate(&self, pos: &Pos, me: u8) -> i32 {
        evaluate_dispatched(pos, me)
    }
}

pub static DISPATCHING_EVALUATOR: DispatchingEvaluator = DispatchingEvaluator;

/// Pinned NNUE evaluator: uses the net when loaded, and falls back to the
/// hand-written evaluator if called without weights (defensive; the pin
/// site only selects this when weights are loaded). Never returns None.
pub struct NnueEvaluator;

impl Evaluator for NnueEvaluator {
    fn evaluate(&self, pos: &Pos, me: u8) -> i32 {
        if let Some(v) = nnue_evaluate(pos, me) {
            return v;
        }
        crate::eval::evaluate(pos, me)
    }
}

pub static NNUE_EVALUATOR: NnueEvaluator = NnueEvaluator;

#[cfg(test)]
pub fn nnue_clear_for_tests() {
    TEST_WEIGHT_SLOT.with(|slot| {
        *slot.borrow_mut() = TestWeightSlot::Unloaded;
    });
}

#[cfg(test)]
pub fn nnue_restore_embedded_for_tests() {
    TEST_WEIGHT_SLOT.with(|slot| {
        *slot.borrow_mut() = TestWeightSlot::Embedded;
    });
}

// ── FEN support (dev/parity only — never used by eval/search paths) ───────
// Parses the trainer's SPEC-dialect FEN board: 8 '/'-separated ranks,
// rank 8 first; 2-letter codes (first-letter case = side, second letter
// lowercase) or 1-letter standard singles; digits = empty counts. Rank
// segmentation follows SPEC §5: a segmentation is valid iff its tokens sum
// to exactly 8 squares, resolved DFS-first with preference code > standard
// single > digit (the same rule as train/src/bin/pretokenize.rs).

fn code_to_key(first: char, second: char) -> Option<PieceKey> {
    if !second.is_ascii_lowercase() {
        return None;
    }
    let up = first.to_ascii_uppercase();
    for &(key, code) in NNUE_TABLE.iter() {
        let b = code.as_bytes();
        if b.len() == 2 && b[0] == up as u8 && b[1] == second as u8 {
            return Some(key);
        }
    }
    None
}

fn std_to_key(c: char) -> Option<PieceKey> {
    use PieceKey::*;
    Some(match c.to_ascii_uppercase() {
        'P' => Pawn,
        'R' => Rook,
        'N' => Knight,
        'B' => Bishop,
        'Q' => Queen,
        'K' => King,
        _ => return Option::None,
    })
}

fn rank_feasible(ch: &[char], pos: usize, used: usize) -> bool {
    if used > 8 {
        return false;
    }
    if pos == ch.len() {
        return used == 8;
    }
    let c = ch[pos];
    if c.is_ascii_digit() {
        return rank_feasible(ch, pos + 1, used + (c as usize - '0' as usize));
    }
    if pos + 1 < ch.len() {
        if code_to_key(c, ch[pos + 1]).is_some()
            && rank_feasible(ch, pos + 2, used + 1)
        {
            return true;
        }
    }
    if std_to_key(c).is_some() {
        return rank_feasible(ch, pos + 1, used + 1);
    }
    false
}

/// Parse a FEN board string into a Pos (stm_white selects pos.stm).
/// Returns None on any malformed rank. Test/parity tooling only.
pub fn fen_to_pos(fen: &str, stm_white: bool) -> Option<Pos> {
    let ranks: Vec<&str> = fen.split('/').collect();
    if ranks.len() != 8 {
        return None;
    }
    let mut pos = Pos::default();
    pos.stm = if stm_white { 0 } else { 1 };
    let mut id: u32 = 1;
    for (r, rk) in ranks.iter().enumerate() {
        let ch: Vec<char> = rk.chars().collect();
        if !rank_feasible(&ch, 0, 0) {
            return None;
        }
        let rank_idx = 7 - r as i32;
        let mut file: i32 = 0;
        let mut i = 0usize;
        while i < ch.len() {
            let c = ch[i];
            if c.is_ascii_digit() {
                file += c as i32 - '0' as i32;
                i += 1;
                continue;
            }
            let mut moved = false;
            if i + 1 < ch.len() {
                if let Some(key) = code_to_key(c, ch[i + 1]) {
                    if rank_feasible(&ch, i + 2, (file + 1) as usize) {
                        let player = if c.is_ascii_uppercase() { 0 } else { 1 };
                        pos.pieces.push(Piece {
                            id,
                            key,
                            sub_key: key,
                            player,
                            x: file as u8,
                            y: rank_idx as u8,
                            alive: true,
                            ..Default::default()
                        });
                        pos.at[((rank_idx as usize) << 3) | (file as usize)] =
                            pos.pieces.len() as u16;
                        id += 1;
                        file += 1;
                        i += 2;
                        moved = true;
                    }
                }
            }
            if !moved {
                if let Some(key) = std_to_key(c) {
                    if rank_feasible(&ch, i + 1, (file + 1) as usize) {
                        let player = if c.is_ascii_uppercase() { 0 } else { 1 };
                        pos.pieces.push(Piece {
                            id,
                            key,
                            sub_key: key,
                            player,
                            x: file as u8,
                            y: rank_idx as u8,
                            alive: true,
                            ..Default::default()
                        });
                        pos.at[((rank_idx as usize) << 3) | (file as usize)] =
                            pos.pieces.len() as u16;
                        id += 1;
                        file += 1;
                        i += 1;
                        moved = true;
                    }
                }
            }
            if !moved {
                return None;
            }
        }
        if file != 8 {
            return None;
        }
    }
    Some(pos)
}

#[cfg(test)]
mod nnue_tests {
    use super::*;
    use crate::eval::evaluate;
    use crate::pos::*;
    use crate::types::*;
    use std::sync::Mutex;

    // Serializes every test that touches the process-global weight store.
    // Pure tests (table, converter, parsing, forward) never take this lock
    // and run in parallel as usual.
    static LOAD_LOCK: Mutex<()> = Mutex::new(());

    fn piece(key: PieceKey, player: u8, x: u8, y: u8) -> Piece {
        Piece {
            id: 1,
            key,
            sub_key: key,
            player,
            x,
            y,
            alive: true,
            ..Default::default()
        }
    }

    fn pos_with(pieces: Vec<Piece>, stm: u8) -> Pos {
        let mut pos = Pos::default();
        pos.stm = stm;
        for (i, mut p) in pieces.into_iter().enumerate() {
            p.id = (i + 1) as u32;
            pos.at[((p.y as usize) << 3) | (p.x as usize)] = (i + 1) as u16;
            pos.pieces.push(p);
        }
        pos
    }

    #[test]
    fn table_covers_exactly_the_trainer_40() {
        // Every SPEC row maps back to its own index, and the table order
        // matches the trainer's CODES order (34 codes) + 6 standards.
        for (i, &(key, _code)) in NNUE_TABLE.iter().enumerate() {
            assert_eq!(piece_type_index(key), Some(i), "{key:?} must map to {i}");
        }
        assert_eq!(NNUE_TABLE.len(), 40);
        // Spot-check values against types.rs discriminants via the table.
        assert_eq!(NNUE_TABLE[27].0 as u16, 39); // PhaseRook
        assert_eq!(NNUE_TABLE[4].0 as u16, 21); // Banker
        assert_eq!(NNUE_TABLE[24].0 as u16, 40); // Pilgrim -> Pg
        assert_eq!(NNUE_TABLE[25].0 as u16, 47); // Pinata -> Pi
        assert_eq!(NNUE_TABLE[8].0 as u16, 27); // Camel -> Ca
        assert_eq!(NNUE_TABLE[9].0 as u16, 28); // Cardinal -> Cd
        assert_eq!(NNUE_TABLE[6].0 as u16, 48); // Blueprint (extended, trained)
    }

    #[test]
    fn untrained_pieces_map_to_nothing() {
        use PieceKey::*;
        for key in [
            Bishopper, Hordeling, TranslucentPawn, Ooze, ExileBishop, HordeKnight,
            CloudKnight, RookZero, Wraith, VoidQueen, NullQueen, WitchQueen, WarKing,
            MeditatingKing, PopeKing, IrreverentKing, Monument, Mushroom, LavaForge,
            Timelord, Druidess, Dryad, Duplicator, Puppet, Samurai, Uni, Dummy, None,
        ] {
            assert_eq!(piece_type_index(key), Option::None, "{key:?} must have no feature");
        }
    }

    #[test]
    fn converter_white_stm_matches_hand_computed_features() {
        // White stm: white K e1 (sq 4), white PhaseRook a1 (sq 0),
        // black K e8 (sq 60), black P a7 (sq 48).
        let pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::PhaseRook, 0, 0, 0),
                piece(PieceKey::King, 1, 4, 7),
                piece(PieceKey::Pawn, 1, 0, 6),
            ],
            0,
        );
        let fs = pos_features(&pos);
        assert_eq!(fs.skipped, 0);
        // plane0 K: (0*40+39)*64+4 = 2500; plane0 Pr: 27*64+0 = 1728;
        // plane1 K: (40+39)*64+60 = 5116; plane1 P: (40+34)*64+48 = 4784.
        assert_eq!(fs.indices, vec![1728, 2500, 4784, 5116]);
    }

    #[test]
    fn converter_black_stm_flips_squares_and_planes() {
        // Same board, black to move: squares 63-sq, planes swapped.
        let pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::PhaseRook, 0, 0, 0),
                piece(PieceKey::King, 1, 4, 7),
                piece(PieceKey::Pawn, 1, 0, 6),
            ],
            1,
        );
        let fs = pos_features(&pos);
        assert_eq!(fs.skipped, 0);
        // white K (enemy): (40+39)*64+(63-4) = 5115; white Pr (enemy):
        // (40+27)*64+63 = 4351; black K (own): 39*64+(63-60) = 2499;
        // black P (own): 34*64+(63-48) = 2191.
        assert_eq!(fs.indices, vec![2191, 2499, 4351, 5115]);
    }

    #[test]
    fn converter_skips_unknown_and_ignores_dead_and_offboard() {
        let base = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let base_feats = pos_features(&base).indices;
        // Add on-board alive unknown pieces: features unchanged, skipped = 2.
        let mut with_unknown = base.clone();
        with_unknown.pieces.push(Piece {
            id: 3,
            key: PieceKey::Bishopper,
            sub_key: PieceKey::Bishopper,
            player: 0,
            x: 0,
            y: 0,
            alive: true,
            ..Default::default()
        });
        with_unknown.pieces.push(Piece {
            id: 4,
            key: PieceKey::WarKing,
            sub_key: PieceKey::WarKing,
            player: 1,
            x: 7,
            y: 7,
            alive: true,
            ..Default::default()
        });
        // Note: (0,0) and (7,7) squares are free in this fixture (kings on e-file).
        let fs = pos_features(&with_unknown);
        assert_eq!(fs.indices, base_feats);
        assert_eq!(fs.skipped, 2);
        // A dead unknown piece and a limbo entry change nothing and are not
        // counted as skipped (skipped counts on-board alive pieces only).
        let mut with_dead = base.clone();
        let mut dead = piece(PieceKey::Ooze, 0, 1, 1);
        dead.alive = false;
        dead.id = 3;
        with_dead.pieces.push(dead);
        with_dead.limbo.push(crate::pos::LimboPiece {
            id: 9,
            key: PieceKey::RookZero,
            sub_key: PieceKey::RookZero,
            player: 1,
            home_x: 2,
            home_y: 2,
        });
        let fs2 = pos_features(&with_dead);
        assert_eq!(fs2.indices, base_feats);
        assert_eq!(fs2.skipped, 0);
    }

    #[test]
    fn converter_uses_effective_key_not_raw_key() {
        // Base-first wire encoding: key=Rook, sub_key=PhaseRook must map
        // to Pr (27), not to R (35).
        let mut p = piece(PieceKey::Rook, 0, 0, 0);
        p.sub_key = PieceKey::PhaseRook;
        let pos = pos_with(vec![p], 0);
        let fs = pos_features(&pos);
        assert_eq!(fs.indices, vec![27 * 64]);
        assert_eq!(fs.skipped, 0);
    }

    #[test]
    fn weights_parse_accepts_exact_size_rejects_bad() {
        let good = vec![0u8; NNUE_MODEL_BYTES];
        assert!(parse_weights(&good).is_some());
        assert!(parse_weights(&vec![0u8; 16]).is_none());
        assert!(parse_weights(&vec![0u8; NNUE_MODEL_BYTES - 4]).is_none());
        assert!(parse_weights(&vec![0u8; NNUE_MODEL_BYTES + 4]).is_none());
        // One NaN weight anywhere rejects the whole blob.
        let mut nan = vec![0u8; NNUE_MODEL_BYTES];
        nan[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(parse_weights(&nan).is_none());
        let mut inf = vec![0u8; NNUE_MODEL_BYTES];
        let last = NNUE_MODEL_BYTES - 4;
        inf[last..].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(parse_weights(&inf).is_none());
    }

    #[test]
    fn prob_to_cp_scale() {
        assert_eq!(prob_to_cp(0.5), 0);
        assert_eq!(prob_to_cp(0.9555), 533);
        assert_eq!(prob_to_cp(0.0399), -553);
        assert!(prob_to_cp(0.9999) <= 1200 && prob_to_cp(0.0001) >= -1200);
        assert!(prob_to_cp(0.9) > prob_to_cp(0.6) && prob_to_cp(0.6) > 0);
    }

    #[test]
    fn dispatch_without_weights_is_exactly_handwritten() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_clear_for_tests();
        assert!(!nnue_is_loaded());
        // Fixtures incl. an unknown-piece position: dispatched must equal
        // handwritten bit-for-bit on every one.
        let positions = vec![
            pos_with(
                vec![
                    piece(PieceKey::King, 0, 4, 0),
                    piece(PieceKey::PhaseRook, 0, 0, 0),
                    piece(PieceKey::King, 1, 4, 7),
                    piece(PieceKey::Pawn, 1, 0, 6),
                ],
                0,
            ),
            pos_with(
                vec![
                    piece(PieceKey::King, 0, 5, 0),
                    piece(PieceKey::Queen, 0, 3, 0),
                    piece(PieceKey::Pilgrim, 0, 5, 3),
                    piece(PieceKey::King, 1, 3, 4),
                    piece(PieceKey::ElectroKnight, 1, 4, 5),
                    piece(PieceKey::Bishopper, 0, 2, 2),
                    piece(PieceKey::Ooze, 1, 6, 6),
                ],
                1,
            ),
        ];
        for pos in &positions {
            for me in [0u8, 1u8] {
                assert_eq!(
                    evaluate_dispatched(pos, me),
                    evaluate(pos, me),
                    "no-weights dispatch must equal handwritten"
                );
            }
        }
    }

    #[test]
    fn load_rejects_bad_bytes_and_keeps_prior_weights() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_clear_for_tests();
        assert!(!nnue_load(&[0u8; 16]));
        assert!(!nnue_is_loaded());
        assert!(nnue_load(&vec![0u8; NNUE_MODEL_BYTES]));
        assert!(nnue_is_loaded());
        let before = nnue_prob(&pos_with(vec![piece(PieceKey::King, 0, 4, 0)], 0));
        // Bad loads must not disturb the loaded weights.
        assert!(!nnue_load(&[0u8; 0]));
        assert!(!nnue_load(&[0u8; 3]));
        let mut nan = vec![0u8; NNUE_MODEL_BYTES];
        nan[100..104].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(!nnue_load(&nan));
        assert!(nnue_is_loaded());
        let after = nnue_prob(&pos_with(vec![piece(PieceKey::King, 0, 4, 0)], 0));
        assert_eq!(before, after);
        nnue_clear_for_tests();
    }

    #[test]
    fn forward_matches_trainer_validator_on_val_split() {
        // Ground truth produced by the trainer's own validate binary on
        // the same files (run 2026-10-25): BCE 0.4121, means lost 0.0399 /
        // draw 0.4933 / won 0.9555 over 23,862 records. The engine forward
        // must reproduce them from the raw record bytes.
        let model_path = std::env::var("NNUE_TEST_MODEL").unwrap_or_else(|_| {
            "/home/hatch/workspace/nnue-phase2/checkpoints/nnue-phase2-final.bin"
                .to_string()
        });
        let val_path = std::env::var("NNUE_TEST_VAL").unwrap_or_else(|_| {
            "/home/hatch/workspace/nnue-phase2/checkpoints/val.bin".to_string()
        });
        let model_bytes = std::fs::read(&model_path)
            .unwrap_or_else(|e| panic!("cannot read model {model_path}: {e}"));
        let w = parse_weights(&model_bytes).expect("final model must parse");
        let data = std::fs::read(&val_path)
            .unwrap_or_else(|e| panic!("cannot read val split {val_path}: {e}"));
        assert_eq!(&data[0..4], b"N2F1");
        let n = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let input_size =
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        assert_eq!(input_size, NNUE_INPUT_SIZE);
        let mut off = 12usize;
        let mut loss = 0.0f64;
        let mut groups: [(u64, f64); 3] = [(0, 0.0), (0, 0.0), (0, 0.0)];
        for _ in 0..n {
            let na = u16::from_le_bytes([data[off], data[off + 1]]) as usize;
            off += 2;
            let mut feats = Vec::with_capacity(na);
            for _ in 0..na {
                feats.push(u16::from_le_bytes([data[off], data[off + 1]]));
                off += 2;
            }
            let label = f32::from_le_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
            ]);
            off += 4;
            let pred = forward(&w, &feats);
            let p = pred.clamp(1e-7, 1.0 - 1e-7);
            loss += (-(label * p.ln() + (1.0 - label) * (1.0 - p).ln())) as f64;
            let gi = if label >= 0.99 {
                2
            } else if label >= 0.49 {
                1
            } else {
                0
            };
            groups[gi].0 += 1;
            groups[gi].1 += pred as f64;
        }
        let mean_loss = loss / n.max(1) as f64;
        let m0 = groups[0].1 / groups[0].0.max(1) as f64;
        let m5 = groups[1].1 / groups[1].0.max(1) as f64;
        let m1 = groups[2].1 / groups[2].0.max(1) as f64;
        eprintln!("engine forward on val split: n={n} loss={mean_loss:.4} lost={m0:.4} draw={m5:.4} won={m1:.4}");
        assert!((mean_loss - 0.4121).abs() < 0.002, "val loss {mean_loss}");
        assert!((m0 - 0.0399).abs() < 0.002, "lost mean {m0}");
        assert!((m5 - 0.4933).abs() < 0.002, "draw mean {m5}");
        assert!((m1 - 0.9555).abs() < 0.002, "won mean {m1}");
        assert!(m1 > m5 && m5 > m0);
    }

    // ── Parity vs an independent SPEC §5 oracle ──────────────────────────
    // The oracle below re-implements the spec text (DP feasibility +
    // code > standard > digit reconstruction, stm-relative planes, black
    // square flip) directly on FEN strings, without touching the engine
    // converter. Agreement on real data lines proves the converter's
    // square/plane/type mapping, not just its internal consistency.
    fn oracle_features(fen: &str, stm_white: bool) -> Option<Vec<u16>> {
        fn code_idx(first: char, second: char) -> Option<usize> {
            if !second.is_ascii_lowercase() {
                return None;
            }
            let up = first.to_ascii_uppercase();
            for (i, &(_, code)) in NNUE_TABLE.iter().enumerate().take(34) {
                let b = code.as_bytes();
                if b[0] == up as u8 && b[1] == second as u8 {
                    return Some(i);
                }
            }
            None
        }
        fn std_idx(c: char) -> Option<usize> {
            Some(match c.to_ascii_uppercase() {
                'P' => 34,
                'R' => 35,
                'N' => 36,
                'B' => 37,
                'Q' => 38,
                'K' => 39,
                _ => return None,
            })
        }
        fn feas(ch: &[char], pos: usize, used: usize) -> bool {
            if used > 8 {
                return false;
            }
            if pos == ch.len() {
                return used == 8;
            }
            let c = ch[pos];
            if c.is_ascii_digit() {
                return feas(ch, pos + 1, used + (c as usize - '0' as usize));
            }
            if pos + 1 < ch.len()
                && code_idx(c, ch[pos + 1]).is_some()
                && feas(ch, pos + 2, used + 1)
            {
                return true;
            }
            std_idx(c).is_some() && feas(ch, pos + 1, used + 1)
        }
        let ranks: Vec<&str> = fen.split('/').collect();
        if ranks.len() != 8 {
            return None;
        }
        let mut feats: Vec<u16> = Vec::new();
        for (r, rk) in ranks.iter().enumerate() {
            let ch: Vec<char> = rk.chars().collect();
            if !feas(&ch, 0, 0) {
                return None;
            }
            let rank_idx = 7 - r as i32;
            let mut file: i32 = 0;
            let mut i = 0usize;
            while i < ch.len() {
                let c = ch[i];
                if c.is_ascii_digit() {
                    file += c as i32 - '0' as i32;
                    i += 1;
                    continue;
                }
                let mut moved = false;
                if i + 1 < ch.len() {
                    if let Some(t) = code_idx(c, ch[i + 1]) {
                        if feas(&ch, i + 2, (file + 1) as usize) {
                            let sq = (rank_idx * 8 + file) as usize;
                            let white = c.is_ascii_uppercase();
                            let own = white == stm_white;
                            let sqp = if stm_white { sq } else { 63 - sq };
                            let plane = if own { 0 } else { 1 };
                            feats.push(((plane * 40 + t) * 64 + sqp) as u16);
                            file += 1;
                            i += 2;
                            moved = true;
                        }
                    }
                }
                if !moved {
                    if let Some(t) = std_idx(c) {
                        if feas(&ch, i + 1, (file + 1) as usize) {
                            let sq = (rank_idx * 8 + file) as usize;
                            let white = c.is_ascii_uppercase();
                            let own = white == stm_white;
                            let sqp = if stm_white { sq } else { 63 - sq };
                            let plane = if own { 0 } else { 1 };
                            feats.push(((plane * 40 + t) * 64 + sqp) as u16);
                            file += 1;
                            i += 1;
                            moved = true;
                        }
                    }
                }
                if !moved {
                    return None;
                }
            }
            if file != 8 {
                return None;
            }
        }
        feats.sort_unstable();
        feats.dedup();
        Some(feats)
    }

    #[test]
    fn parity_converter_matches_spec_oracle_on_real_lines() {
        let path = std::env::var("NNUE_TEST_RAW").unwrap_or_else(|_| {
            "/home/hatch/workspace/pxb-selfplay-v2/combined_raw.txt".to_string()
        });
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read raw data {path}: {e}"));
        let mut checked = 0usize;
        for line in text.lines().take(200) {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() < 2 {
                continue;
            }
            let left: Vec<&str> = parts[0].trim().split_whitespace().collect();
            if left.len() < 2 {
                continue;
            }
            let (fen, stm_white) = (left[0], left[1] == "w");
            let pos = match fen_to_pos(fen, stm_white) {
                Some(p) => p,
                None => continue,
            };
            let expected = match oracle_features(fen, stm_white) {
                Some(f) => f,
                None => continue,
            };
            let got = pos_features(&pos);
            assert_eq!(got.indices, expected, "parity mismatch on line: {line}");
            // The oracle covers exactly the trained set, so any engine
            // skipped piece on these lines would be a mapping gap.
            assert_eq!(got.skipped, 0, "unexpected skipped piece on line: {line}");
            checked += 1;
        }
        assert!(checked >= 150, "too few parity lines checked: {checked}");
        eprintln!("parity: {checked} real lines agree with the SPEC oracle");
    }

    #[test]
    fn test_nnue_values_unmapped_skip_pieces() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let base_pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let with_bishopper = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::Bishopper, 0, 2, 0),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let with_ooze = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::Ooze, 0, 3, 1),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let with_warking = pos_with(
            vec![
                piece(PieceKey::WarKing, 0, 4, 0),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let ev_base = nnue_evaluate(&base_pos, 0).unwrap();
        let ev_bish = nnue_evaluate(&with_bishopper, 0).unwrap();
        let ev_ooze = nnue_evaluate(&with_ooze, 0).unwrap();
        assert_eq!(ev_bish - ev_base, 379, "Unmapped Bishopper must add +379cp");
        assert_eq!(ev_ooze - ev_base, 300, "Unmapped Ooze must add +300cp");
        assert_eq!(unmapped_material_delta_cp(&with_warking, 0), 0, "WarKing must not add 20000cp");
    }

    #[test]
    fn test_embedded_weights_equal_file_loaded_zero_cp_diff() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let p = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::Queen, 0, 3, 0),
                piece(PieceKey::Knight, 0, 6, 0),
                piece(PieceKey::King, 1, 4, 7),
                piece(PieceKey::Rook, 1, 0, 7),
                piece(PieceKey::Pilgrim, 1, 2, 7),
            ],
            0,
        );
        let embedded_eval = nnue_evaluate(&p, 0).unwrap();
        assert!(nnue_load(EMBEDDED_MODEL_BYTES));
        let loaded_eval = nnue_evaluate(&p, 0).unwrap();
        assert_eq!(embedded_eval, loaded_eval, "Embedded Unified Brain must have 0cp diff vs file-loaded NNUE");
    }

    #[test]
    fn pixie_state_calibration_probe() {
        // Calibration probe for the pixie-state delta (Marauder kills,
        // Pilgrim odo, Reactor caps, ElectroKnight quiet, Dancer encore).
        // Uses the PURE NET path (nnue_prob -> prob_to_cp, no deltas) so
        // these numbers are the net's own values, independent of any delta.
        // Prints values for calibration; asserts net blindness to state.
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let net = |pieces: Vec<Piece>| -> i32 {
            let pos = pos_with(pieces, 0);
            prob_to_cp(nnue_prob(&pos).unwrap())
        };
        let kings = |mut v: Vec<Piece>| -> Vec<Piece> {
            v.insert(0, piece(PieceKey::King, 0, 4, 0));
            v.push(piece(PieceKey::King, 1, 4, 7));
            v
        };
        let base = net(kings(vec![]));
        let mut ma = piece(PieceKey::Marauder, 0, 3, 3);
        let mut vals = Vec::new();
        for k in 0..=3u8 {
            ma.kills = k;
            vals.push(net(kings(vec![ma.clone()])));
        }
        eprintln!("CALIB net kings-only={} marauder kills0..3={:?}", base, vals);
        assert_eq!(vals[0], vals[1], "net must be kills-blind");
        assert_eq!(vals[0], vals[2]);
        assert_eq!(vals[0], vals[3]);
        let q = net(kings(vec![piece(PieceKey::Queen, 0, 3, 3)]));
        eprintln!("CALIB net queen@d4={} queen-marauder gap={}", q, q - vals[0]);
        let mut pg = piece(PieceKey::Pilgrim, 0, 3, 3);
        let mut pvals = Vec::new();
        for o in [0u8, 10, 19] {
            pg.odo = o;
            pg.state_flags = 0;
            pvals.push(net(kings(vec![pg.clone()])));
        }
        pg.odo = 20;
        pg.state_flags = 1 << 8;
        let pg_charged = net(kings(vec![pg.clone()]));
        eprintln!("CALIB net pilgrim odo0/10/19={:?} charged={} queen-pilgrim gap={}",
            pvals, pg_charged, q - pvals[0]);
        assert_eq!(pvals[0], pvals[1]);
        assert_eq!(pvals[0], pvals[2]);
        assert_eq!(pvals[0], pg_charged, "net must be odo-blind");
        let mut fr = piece(PieceKey::FissionReactor, 0, 3, 3);
        fr.caps = 0;
        let fr0 = net(kings(vec![fr.clone()]));
        fr.caps = 4;
        let fr4 = net(kings(vec![fr.clone()]));
        eprintln!("CALIB net reactor caps0={} caps4={}", fr0, fr4);
        assert_eq!(fr0, fr4, "net must be caps-blind");
        let mut ek = piece(PieceKey::ElectroKnight, 0, 3, 3);
        ek.quiet = 0;
        let ek0 = net(kings(vec![ek.clone()]));
        ek.quiet = 3;
        let ek3 = net(kings(vec![ek.clone()]));
        eprintln!("CALIB net electroknight quiet0={} quiet3={}", ek0, ek3);
        assert_eq!(ek0, ek3, "net must be quiet-blind");
        let mut da = piece(PieceKey::Dancer, 0, 3, 3);
        da.state_flags = 0;
        let da0 = net(kings(vec![da.clone()]));
        da.state_flags = 1 << 7;
        let da1 = net(kings(vec![da.clone()]));
        eprintln!("CALIB net dancer plain={} encore_pending={}", da0, da1);
        assert_eq!(da0, da1, "net must be encore-blind");
    }

    #[test]
    fn pixie_state_delta_marauder_direction() {
        // Net is kills-blind, so the nnue_evaluate diff across kills must
        // be exactly the delta's kills term (no victims -> no threat term).
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let mut ma = piece(PieceKey::Marauder, 0, 3, 3);
        ma.kills = 0;
        let p0 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                ma.clone(),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        ma.kills = 3;
        let p3 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                ma,
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let d0 = nnue_evaluate(&p0, 0).unwrap();
        let d3 = nnue_evaluate(&p3, 0).unwrap();
        assert_eq!(d3 - d0, 203, "3-kill Marauder must gain the calibrated 203 gap (v3)");
    }

    #[test]
    fn pixie_state_delta_pilgrim_direction() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let mut pg = piece(PieceKey::Pilgrim, 0, 3, 3);
        pg.odo = 0;
        pg.state_flags = 0;
        let p0 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                pg.clone(),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        pg.odo = 20;
        pg.state_flags = 1 << 8;
        let pc = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                pg,
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let d0 = nnue_evaluate(&p0, 0).unwrap();
        let dc = nnue_evaluate(&pc, 0).unwrap();
        assert_eq!(dc - d0, 240, "charged Pilgrim must gain the calibrated 240 gap (v3)");
    }

    #[test]
    fn pixie_state_delta_reactor_threat() {
        // Black Reactor on d4, White Queen diagonally adjacent on c5.
        // caps 4: +100 suicide discount (enemy reactor dying is good for
        // White) - 349 blast threat (40% of 873) = -249 vs caps 0.
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let mut fr = piece(PieceKey::FissionReactor, 1, 3, 3);
        fr.caps = 0;
        let p0 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::Queen, 0, 2, 4),
                fr.clone(),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        fr.caps = 4;
        let p4 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::Queen, 0, 2, 4),
                fr,
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let d0 = nnue_evaluate(&p0, 0).unwrap();
        let d4 = nnue_evaluate(&p4, 0).unwrap();
        assert_eq!(d4 - d0, -249, "4-cap Reactor must net -249 for White");
    }

    #[test]
    fn pixie_state_delta_ek_and_dancer_values() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let mut ek = piece(PieceKey::ElectroKnight, 0, 3, 3);
        ek.quiet = 0;
        let e0 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                ek.clone(),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        ek.quiet = 3;
        let e3 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                ek,
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        assert_eq!(
            nnue_evaluate(&e3, 0).unwrap() - nnue_evaluate(&e0, 0).unwrap(),
            150,
            "charged ElectroKnight must gain +150"
        );
        let mut da = piece(PieceKey::Dancer, 0, 3, 3);
        da.state_flags = 0;
        let d0 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                da.clone(),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        da.state_flags = 1 << 7;
        let d1 = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                da,
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        assert_eq!(
            nnue_evaluate(&d1, 0).unwrap() - nnue_evaluate(&d0, 0).unwrap(),
            225,
            "encore-pending Dancer must gain +150 value + 75 threat"
        );
    }

    #[test]
    fn pixie_state_delta_antisymmetric() {
        // White Marauder (kills 2) on d4, Black charged Pilgrim on e5.
        // me=0: +145 (kills, v3) + 69 (Marauder ray on 345 Pilgrim)
        //      - 240 (charged Pilgrim, v3) - 79 (Pilgrim ray on 396 Marauder)
        //      = -105; me=1 must be exactly +105.
        let mut ma = piece(PieceKey::Marauder, 0, 3, 3);
        ma.kills = 2;
        let mut pg = piece(PieceKey::Pilgrim, 1, 4, 4);
        pg.odo = 20;
        pg.state_flags = 1 << 8;
        let pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                ma,
                pg,
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        assert_eq!(pixie_state_nnue_delta(&pos, 0), -105);
        assert_eq!(pixie_state_nnue_delta(&pos, 1), 105);
        assert_eq!(
            pixie_state_nnue_delta(&pos, 0),
            -pixie_state_nnue_delta(&pos, 1)
        );
    }

    #[test]
    fn pixie_state_delta_bounded() {
        // Kitchen sink: every state piece maxed on both sides plus victims,
        // a limbo Djinn and an offboard Knightmare. The delta must stay
        // bounded; the evaluate ceiling (±20000) bounds the total.
        let mut ma = piece(PieceKey::Marauder, 0, 3, 3);
        ma.kills = 3;
        let mut pg = piece(PieceKey::Pilgrim, 0, 2, 2);
        pg.odo = 20;
        pg.state_flags = 1 << 8;
        let mut fr = piece(PieceKey::FissionReactor, 1, 5, 5);
        fr.caps = 4;
        let mut ek = piece(PieceKey::ElectroKnight, 1, 3, 5);
        ek.quiet = 3;
        let mut da = piece(PieceKey::Dancer, 1, 5, 3);
        da.state_flags = 1 << 7;
        let mut pos = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                ma,
                pg,
                piece(PieceKey::Queen, 0, 4, 4),
                piece(PieceKey::HordeMother, 0, 1, 1),
                piece(PieceKey::Hordeling, 0, 0, 1),
                piece(PieceKey::Hordeling, 0, 1, 0),
                piece(PieceKey::Hordeling, 0, 2, 1),
                fr,
                ek,
                da,
                piece(PieceKey::Queen, 1, 4, 6),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        pos.limbo.push(crate::pos::LimboPiece {
            id: 99,
            key: PieceKey::Djinn,
            sub_key: PieceKey::Djinn,
            player: 1,
            home_x: 4,
            home_y: 4,
        });
        pos.offboard.push(crate::pos::OffboardPiece {
            id: 98,
            key: PieceKey::Knightmare,
            sub_key: PieceKey::Knightmare,
            player: 0,
            off_since: 1,
            ox: -1,
            oy: 2,
        });
        let d0 = pixie_state_nnue_delta(&pos, 0);
        let d1 = pixie_state_nnue_delta(&pos, 1);
        assert!(d0.abs() < 2000, "delta must stay bounded, got {d0}");
        assert_eq!(d0, -d1, "delta must stay antisymmetric");
    }

    #[test]
    fn pixie_state_delta_antiviolence_protection() {
        // 2026-10-31 AV-undervalued fix: an AV shielding attacked
        // friendlies must price above a bare AV (and above a Knight
        // trade), capped so it cannot run away. Black pawn d5 hits e4;
        // knights c4/c5 hit e3/d3 (leapers, unblocked by AV on d4).
        let kings = |mut v: Vec<Piece>| -> Vec<Piece> {
            v.insert(0, piece(PieceKey::King, 0, 4, 0));
            v.push(piece(PieceKey::King, 1, 4, 7));
            v
        };
        // AV alone: no adjacent enemy, no protection.
        let alone = pos_with(
            kings(vec![piece(PieceKey::AntiViolence, 0, 3, 3)]),
            0,
        );
        assert_eq!(pixie_state_nnue_delta(&alone, 0), 0);
        // AV d4 shielding one attacked pawn (e4, hit by d5): +25.
        let one = pos_with(
            kings(vec![
                piece(PieceKey::AntiViolence, 0, 3, 3),
                piece(PieceKey::Pawn, 0, 4, 3),
                piece(PieceKey::Pawn, 1, 3, 4),
            ]),
            0,
        );
        assert_eq!(pixie_state_nnue_delta(&one, 0), 25);
        assert_eq!(pixie_state_nnue_delta(&one, 1), -25);
        // Three shielded pawns (e4/e3/d3, hit by pawn d5 and knights
        // c4/c5; leapers, so AV on d4 does not block their rays): 75 -> 60.
        let three = pos_with(
            kings(vec![
                piece(PieceKey::AntiViolence, 0, 3, 3),
                piece(PieceKey::Pawn, 0, 4, 3),
                piece(PieceKey::Pawn, 0, 4, 2),
                piece(PieceKey::Pawn, 0, 3, 2),
                piece(PieceKey::Pawn, 1, 3, 4),
                piece(PieceKey::Knight, 1, 2, 3),
                piece(PieceKey::Knight, 1, 2, 4),
            ]),
            0,
        );
        assert_eq!(pixie_state_nnue_delta(&three, 0), 60);
        assert_eq!(pixie_state_nnue_delta(&three, 1), -60);
    }

    #[test]
    fn eval_total_cp_never_flat_clamps_at_1200() {
        // 2026-10-31 mirage fix: the total must NOT flat-clamp at ±1200.
        assert_eq!(eval_total_cp(1200, 828), 2028);
        assert_eq!(eval_total_cp(1200, -873), 327);
        assert_eq!(eval_total_cp(-1200, -828), -2028);
        // Safety ceiling only, far below mate and veto magnitudes.
        assert_eq!(eval_total_cp(1200, 50000), 20000);
    }

    #[test]
    fn mirage_saturated_position_scores_above_1200_and_loses_queen_value() {
        // Stacked winning position: kings + Queen + charged Pilgrim +
        // 3-kill Marauder + charged ElectroKnight + encore-pending Dancer
        // (all White) vs bare Black king. Pre-fix this scored exactly 1200
        // (clamped) and hanging the Queen barely moved it; post-fix the
        // score is unclamped and losing the Queen costs hundreds of cp.
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let mut pg = piece(PieceKey::Pilgrim, 0, 3, 3);
        pg.odo = 20;
        pg.state_flags = 1 << 8;
        let mut ma = piece(PieceKey::Marauder, 0, 2, 2);
        ma.kills = 3;
        let mut ek = piece(PieceKey::ElectroKnight, 0, 5, 2);
        ek.quiet = 3;
        let mut da = piece(PieceKey::Dancer, 0, 1, 2);
        da.state_flags = 1 << 7;
        let full = vec![
            piece(PieceKey::King, 0, 4, 0),
            piece(PieceKey::Queen, 0, 3, 4),
            pg,
            ma,
            ek,
            da,
            piece(PieceKey::King, 1, 4, 7),
        ];
        let mut no_queen = full.clone();
        no_queen.retain(|p| !(p.key == PieceKey::Queen && p.player == 0));
        let a = nnue_evaluate(&pos_with(full, 0), 0).unwrap();
        let b = nnue_evaluate(&pos_with(no_queen, 0), 0).unwrap();
        eprintln!("MIRAGE stacked full={a} no-queen={b} drop={}", a - b);
        assert!(a > 1200, "stacked winning position must score above the old 1200 ceiling, got {a}");
        assert!(a - b > 200, "losing the Queen must cost hundreds of cp, got drop {}", a - b);
    }

    #[test]
    fn v3_recalibration_gap_pieces_match_queen_on_d4() {
        // v3 recalibration (2026-10-31): on v3's net, Queen@d4 = 526,
        // Marauder@d4 = 323, Pilgrim@d4 = 286 (kings e1/e8, probe). The
        // gap-derived delta terms must restore Queen parity exactly:
        // Marauder 3 kills +203, charged Pilgrim +240.
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let kings = |mut v: Vec<Piece>| -> Vec<Piece> {
            v.insert(0, piece(PieceKey::King, 0, 4, 0));
            v.push(piece(PieceKey::King, 1, 4, 7));
            v
        };
        let q = nnue_evaluate(&pos_with(kings(vec![piece(PieceKey::Queen, 0, 3, 3)]), 0), 0).unwrap();
        let mut ma = piece(PieceKey::Marauder, 0, 3, 3);
        ma.kills = 3;
        let m = nnue_evaluate(&pos_with(kings(vec![ma]), 0), 0).unwrap();
        let mut pg = piece(PieceKey::Pilgrim, 0, 3, 3);
        pg.odo = 20;
        pg.state_flags = 1 << 8;
        let p = nnue_evaluate(&pos_with(kings(vec![pg]), 0), 0).unwrap();
        eprintln!("RECALIB queen={q} marauder3={m} pilgrim-charged={p}");
        assert_eq!(m, q, "3-kill Marauder must match Queen on d4 after v3 recalibration");
        assert_eq!(p, q, "charged Pilgrim must match Queen on d4 after v3 recalibration");
    }

    #[test]
    fn nnue_evaluate_includes_antiviolence_aura_delta() {
        let _g = LOAD_LOCK.lock().unwrap();
        nnue_restore_embedded_for_tests();
        let p = pos_with(
            vec![
                piece(PieceKey::King, 0, 4, 0),
                piece(PieceKey::AntiViolence, 0, 3, 3),
                piece(PieceKey::Queen, 1, 4, 4),
                piece(PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let acc = crate::nnue::with_weights(|w| crate::accumulator::NnueAccumulator::refresh(w, &p)).unwrap();
        assert_eq!(nnue_evaluate(&p, 0), nnue_evaluate_with_acc(&p, &acc, 0));
    }
}
