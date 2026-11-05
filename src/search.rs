// ============================================================================
// PIXIE CHESS — TACTICAL ALPHA-BETA & QUIESCENCE SEARCH ENGINE (v2.1, WASM)
// ============================================================================
// Implements all four tactical improvements:
// 1. Deeper Search:
//    - Generation-stamped 4M-entry Transposition Table (`TTable::clear` is O(1)
//      instead of a 56 MB memset, preserving TT and Countermoves across turns).
//    - Automatic Root Aspiration Windows at `ply == 0` (`depth >= 4`).
//    - Root move ordering from the previous iteration's `last_root_scored` +
//      multi-candidate root window so `s.root_scored` has true differentiated
//      scores for `veto_score` re-ranking (fixing `ambiguous move` ties).
//    - Mid-iteration timeout recovery: if `s.stop` triggers mid-depth at `ply == 0`,
//      restores the complete `s.root_scored` list from the last completed depth
//      (fixing `v=0/0`, `v=1/0`, `v=2/0` partial-depth veto blunders).
//    - Fixes the `depth <= 0 && in_chk && ext >= 2` leaf bug that returned static
//      eval on checkmated positions instead of transitioning to `qsearch`.
// 2. Adaptive Time & Node Budget Management:
//    - Cuts WASM->JS clock FFI overhead by 8x (`nodes & 511 == 0`).
//    - Single legal root move early exit at `depth == 2`.
//    - Stable-PV soft time cutoff (52%–72% of `budget_ms`) to keep total game
//      time within a 10–12s phone budget while extending up to 100% on tactical
//      score drops or checks.
// 3. Deep Tactical Quiescence Search (`QDEPTH = 8`, `QCAP = 16`):
//    - Uses `gen_moves(pos, me, Some(0))` to cover all standard and Pixie captures,
//      promotions, `Gunslinger` standoffs, `HordeMother` spawns, `ElectroKnight`
//      AoE captures, `Shrike` double-captures, `Dancer` double-moves, `SumoRook`
//      pushes/ejections, `Bishopper` leaps, and `EpeePawn` global en-passant.
//    - Adds `BladeRunner` diagonal pass-through delayed kills (`doomed` victims)
//      and root-of-quiescence checking moves (including `HeroPawn` auto-Queen
//      checks and mating checks like `Rb4-b8+`).
//    - Full check-evasion search when `in_chk` with exact checkmate detection.
// 4. Upgraded Move Ordering (`ab_order`):
//    - TT move -> Pixie-aware MVV-LVA + SEE winning/equal captures & power
//      activations -> 2 Killers -> Counter-move heuristic -> Combined Butterfly +
//      Piece-To History with gravity bonus/malus -> losing captures.
// ============================================================================

use crate::pos::*;
use crate::types::*;
use crate::apply::*;
use crate::apply_full::*;
use crate::journal::*;
use crate::eval::*;
use crate::accumulator::NnueAccumulator;
use crate::nnue::{nnue_evaluate_with_acc, with_weights, DISPATCHING_EVALUATOR, NNUE_EVALUATOR};
use crate::attacks::*;
use crate::movegen::gen_moves;
use crate::movegen::piece_attacks;

pub const QDEPTH: i32 = 8;
pub const QCAP:   usize = 16;
/// Max nodes in one qsearch tree (2026-10-31 depth-collapse fix). Normal
/// qsearch trees are hundreds to low thousands of nodes; only pathological
/// Pixie explosions (HordeMother spawns x check evasions) approach this.
/// See the envelope check in qsearch_acc.
pub const QSEARCH_MAX_NODES: u64 = 65_536;
pub const MATE:   i32 = 1_000_000;
pub const ASP_DELTA: i32 = 40;

pub const TT_BITS: u32 = 22;
pub const TT_SIZE: usize = 1 << TT_BITS;
pub const TT_MASK: u32 = (TT_SIZE - 1) as u32;

// ── Search tuning constants (integer centipawn scale) ─────────────────────
const RAZOR_MARGIN: i32 = 300;
const RFP_MARGIN_PER_DEPTH: i32 = 135;
const FUTILITY_BASE: i32 = 85;
const FUTILITY_PER_DEPTH: i32 = 125;
const DELTA_MARGIN: i32 = 220;
const HIST_MAX: i32 = 32_768;
const MAX_NODES_HARD: u64 = u64::MAX; // GENERATOR-ONLY probe: no node cap (live engine keeps 1.5M)
const PIECE_HIST_SIZE: usize = 68 * 64;

pub struct TTable {
    key:    Vec<u32>,
    depth:  Vec<i8>,
    flag:   Vec<i8>,
    score:  Vec<i32>,
    mvfrom: Vec<i8>,
    mvto:   Vec<i8>,
    occ:    Vec<u8>,
    gen:    u8,
    // Internal search heuristics & countermove tables (private to TTable so Search struct fields stay unchanged)
    cm_from: Vec<u8>,
    cm_to:   Vec<u8>,
    piece_hist: Vec<i32>,
    prev_from: [u8; 64],
    prev_to:   [u8; 64],
    prev_key:  [u8; 64],
    // Root state across iterative deepening depths within a move
    pub last_root_scored: Vec<(Move, i32)>,
    last_best_root: Option<Move>,
    last_best_score: i32,
    stable_root_count: u8,
    pub score_drop: i32,
}

impl TTable {
    pub fn new() -> Self {
        Self {
            key:    vec![0; TT_SIZE],
            depth:  vec![0; TT_SIZE],
            flag:   vec![0; TT_SIZE],
            score:  vec![0; TT_SIZE],
            mvfrom: vec![-128; TT_SIZE],
            mvto:   vec![0; TT_SIZE],
            occ:    vec![0; TT_SIZE],
            gen:    1,
            cm_from: vec![SQ_NONE; 4096],
            cm_to:   vec![SQ_NONE; 4096],
            piece_hist: vec![0; PIECE_HIST_SIZE],
            prev_from: [SQ_NONE; 64],
            prev_to:   [SQ_NONE; 64],
            prev_key:  [0; 64],
            last_root_scored: Vec::with_capacity(64),
            last_best_root: None,
            last_best_score: 0,
            stable_root_count: 0,
            score_drop: 0,
        }
    }

    /// Rank 5 Fix: Generation-stamped per-think reset that strictly invalidates
    /// prior-turn TT entries (`probe` and `probe_move` require `self.occ[i] == self.gen`)
    /// and clears countermove and piece-history tables so warm instances are
    /// 100% deterministic with cold instances.
    pub fn clear(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.gen = 1;
            self.occ.fill(0);
        }
        self.cm_from.fill(SQ_NONE);
        self.cm_to.fill(SQ_NONE);
        self.piece_hist.fill(0);
        self.prev_from = [SQ_NONE; 64];
        self.prev_to = [SQ_NONE; 64];
        self.prev_key = [0; 64];
        self.last_root_scored.clear();
        self.last_best_root = None;
        self.last_best_score = 0;
        self.stable_root_count = 0;
        self.score_drop = 0;
    }

    /// Full cold reset of all TT and heuristic buffers.
    pub fn reset_cold(&mut self) {
        self.key.fill(0);
        self.depth.fill(0);
        self.flag.fill(0);
        self.score.fill(0);
        self.mvfrom.fill(-128);
        self.mvto.fill(0);
        self.occ.fill(0);
        self.gen = 1;
        self.cm_from.fill(SQ_NONE);
        self.cm_to.fill(SQ_NONE);
        self.piece_hist.fill(0);
        self.prev_from = [SQ_NONE; 64];
        self.prev_to = [SQ_NONE; 64];
        self.prev_key = [0; 64];
        self.last_root_scored.clear();
        self.last_best_root = None;
        self.last_best_score = 0;
        self.stable_root_count = 0;
        self.score_drop = 0;
    }

    #[inline]
    pub fn probe(&self, h: u32, depth: i32, alpha: i32, beta: i32) -> Option<i32> {
        let i = (h & TT_MASK) as usize;
        if self.occ[i] != self.gen || self.key[i] != h {
            return None;
        }
        if (self.depth[i] as i32) < depth {
            return None;
        }
        let f = self.flag[i];
        let sc = self.score[i];
        if f == 0 {
            return Some(sc);
        }
        if f == 1 && sc >= beta {
            return Some(sc);
        }
        if f == 2 && sc <= alpha {
            return Some(sc);
        }
        None
    }

    /// Lever-2 experiment: exact-only TT probe for PV nodes. Non-PV nodes
    /// already use `probe` (bounds are valid cutoffs there); PV nodes
    /// previously skipped the probe entirely, re-searching positions whose
    /// exact score was already in the table. An exact entry (flag == 0,
    /// searched to at least this depth) is window-independent, and mate
    /// scores are never stored (`store` refuses |score| > 900_000), so no
    /// ply adjustment is needed. Bounds are NOT returned here: a PV
    /// fail-high/fail-low must come from the search itself.
    #[inline]
    pub fn probe_exact(&self, h: u32, depth: i32) -> Option<i32> {
        let i = (h & TT_MASK) as usize;
        if self.occ[i] != self.gen || self.key[i] != h {
            return None;
        }
        if (self.depth[i] as i32) < depth {
            return None;
        }
        if self.flag[i] == 0 {
            Some(self.score[i])
        } else {
            None
        }
    }

    #[inline]
    pub fn probe_move(&self, h: u32) -> Option<(u8, u8)> {
        let i = (h & TT_MASK) as usize;
        if self.occ[i] != self.gen || self.key[i] != h || self.mvfrom[i] == -128 {
            return None;
        }
        Some((self.mvfrom[i] as u8, self.mvto[i] as u8))
    }

    pub fn store(&mut self, h: u32, depth: i32, score: i32, flag: i8, best: Option<&Move>) {
        if score.abs() > 900_000 {
            return;
        }
        let i = (h & TT_MASK) as usize;
        let same_gen = self.occ[i] == self.gen;
        if self.occ[i] != 0 && same_gen {
            let old_d = self.depth[i] as i32;
            let exact_bonus = if flag == 0 { 2 } else { 0 };
            let old_exact_bonus = if self.flag[i] == 0 { 1 } else { 0 };
            if self.key[i] == h {
                if old_d + old_exact_bonus > depth + exact_bonus {
                    if self.mvfrom[i] == -128 {
                        if let Some(mv) = best {
                            self.mvfrom[i] = mv.from as i8;
                            self.mvto[i] = mv.to as i8;
                        }
                    }
                    return;
                }
            } else if old_d + old_exact_bonus > depth + exact_bonus + 2 {
                return;
            }
        }
        let preserve_old_move = self.occ[i] != 0 && self.key[i] == h && best.is_none();
        self.key[i] = h;
        self.depth[i] = depth.clamp(-127, 127) as i8;
        self.flag[i] = flag;
        self.score[i] = score;
        self.occ[i] = self.gen;
        if let Some(mv) = best {
            self.mvfrom[i] = mv.from as i8;
            self.mvto[i] = mv.to as i8;
        } else if !preserve_old_move {
            self.mvfrom[i] = -128;
        }
    }
}

pub struct Search {
    pub nodes: u64,
    pub max_nodes: u64,
    pub fast_qsearch: bool,
    pub stop: bool,
    pub t0_ms: u64,
    pub budget_ms: u64,
    pub killers: [[Option<Move>; 2]; 64],
    pub hist: [i32; 4096],
    pub no_pvs: bool,
    pub no_tt: bool,
    pub null_ply: i32,
    pub journal: Journal,
    pub tt: TTable,
    pub best_root: Option<Move>,
    pub best_root_score: i32,
    /// Root moves with search scores from the last completed depth,
    /// used for veto re-ranking. Cleared at the start of each depth.
    pub root_scored: Vec<(Move, i32)>,
    /// Veto observability: how many root moves were veto-checked,
    /// and how many were rejected (below VETO_THRESHOLD).
    pub veto_checks: u64,
    pub veto_rejects: u64,
    /// Deepest fully completed iterative-deepening depth.
    pub completed_depth: i32,
    /// The single evaluation entry point for this search (NNUE-ready seam).
    /// Leaf scoring in qsearch and rel_eval goes through this evaluator, never
    /// through a second scorer. Today it is always the hand-written evaluator;
    /// a future NNUE implementation plugs in here without touching think,
    /// search, or the wrapper.
    pub evaluator: &'static dyn Evaluator,
    /// Rank 1: Persistent cross-turn game history of 64-bit Zobrist hashes.
    /// Automatically resets when a new game is detected (`move_no <= 2` or piece count jumps).
    pub game_history: Vec<u64>,
    pub last_game_piece_count: usize,
    pub last_game_move_no: u32,
    pub last_root_move: [Option<(u8, u8)>; 2],
    pub rep_hits: u64,
    /// Rank 2: Ply-indexed incremental NNUE accumulator stack (`acc_stack[0..128]`).
    /// Eliminates `pos_features(pos)` heap allocations and full sparse matrix-vector
    /// multiplies on journaled search/qsearch plies while preserving 0cp bit-exactness.
    pub acc_stack: Vec<NnueAccumulator>,
    pub use_acc_stack: bool,
    pub disable_acc_stack: bool,
    /// Cluster-worker subset search: this Search is searching a 1/5
    /// round-robin slice of the root moves (coordinator local group or a
    /// remote worker's /think with server_pairs), not the full root list.
    /// Narrows the full-window root-move count (see root_full_window).
    /// Set per-request by the service; never reset by reset_for_think
    /// (the service sets it explicitly on every think path).
    pub cluster_worker: bool,
    /// Root move filter for cluster worker slices (`ply == 0`).
    pub root_filter: Option<Vec<Move>>,
}

impl Search {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            max_nodes: MAX_NODES_HARD,
            fast_qsearch: false,
            stop: false,
            t0_ms: 0,
            budget_ms: 0,
            killers: [[None; 2]; 64],
            hist: [0; 4096],
            no_pvs: false,
            no_tt: false,
            null_ply: -1,
            journal: Journal::default(),
            tt: TTable::new(),
            best_root: None,
            best_root_score: 0,
            root_scored: Vec::new(),
            veto_checks: 0,
            veto_rejects: 0,
            completed_depth: 0,
            evaluator: &DISPATCHING_EVALUATOR,
            game_history: Vec::with_capacity(256),
            last_game_piece_count: 0,
            last_game_move_no: 0,
            last_root_move: [None, None],
            rep_hits: 0,
            acc_stack: vec![NnueAccumulator::new(); 128],
            use_acc_stack: false,
            disable_acc_stack: false,
            cluster_worker: false,
            root_filter: None,
        }
    }

    /// Rank 5: Per-think reset of search state. Generation-bumps the TT (`probe`
    /// and `probe_move` strictly check `occ[i] == gen`), zeroes killers, butterfly
    /// history, piece-to history, and countermoves so warm instances produce
    /// identical trees to freshly constructed `Search::new()` instances.
    pub fn reset_for_think(&mut self, budget_ms: u64) {
        self.t0_ms = crate::now_ms();
        self.budget_ms = budget_ms;
        self.nodes = 0;
        self.stop = false;
        self.best_root = None;
        self.best_root_score = 0;
        self.no_tt = false;
        self.no_pvs = false;
        self.null_ply = -1;
        self.killers = [[None; 2]; 64];
        self.hist.fill(0);
        self.tt.clear();
        self.journal.v.clear();
        self.root_scored.clear();
        self.veto_checks = 0;
        self.veto_rejects = 0;
        self.completed_depth = 0;
        self.rep_hits = 0;
    }

    /// Rank 5: Complete cold reset (clears `game_history` as well as all TT/history buffers).
    pub fn reset_full_cold(&mut self) {
        self.reset_for_think(0);
        self.tt.reset_cold();
        self.game_history.clear();
        self.last_game_piece_count = 0;
        self.last_game_move_no = 0;
        self.last_root_move = [None, None];
    }

    /// Rank 2: Initialize the root accumulator at `acc_stack[0]` for `pos`.
    #[inline]
    pub fn init_root_accumulator(&mut self, pos: &Pos) {
        if self.disable_acc_stack {
            self.use_acc_stack = false;
            return;
        }
        let is_nnue = core::ptr::eq(
            self.evaluator as *const dyn Evaluator as *const (),
            &NNUE_EVALUATOR as *const dyn Evaluator as *const (),
        ) || (core::ptr::eq(
            self.evaluator as *const dyn Evaluator as *const (),
            &DISPATCHING_EVALUATOR as *const dyn Evaluator as *const (),
        ) && crate::nnue::nnue_is_loaded());
        if is_nnue {
            let ok = with_weights(|w| {
                self.acc_stack[0].refresh_in_place(w, pos);
            })
            .is_some();
            self.use_acc_stack = ok;
        } else {
            self.use_acc_stack = false;
        }
    }

    /// Rank 2: Evaluate `pos` at `acc_idx` using the incremental accumulator when active,
    /// falling back to `self.evaluator.evaluate(pos, me)`.
    #[inline]
    pub fn eval_at(&self, pos: &Pos, me: u8, acc_idx: usize) -> i32 {
        if self.use_acc_stack && acc_idx < self.acc_stack.len() {
            if let Some(cp) = nnue_evaluate_with_acc(pos, &self.acc_stack[acc_idx], me) {
                return cp;
            }
        }
        self.evaluator.evaluate(pos, me)
    }

    #[inline]
    pub fn rel_eval_at(&self, pos: &Pos, stm: u8, me: u8, acc_idx: usize) -> i32 {
        let sc = self.eval_at(pos, me, acc_idx);
        if stm == me { sc } else { -sc }
    }

    #[inline]
    fn check_time(&mut self) {
        if self.nodes >= self.max_nodes {
            self.stop = true;
            return;
        }
        if (self.nodes & 511) == 0 && crate::now_ms().saturating_sub(self.t0_ms) >= self.budget_ms {
            self.stop = true;
        }
    }
}

#[inline(always)]
fn hist_index(from: u8, to: u8) -> usize {
    (((from as usize) & 63) << 6) | ((to as usize) & 63)
}

#[inline(always)]
fn piece_hist_index(pk: u8, to: u8) -> usize {
    ((pk as usize).min(67) << 6) | ((to as usize) & 63)
}

#[inline(always)]
fn update_history_entry(entry: &mut i32, bonus: i32) {
    let clamped = bonus.clamp(-2000, 2000);
    *entry += clamped - (*entry * clamped.abs()) / HIST_MAX;
}

#[inline(always)]
fn promo_bonus_of(promo: u8) -> i32 {
    if promo == PieceKey::Queen as u8 {
        800
    } else if promo == PieceKey::Rook as u8 {
        450
    } else if promo == PieceKey::Bishop as u8 || promo == PieceKey::Knight as u8 {
        280
    } else {
        750
    }
}

#[inline(always)]
fn same_move(a: &Move, b: &Move) -> bool {
    a.from == b.from
        && a.to == b.to
        && a.flags == b.flags
        && a.promo == b.promo
        && a.special_a == b.special_a
}

/// Computes the immediate material swing (in centipawns) of a tactical move `mv`,
/// including all Pixie power activations (`ElectroKnight` AoE, `HordeMother` spawn,
/// `Gunslinger` standoff, `Shrike` double-capture, `Dancer` double-move,
/// `SumoRook` shove/ejection, `BladeRunner` pass-through doom, `Banker` GoldenPawn
/// conversion, and `GoldenPawn` instant-win promotion).
#[inline]
fn tactical_move_gain(pos: &Pos, mv: &Move, me: u8, opp: u8) -> i32 {
    let mut gain = 0i32;

    if mv.flags & MF_GUN_STANDOFF != 0 {
        let vic_v = if mv.special_a < 64 {
            slot_of(pos, mv.special_a)
                .map(|si| piece_value(&pos.pieces[si]))
                .unwrap_or(0)
        } else {
            0
        };
        let gun_v = if mv.from < 64 {
            slot_of(pos, mv.from)
                .map(|si| piece_value(&pos.pieces[si]))
                .unwrap_or(325)
        } else {
            325
        };
        return vic_v - gun_v;
    }

    if mv.flags & MF_IS_PUSH != 0 {
        if mv.to < 64 {
            if let Some(vs) = slot_of(pos, mv.to) {
                let v = &pos.pieces[vs];
                if v.alive && v.player == opp {
                    let nx = sq_x(mv.to) as i8 + mv.push_dx;
                    let ny = sq_y(mv.to) as i8 + mv.push_dy;
                    if !on_board(nx, ny) {
                        return if is_king_piece(v) { 10_000 } else { piece_value(v) };
                    }
                    let nnx = nx + mv.push_dx;
                    let nny = ny + mv.push_dy;
                    if !on_board(nnx, nny) {
                        if let Some(b_slot) = slot_of(pos, sq(nx, ny)) {
                            let bp = &pos.pieces[b_slot];
                            if bp.alive {
                                if bp.player == opp {
                                    return if is_king_piece(bp) { 10_000 } else { piece_value(bp) };
                                } else {
                                    return -piece_value(bp);
                                }
                            }
                        }
                    }
                    return 60;
                }
            }
        }
        return 40;
    }

    if mv.flags & MF_EP != 0 {
        gain += if mv.ep_victim < 64 {
            slot_of(pos, mv.ep_victim)
                .map(|si| piece_value(&pos.pieces[si]))
                .unwrap_or(100)
        } else {
            100
        };
    } else if mv.flags & MF_CAPTURE != 0 && mv.to < 64 {
        if let Some(vs) = slot_of(pos, mv.to) {
            let v = &pos.pieces[vs];
            if v.alive && v.player == opp {
                gain += piece_value(v);
                // Killing a HordeMother or Hordeling wipes the entire connected horde
                if v.horde_id != 0 {
                    let hid = v.horde_id;
                    let vid = v.id;
                    for p in &pos.pieces {
                        if p.alive && p.horde_id == hid && p.id != vid {
                            gain += piece_value(p);
                        }
                    }
                }
            }
        }
    }

    // Shrike first-move 2-square pounce can also kill an enemy piece on the intermediate square
    if mv.flags & MF_SHRIKE_FIRST != 0 && mv.from < 64 && mv.to < 64 {
        let d_y: i8 = if me == 0 { 1 } else { -1 };
        if (sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs() == 2 {
            let mid_sq = sq(sq_x(mv.from) as i8, sq_y(mv.from) as i8 + d_y);
            if let Some(ms) = slot_of(pos, mid_sq) {
                let mp = &pos.pieces[ms];
                if mp.alive && mp.player == opp && !is_king_piece(mp) {
                    gain += piece_value(mp);
                }
            }
        }
    }

    // ElectroKnight AoE secondary target on `special_a`
    if mv.flags & MF_ELECTRO != 0 && mv.special_a < 64 {
        if let Some(es) = slot_of(pos, mv.special_a) {
            let ep = &pos.pieces[es];
            if ep.alive && ep.player == opp && !is_king_piece(ep) {
                gain += piece_value(ep);
            }
        }
    }

    // Dancer encore double-move: leg 1 is `mv.to`, leg 2 is `mv.special_a`
    if mv.flags & MF_DANCER_DOUBLE != 0 && mv.special_a < 64 {
        if let Some(s2) = slot_of(pos, mv.special_a) {
            let v2 = &pos.pieces[s2];
            if v2.alive && v2.player == opp && !is_king_piece(v2) {
                gain += piece_value(v2);
            }
        }
    }

    // HordeMother capture spawns a Hordeling on `special_a`
    if mv.flags & MF_HORDE_SPAWN != 0 {
        gain += 95;
    }

    // Promotions (including GoldenPawn instant win)
    if mv.flags & MF_PROMO != 0 {
        let is_golden = mv.from < 64
            && slot_of(pos, mv.from).map_or(false, |si| {
                let p = &pos.pieces[si];
                p.key == PieceKey::GoldenPawn || p.sub_key == PieceKey::GoldenPawn
            });
        if is_golden && !enemy_aristocrat(pos, opp) {
            gain += 8_000;
        } else {
            gain += promo_bonus_of(mv.promo);
        }
    }

    // Banker capturing an enemy pawn promotes our most advanced pawn into a GoldenPawn
    if mv.flags & MF_CAPTURE != 0 && mv.from < 64 && mv.to < 64 {
        if let Some(ms) = slot_of(pos, mv.from) {
            let mp = &pos.pieces[ms];
            if mp.key == PieceKey::Banker || mp.sub_key == PieceKey::Banker {
                if let Some(vs) = slot_of(pos, mv.to) {
                    if is_piece_pawn_like(&pos.pieces[vs]) {
                        gain += 90;
                    }
                }
            }
        }
    }

    // State-power gains (2026-10-31, derived from pixie_state_nnue_delta
    // at the v3 scale, NOT from Argon's uncalibrated numbers): a move that
    // changes a piece's state value is worth more than its victim value
    // alone. Ordering-only: feeds move ordering and qsearch pruning, never
    // the eval itself.
    if mv.from < 64 && mv.to < 64 {
        if let Some(ms) = slot_of(pos, mv.from) {
            let mp = &pos.pieces[ms];
            let ek = effective_key(mp);
            // 1. Marauder range-expansion on capture (range/7 x 203 gap):
            // 1st kill +87, 2nd +58, 3rd +58, further kills +0 (capped).
            if (mv.flags & MF_CAPTURE) != 0 && ek == PieceKey::Marauder {
                gain += match mp.kills {
                    0 => 87,
                    1 => 58,
                    2 => 58,
                    _ => 0,
                };
            }
            // 2. Pilgrim odometer charge completion (any move type): the
            // exact delta difference, 240 minus the pre-move ramp + pen.
            // Can be negative at high odo (the pen already priced it as
            // nearly charged); that is correct, not a bug.
            if ek == PieceKey::Pilgrim && !mp.pilgrim_charged() && mp.odo < 20 {
                let dist = (sq_x(mv.to) as i8 - sq_x(mv.from) as i8)
                    .abs()
                    .max((sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs()) as i32;
                if mp.odo as i32 + dist >= 20 {
                    let o = mp.odo as i32;
                    let pen = if o >= 15 { 75 } else if o >= 10 { 30 } else { 0 };
                    gain += 240 - ((o * o * 240) / 400 + pen);
                }
            }
            // 3. ElectroKnight 3rd quiet move charges it (+150 vs +50).
            if ek == PieceKey::ElectroKnight
                && !mp.echarged()
                && mp.quiet == 2
                && (mv.flags & MF_CAPTURE) == 0
            {
                gain += 100;
            }
            // 4. FissionReactor 5th capture self-destructs: it dies and
            // kills all non-King pieces diagonally adjacent to mv.to.
            // (The victim on mv.to itself is already counted above.)
            if (mv.flags & MF_CAPTURE) != 0 && ek == PieceKey::FissionReactor && mp.caps >= 4 {
                let mut blast_net = -piece_value(mp);
                let tx = sq_x(mv.to) as i8;
                let ty = sq_y(mv.to) as i8;
                for (dx, dy) in DIRS_BISHOP {
                    let nx = tx + dx;
                    let ny = ty + dy;
                    if !on_board(nx, ny) {
                        continue;
                    }
                    let nsq = sq(nx, ny);
                    if nsq == mv.from {
                        continue;
                    }
                    if let Some(bs) = slot_of(pos, nsq) {
                        let bp = &pos.pieces[bs];
                        if bp.alive && !is_king_piece(bp) {
                            if bp.player == opp {
                                blast_net += piece_value(bp);
                            } else {
                                blast_net -= piece_value(bp);
                            }
                        }
                    }
                }
                gain += blast_net;
            }
        }
    }

    gain
}

/// Checks if a `BladeRunner` move from `mv.from` to `mv.to` slides through at least
/// one enemy piece (marking it `doomed`), returning the total value of doomed victims.
#[inline]
fn blade_runner_doomed_gain(pos: &Pos, mv: &Move, me: u8, opp: u8) -> i32 {
    if mv.from >= 64 || mv.to >= 64 {
        return 0;
    }
    let ms = match slot_of(pos, mv.from) {
        Some(s) => s,
        None => return 0,
    };
    let mp = &pos.pieces[ms];
    if mp.key != PieceKey::BladeRunner && mp.sub_key != PieceKey::BladeRunner {
        return 0;
    }
    let dx = sq_x(mv.to) as i8 - sq_x(mv.from) as i8;
    let dy = sq_y(mv.to) as i8 - sq_y(mv.from) as i8;
    if dx == 0 || dy == 0 || dx.abs() != dy.abs() {
        return 0;
    }
    let sx = dx.signum();
    let sy = dy.signum();
    let mut cx = sq_x(mv.from) as i8 + sx;
    let mut cy = sq_y(mv.from) as i8 + sy;
    let mut doomed_val = 0i32;
    while cx != sq_x(mv.to) as i8 || cy != sq_y(mv.to) as i8 {
        if let Some(vs) = slot_of(pos, sq(cx, cy)) {
            let v = &pos.pieces[vs];
            if v.alive && v.player == opp && !is_king_piece(v) {
                let already = pos.doomed.iter().any(|d| d.id == v.id);
                if !already {
                    doomed_val += piece_value(v);
                }
            }
        }
        cx += sx;
        cy += sy;
    }
    let _ = me;
    doomed_val
}

#[inline]
/// True when a quiet move completes a state charge: a Pilgrim crossing
/// odo >= 20 (permanent queen rays) or an ElectroKnight's 3rd quiet move
/// (charged). Such moves are exempt from LMR: their value is a permanent
/// power-up, and reducing them buries the tactic at move 15+. Captures are
/// already LMR-exempt (tactical), so this only matters for quiet moves.
/// Must be called pre-apply: after the move, mv.from is empty and the
/// piece's state has already advanced.
fn completes_state_charge(pos: &Pos, mv: &Move) -> bool {
    if mv.from >= 64 || mv.to >= 64 || (mv.flags & MF_CAPTURE) != 0 {
        return false;
    }
    let ms = match slot_of(pos, mv.from) {
        Some(s) => s,
        None => return false,
    };
    let mp = &pos.pieces[ms];
    match effective_key(mp) {
        PieceKey::Pilgrim => {
            if mp.pilgrim_charged() || mp.odo >= 20 {
                return false;
            }
            let dist = (sq_x(mv.to) as i8 - sq_x(mv.from) as i8)
                .abs()
                .max((sq_y(mv.to) as i8 - sq_y(mv.from) as i8).abs()) as i32;
            mp.odo as i32 + dist >= 20
        }
        PieceKey::ElectroKnight => !mp.echarged() && mp.quiet == 2,
        _ => false,
    }
}

fn lmr_reduction(depth: i32, legal: i32, is_pv: bool, is_killer_or_cm: bool, hist_sc: i32) -> i32 {
    // Lever-2 experiment: a deeper tier for very late quiet moves. A move
    // searched 20+ deep in the move list at depth >= 5 is overwhelmingly a
    // fail-low candidate; reducing it by 3 instead of 2 saves its subtree,
    // and the existing re-search on sc_r > alpha restores full depth for
    // any reduced move that actually looks good, bounding strength risk.
    let mut r = if depth >= 7 && legal > 14 {
        3
    } else if depth >= 5 && legal > 20 {
        3
    } else if depth >= 5 && legal > 7 {
        2
    } else {
        1
    };
    if is_pv {
        r -= 1;
    }
    if is_killer_or_cm {
        r -= 1;
    }
    if hist_sc > 3_500 {
        r -= 1;
    } else if hist_sc < -2_000 && !is_pv && legal > 6 {
        r += 1;
    }
    r.clamp(0, (depth - 2).max(0))
}

/// Counts how many times hash `h` appears in `pos.hist_hashes`.
/// Note: `pos.hist_hashes` already includes the current position's hash
/// (pushed by `pxb_pos_create`, `make_journaled`, or `apply_full`), plus any
/// cross-turn `game_history` hashes seeded at the start of `run_ranked_root_search`.
/// Therefore `rep_count >= 2` means the current position has occurred before
/// in the game or current search branch (2-fold repetition).
pub fn rep_count(pos: &Pos, stm: u8, pre: u32) -> usize {
    let mut n = 0;
    let h = if pre != 0 { pre } else { rep_hash(pos, stm) };
    for &x in pos.hist_hashes.iter().rev() {
        if x == h {
            n += 1;
        }
    }
    n
}

pub fn rel_eval(pos: &Pos, stm: u8, me: u8, ev: &dyn Evaluator) -> i32 {
    let s = ev.evaluate(pos, me);
    if stm == me { s } else { -s }
}

/// Snapshot pre-move information needed for an `O(NNUE_HIDDEN_SIZE)` incremental
/// accumulator update before `make_journaled` mutates `pos`.
#[inline]
fn snapshot_journal_delta(
    pos: &Pos,
    mv: &Move,
    stm: u8,
    ntm: u8,
) -> Option<(usize, PieceKey, PieceKey, u8, u8, Option<(PieceKey, u8, u8)>)> {
    if mv.from >= 64 || mv.to >= 64 {
        return None;
    }
    let ms = slot_of(pos, mv.from)?;
    let mover = &pos.pieces[ms];
    let old_key = effective_key(mover);
    let new_key = if (mv.flags & MF_PROMO) != 0 && mv.promo != 0 {
        PieceKey::from_u16(mv.promo as u16)
    } else {
        old_key
    };
    let cap = if (mv.flags & MF_EP) != 0 {
        if mv.ep_victim < 64 {
            slot_of(pos, mv.ep_victim).map(|cs| (effective_key(&pos.pieces[cs]), ntm, mv.ep_victim))
        } else {
            None
        }
    } else {
        slot_of(pos, mv.to).map(|cs| (effective_key(&pos.pieces[cs]), pos.pieces[cs].player, mv.to))
    };
    let _ = stm;
    Some((ms, old_key, new_key, mv.from, mv.to, cap))
}

#[inline]
fn push_acc_journaled(
    s: &mut Search,
    parent_idx: usize,
    child_idx: usize,
    stm: u8,
    delta: Option<(usize, PieceKey, PieceKey, u8, u8, Option<(PieceKey, u8, u8)>)>,
    pos_after: &Pos,
) {
    if !s.use_acc_stack || child_idx >= s.acc_stack.len() {
        return;
    }
    if let Some((ms, old_k, fallback_new_k, from_sq, to_sq, cap)) = delta {
        let actual_new_k = pos_after
            .pieces
            .get(ms)
            .map(|p| effective_key(p))
            .unwrap_or(fallback_new_k);
        let mut next_acc = s.acc_stack[parent_idx];
        with_weights(|w| {
            next_acc.update_for_journaled_move(w, old_k, actual_new_k, stm, from_sq, to_sq, cap);
        });
        s.acc_stack[child_idx] = next_acc;
    } else {
        with_weights(|w| {
            s.acc_stack[child_idx].refresh_in_place(w, pos_after);
        });
    }
}

#[inline]
fn push_acc_refresh(s: &mut Search, child_idx: usize, pos_after: &Pos) {
    if !s.use_acc_stack || child_idx >= s.acc_stack.len() {
        return;
    }
    with_weights(|w| {
        s.acc_stack[child_idx].refresh_in_place(w, pos_after);
    });
}

// ── 3. QUIESCENCE SEARCH (CAPTURES, CHECKS, EVASIONS & PIXIE POWERS) ────────
pub fn qsearch(
    s: &mut Search,
    pos: &mut Pos,
    me: u8,
    opp: u8,
    mut alpha: i32,
    beta: i32,
    depth: i32,
    qext: i32,
    pre_in_chk: Option<bool>,
) -> i32 {
    qsearch_acc(s, pos, me, opp, alpha, beta, depth, qext, pre_in_chk, 0, s.nodes)
}

pub fn qsearch_acc(
    s: &mut Search,
    pos: &mut Pos,
    me: u8,
    opp: u8,
    mut alpha: i32,
    beta: i32,
    depth: i32,
    qext: i32,
    pre_in_chk: Option<bool>,
    acc_idx: usize,
    qroot_nodes: u64,
) -> i32 {
    s.nodes += 1;
    s.check_time();
    if s.stop {
        return alpha;
    }

    let qply_i = (QDEPTH - depth).clamp(0, 64) + 16;

    if let Some(kt) = pos.king_thrown {
        return if kt == me { MATE - qply_i } else { -(MATE - qply_i) };
    }
    if let Some(gw) = pos.gold_win {
        return if gw == me { MATE - qply_i } else { -(MATE - qply_i) };
    }

    let occ_local = occ(pos);
    let in_chk = pre_in_chk.unwrap_or_else(|| king_attacked(pos, me, opp, Some(&occ_local)));

    let mut best: i32;
    let mut stand_pat = 0i32;

    if in_chk {
        // When in check, we cannot stand pat — we must search evasions to detect checkmate.
        // Only bail out at extreme depth ceiling (-6) to prevent infinite check loops.
        if depth <= -6 {
            return s.eval_at(pos, me, acc_idx);
        }
        best = -(MATE - qply_i);
    } else {
        stand_pat = s.eval_at(pos, me, acc_idx);
        if depth <= 0 {
            return stand_pat;
        }
        if stand_pat >= beta {
            return stand_pat; // fail-soft
        }
        best = stand_pat;
        if stand_pat > alpha {
            alpha = stand_pat;
        }
        // Node envelope (2026-10-31 depth-collapse fix): one qsearch tree
        // is capped at QSEARCH_MAX_NODES. On breach, fail low with
        // stand_pat (never invent a score, never set stop), preserving the
        // main tree's budget. In-check positions are exempt: mate
        // detection must stay exact (their non-check descendants check).
        if s.nodes.saturating_sub(qroot_nodes) > QSEARCH_MAX_NODES {
            return stand_pat.max(alpha);
        }
    }

    let mut candidates: Vec<(Move, i32)> = Vec::with_capacity(24);

    if in_chk {
        // Generate all pseudo-legal moves and score them; legality is tested once in the loop below
        let gm = gen_moves(pos, me, None);
        for m in gm {
            let is_cap = m.flags & MF_CAPTURE != 0;
            let is_promo = m.flags & MF_PROMO != 0;
            let atk_v = if m.from < 64 {
                slot_of(pos, m.from)
                    .map(|si| piece_value(&pos.pieces[si]))
                    .unwrap_or(100)
            } else {
                300
            };
            let sc = if is_cap || is_promo {
                let gain = tactical_move_gain(pos, &m, me, opp);
                100_000 + gain * 16 - atk_v
            } else if m.from < 64 && m.to < 64 {
                s.hist[hist_index(m.from, m.to)]
            } else {
                0
            };
            candidates.push((m, sc));
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
    } else {
        // 1. Stage-0 tactical moves (all standard & Pixie captures, promotions, Gunslinger
        //    standoffs, HordeMother spawns, ElectroKnight AoE, Shrike double-captures,
        //    SumoRook pushes, Dancer double-moves, Bishopper leap captures, EpeePawn EP)
        let stage0 = gen_moves(pos, me, Some(0));
        let mut horde_seen = [0u64; 64];
        for m in stage0 {
            // Skip non-tactical quiet special abilities that stage 0 also emits
            if (m.flags & (MF_ROCKET_BLAST | MF_DJINN_DISSIPATE | MF_KNIGHTMARE_OFF | MF_BISHOPPER_CROUCH)) != 0 {
                continue;
            }
            if (m.flags & MF_BISHOPPER_LEAP) != 0 && (m.flags & (MF_CAPTURE | MF_PROMO)) == 0 {
                continue;
            }
            if (m.flags & MF_DANCER_DOUBLE) != 0 && (m.flags & MF_CAPTURE) == 0 {
                let leg2_cap = m.special_a < 64 && slot_of(pos, m.special_a).is_some();
                if !leg2_cap {
                    continue;
                }
            }

            // HordeMother spawn dedup (qsearch only): one (from, to)
            // capture emits up to 48 moves differing only in the spawn
            // square (all free ranks 2-7 squares, Sandbox 2026-11-03).
            // In quiescence, searching all 48 multiplies the tree by 48x
            // for one capture; keep the first spawn square. The main
            // search still sees all 48 (exactness preserved there).
            if (m.flags & MF_HORDE_SPAWN) != 0 && m.from < 64 && m.to < 64 {
                let bit = 1u64 << m.to;
                if horde_seen[m.from as usize] & bit != 0 {
                    continue;
                }
                horde_seen[m.from as usize] |= bit;
            }

            let gain = tactical_move_gain(pos, &m, me, opp);
            if (m.flags & MF_GUN_STANDOFF) != 0 && gain < -30 {
                continue; // Don't trade Gunslinger for a much cheaper pawn in qsearch
            }

            let atk_v = if m.from < 64 {
                slot_of(pos, m.from)
                    .map(|si| piece_value(&pos.pieces[si]))
                    .unwrap_or(100)
            } else {
                300
            };

            let is_promo = (m.flags & MF_PROMO) != 0;
            let is_multi_kill = (m.flags & (MF_ELECTRO | MF_HORDE_SPAWN | MF_DANCER_DOUBLE | MF_SHRIKE_FIRST | MF_IS_PUSH | MF_GUN_STANDOFF)) != 0;

            // Delta pruning (only when no doomed pieces are waiting to die)
            if !is_promo && !is_multi_kill && pos.doomed.is_empty() && stand_pat + gain + DELTA_MARGIN < alpha {
                continue;
            }

            // Fast SEE filter: skip clearly losing captures (e.g. Queen/Rook capturing defended Pawn)
            // unless it's a multi-kill power or promotion
            let DestDefended = m.to < 64 && square_attacked(pos, m.to, opp, &occ_local);
            if !is_promo && !is_multi_kill && DestDefended && atk_v - gain >= 220 {
                // Only prune if we don't have a second attacker or if the attacker is much more valuable
                if atk_v - gain >= 380 {
                    continue;
                }
            }

            let order_sc = if DestDefended && atk_v > gain + 80 {
                10_000 + gain * 16 - atk_v
            } else {
                100_000 + gain * 16 - atk_v
            };
            candidates.push((m, order_sc));
        }

        // 2. Check if we have an alive BladeRunner or if we are at the root of qsearch
        //    (`depth == QDEPTH && qext > 0`) where quiet checking moves & HeroPawn check-promos
        //    can force mate or win material!
        let has_br = pos.pieces.iter().any(|p| {
            p.alive
                && p.player == me
                && xy_on_board(p.x, p.y)
                && (p.key == PieceKey::BladeRunner || p.sub_key == PieceKey::BladeRunner)
        });
        // 2026-10-31 mate fix: search quiet checks on the first TWO
        // qsearch plies (mates hide one ply deeper) and drop the stand_pat
        // gate: a mate wins regardless of stand_pat.
        let search_quiet_checks = !s.fast_qsearch && depth >= QDEPTH - 1 && qext > 0;

        if has_br || search_quiet_checks {
            let opp_ksq = find_king(pos, opp).map(|ki| sq(pos.pieces[ki].x as i8, pos.pieces[ki].y as i8));
            let stage1 = gen_moves(pos, me, Some(1));
            for m in stage1 {
                if has_br {
                    let br_gain = blade_runner_doomed_gain(pos, &m, me, opp);
                    if br_gain > 0 {
                        candidates.push((m, 95_000 + br_gain * 12));
                        continue;
                    }
                }
                if search_quiet_checks && m.from < 64 && m.to < 64 {
                    if let Some(ksq) = opp_ksq {
                        // Fast geometric pre-filter for quiet checking moves
                        let tx = sq_x(m.to) as i8;
                        let ty = sq_y(m.to) as i8;
                        let kx = sq_x(ksq) as i8;
                        let ky = sq_y(ksq) as i8;
                        let dx = (tx - kx).abs();
                        let dy = (ty - ky).abs();
                        let aligned_or_leap = dx == 0
                            || dy == 0
                            || dx == dy
                            || (dx == 1 && dy == 2)
                            || (dx == 2 && dy == 1)
                            || (dx == 1 && dy == 3)
                            || (dx == 3 && dy == 1);
                        // Contact checks adjacent to the enemy King were
                        // wrongly rejected: the King itself "attacks" those
                        // squares, so !square_attacked was always false for
                        // them. Allow a contact check when another friendly
                        // piece defends the square (a supported mate); a
                        // hung contact check still resolves correctly in
                        // search (the King can capture it), it only costs
                        // nodes. 2026-10-31 mate fix (ds6CQR/Oq7FHN).
                        let is_contact = dx <= 1 && dy <= 1;
                        let safe_or_supported = !square_attacked(pos, m.to, opp, &occ_local)
                            || (is_contact && square_attacked(pos, m.to, me, &occ_local));
                        if aligned_or_leap && safe_or_supported {
                            let is_hero = slot_of(pos, m.from).map_or(false, |si| {
                                let p = &pos.pieces[si];
                                p.key == PieceKey::HeroPawn || p.sub_key == PieceKey::HeroPawn
                            });
                            let bonus = if is_hero { 98_000 } else { 40_000 };
                            // Marked with negative sentinel in `special_b = 1` so we only recurse if it actually gives check
                            let mut m_chk = m;
                            m_chk.special_b = 1;
                            candidates.push((m_chk, bonus));
                        }
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        if candidates.len() > QCAP {
            candidates.truncate(QCAP);
        }
    }

    if candidates.is_empty() && !in_chk {
        return best;
    }

    let mut legal_moves = 0usize;

    for (raw_mv, _) in &candidates {
        let require_check = !in_chk && raw_mv.special_b == 1;
        let mut mv = *raw_mv;
        if require_check {
            mv.special_b = 0;
        }

        let j_delta = if s.use_acc_stack {
            snapshot_journal_delta(pos, &mv, me, opp)
        } else {
            None
        };
        let mark = s.journal.mark();
        let use_j = make_journaled(pos, &mv, me, opp, &mut s.journal);
        let mut p2_owned: Option<Pos>;
        let sc;
        let next_acc_idx = (acc_idx + 1).min(127);

        if use_j {
            // Lever-3: derived child occupancy (exact on the journaled path).
            let p2_occ = occ_after_journaled(&occ_local, &mv);
            if king_attacked(pos, me, opp, Some(&p2_occ)) {
                s.journal.unwind(pos, mark);
                continue;
            }
            let child_chk = if require_check || (!in_chk && depth > 0 && qext > 0) {
                Some(gives_check(pos, me, opp, Some(&p2_occ)))
            } else {
                None
            };
            if require_check && child_chk != Some(true) {
                s.journal.unwind(pos, mark);
                continue;
            }
            legal_moves += 1;
            push_acc_journaled(s, acc_idx, next_acc_idx, me, j_delta, pos);
            let (qnd, qnq) = if let Some(true) = child_chk {
                (depth, qext - 1)
            } else {
                (depth - 1, qext)
            };
            sc = -qsearch_acc(s, pos, opp, me, -beta, -alpha, qnd, qnq, child_chk, next_acc_idx, qroot_nodes);
            s.journal.unwind(pos, mark);
        } else {
            p2_owned = apply_full(pos, &mv, me, opp);
            let p2 = match p2_owned.as_mut() {
                Some(p) => p,
                None => {
                    s.journal.unwind(pos, mark);
                    continue;
                }
            };
            let p2_occ = occ(p2);
            if king_attacked(p2, me, opp, Some(&p2_occ)) {
                s.journal.unwind(pos, mark);
                continue;
            }
            let child_chk = if require_check || (!in_chk && depth > 0 && qext > 0) {
                Some(gives_check(p2, me, opp, Some(&p2_occ)))
            } else {
                None
            };
            if require_check && child_chk != Some(true) {
                s.journal.unwind(pos, mark);
                continue;
            }
            legal_moves += 1;
            push_acc_refresh(s, next_acc_idx, p2);
            let (qnd, qnq) = if let Some(true) = child_chk {
                (depth, qext - 1)
            } else {
                (depth - 1, qext)
            };
            sc = -qsearch_acc(s, p2, opp, me, -beta, -alpha, qnd, qnq, child_chk, next_acc_idx, qroot_nodes);
            s.journal.unwind(pos, mark);
        }

        if s.stop {
            return best;
        }
        if sc > best {
            best = sc;
            if sc > alpha {
                alpha = sc;
                if alpha >= beta {
                    return best; // fail-soft
                }
            }
        }
    }

    if in_chk && legal_moves == 0 {
        return -(MATE - qply_i);
    }

    best
}

// ── 1 & 2. ROOT ITERATIVE DEEPENING CONTROLLER & CORE ALPHA-BETA SEARCH ────

/// Helper to restore complete root scores when an iteration aborts mid-depth on timeout.
fn restore_root_on_abort(s: &mut Search) {
    if s.tt.last_root_scored.is_empty() {
        return;
    }
    // Merge any root move from the aborted iteration that beat `last_best_score`
    // while preserving the complete set of all legal root moves from `last_root_scored`.
    let partial = core::mem::take(&mut s.root_scored);
    let mut merged = s.tt.last_root_scored.clone();

    if let Some(cur_best) = s.best_root {
        if s.best_root_score >= s.tt.last_best_score - 20 {
            if let Some(slot) = merged.iter_mut().find(|(m, _)| same_move(m, &cur_best)) {
                slot.1 = s.best_root_score.max(slot.1);
            } else {
                merged.push((cur_best, s.best_root_score));
            }
        } else {
            s.best_root = s.tt.last_best_root;
            s.best_root_score = s.tt.last_best_score;
        }
    } else {
        s.best_root = s.tt.last_best_root;
        s.best_root_score = s.tt.last_best_score;
    }

    for (pm, psc) in partial {
        if psc > s.tt.last_best_score {
            if let Some(slot) = merged.iter_mut().find(|(m, _)| same_move(m, &pm)) {
                slot.1 = psc;
            }
        }
    }

    s.root_scored = merged;
}

pub fn abSearch(
    s: &mut Search,
    pos: &mut Pos,
    depth: i32,
    alpha: i32,
    beta: i32,
    stm: u8,
    ntm: u8,
    me: u8,
    ply: u32,
    ext: i32,
    pre_in_chk: Option<bool>,
) -> i32 {
    // Intercept top-level root calls (`ply == 0`) from `ffi.rs` to provide:
    // 1. Adaptive soft/hard time management
    // 2. Automatic Aspiration Windows (`depth >= 4`)
    // 3. Mid-iteration abort recovery so `s.root_scored` is never left empty or partial!
    if ply == 0 {
        s.init_root_accumulator(pos);
    }
    if ply == 0 && alpha <= -500_000_000 && beta >= 500_000_000 {
        if depth <= 1 {
            s.tt.last_root_scored.clear();
            s.tt.last_best_root = None;
            s.tt.last_best_score = 0;
            s.tt.stable_root_count = 0;
            s.tt.score_drop = 0;
        } else if !s.tt.last_root_scored.is_empty() {
            // 1. Single legal move at root: no need to search past depth 2
            //    (only when searching full root, not on a 1-move cluster slice).
            if s.root_filter.is_none() && s.tt.last_root_scored.len() == 1 && depth >= 3 {
                restore_root_on_abort(s);
                s.stop = true;
                return s.tt.last_best_score;
            }

            // 2. Adaptive soft time cutoff: don't launch a new depth that cannot finish
            let elapsed = crate::now_ms().saturating_sub(s.t0_ms);
            let soft_pct: u64 = if s.tt.score_drop >= 60 {
                88 // Tactical crisis / score drop: use almost full budget
            } else if depth >= 5 && s.tt.stable_root_count >= 3 && s.tt.score_drop <= 20 {
                52 // Stable best move for 3+ depths: save clock time
            } else if depth >= 4 {
                70 // Normal soft limit (next depth takes ~2.2x previous cumulative time)
            } else {
                95
            };
            if s.budget_ms < 500_000_000 && elapsed >= (s.budget_ms * soft_pct) / 100 {
                restore_root_on_abort(s);
                s.stop = true;
                return s.tt.last_best_score;
            }
        }

        // 3. Aspiration Windows for `depth >= 4`
        let use_asp = depth >= 4
            && !s.no_pvs
            && s.tt.last_best_root.is_some()
            && s.tt.last_best_score.abs() < MATE - 100_000;

        let mut delta = ASP_DELTA;
        let mut asp_a = if use_asp {
            (s.tt.last_best_score - delta).max(alpha)
        } else {
            alpha
        };
        let mut asp_b = if use_asp {
            (s.tt.last_best_score + delta).min(beta)
        } else {
            beta
        };

        let final_sc;
        loop {
            s.root_scored.clear();
            s.best_root = None;
            let sc = ab_search_core(s, pos, depth, asp_a, asp_b, stm, ntm, me, 0, ext, pre_in_chk);

            if s.stop {
                restore_root_on_abort(s);
                return s.best_root_score;
            }

            if sc <= asp_a && asp_a > alpha {
                // Failed low: record score drop and widen window downward
                let drop = s.tt.last_best_score - sc;
                if drop > s.tt.score_drop {
                    s.tt.score_drop = drop;
                }
                asp_b = (asp_a + asp_b) / 2;
                delta = delta.saturating_mul(3);
                asp_a = if delta > 1000 { alpha } else { (sc - delta).max(alpha) };
                continue;
            }

            if sc >= asp_b && asp_b < beta {
                // Failed high: widen window upward
                delta = delta.saturating_mul(3);
                asp_b = if delta > 1000 { beta } else { (sc + delta).min(beta) };
                continue;
            }

            final_sc = sc;
            break;
        }

        // Depth completed cleanly! Update stability tracking and save full root scores.
        if let Some(bm) = s.best_root {
            if s.tt.last_best_root.map_or(false, |prev| same_move(&prev, &bm)) {
                s.tt.stable_root_count = s.tt.stable_root_count.saturating_add(1);
            } else {
                s.tt.stable_root_count = 1;
            }
            if depth > 1 {
                let drop = s.tt.last_best_score - final_sc;
                s.tt.score_drop = drop.max(0);
            }
            s.tt.last_best_root = Some(bm);
            s.tt.last_best_score = final_sc;
        }
        if !s.root_scored.is_empty() {
            s.tt.last_root_scored = s.root_scored.clone();
        }
        return final_sc;
    }

    ab_search_core(s, pos, depth, alpha, beta, stm, ntm, me, ply, ext, pre_in_chk)
}

fn ab_search_core(
    s: &mut Search,
    pos: &mut Pos,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    stm: u8,
    ntm: u8,
    me: u8,
    ply: u32,
    ext: i32,
    pre_in_chk: Option<bool>,
) -> i32 {
    s.nodes += 1;
    s.check_time();
    if s.stop {
        return alpha;
    }

    let ply_i = ply as i32;
    let ply_idx = (ply as usize).min(63);

    if let Some(kt) = pos.king_thrown {
        return if kt == stm {
            MATE - ply_i
        } else {
            -(MATE - ply_i)
        };
    }
    if let Some(gw) = pos.gold_win {
        return if gw == stm {
            MATE - ply_i
        } else {
            -(MATE - ply_i)
        };
    }
    if pos.halfmove >= 100 {
        return 0;
    }

    let acc_idx = (ply as usize).min(126);
    let next_acc_idx = (acc_idx + 1).min(127);

    // Rank 1 Fix: Detect 2-fold repetition in-tree or against seeded `game_history`!
    // Previously `rep_count >= 3` required the exact same position to occur 3 times
    // inside a single search call, making 2-ply (`Ra8-a7-a8`) and 4-ply loops invisible.
    // We also apply a small anti-shuffle contempt (-28cp for the root player initiating
    // a repeat) so the bot refuses sterile Rook/King shuffles unless otherwise losing.
    // Lever-3: fused single-loop hash (bit-identical to the two calls above).
    let (h_signed, rep_h) = zobrist_both(pos, stm);
    let h = h_signed as u32;
    if ply > 0 && rep_count(pos, stm, rep_h) >= 2 {
        s.rep_hits += 1;
        return if stm == me { -28 } else { 28 };
    }

    let is_root = ply == 0;
    let is_pv = beta - alpha > 1;

    let mut tt_move: Option<(u8, u8)> = None;
    if !is_root && !s.no_tt {
        tt_move = s.tt.probe_move(h);
        if !is_pv {
            if let Some(sc) = s.tt.probe(h, depth, alpha, beta) {
                return sc;
            }
        } else if let Some(sc) = s.tt.probe_exact(h, depth) {
            return sc;
        }
    } else if is_root && !s.no_tt {
        tt_move = s.tt.last_best_root.map(|m| (m.from, m.to)).or_else(|| s.tt.probe_move(h));
    }

    let occ_local = occ(pos);
    let in_chk = pre_in_chk.unwrap_or_else(|| king_attacked(pos, stm, ntm, Some(&occ_local)));
    let orig_alpha = alpha;

    // Leaf transition: NEVER return static `rel_eval` when `in_chk == true`!
    // Transitioning to `qsearch(..., Some(true))` guarantees all check evasions
    // and checkmates are resolved accurately.
    if depth <= 0 {
        if in_chk && ext < 3 {
            return ab_search_core(
                s,
                pos,
                1,
                alpha,
                beta,
                stm,
                ntm,
                me,
                ply,
                ext + 1,
                Some(true),
            );
        }
        return qsearch_acc(s, pos, stm, ntm, alpha, beta, QDEPTH, 2, Some(in_chk), acc_idx, s.nodes);
    }

    let mut static_ev = 0i32;
    let mut have_static_ev = false;

    if !in_chk && ply > 0 {
        static_ev = s.rel_eval_at(pos, stm, me, acc_idx);
        have_static_ev = true;

        // 1. Razoring at depth 1
        if !is_pv
            && depth == 1
            && pos.doomed.is_empty()
            && alpha.abs() < MATE - 100_000
            && static_ev + RAZOR_MARGIN <= alpha
        {
            let qsc = qsearch_acc(s, pos, stm, ntm, alpha, beta, QDEPTH, 1, Some(false), acc_idx, s.nodes);
            if qsc <= alpha {
                return qsc;
            }
        }

        // 2. Reverse Futility Pruning (Static Null Move Pruning)
        if !is_pv
            && depth <= 3
            && pos.doomed.is_empty()
            && static_ev - RFP_MARGIN_PER_DEPTH * depth >= beta
            && beta.abs() < MATE - 100_000
        {
            return static_ev;
        }

        // 3. Adaptive Null-Move Pruning (disabled when doomed pieces exist)
        if !is_pv
            && depth >= 3
            && s.null_ply != (ply_i - 1)
            && pos.doomed.is_empty()
            && has_non_pawn_material(pos, stm)
            && static_ev >= beta
        {
            let saved_ep = pos.ep;
            let saved_stm = pos.stm;
            let saved_np = s.null_ply;
            let saved_pf = s.tt.prev_from[ply_idx];
            let saved_pt = s.tt.prev_to[ply_idx];
            let saved_pk = s.tt.prev_key[ply_idx];
            pos.ep = None;
            pos.stm = ntm;
            s.null_ply = ply_i;
            s.tt.prev_from[ply_idx] = SQ_NONE;
            s.tt.prev_to[ply_idx] = SQ_NONE;
            s.tt.prev_key[ply_idx] = 0;

            if s.use_acc_stack {
                let mut null_acc = s.acc_stack[acc_idx];
                null_acc.flip_stm();
                s.acc_stack[next_acc_idx] = null_acc;
            }

            let r = if depth >= 6 { 3 } else { 2 }
                + if static_ev - beta >= 300 && depth >= 4 {
                    1
                } else {
                    0
                };
            let nms_depth = (depth - 1 - r).max(1);
            let nms = -ab_search_core(
                s,
                pos,
                nms_depth,
                -beta,
                1 - beta,
                ntm,
                stm,
                me,
                ply + 1,
                ext,
                Some(false),
            );
            pos.ep = saved_ep;
            pos.stm = saved_stm;
            s.null_ply = saved_np;
            s.tt.prev_from[ply_idx] = saved_pf;
            s.tt.prev_to[ply_idx] = saved_pt;
            s.tt.prev_key[ply_idx] = saved_pk;
            if s.stop {
                return alpha;
            }
            if nms >= beta && nms < MATE - 100_000 {
                return nms; // fail-soft
            }
        }

        // 4. Internal Iterative Reduction (IIR): if we have no TT move at a high-depth node,
        //    reduce depth by 1 so we don't blow the node budget with blind move ordering.
        if tt_move.is_none() && depth >= 5 && is_pv {
            let _ = ab_search_core(
                s,
                pos,
                depth - 2,
                alpha,
                beta,
                stm,
                ntm,
                me,
                ply,
                ext,
                Some(false),
            );
            tt_move = s.tt.probe_move(h);
        }
    }

    // Look up opponent's previous move for countermove heuristic & recapture extensions
    let (prev_f, prev_t) = if ply > 0 {
        let p_idx = ((ply - 1) as usize).min(63);
        (s.tt.prev_from[p_idx], s.tt.prev_to[p_idx])
    } else {
        (SQ_NONE, SQ_NONE)
    };
    let cm_move: Option<(u8, u8)> = if prev_f < 64 && prev_t < 64 {
        let cm_idx = hist_index(prev_f, prev_t);
        let cf = s.tt.cm_from[cm_idx];
        let ct = s.tt.cm_to[cm_idx];
        if cf != SQ_NONE && ct != SQ_NONE {
            Some((cf, ct))
        } else {
            None
        }
    } else {
        None
    };

    let mut best = -1_000_000_000i32;
    let mut best_move: Option<Move> = None;
    let mut best_reduced = false;
    let mut legal = 0i32;
    let mut searched_quiets: [(u8, u8, u8); 32] = [(SQ_NONE, SQ_NONE, 0); 32];
    let mut n_searched_quiets: usize = 0;

    // At root (`ply == 0`), generate all moves in a single pass so `last_root_scored`
    // from `depth - 1` can order quiet and tactical root moves together in true best-first order!
    let n_stages: u8 = if is_root { 1 } else { 2 };

    for stage in 0..n_stages {
        let mut moves = if is_root {
            let mut rm = gen_moves(pos, stm, None);
            if let Some(ref allowed) = s.root_filter {
                rm.retain(|m| allowed.contains(m));
            }
            rm
        } else {
            gen_moves(pos, stm, Some(stage))
        };

        ab_order(
            &mut moves,
            pos,
            stm,
            ntm,
            &occ_local,
            ply_idx,
            tt_move,
            cm_move,
            &s.killers[ply_idx],
            &s.hist,
            &s.tt.piece_hist,
            if is_root { Some(&s.tt.last_root_scored) } else { None },
            if is_root { s.tt.last_best_root } else { None },
        );

        if ply == 0 {
            s.journal.v.clear();
        }

        let mut horde_seen_interior = [0u64; 64];
        let mut rocket_seen_interior = false;
        let mut pilgrim_seen_interior = [0u64; 64];

        for mv in &moves {
            // At non-root interior nodes (`ply > 0`), collapse combinatorial random/spawn
            // variants (48 HordeMother spawn squares, ~50 Rocketman blast squares, and
            // K random Pilgrim resurrection targets) to the highest-ordered representative
            // per (from, to). Root (`ply == 0`) still searches all variants exactly.
            // Generator-only: gated on fast_qsearch, so the live engine is untouched.
            if !is_root && s.fast_qsearch {
                if (mv.flags & MF_HORDE_SPAWN) != 0 && mv.from < 64 && mv.to < 64 {
                    let bit = 1u64 << mv.to;
                    if horde_seen_interior[mv.from as usize] & bit != 0 {
                        continue;
                    }
                    horde_seen_interior[mv.from as usize] |= bit;
                } else if (mv.flags & MF_ROCKET_BLAST) != 0 {
                    if rocket_seen_interior {
                        continue;
                    }
                    rocket_seen_interior = true;
                } else if (mv.flags & MF_PILGRIM_RESURRECT) != 0 && mv.from < 64 && mv.to < 64 {
                    let bit = 1u64 << mv.to;
                    if pilgrim_seen_interior[mv.from as usize] & bit != 0 {
                        continue;
                    }
                    pilgrim_seen_interior[mv.from as usize] |= bit;
                }
            }

            let is_cap = mv.flags & MF_CAPTURE != 0;
            let is_promo = mv.flags & MF_PROMO != 0;
            let is_push = mv.flags & MF_IS_PUSH != 0;
            let is_drop = mv.flags & MF_KNIGHTMARE_DROP != 0;
            let is_standoff = mv.flags & MF_GUN_STANDOFF != 0;
            let is_dancer = mv.flags & MF_DANCER_DOUBLE != 0;
            let is_tactical = is_cap || is_promo || is_push || is_drop || is_standoff || is_dancer;

            // State-charge completion (Pilgrim odo crossing 20, ElectroKnight
            // 3rd quiet move) is exempt from LMR below. Computed pre-apply:
            // after make_journaled, mv.from is empty.
            let charge_move = completes_state_charge(pos, mv);

            let mover_pk: u8 = if mv.from < 64 {
                slot_of(pos, mv.from)
                    .map(|si| effective_key(&pos.pieces[si]) as u8)
                    .unwrap_or(0)
            } else {
                PieceKey::Knightmare as u8
            };

            let pre_vic_val = if is_cap && mv.to < 64 {
                slot_of(pos, mv.to)
                    .map(|si| piece_value(&pos.pieces[si]))
                    .unwrap_or(0)
            } else {
                0
            };
            let is_recapture = is_pv && is_cap && prev_t == mv.to && pre_vic_val >= 280;
            let is_7th_push = !is_promo
                && mv.from < 64
                && mv.to < 64
                && slot_of(pos, mv.from).map_or(false, |si| {
                    let p = &pos.pieces[si];
                    is_piece_pawn_like(p)
                        && p.key != PieceKey::IronPawn
                        && p.sub_key != PieceKey::IronPawn
                        && (sq_y(mv.to) as i8 == if stm == 0 { 6 } else { 1 }
                            || ((p.key == PieceKey::GoldenPawn || p.sub_key == PieceKey::GoldenPawn)
                                && (sq_y(mv.to) as i8 == if stm == 0 { 5 } else { 2 })))
                });

            let j_delta = if s.use_acc_stack {
                snapshot_journal_delta(pos, mv, stm, ntm)
            } else {
                None
            };
            let mark = s.journal.mark();
            let use_j = make_journaled(pos, mv, stm, ntm, &mut s.journal);
            let mut p2_owned: Option<Pos> = None;
            let sc;
            let mut searched_zero_window_only = false;

            if use_j {
                // Lever-3: derived child occupancy (exact on the journaled path).
                let p2_occ = occ_after_journaled(&occ_local, mv);
                if king_attacked(pos, stm, ntm, Some(&p2_occ)) {
                    s.journal.unwind(pos, mark);
                    continue;
                }
                legal += 1;
                let child_chk = gives_check(pos, stm, ntm, Some(&p2_occ));

                let lmr_killer = s.killers[ply_idx]
                    .iter()
                    .any(|k| k.map_or(false, |km| km.from == mv.from && km.to == mv.to));
                let is_cm = cm_move.map_or(false, |(cf, ct)| cf == mv.from && ct == mv.to);
                let h_sc = if mv.from < 64 && mv.to < 64 {
                    s.hist[hist_index(mv.from, mv.to)]
                        + s.tt.piece_hist[piece_hist_index(mover_pk, mv.to)]
                } else {
                    0
                };

                // Futility & Late Move Pruning for quiet non-checking moves at shallow depth
                // Lever-2 experiment: extended from depth <= 2 to depth <= 3.
                // At depth 3, state-charge completions (Pilgrim odo >= 20,
                // ElectroKnight 3rd quiet) are exempt: a charge is a
                // permanent power-up worth up to ~240cp, which a plain
                // futility margin cannot safely bound. Depth <= 2 behaviour
                // is unchanged.
                if !is_root
                    && !is_pv
                    && !in_chk
                    && !child_chk
                    && !is_tactical
                    && !lmr_killer
                    && !is_cm
                    && depth <= 3
                    && (depth <= 2 || !charge_move)
                    && legal >= 2
                    && best > -(MATE - 100_000)
                {
                    if legal > 8 + depth * 6 && h_sc < 0 {
                        s.journal.unwind(pos, mark);
                        continue;
                    }
                    if !have_static_ev {
                        static_ev = s.rel_eval_at(pos, stm, me, acc_idx);
                        have_static_ev = true;
                    }
                    let f_margin = FUTILITY_BASE + FUTILITY_PER_DEPTH * depth;
                    if static_ev + f_margin <= alpha {
                        s.journal.unwind(pos, mark);
                        continue;
                    }
                }

                push_acc_journaled(s, acc_idx, next_acc_idx, stm, j_delta, pos);

                let ext_add = if child_chk && depth >= 1 && ext < 5 {
                    1
                } else if (is_recapture || (is_pv && is_7th_push)) && depth >= 2 && ext < 4 {
                    1
                } else {
                    0
                };
                let n_ext = ext + ext_add;
                let full_depth = depth - 1 + ext_add;

                s.tt.prev_from[ply_idx] = mv.from;
                s.tt.prev_to[ply_idx] = mv.to;
                s.tt.prev_key[ply_idx] = mover_pk;

                // Cluster workers search a 1/5 subset, so the first 3 moves of EVERY
                // worker got full windows (15 total vs 3 in single mode).
                // Only the first move per worker gets a full window in
                // cluster mode; PVS re-search on fail-high keeps this exact.
                let full_n = if s.cluster_worker { 1 } else { 3 };
                let root_full_window = is_root && (depth <= 2 || legal <= full_n);

                if !s.no_pvs && !root_full_window {
                    if legal == 1 {
                        sc = -ab_search_core(
                            s,
                            pos,
                            full_depth,
                            -beta,
                            -alpha,
                            ntm,
                            stm,
                            me,
                            ply + 1,
                            n_ext,
                            Some(child_chk),
                        );
                    } else {
                        let do_lmr = depth >= 3
                            && legal > 3
                            && !is_tactical
                            && !lmr_killer
                            && !child_chk
                            && !in_chk
                            && !charge_move;
                        let red = if do_lmr {
                            lmr_reduction(depth, legal, is_pv, lmr_killer || is_cm, h_sc)
                        } else {
                            0
                        };
                        searched_zero_window_only = true;
                        let mut sc_r = -ab_search_core(
                            s,
                            pos,
                            full_depth - red,
                            -alpha - 1,
                            -alpha,
                            ntm,
                            stm,
                            me,
                            ply + 1,
                            n_ext,
                            Some(child_chk),
                        );
                        if s.stop {
                            s.journal.unwind(pos, mark);
                            return alpha;
                        }
                        if red > 0 && sc_r > alpha {
                            sc_r = -ab_search_core(
                                s,
                                pos,
                                full_depth,
                                -alpha - 1,
                                -alpha,
                                ntm,
                                stm,
                                me,
                                ply + 1,
                                n_ext,
                                Some(child_chk),
                            );
                        }
                        if sc_r > alpha && sc_r < beta {
                            searched_zero_window_only = false;
                            sc_r = -ab_search_core(
                                s,
                                pos,
                                full_depth,
                                -beta,
                                -alpha,
                                ntm,
                                stm,
                                me,
                                ply + 1,
                                n_ext,
                                Some(child_chk),
                            );
                        }
                        sc = sc_r;
                    }
                } else {
                    let low_bound = if is_root && legal > 1 && depth >= 3 {
                        (alpha - 140).max(-1_000_000_000)
                    } else {
                        alpha
                    };
                    sc = -ab_search_core(
                        s,
                        pos,
                        full_depth,
                        -beta,
                        -low_bound,
                        ntm,
                        stm,
                        me,
                        ply + 1,
                        n_ext,
                        Some(child_chk),
                    );
                }
                s.journal.unwind(pos, mark);
            } else {
                p2_owned = apply_full(pos, &mv, stm, ntm);
                let p2 = match p2_owned.as_mut() {
                    Some(p) => p,
                    None => {
                        continue;
                    }
                };
                let p2_occ = occ(p2);
                if king_attacked(p2, stm, ntm, Some(&p2_occ)) {
                    continue;
                }

                // EP field
                if mv.from < 64 && mv.to < 64 {
                    if let Some(ms) = slot_of(pos, mv.from) {
                        let p = &pos.pieces[ms];
                        if is_piece_pawn_like(p)
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
                            p2.ep = None;
                        }
                    }
                }

                legal += 1;
                let child_chk = gives_check(p2, stm, ntm, Some(&p2_occ));

                let lmr_killer = s.killers[ply_idx]
                    .iter()
                    .any(|k| k.map_or(false, |km| km.from == mv.from && km.to == mv.to));
                let is_cm = cm_move.map_or(false, |(cf, ct)| cf == mv.from && ct == mv.to);
                let h_sc = if mv.from < 64 && mv.to < 64 {
                    s.hist[hist_index(mv.from, mv.to)]
                        + s.tt.piece_hist[piece_hist_index(mover_pk, mv.to)]
                } else {
                    0
                };

                // Futility & Late Move Pruning (lever-2: depth <= 3, state-charge
                // completions exempt at depth 3 — see journaled path above).
                if !is_root
                    && !is_pv
                    && !in_chk
                    && !child_chk
                    && !is_tactical
                    && !lmr_killer
                    && !is_cm
                    && depth <= 3
                    && (depth <= 2 || !charge_move)
                    && legal >= 2
                    && best > -(MATE - 100_000)
                {
                    if legal > 8 + depth * 6 && h_sc < 0 {
                        continue;
                    }
                    if !have_static_ev {
                        static_ev = s.rel_eval_at(pos, stm, me, acc_idx);
                        have_static_ev = true;
                    }
                    let f_margin = FUTILITY_BASE + FUTILITY_PER_DEPTH * depth;
                    if static_ev + f_margin <= alpha {
                        continue;
                    }
                }

                push_acc_refresh(s, next_acc_idx, p2);

                let ext_add = if child_chk && depth >= 1 && ext < 5 {
                    1
                } else if (is_recapture || (is_pv && is_7th_push)) && depth >= 2 && ext < 4 {
                    1
                } else {
                    0
                };
                let n_ext = ext + ext_add;
                let full_depth = depth - 1 + ext_add;

                s.tt.prev_from[ply_idx] = mv.from;
                s.tt.prev_to[ply_idx] = mv.to;
                s.tt.prev_key[ply_idx] = mover_pk;

                // Cluster workers search a 1/5 subset, so the first 3 moves of EVERY
                // worker got full windows (15 total vs 3 in single mode).
                // Only the first move per worker gets a full window in
                // cluster mode; PVS re-search on fail-high keeps this exact.
                let full_n = if s.cluster_worker { 1 } else { 3 };
                let root_full_window = is_root && (depth <= 2 || legal <= full_n);

                if !s.no_pvs && !root_full_window {
                    if legal == 1 {
                        sc = -ab_search_core(
                            s,
                            p2,
                            full_depth,
                            -beta,
                            -alpha,
                            ntm,
                            stm,
                            me,
                            ply + 1,
                            n_ext,
                            Some(child_chk),
                        );
                    } else {
                        let do_lmr = depth >= 3
                            && legal > 3
                            && !is_tactical
                            && !lmr_killer
                            && !child_chk
                            && !in_chk
                            && !charge_move;
                        let red = if do_lmr {
                            lmr_reduction(depth, legal, is_pv, lmr_killer || is_cm, h_sc)
                        } else {
                            0
                        };
                        searched_zero_window_only = true;
                        let mut sc_r = -ab_search_core(
                            s,
                            p2,
                            full_depth - red,
                            -alpha - 1,
                            -alpha,
                            ntm,
                            stm,
                            me,
                            ply + 1,
                            n_ext,
                            Some(child_chk),
                        );
                        if s.stop {
                            return alpha;
                        }
                        if red > 0 && sc_r > alpha {
                            sc_r = -ab_search_core(
                                s,
                                p2,
                                full_depth,
                                -alpha - 1,
                                -alpha,
                                ntm,
                                stm,
                                me,
                                ply + 1,
                                n_ext,
                                Some(child_chk),
                            );
                        }
                        if sc_r > alpha && sc_r < beta {
                            searched_zero_window_only = false;
                            sc_r = -ab_search_core(
                                s,
                                p2,
                                full_depth,
                                -beta,
                                -alpha,
                                ntm,
                                stm,
                                me,
                                ply + 1,
                                n_ext,
                                Some(child_chk),
                            );
                        }
                        sc = sc_r;
                    }
                } else {
                    let low_bound = if is_root && legal > 1 && depth >= 3 {
                        (alpha - 140).max(-1_000_000_000)
                    } else {
                        alpha
                    };
                    sc = -ab_search_core(
                        s,
                        p2,
                        full_depth,
                        -beta,
                        -low_bound,
                        ntm,
                        stm,
                        me,
                        ply + 1,
                        n_ext,
                        Some(child_chk),
                    );
                }
            }

            if s.stop {
                return alpha;
            }

            if !is_tactical && mv.from < 64 && mv.to < 64 && n_searched_quiets < 32 {
                searched_quiets[n_searched_quiets] = (mv.from, mv.to, mover_pk);
                n_searched_quiets += 1;
            }

            if sc > best {
                best = sc;
                best_move = Some(*mv);
                best_reduced = false;
                if ply == 0 {
                    s.best_root = Some(*mv);
                    s.best_root_score = sc;
                }
            }

            // Record every root move with its score for veto re-ranking.
            // If a move at `ply == 0` was only searched with a PVS zero-window and failed
            // low (`sc <= alpha`), `sc` is an upper bound — cap it strictly below `alpha`
            // (blended with its `depth - 1` score) so zero-window fail-low moves never tie
            // with the best move (`ambiguous move`) or jump ahead on small soft veto penalties!
            if ply == 0 {
                let mv_c = *mv;
                let recorded_sc = if searched_zero_window_only && sc <= alpha {
                    let prev_sc = s
                        .tt
                        .last_root_scored
                        .iter()
                        .find(|(m, _)| same_move(m, &mv_c))
                        .map(|(_, psc)| *psc)
                        .unwrap_or(sc - 35);
                    let mut r = sc.min(alpha - 25 - (legal * 2).min(60)).min(prev_sc.max(sc - 120));
                    // A zero-window fail-low score is only an upper bound, never
                    // a proven score. Once an earlier root move found a genuine
                    // mate, alpha itself is mate-scale, so "just below alpha"
                    // is still mate-scale — and an unproven move would rank as
                    // a forced mate (Qt7zbC collapse: after c2->c1 set
                    // alpha=MATE-1, 25 moves were recorded ~MATE-120 and the bot
                    // played e6->c4 believing it mated). Never record a bound
                    // as a mate: fall back to the previous depth's exact score,
                    // or to just below the mate threshold when that is itself
                    // mate-scale. (Negative mate-scale bounds are left alone:
                    // a near-mate loss already ranks at the bottom, correctly.)
                    if r >= MATE - 100_000 {
                        r = if prev_sc < MATE - 100_000 {
                            prev_sc
                        } else {
                            MATE - 100_000 - 1
                        };
                    }
                    r
                } else {
                    sc
                };
                match s.root_scored.iter_mut().find(|(m, _)| same_move(m, &mv_c)) {
                    Some(slot) => {
                        slot.1 = recorded_sc;
                    }
                    None => {
                        s.root_scored.push((mv_c, recorded_sc));
                    }
                }
            }

            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                let mv_c = *mv;
                if !is_cap {
                    // Update 2 killer slots without duplicating the same move
                    let kl = &mut s.killers[ply_idx];
                    let same_k0 = kl[0].map_or(false, |k| k.from == mv_c.from && k.to == mv_c.to);
                    if !same_k0 {
                        kl[1] = kl[0];
                        kl[0] = Some(mv_c);
                    }
                    // Update butterfly history + piece-to history with gravity + malus for prior failed quiets
                    if mv_c.from < 64 && mv_c.to < 64 {
                        let bonus = (depth * depth * 16 + depth * 32).min(2000);
                        let win_idx = hist_index(mv_c.from, mv_c.to);
                        let win_p_idx = piece_hist_index(mover_pk, mv_c.to);
                        update_history_entry(&mut s.hist[win_idx], bonus);
                        update_history_entry(&mut s.tt.piece_hist[win_p_idx], bonus);

                        for q_i in 0..n_searched_quiets.saturating_sub(1) {
                            let (qf, qt, qpk) = searched_quiets[q_i];
                            let q_idx = hist_index(qf, qt);
                            let qp_idx = piece_hist_index(qpk, qt);
                            if q_idx != win_idx {
                                update_history_entry(&mut s.hist[q_idx], -bonus / 2);
                            }
                            if qp_idx != win_p_idx {
                                update_history_entry(&mut s.tt.piece_hist[qp_idx], -bonus / 2);
                            }
                        }
                        // Store countermove against opponent's previous move
                        if prev_f < 64 && prev_t < 64 {
                            let cm_idx = hist_index(prev_f, prev_t);
                            s.tt.cm_from[cm_idx] = mv_c.from;
                            s.tt.cm_to[cm_idx] = mv_c.to;
                        }
                    }
                }
                break;
            }
        }
        if s.stop {
            return alpha;
        }
        if alpha >= beta {
            break;
        }
    }

    if legal == 0 {
        return if in_chk { -(MATE - ply_i) } else { 0 };
    }

    if !s.no_tt && h != 0 && !best_reduced {
        let flag: i8 = if best <= orig_alpha {
            2
        } else if best >= beta {
            1
        } else {
            0
        };
        s.tt.store(h, depth, best, flag, best_move.as_ref());
    }
    best
}

// ── 4. MOVE ORDERING (TT, MVV-LVA + SEE, PIXIE POWERS, KILLERS, CM, HISTORY) ──
fn ab_order(
    moves: &mut Vec<Move>,
    pos: &Pos,
    stm: u8,
    ntm: u8,
    occ_local: &[bool; 64],
    ply_idx: usize,
    tt_move: Option<(u8, u8)>,
    cm_move: Option<(u8, u8)>,
    killers: &[Option<Move>; 2],
    hist: &[i32; 4096],
    piece_hist: &[i32],
    prev_root_scores: Option<&Vec<(Move, i32)>>,
    prev_best_root: Option<Move>,
) {
    let _ = ply_idx;
    let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());

    // Enemy king square, once per call, for the King-zone quiet bonus below.
    let opp_ksq = find_king(pos, ntm).map(|ki| sq(pos.pieces[ki].x as i8, pos.pieces[ki].y as i8));

    for mv in moves.drain(..) {
        let mut sc = 0i32;

        // 1. Root Best Move & Previous-Iteration Root Score Ordering
        if let Some(pbr) = prev_best_root {
            if same_move(&mv, &pbr) {
                sc = 10_000_000;
            }
        }
        if sc == 0 {
            if let Some(prs) = prev_root_scores {
                if let Some((_, prev_sc)) = prs.iter().find(|(m, _)| same_move(m, &mv)) {
                    sc = 5_000_000 + prev_sc.clamp(&-100_000, &100_000) * 16;
                }
            }
        }

        // 2. Transposition Table Move
        if sc == 0 {
            if let Some((f, t)) = tt_move {
                if f == mv.from && t == mv.to {
                    sc = 2_000_000;
                }
            }
        }

        if sc == 0 {
            let is_cap = mv.flags & MF_CAPTURE != 0;
            let is_promo = mv.flags & MF_PROMO != 0;
            let is_standoff = mv.flags & MF_GUN_STANDOFF != 0;
            let is_push = mv.flags & MF_IS_PUSH != 0;
            let is_dancer = mv.flags & MF_DANCER_DOUBLE != 0;

            let (atk_val, mover_pk) = if mv.from < 64 {
                slot_of(pos, mv.from)
                    .map(|i| {
                        let p = &pos.pieces[i];
                        (piece_value(p), effective_key(p) as u8)
                    })
                    .unwrap_or((100, 0))
            } else {
                (300, PieceKey::Knightmare as u8)
            };

            if is_cap || is_standoff || (is_dancer && mv.special_a < 64 && slot_of(pos, mv.special_a).is_some()) {
                let gain = tactical_move_gain(pos, &mv, stm, ntm);
                let bad_capture = !is_standoff
                    && gain + 90 < atk_val
                    && mv.to < 64
                    && square_attacked(pos, mv.to, ntm, occ_local);
                if !bad_capture {
                    sc = 1_000_000 + gain * 16 - atk_val;
                } else {
                    sc = 150_000 + gain * 16 - atk_val;
                }
            } else if is_promo {
                let gain = tactical_move_gain(pos, &mv, stm, ntm);
                sc = 960_000 + gain * 8;
            } else if is_push {
                let gain = tactical_move_gain(pos, &mv, stm, ntm);
                sc = 920_000 + gain * 12;
            } else {
                let br_gain = if mover_pk == PieceKey::BladeRunner as u8 {
                    blade_runner_doomed_gain(pos, &mv, stm, ntm)
                } else {
                    0
                };
                if br_gain > 0 {
                    sc = 910_000 + br_gain * 12;
                } else if killers[0].map_or(false, |k| k.from == mv.from && k.to == mv.to) {
                    sc = 880_000;
                } else if killers[1].map_or(false, |k| k.from == mv.from && k.to == mv.to) {
                    sc = 840_000;
                } else if cm_move.map_or(false, |(cf, ct)| cf == mv.from && ct == mv.to) {
                    sc = 800_000;
                } else if mv.flags & MF_KNIGHTMARE_DROP != 0 {
                    sc = 450_000;
                } else if mv.from < 64 && mv.to < 64 {
                    let h_b = hist[hist_index(mv.from, mv.to)];
                    let h_p = piece_hist[piece_hist_index(mover_pk, mv.to)];
                    sc = 300_000 + (h_b + h_p).clamp(-250_000, 450_000);
                    // King-zone quiet checks first among quiets: a Queen,
                    // Rook, or Pixie sliding into the enemy king's zone is
                    // a mating net in the making. Geometric only (no
                    // gives_check cost); bounded below captures (1M+) and
                    // killer[0] (880k). 2026-10-31 mate fix.
                    if let Some(ksq) = opp_ksq {
                        if (mover_pk == PieceKey::Queen as u8
                            || mover_pk == PieceKey::Rook as u8
                            || mover_pk > PieceKey::King as u8)
                            && ((KING_ZONE_BB[(ntm & 1) as usize][ksq as usize] >> mv.to) & 1) == 1
                        {
                            sc += 120_000;
                        }
                    }
                }
            }
        }
        scored.push((mv, sc));
    }

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    moves.extend(scored.into_iter().map(|(m, _)| m));
}

#[cfg(test)]
mod search_audit_tests {
    use super::*;
    use crate::nnue::NNUE_EVALUATOR;

    fn build_pos(pieces: &[(PieceKey, u8, u8, u8)], stm: u8) -> Pos {
        let mut pos = Pos {
            at: [0; 64],
            pieces: Vec::new(),
            limbo: Vec::new(),
            offboard: Vec::new(),
            doomed: Vec::new(),
            hist_hashes: Vec::new(),
            ep: None,
            halfmove: 0,
            eval_sum: 0,
            ket_x2: 0,
            queen_count: 0,
            king_thrown: None,
            gold_win: None,
            last_cap: [(PieceKey::None, PieceKey::None); 2],
            horde_seq: 0,
            id_seq: 0,
            stm,
        };
        for (i, &(key, player, x, y)) in pieces.iter().enumerate() {
            pos.pieces.push(Piece {
                id: (i + 1) as u32,
                key,
                sub_key: key,
                player,
                x,
                y,
                alive: true,
                ..Default::default()
            });
            pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
        }
        init_eval_sum(&mut pos);
        pos.hist_hashes.push(rep_hash(&pos, stm));
        pos
    }

    #[test]
    fn test_incremental_search_bit_identical() {
        // Rank 2 Proof: Search with incremental `acc_stack` enabled vs full-recompute
        // `NNUE_EVALUATOR` produces bit-identical root scores (0cp diff) and identical best move.
        let pos0 = build_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::Rook, 0, 0, 0),
                (PieceKey::Knight, 0, 6, 0),
                (PieceKey::Pawn, 0, 4, 3),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Queen, 1, 3, 7),
                (PieceKey::Rook, 1, 0, 7),
                (PieceKey::Pilgrim, 1, 2, 7),
                (PieceKey::Pawn, 1, 4, 6),
            ],
            1,
        );

        let mut s_inc = Search::new();
        s_inc.reset_for_think(5_000);
        s_inc.evaluator = &NNUE_EVALUATOR;
        s_inc.init_root_accumulator(&pos0);
        assert!(s_inc.use_acc_stack);
        let mut p_inc = pos0.clone();
        let sc_inc = abSearch(&mut s_inc, &mut p_inc, 3, -1_000_000_000, 1_000_000_000, 1, 0, 1, 0, 0, None);

        let mut s_full = Search::new();
        s_full.reset_for_think(5_000);
        s_full.evaluator = &NNUE_EVALUATOR;
        s_full.disable_acc_stack = true;
        let mut p_full = pos0.clone();
        let sc_full = abSearch(&mut s_full, &mut p_full, 3, -1_000_000_000, 1_000_000_000, 1, 0, 1, 0, 0, None);
        assert!(!s_full.use_acc_stack);

        assert_eq!(
            sc_inc, sc_full,
            "Incremental acc_stack search must be bit-identical (0cp diff) to full recompute"
        );
        assert_eq!(s_inc.best_root, s_full.best_root, "Best root move must match identically");
        assert_eq!(s_inc.nodes, s_full.nodes, "Node count must match identically");
    }

    #[test]
    fn test_cluster_worker_window_narrowing_is_exact() {
        // 2026-10-31 cluster root-window fix: a cluster worker's subset
        // gets a full window only on its first root move, not the first 3
        // (15 wide-window moves per iteration across 5 workers vs 3 in
        // single mode). PVS re-search on fail-high must make this exact:
        // same best move and same score as the single-mode window, only
        // fewer nodes. If this fails, the narrowing is NOT safe to ship.
        let pos0 = build_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::Rook, 0, 0, 0),
                (PieceKey::Knight, 0, 6, 0),
                (PieceKey::Pawn, 0, 4, 3),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Queen, 1, 3, 7),
                (PieceKey::Rook, 1, 0, 7),
                (PieceKey::Pilgrim, 1, 2, 7),
                (PieceKey::Pawn, 1, 4, 6),
            ],
            1,
        );
        let run = |cluster: bool| -> (Option<Move>, i32, u64) {
            let mut s = Search::new();
            s.reset_for_think(5_000);
            s.evaluator = &NNUE_EVALUATOR;
            s.disable_acc_stack = true;
            s.cluster_worker = cluster;
            let mut p = pos0.clone();
            let sc = abSearch(&mut s, &mut p, 3, -1_000_000_000, 1_000_000_000, 1, 0, 1, 0, 0, None);
            (s.best_root, sc, s.nodes)
        };
        let (b_single, sc_single, _n_single) = run(false);
        let (b_cluster, sc_cluster, _n_cluster) = run(true);
        assert_eq!(
            b_single, b_cluster,
            "cluster window narrowing must not change the best move"
        );
        assert_eq!(
            sc_single, sc_cluster,
            "cluster window narrowing must not change the score"
        );
    }

    #[test]
    fn test_qsearch_finds_defended_contact_mate() {
        // 2026-10-31 mate fix (ds6CQR/Oq7FHN): White King f6, Queen d7,
        // Black King h8, White to move. Qd7-g7 is checkmate, defended by
        // the King on f6. Pre-fix, qsearch rejected it: the enemy King
        // "attacks" g7, so !square_attacked was false for every contact
        // check. Post-fix, the defended contact check is searched and the
        // mate score is returned.
        let mut pos = build_pos(
            &[
                (PieceKey::King, 0, 5, 5),
                (PieceKey::Queen, 0, 3, 6),
                (PieceKey::King, 1, 7, 7),
            ],
            0,
        );
        let mut s = Search::new();
        s.reset_for_think(5_000);
        s.evaluator = &NNUE_EVALUATOR;
        s.disable_acc_stack = true;
        let sc = qsearch(&mut s, &mut pos, 0, 1, -1_000_000_000, 1_000_000_000, QDEPTH, 2, None);
        assert!(
            sc >= MATE - 1000,
            "qsearch must find the defended contact mate Qg7, got score {sc}"
        );
    }

    #[test]
    fn test_warm_vs_cold_determinism() {
        // Rank 5 Proof: Running 10 prior positions on a warm `Search` instance then
        // searching target position P produces the exact same best move, score, and
        // node count as a freshly allocated `Search::new()` instance.
        let target = build_pos(
            &[
                (PieceKey::King, 0, 6, 0),
                (PieceKey::Rook, 0, 3, 0),
                (PieceKey::Bishop, 0, 2, 3),
                (PieceKey::Pawn, 0, 5, 1),
                (PieceKey::King, 1, 6, 7),
                (PieceKey::Rook, 1, 0, 7),
                (PieceKey::Knight, 1, 5, 5),
                (PieceKey::Pawn, 1, 6, 6),
            ],
            1,
        );

        let mut cold_s = Search::new();
        let mut p_cold = target.clone();
        let cold_out = crate::think::run_ranked_root_search(&mut cold_s, &mut p_cold, 1, 60_000, 3);

        let mut warm_s = Search::new();
        for i in 0..10u8 {
            let mut noise_pos = build_pos(
                &[
                    (PieceKey::King, 0, i % 6, 0),
                    (PieceKey::Queen, 0, (i + 2) % 8, 1),
                    (PieceKey::King, 1, 7 - (i % 6), 7),
                    (PieceKey::Rook, 1, (i + 3) % 8, 6),
                ],
                i & 1,
            );
            let _ = crate::think::run_ranked_root_search(&mut warm_s, &mut noise_pos, i & 1, 60_000, 2);
        }
        // Clear cross-turn game_history when comparing single-position cold vs warm determinism
        warm_s.game_history.clear();
        let mut p_warm = target.clone();
        let warm_out = crate::think::run_ranked_root_search(&mut warm_s, &mut p_warm, 1, 60_000, 3);

        assert_eq!(cold_out.legacy_pick, warm_out.legacy_pick, "Warm and cold best move must be identical");
        assert_eq!(cold_out.legacy_score, warm_out.legacy_score, "Warm and cold score must be identical");
        assert_eq!(cold_out.nodes, warm_out.nodes, "Warm and cold node counts must be identical");
    }

    #[test]
    fn test_occ_after_journaled_matches_fresh_occ() {
        // Lever-3 proof: for every journaled root move (captures,
        // promotions, Fish/EK-adjacent quiets), the derived child
        // occupancy must equal a fresh occ() scan of the child board.
        use crate::apply::make_journaled;
        use crate::journal::Journal;
        let pos0 = build_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::Rook, 0, 0, 0),
                (PieceKey::Fish, 0, 1, 0),
                (PieceKey::ElectroKnight, 0, 5, 2),
                (PieceKey::Pawn, 0, 4, 3),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Queen, 1, 3, 7),
                (PieceKey::Rook, 1, 0, 7),
                (PieceKey::Pawn, 1, 4, 6),
                (PieceKey::Pawn, 1, 3, 4),
            ],
            0,
        );
        let parent_occ = occ(&pos0);
        let moves = gen_moves(&pos0, 0, None);
        assert!(!moves.is_empty());
        let mut checked = 0usize;
        for mv in &moves {
            let mut p = pos0.clone();
            let mut j = Journal::default();
            let mark = j.mark();
            if make_journaled(&mut p, mv, 0, 1, &mut j) {
                assert_eq!(
                    occ_after_journaled(&parent_occ, mv),
                    occ(&p),
                    "derived occ must match fresh occ for move {mv:?}"
                );
                checked += 1;
                j.unwind(&mut p, mark);
            }
        }
        assert!(checked > 10, "expected several journaled moves, got {checked}");
    }
}
