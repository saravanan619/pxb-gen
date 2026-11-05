// ── ONE-ENGINE THINK ENTRY ─────────────────────────────────────────────────
// `pxb_think` is the single decision entry point of the engine. Inside one
// call it: generates moves, searches with the veto, applies mate/hang safety
// through that same veto, and falls back to a legal root move on failure or
// timeout. It returns a RANKED candidate list (best-first), already filtered
// to the server's legal moves when a server list is supplied. The wrapper is
// pure transport around this call: it never generates, scores, ranks, vetoes,
// or judges a move.
//
// Design notes (see docs/think-ffi.md for the wire contract):
// - The search + veto core here is the SAME procedure `pxb_search_root` has
//   always run (iterative deepening with completed-depth snapshots, then
//   veto re-ranking with soft penalties folded into an effective score).
//   `pxb_search_root` is refactored onto this core, so the legacy pick is
//   bit-for-bit what it was; think additionally preserves the whole ranked
//   list instead of only the pick.
// - Mate/hang safety is the veto itself, not a second scan: `veto_score`
//   covers king-into-check, hanging pieces, paranoid SEE, pin-mate threats,
//   and opponent mate-in-1 (see veto.rs). A candidate that fails the veto
//   never appears in the ranked list.
// - The fallback is `fallback_root_move` (veto-preferring legal move, else
//   first legal move), which is what the legacy path already pre-wrote on
//   zero-budget/timeout. Think returns it as a single-candidate list with
//   status FALLBACK_ONLY when search produced no ranked candidates.

use crate::apply_full::{apply_full, apply_full_no_wa};
use crate::attacks::king_attacked;
use crate::eval::{pixie_val, rep_hash, std_val, zobrist_u64, HANDWRITTEN_EVALUATOR};
use crate::nnue::{nnue_is_loaded, NNUE_EVALUATOR};
use crate::movegen::gen_moves;
use crate::pos::*;
use crate::search::{abSearch, Search};
use crate::types::*;
use crate::veto::{veto_score, VETO_THRESHOLD};

// ── Wire constants (docs/think-ffi.md) ─────────────────────────────────────
pub const THINK_MAGIC: u32 = 0x5058;
pub const THINK_VERSION: u32 = 1;
pub const THINK_HEADER_BYTES: usize = 48;
pub const THINK_RECORD_BYTES: usize = 24;

// Status codes (header field).
pub const THINK_OK: i32 = 0;
pub const THINK_NO_SERVER_LEGAL: i32 = 1;
pub const THINK_NO_LEGAL_MOVES: i32 = 2;
pub const THINK_FALLBACK_ONLY: i32 = 3;

// Blind-clock budget cap (ms) when the caller reports no clock, matching the
// wrapper's historical BLIND_SEARCH_MS.
pub const THINK_BLIND_BUDGET_MS: u64 = 4000;

// ── Budget policy ──────────────────────────────────────────────────────────
// Pure function, unit-tested below. Divides the remaining clock over an
// estimated number of moves left (60 minus moves played, floored at 20),
// clamped to [400ms, 6000ms], and never more than a third of the remaining
// clock on one move (so a low clock can never be flagged by one think).
// Unknown clock (0) means the server sent no clock data: use the blind cap.
pub fn think_budget_ms(clock_ms: u64, move_no: u32) -> u64 {
    if clock_ms == 0 {
        return THINK_BLIND_BUDGET_MS;
    }
    let remaining_moves = 60u64.saturating_sub(move_no as u64).max(20);
    let paced = clock_ms / remaining_moves;
    paced.clamp(400, 6000).min(clock_ms / 3)
}

// ── Ranked candidates ──────────────────────────────────────────────────────
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankedCandidate {
    pub mv: Move,
    pub search_score: i32,
    pub veto_score: i32,
}

impl RankedCandidate {
    pub fn effective(&self) -> i32 {
        self.search_score.saturating_add(self.veto_score)
    }
}

/// Rocketman teleport is RANDOM in the real game (Active button -> random
/// tile), so no single destination may outrank other moves: every teleport
/// candidate shares the mean effective score (the expected value of a random
/// landing). Without this the search cherry-picks the best square, which the
/// server will never deliver.
pub fn average_teleport_scores(ranked: &mut [RankedCandidate]) {
    let tel: Vec<i32> = ranked
        .iter()
        .filter(|c| c.mv.flags & MF_ROCKET_BLAST != 0)
        .map(|c| c.effective())
        .collect();
    if tel.len() <= 1 {
        return;
    }
    let mean = tel.iter().sum::<i32>() / tel.len() as i32;
    for c in ranked.iter_mut() {
        if c.mv.flags & MF_ROCKET_BLAST != 0 {
            // Keep effective() == mean by moving the difference into
            // search_score (the reported search value for a teleport IS its
            // expected value, so this stays honest in logs).
            c.search_score = mean - c.veto_score;
        }
    }
}

/// Random-effect moves whose outcome the player does NOT choose share their
/// group's mean effective score. Unlike Rocketman (one teleport choice, all
/// destinations random), an ElectroKnight capture is a player choice of
/// (from, to) while only the lightning target among the orthogonally
/// adjacent enemies is random, so the grouping key is (from, to): moves
/// with different destinations are different player choices and must NOT
/// be averaged together. The same grouping serves Pilgrim resurrect moves
/// (player chooses the Pilgrim move; the resurrected ally is random).
/// Grouping by (from, to) also keeps cluster averaging sound: the cluster
/// splits root PAIRS, so every special_a variant of one pair lands on the
/// same worker and the group is complete there. The chosen move's
/// special_a is search-only live: the wrapper sends only from/to and the
/// server picks the random target itself.
pub fn average_grouped_random_scores(ranked: &mut [RankedCandidate], flag: u32) {
    use std::collections::HashMap;
    let mut groups: HashMap<(u8, u8), Vec<i32>> = HashMap::new();
    for c in ranked.iter() {
        if c.mv.flags & flag != 0 {
            groups.entry((c.mv.from, c.mv.to)).or_default().push(c.effective());
        }
    }
    if groups.is_empty() { return; }
    let means: HashMap<(u8, u8), i32> = groups
        .iter()
        .filter(|(_, v)| v.len() > 1)
        .map(|(k, v)| (*k, v.iter().sum::<i32>() / v.len() as i32))
        .collect();
    if means.is_empty() { return; }
    for c in ranked.iter_mut() {
        if c.mv.flags & flag != 0 {
            if let Some(mean) = means.get(&(c.mv.from, c.mv.to)) {
                c.search_score = *mean - c.veto_score;
            }
        }
    }
}

pub struct RankedOutcome {
    pub ranked: Vec<RankedCandidate>,
    pub legacy_pick: Option<Move>,
    pub legacy_score: i32,
    pub fallback: Option<Move>,
    pub depth: i32,
    pub nodes: u64,
    pub veto_checks: u64,
    pub veto_rejects: u64,
    pub no_moves: bool,
}

// Choose a legal root move before searching so callers never receive
// uninitialized linear-memory bytes when the clock budget expires before
// depth 1 completes (or max_depth is zero). Prefer a move that passes the
// engine veto, but preserve a legal fallback if every move is vetoed.
// (Moved here verbatim from ffi.rs so think and the legacy path share it.)
pub(crate) fn fallback_root_move(pos: &Pos, seat: u8, opp: u8) -> Option<Move> {
    fallback_root_move_filtered(pos, seat, opp, None)
}

pub(crate) fn fallback_root_move_filtered(
    pos: &Pos,
    seat: u8,
    opp: u8,
    allowed: Option<&[Move]>,
) -> Option<Move> {
    let mut first_legal = None;
    for mv in gen_moves(pos, seat, None) {
        if let Some(list) = allowed {
            if !list.contains(&mv) {
                continue;
            }
        }
        let child = match apply_full_no_wa(pos, &mv, seat, opp) {
            Some(p) => p,
            None => continue,
        };
        if king_attacked(&child, seat, opp, None) {
            continue;
        }
        if first_legal.is_none() {
            first_legal = Some(mv);
        }
        if veto_score(pos, &mv, seat, opp) >= VETO_THRESHOLD {
            return Some(mv);
        }
    }
    first_legal
}

// Core ranked search: iterative deepening with completed-depth snapshots,
// then veto re-ranking. This is the procedure `pxb_search_root` has always
// run; it is shared (not duplicated) so the legacy pick and think's ranking
// can never drift apart.
/// Helper: count alive on-board pieces to detect a new game boundary automatically.
fn alive_board_pieces(pos: &Pos) -> usize {
    pos.pieces
        .iter()
        .filter(|p| p.alive && xy_on_board(p.x, p.y))
        .count()
}

/// Rank 1: Seed `pos.hist_hashes` with prior-turn hashes from `s.game_history`
/// so `rep_count` in `search.rs` detects cross-turn 2-fold and 3-fold repetitions
/// even when the JS caller sends `n_hist = 0`.
pub fn seed_pos_history_from_game(s: &mut Search, pos: &mut Pos, seat: u8) {
    let cur_pieces = alive_board_pieces(pos);
    // If the board suddenly has many more pieces than the previous turn (e.g., new game started),
    // clear `game_history` automatically.
    if s.last_game_piece_count > 0 && cur_pieces > s.last_game_piece_count + 4 {
        s.game_history.clear();
        s.last_root_move = [None, None];
    }
    s.last_game_piece_count = cur_pieces;

    let root_h64 = zobrist_u64(pos, seat);
    let root_h32 = rep_hash(pos, seat);
    if s.game_history.last().copied() != Some(root_h64) {
        s.game_history.push(root_h64);
    }
    if s.game_history.len() > 256 {
        let drop_n = s.game_history.len() - 256;
        s.game_history.drain(0..drop_n);
    }
    // Seed `pos.hist_hashes` with the low-32 Zobrist hashes from `game_history`
    // if the caller only provided the single root hash (`n_hist == 0`).
    if pos.hist_hashes.len() <= 1 {
        pos.hist_hashes.clear();
        for &gh in &s.game_history {
            pos.hist_hashes.push(gh as u32);
        }
        if pos.hist_hashes.last().copied() != Some(root_h32) {
            pos.hist_hashes.push(root_h32);
        }
    }
}

/// Interrupted-iteration rule: the last completed depth's root list wins,
/// unless no depth ever completed and the partial iteration did score some
/// root moves — then the partial list wins over a blind fallback move.
/// Returns true when the partial list should be kept.
fn keep_partial_root_list(completed: &[(Move, i32)], partial: &[(Move, i32)]) -> bool {
    completed.is_empty() && !partial.is_empty()
}

pub fn run_ranked_root_search(
    s: &mut Search,
    pos: &mut Pos,
    seat: u8,
    budget_ms: u64,
    max_depth: u8,
) -> RankedOutcome {
    run_ranked_root_search_with_server(s, pos, seat, budget_ms, max_depth, None)
}

pub fn run_ranked_root_search_with_server(
    s: &mut Search,
    pos: &mut Pos,
    seat: u8,
    budget_ms: u64,
    max_depth: u8,
    server: Option<&[Move]>,
) -> RankedOutcome {
    s.reset_for_think(budget_ms);
    s.root_filter = server.map(|sl| sl.to_vec());

    // Pin one evaluator for the whole think: NNUE when weights are
    // loaded, handwritten otherwise. A load completing mid-think must
    // not flip the scorer inside a single search (and the TT is cleared
    // per think, so no cross-scorer TT contamination either).
    s.evaluator = if nnue_is_loaded() {
        &NNUE_EVALUATOR
    } else {
        &HANDWRITTEN_EVALUATOR
    };
    s.init_root_accumulator(pos);
    seed_pos_history_from_game(s, pos, seat);

    let opp = 1 - seat;
    let fallback = fallback_root_move_filtered(pos, seat, opp, server)
        .or_else(|| fallback_root_move(pos, seat, opp));
    if fallback.is_none() {
        s.root_filter = None;
    }
    let fallback = match fallback {
        Some(mv) => mv,
        None => {
            return RankedOutcome {
                ranked: Vec::new(),
                legacy_pick: None,
                legacy_score: 0,
                fallback: None,
                depth: 0,
                nodes: 0,
                veto_checks: 0,
                veto_rejects: 0,
                no_moves: true,
            };
        }
    };

    // Snapshot of the last fully completed iteration. An interrupted
    // iteration may have only a partial root_scored list; never let that
    // partial list replace the best result from the previous completed depth.
    let mut completed_root_scored: Vec<(Move, i32)> = Vec::new();
    let mut completed_best_root: Option<Move> = None;
    let mut completed_best_root_score = 0i32;

    for depth in 1..=max_depth as i32 {
        if s.stop || crate::now_ms().saturating_sub(s.t0_ms) >= s.budget_ms {
            break;
        }
        s.best_root = None;
        s.root_scored.clear();
        let _d_score = abSearch(
            s, pos, depth, -1_000_000_000, 1_000_000_000, seat, opp, seat, 0, 0, None,
        );
        if !s.stop {
            s.completed_depth = depth;
            completed_root_scored = s.root_scored.clone();
            completed_best_root = s.best_root;
            completed_best_root_score = s.best_root_score;
        } else {
            // Interrupted iteration: restore the last completed depth's
            // results, unless no depth ever completed and this partial
            // iteration scored some root moves — discarding those would
            // force a blind fallback move even though searched moves exist
            // that still pass through veto re-ranking + the shuffle guard.
            if keep_partial_root_list(&completed_root_scored, &s.root_scored) {
                completed_root_scored = s.root_scored.clone();
                completed_best_root = s.best_root;
                completed_best_root_score = s.best_root_score;
                // completed_depth stays 0: no depth fully completed, but the
                // partial root scores flow into veto re-ranking below.
            } else {
                s.root_scored = completed_root_scored.clone();
                s.best_root = completed_best_root;
                s.best_root_score = completed_best_root_score;
            }
            break;
        }
    }

    // ── Veto re-ranking ──────────────────────────────────────────────
    // Sort ALL root moves from the last completed depth by SEARCH score
    // (descending), then keep the ones that pass the veto, ranked by
    // effective score (search + soft veto penalty). Hard vetoes
    // (veto_score < VETO_THRESHOLD: mate-in-1, king unsafe, hanging,
    // pin-mate) reject outright.
    let mut ranked: Vec<RankedCandidate> = Vec::new();
    let legacy_pick: Option<Move>;
    let legacy_score: i32;
    if !s.root_scored.is_empty() {
        s.root_scored.sort_by(|a, b| b.1.cmp(&a.1));
        for (rm, rscore) in s.root_scored.iter() {
            s.veto_checks += 1;
            let mut vs = veto_score(pos, rm, seat, opp);
            if vs < VETO_THRESHOLD {
                s.veto_rejects += 1;
                continue;
            }
            // Rank 1 Anti-Shuffle Guard: If playing `rm` immediately lands on a
            // position hash already seen in `s.game_history` (or reverses the last
            // quiet piece move back and forth), apply a -110cp repetition penalty per
            // prior occurrence so non-repeating developing moves win ties and near-ties.
            if let Some(child) = apply_full(pos, rm, seat, opp) {
                let child_h64 = zobrist_u64(&child, opp);
                let child_h32 = rep_hash(&child, opp);
                let seen_64 = s.game_history.iter().filter(|&&h| h == child_h64).count();
                let seen_32 = pos.hist_hashes.iter().filter(|&&h| h == child_h32).count();
                let repeats = seen_64.max(seen_32);
                if repeats >= 1 {
                    s.rep_hits += 1;
                    vs -= 110 * (repeats as i32);
                }
                // Direct 1-turn piece ping-pong guard (e.g., Anti-Violence jumping A -> B
                // then immediately B -> A on the next turn while not capturing or escaping attack):
                let seat_idx = (seat & 1) as usize;
                if let Some((prev_from, prev_to)) = s.last_root_move[seat_idx] {
                    let is_quiet = (rm.flags & (MF_CAPTURE | MF_PROMO | MF_IS_PUSH | MF_HORDE_SPAWN | MF_ELECTRO | MF_GUN_STANDOFF)) == 0;
                    if is_quiet && rm.from == prev_to && rm.to == prev_from {
                        let occ_before = occ(pos);
                        let was_attacked = crate::attacks::square_attacked(pos, rm.from, opp, &occ_before);
                        if !was_attacked {
                            s.rep_hits += 1;
                            vs -= 140;
                        }
                    }
                }
            }
            ranked.push(RankedCandidate {
                mv: *rm,
                search_score: *rscore,
                veto_score: vs,
            });
        }
        average_teleport_scores(&mut ranked);
        // NOTE (Sandbox-verified 2026-11-03): ElectroKnight lightning is
        // DETERMINISTIC (first destructible neighbour clockwise from north),
        // so EK moves are NOT averaged here anymore. Movegen emits exactly
        // one EK move per capture target; averaging would only be correct
        // for a random outcome, which the Sandbox refuted. Pilgrim
        // resurrect variants ARE still averaged (the real game picks the
        // revived ally at random).
        average_grouped_random_scores(&mut ranked, MF_PILGRIM_RESURRECT);
        ranked.sort_by(|a, b| b.effective().cmp(&a.effective()));
        if let Some(top) = ranked.first() {
            legacy_pick = Some(top.mv);
            legacy_score = top.search_score;
            s.best_root = Some(top.mv);
            s.best_root_score = top.search_score;
            s.last_root_move[(seat & 1) as usize] = Some((top.mv.from, top.mv.to));
            // Record the chosen child position in `s.game_history` so the next turn
            // knows the exact board state we played into!
            if let Some(chosen_child) = apply_full(pos, &top.mv, seat, opp) {
                let ch64 = zobrist_u64(&chosen_child, opp);
                s.game_history.push(ch64);
            }
        } else {
            // Every root move vetoed: legacy behavior keeps the original
            // best_root from the last completed depth.
            legacy_pick = completed_best_root;
            legacy_score = s.best_root_score;
        }
    } else if let Some(bm) = s.best_root {
        // Legacy edge path: no scored root moves (e.g. depth-1 timeout
        // edge). Veto-check just the best move; if it is vetoed, scan the
        // full move list for the best veto-passing alternative.
        s.veto_checks += 1;
        let vscore = veto_score(pos, &bm, seat, opp);
        if vscore < VETO_THRESHOLD {
            s.veto_rejects += 1;
            let root_moves = gen_moves(pos, seat, None);
            let mut best_alt: Option<Move> = None;
            let mut best_vscore = i32::MIN;
            for rm in root_moves {
                s.veto_checks += 1;
                let vs = veto_score(pos, &rm, seat, opp);
                if vs >= VETO_THRESHOLD && vs > best_vscore {
                    best_vscore = vs;
                    best_alt = Some(rm);
                }
            }
            legacy_pick = best_alt.or(Some(bm));
        } else {
            legacy_pick = Some(bm);
        }
        legacy_score = s.best_root_score;
    } else {
        // No depth completed: legacy behavior leaves the pre-written
        // fallback as the move.
        legacy_pick = Some(fallback);
        legacy_score = 0;
    }

    s.root_filter = None;
    RankedOutcome {
        ranked,
        legacy_pick,
        legacy_score,
        fallback: Some(fallback),
        depth: s.completed_depth,
        nodes: s.nodes,
        veto_checks: s.veto_checks,
        veto_rejects: s.veto_rejects,
        no_moves: false,
    }
}

// ── Think candidate construction ───────────────────────────────────────────
// Turns a ranked outcome into the candidate list think returns:
// veto-passing ranked moves when search produced them; otherwise the single
// fallback move (statically scored through the same evaluator — never a
// second scorer); nothing when there are no legal moves. When a server list
// is supplied, the list is filtered to it; an empty result after filtering
// is a server-mismatch (status NO_SERVER_LEGAL), never an invented move.
pub fn think_candidates(
    s: &Search,
    pos: &Pos,
    seat: u8,
    outcome: &RankedOutcome,
    server: Option<&[Move]>,
) -> (Vec<RankedCandidate>, i32) {
    if outcome.no_moves {
        return (Vec::new(), THINK_NO_LEGAL_MOVES);
    }
    let opp = 1 - seat;
    let (mut cands, mut status) = if !outcome.ranked.is_empty() {
        (outcome.ranked.clone(), THINK_OK)
    } else {
        match outcome.fallback {
            Some(fb) => {
                let (search_score, veto) = match apply_full(pos, &fb, seat, opp) {
                    Some(child) => (s.evaluator.evaluate(&child, seat), veto_score(pos, &fb, seat, opp)),
                    None => (0, veto_score(pos, &fb, seat, opp)),
                };
                (
                    vec![RankedCandidate {
                        mv: fb,
                        search_score,
                        veto_score: veto,
                    }],
                    THINK_FALLBACK_ONLY,
                )
            }
            None => (Vec::new(), THINK_NO_LEGAL_MOVES),
        }
    };
    if let Some(list) = server {
        // RUN16: the server contract is (from,to) squares — server pairs
        // carry no variant fields, and the wrapper sends from/to only.
        // Exact full-field equality falsely rejected ranked candidates
        // whose variant fields (special_b, push, ep_victim, offboard_id)
        // differed from the gen copy match_group stored, even though the
        // squares matched a server pair (Codex consult + the same_move
        // asymmetry: same_move compares 5 fields, derived PartialEq 10).
        // Retain by squares; log any candidate that survives by squares
        // but not by exact equality, so variant conflation stays visible
        // in the service log instead of silently passing.
        let mut kept: Vec<RankedCandidate> = Vec::with_capacity(cands.len());
        for c in cands.into_iter() {
            if list.contains(&c.mv) {
                kept.push(c);
                continue;
            }
            if let Some(m) = list.iter().find(|m| m.from == c.mv.from && m.to == c.mv.to) {
                eprintln!("[think-candidates] square-match-only from={} to={} cand=(flags={} promo={} ep={} sp=({},{}) push=({},{}) off={}) list=(flags={} promo={} ep={} sp=({},{}) push=({},{}) off={})",
                    c.mv.from, c.mv.to,
                    c.mv.flags, c.mv.promo, c.mv.ep_victim, c.mv.special_a, c.mv.special_b, c.mv.push_dx, c.mv.push_dy, c.mv.offboard_id,
                    m.flags, m.promo, m.ep_victim, m.special_a, m.special_b, m.push_dx, m.push_dy, m.offboard_id);
                kept.push(c);
            }
        }
        cands = kept;
        if cands.is_empty() {
            return (Vec::new(), THINK_NO_SERVER_LEGAL);
        }
        if status == THINK_FALLBACK_ONLY {
            // Fallback survived filtering: it is still the only candidate.
            status = THINK_FALLBACK_ONLY;
        }
    }
    (cands, status)
}

// ── Move byte codec (alignment-proof) ──────────────────────────────────────
// Move is #[repr(C)] 16 bytes with no padding (flags u32, offboard_id u32,
// from/to/promo/ep_victim/special_a/special_b u8, push_dx/push_dy i8), but
// host buffers are never assumed aligned: both directions go through bytes.
pub fn move_to_bytes(mv: &Move) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&mv.flags.to_le_bytes());
    b[4..8].copy_from_slice(&mv.offboard_id.to_le_bytes());
    b[8] = mv.from;
    b[9] = mv.to;
    b[10] = mv.promo;
    b[11] = mv.ep_victim;
    b[12] = mv.special_a;
    b[13] = mv.special_b;
    b[14] = mv.push_dx as u8;
    b[15] = mv.push_dy as u8;
    b
}

pub fn move_from_bytes(b: &[u8]) -> Option<Move> {
    if b.len() < 16 {
        return None;
    }
    Some(Move {
        flags: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        offboard_id: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        from: b[8],
        to: b[9],
        promo: b[10],
        ep_victim: b[11],
        special_a: b[12],
        special_b: b[13],
        push_dx: b[14] as i8,
        push_dy: b[15] as i8,
    })
}

// ── Think result serialization ─────────────────────────────────────────────
// Header (48 bytes, little-endian):
//   u32 magic (0x5058), u32 version (1), u32 count (records written),
//   i32 status, i32 depth, i32 best_score (first candidate's search score,
//   or 0 when empty), u64 nodes, u64 veto_checks, u64 veto_rejects.
// Records (24 bytes each, best-first): Move 16B, i32 search_score,
// i32 veto_score (effective score = search_score + veto_score).
// If the caller's buffer is too small for all records, the best-first prefix
// that fits is written; count reports records actually written, so truncation
// drops the worst candidates, never the best.
pub fn serialize_think(
    cands: &[RankedCandidate],
    status: i32,
    depth: i32,
    nodes: u64,
    veto_checks: u64,
    veto_rejects: u64,
    out_cap: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    if out_cap < THINK_HEADER_BYTES {
        return out;
    }
    let writable = (out_cap - THINK_HEADER_BYTES) / THINK_RECORD_BYTES;
    let n = cands.len().min(writable);
    let best_score = cands.first().map(|c| c.search_score).unwrap_or(0);
    out.extend_from_slice(&THINK_MAGIC.to_le_bytes());
    out.extend_from_slice(&THINK_VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&status.to_le_bytes());
    out.extend_from_slice(&depth.to_le_bytes());
    out.extend_from_slice(&best_score.to_le_bytes());
    out.extend_from_slice(&nodes.to_le_bytes());
    out.extend_from_slice(&veto_checks.to_le_bytes());
    out.extend_from_slice(&veto_rejects.to_le_bytes());
    debug_assert_eq!(out.len(), THINK_HEADER_BYTES);
    for c in &cands[..n] {
        out.extend_from_slice(&move_to_bytes(&c.mv));
        out.extend_from_slice(&c.search_score.to_le_bytes());
        out.extend_from_slice(&c.veto_score.to_le_bytes());
    }
    out
}

// ── Input / ability choice ─────────────────────────────────────────────────
// Picks an option by engine material value (highest wins, ties keep the
// earliest option). Option values are engine PieceKey codes (u16): this
// reproduces the legacy "prefer queen" behavior through values instead of
// label matching (Queen outvalues every other piece). Returns the chosen
// option INDEX, or -1 when there are no options. Non-piece options cannot
// be decided by the engine; the caller keeps the legacy first-option
// fallback for those (transport default, no scoring involved).
fn key_value(k: PieceKey) -> i32 {
    if let Some(v) = pixie_val(k) {
        return v;
    }
    if let Some(v) = std_val(k) {
        return v;
    }
    0
}

pub fn choose_input(option_keys: &[u16]) -> i32 {
    if option_keys.is_empty() {
        return -1;
    }
    let mut best_idx = 0i32;
    let mut best_val = i32::MIN;
    for (i, &raw) in option_keys.iter().enumerate() {
        let v = key_value(PieceKey::from_u16(raw));
        if v > best_val {
            best_val = v;
            best_idx = i as i32;
        }
    }
    best_idx
}

#[cfg(test)]
mod think_tests {
    use super::*;
    use crate::eval::init_eval_sum;

    fn test_pos(pieces: &[(PieceKey, u8, u8, u8)], stm: u8) -> Pos {
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
    fn budget_policy_cases() {
        // (clock_ms, move_no) -> budget_ms
        assert_eq!(think_budget_ms(0, 0), 4000); // blind cap
        assert_eq!(think_budget_ms(473_000, 0), 6000); // 473000/60, capped at 6000
        assert_eq!(think_budget_ms(480_000, 0), 6000); // clamped
        assert_eq!(think_budget_ms(6_000, 80), 400); // low clock floor
        assert_eq!(think_budget_ms(1_000, 0), 333); // never > clock/3
        assert_eq!(think_budget_ms(30_000, 40), 1500); // 30000/20
        assert_eq!(think_budget_ms(120_000, 55), 6000); // 120000/20
    }

    #[test]
    fn move_codec_roundtrip() {
        let mv = Move {
            flags: 0x12345678,
            offboard_id: 42,
            from: 12,
            to: 28,
            promo: 5,
            ep_victim: 0,
            special_a: 3,
            special_b: 0,
            push_dx: -1,
            push_dy: 2,
        };
        assert_eq!(move_from_bytes(&move_to_bytes(&mv)).unwrap(), mv);
        assert!(move_from_bytes(&[0u8; 15]).is_none());
    }

    #[test]
    fn think_basic_ranking_matches_legacy_pick() {
        // White: K e1, Q d1, R a1, P e2. Black: K e8, R a8, P e7.
        let mut pos = test_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::Rook, 0, 0, 0),
                (PieceKey::Pawn, 0, 4, 1),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Rook, 1, 0, 7),
                (PieceKey::Pawn, 1, 4, 6),
            ],
            0,
        );
        let mut s = Search::new();
        let outcome = run_ranked_root_search(&mut s, &mut pos, 0, 500, 4);
        assert!(!outcome.no_moves);
        assert!(!outcome.ranked.is_empty());
        // Ranked best-first by effective score.
        for w in outcome.ranked.windows(2) {
            assert!(w[0].effective() >= w[1].effective());
        }
        // Every ranked candidate passes the veto.
        for c in &outcome.ranked {
            assert!(c.veto_score >= VETO_THRESHOLD);
        }
        // Think top == legacy pick, with the same search score.
        let (cands, status) = think_candidates(&s, &pos, 0, &outcome, None);
        assert_eq!(status, THINK_OK);
        assert_eq!(cands[0].mv, outcome.legacy_pick.unwrap());
        assert_eq!(cands[0].search_score, outcome.legacy_score);
    }

    #[test]
    fn think_veto_excludes_king_into_check() {
        // White King e4, Black Bishop d6 attacks e5; Black King a8.
        // King move e4->e5 walks into check and must never appear ranked,
        // whether the search or the veto layer excludes it.
        let mut pos = test_pos(
            &[
                (PieceKey::King, 0, 4, 3),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::King, 1, 0, 7),
                (PieceKey::Bishop, 1, 3, 5),
            ],
            0,
        );
        // Direct veto check: e4 = sq 28, e5 = sq 36.
        let king_step = Move {
            from: 28,
            to: 36,
            ..Default::default()
        };
        assert!(
            veto_score(&pos, &king_step, 0, 1) < VETO_THRESHOLD,
            "veto must hard-reject a king step into an attacked square"
        );
        let mut s = Search::new();
        let outcome = run_ranked_root_search(&mut s, &mut pos, 0, 500, 4);
        assert!(!outcome.no_moves);
        for c in &outcome.ranked {
            assert!(
                !(c.mv.from == 28 && c.mv.to == 36),
                "king-into-check move must never appear in the ranking"
            );
        }
    }

    #[test]
    fn think_server_filter_and_mismatch() {
        let mut pos = test_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::King, 1, 4, 7),
                (PieceKey::Rook, 1, 0, 7),
            ],
            0,
        );
        let mut s = Search::new();
        let outcome = run_ranked_root_search(&mut s, &mut pos, 0, 500, 4);
        assert!(!outcome.ranked.is_empty());
        // Restrict to the top move only: exactly one candidate survives.
        let top = outcome.ranked[0].mv;
        let (cands, status) = think_candidates(&s, &pos, 0, &outcome, Some(&[top]));
        assert_eq!(status, THINK_OK);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].mv, top);
        // A server list with no ranked move in it is a mismatch, never an
        // invented move.
        let alien = Move {
            from: 0,
            to: 1,
            ..Default::default()
        };
        let (cands, status) = think_candidates(&s, &pos, 0, &outcome, Some(&[alien]));
        assert_eq!(status, THINK_NO_SERVER_LEGAL);
        assert!(cands.is_empty());
    }

    #[test]
    fn think_server_filter_matches_by_squares_not_full_fields() {
        // RUN16: a ranked candidate whose variant fields differ from the
        // gen copy (e.g. push_dx, special_b) must still survive when its
        // squares match a server pair — the server contract is squares.
        let pos = test_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let s = Search::new();
        let ranked_mv = Move {
            from: 4,
            to: 5,
            push_dx: 1,
            special_b: 7,
            ..Default::default()
        };
        let listed_mv = Move {
            from: 4,
            to: 5,
            ..Default::default()
        };
        let outcome = RankedOutcome {
            ranked: vec![RankedCandidate {
                mv: ranked_mv,
                search_score: 10,
                veto_score: 0,
            }],
            legacy_pick: Some(ranked_mv),
            legacy_score: 10,
            fallback: None,
            depth: 1,
            nodes: 1,
            veto_checks: 0,
            veto_rejects: 0,
            no_moves: false,
        };
        let (cands, status) = think_candidates(&s, &pos, 0, &outcome, Some(&[listed_mv]));
        assert_eq!(status, THINK_OK);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].mv, ranked_mv);
    }

    #[test]
    fn think_zero_budget_falls_back_to_legal_move() {
        let mut pos = test_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Queen, 0, 3, 0),
                (PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let mut s = Search::new();
        let outcome = run_ranked_root_search(&mut s, &mut pos, 0, 0, 12);
        assert!(outcome.ranked.is_empty());
        assert!(outcome.fallback.is_some());
        let (cands, status) = think_candidates(&s, &pos, 0, &outcome, None);
        assert_eq!(status, THINK_FALLBACK_ONLY);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].mv, outcome.fallback.unwrap());
    }

    #[test]
    fn interrupted_depth1_keeps_partial_root_list() {
        let partial = vec![(Move { from: 12, to: 28, ..Default::default() }, 42)];
        assert!(keep_partial_root_list(&[], &partial));
    }

    #[test]
    fn interrupted_later_depth_restores_completed_list() {
        let completed = vec![(Move { from: 12, to: 28, ..Default::default() }, 42)];
        let partial = vec![(Move { from: 13, to: 29, ..Default::default() }, 10)];
        assert!(!keep_partial_root_list(&completed, &partial));
    }

    #[test]
    fn interrupted_empty_iteration_keeps_nothing() {
        let empty: Vec<(Move, i32)> = Vec::new();
        assert!(!keep_partial_root_list(&empty, &empty));
    }

    #[test]
    fn choose_input_prefers_queen_and_first_on_ties() {
        use PieceKey::*;
        assert_eq!(choose_input(&[]), -1);
        assert_eq!(choose_input(&[Rook as u16, Queen as u16, Knight as u16]), 1);
        assert_eq!(choose_input(&[Knight as u16, Knight as u16]), 0);
        assert_eq!(choose_input(&[999, Queen as u16]), 1);
    }

    #[test]
    fn serialize_think_layout_and_truncation() {
        let mv = Move {
            from: 12,
            to: 28,
            ..Default::default()
        };
        let cands = vec![
            RankedCandidate {
                mv,
                search_score: 250,
                veto_score: -10,
            },
            RankedCandidate {
                mv,
                search_score: 100,
                veto_score: 0,
            },
        ];
        let buf = serialize_think(&cands, THINK_OK, 9, 12345, 24, 1, 10_000);
        assert_eq!(buf.len(), THINK_HEADER_BYTES + 2 * THINK_RECORD_BYTES);
        assert_eq!(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), THINK_MAGIC);
        assert_eq!(u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]), 2);
        assert_eq!(i32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]), 9);
        assert_eq!(i32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]), 250);
        // Truncation to one record keeps the best-first prefix.
        let buf2 = serialize_think(&cands, THINK_OK, 9, 12345, 24, 1, THINK_HEADER_BYTES + THINK_RECORD_BYTES);
        assert_eq!(buf2.len(), THINK_HEADER_BYTES + THINK_RECORD_BYTES);
        assert_eq!(u32::from_le_bytes([buf2[8], buf2[9], buf2[10], buf2[11]]), 1);
        // Header-only buffer request yields nothing (caller must size >= 48).
        assert!(serialize_think(&cands, THINK_OK, 9, 0, 0, 0, 47).is_empty());
    }

    #[test]
    fn test_root_avoids_game_history_shuffle() {
        // Rank 1 Proof: Replay Game 153 Turn 39 (Black to move after 37...Ra8->a7).
        // Seed `s.game_history` with the Turn 37 position where Black's Rook was on a8.
        // Verify that `Ra7->a8` (from=48, to=56) is penalized for repeating game history
        // and never selected as the top ranked candidate.
        let pos_t37 = test_pos(
            &[
                (PieceKey::King, 0, 6, 0),
                (PieceKey::Rook, 0, 3, 0),
                (PieceKey::Bishop, 0, 2, 3),
                (PieceKey::Pawn, 0, 5, 1),
                (PieceKey::King, 1, 6, 7),
                (PieceKey::Rook, 1, 0, 7), // Ra8
                (PieceKey::Bishop, 1, 5, 7), // Bf8
                (PieceKey::Pilgrim, 1, 2, 7), // Pic8
                (PieceKey::Queen, 1, 3, 7), // Qd8
                (PieceKey::Pawn, 1, 5, 6), // f7 (shields Kg8 from Bc4 check so Ra7->a8 is legal)
            ],
            1,
        );
        let pos_t39 = test_pos(
            &[
                (PieceKey::King, 0, 6, 0),
                (PieceKey::Rook, 0, 3, 0),
                (PieceKey::Bishop, 0, 2, 3),
                (PieceKey::Pawn, 0, 5, 1),
                (PieceKey::King, 1, 6, 7),
                (PieceKey::Rook, 1, 0, 6), // Ra7
                (PieceKey::Bishop, 1, 5, 7),
                (PieceKey::Pilgrim, 1, 2, 7),
                (PieceKey::Queen, 1, 3, 7),
                (PieceKey::Pawn, 1, 5, 6), // f7 (shields Kg8 from Bc4 check so Ra7->a8 is legal)
            ],
            1,
        );

        let mut s = Search::new();
        // Record Turn 37 position (and the post-Ra7->a8 position with White to move) in game_history
        s.game_history.push(zobrist_u64(&pos_t37, 1));
        s.game_history.push(zobrist_u64(&pos_t37, 0));

        let mut p = pos_t39.clone();
        let out = run_ranked_root_search(&mut s, &mut p, 1, 1_500, 3);
        let top = out.ranked.first().expect("must have ranked moves");
        assert!(
            !(top.mv.from == 48 && top.mv.to == 56),
            "Black must not shuffle Ra7->a8 (48->56); picked {}->{}",
            top.mv.from,
            top.mv.to
        );
        assert!(s.rep_hits > 0, "Repetition detector must record hits on the shuffle line");
    }

    #[test]
    fn teleport_candidates_share_mean_score() {
        use crate::types::MF_ROCKET_BLAST;
        let mut ranked = vec![
            RankedCandidate {
                mv: Move { from: 0, to: 10, flags: MF_ROCKET_BLAST, ..Default::default() },
                search_score: 100, veto_score: 0,
            },
            RankedCandidate {
                mv: Move { from: 0, to: 20, flags: MF_ROCKET_BLAST, ..Default::default() },
                search_score: 300, veto_score: 0,
            },
            RankedCandidate {
                mv: Move { from: 1, to: 9, flags: 0, ..Default::default() },
                search_score: 200, veto_score: 0,
            },
        ];
        average_teleport_scores(&mut ranked);
        assert_eq!(ranked[0].effective(), 200, "teleport must score its mean, not its best square");
        assert_eq!(ranked[1].effective(), 200);
        assert_eq!(ranked[2].effective(), 200, "non-teleport scores must be untouched");
    }

    // NOTE (Sandbox-verified 2026-11-03): the old electro-averaging test
    // was retired with the random-zap model. EK lightning is deterministic
    // (first destructible neighbour clockwise from north, exactly one move
    // per capture target), so there is nothing to average. The grouping
    // function itself is still covered by the Pilgrim test below.

    #[test]
    fn pilgrim_resurrect_candidates_share_mean_per_move() {
        use crate::types::MF_PILGRIM_RESURRECT;
        let mut ranked = vec![
            RankedCandidate {
                mv: Move { from: 0, to: 10, flags: MF_PILGRIM_RESURRECT, special_a: 5, ..Default::default() },
                search_score: 100, veto_score: 0,
            },
            RankedCandidate {
                mv: Move { from: 0, to: 10, flags: MF_PILGRIM_RESURRECT, special_a: 6, ..Default::default() },
                search_score: 300, veto_score: 0,
            },
        ];
        average_grouped_random_scores(&mut ranked, MF_PILGRIM_RESURRECT);
        assert_eq!(ranked[0].effective(), 200, "random resurrect targets share their mean");
        assert_eq!(ranked[1].effective(), 200);
    }

    #[test]
    fn filtered_single_move_slice_searches_past_depth_2() {
        let mut pos = test_pos(
            &[
                (PieceKey::King, 0, 4, 0),
                (PieceKey::Rook, 0, 0, 0),
                (PieceKey::King, 1, 4, 7),
            ],
            0,
        );
        let slice = vec![Move { from: sq(0, 0), to: sq(0, 1), ..Default::default() }];
        let mut s = Search::new();
        let out = run_ranked_root_search_with_server(&mut s, &mut pos, 0, 5_000, 5, Some(&slice));
        assert_eq!(out.ranked.len(), 1, "root_filter must restrict search to the 1-move worker slice");
        assert!(out.depth >= 4, "1-move worker slice must not early-exit at depth 2, got depth {}", out.depth);
    }
}
