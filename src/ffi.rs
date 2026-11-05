use crate::veto::{veto_score, VETO_THRESHOLD};
use crate::pos::*;
use crate::types::*;
use crate::movegen::gen_moves;
use crate::apply_full::apply_full;
use crate::attacks::{king_attacked, gives_check, has_legal_move};
use crate::nnue::{evaluate_dispatched, nnue_is_loaded, nnue_load};
use crate::search::{Search, abSearch};

// The browser host supplies the allocator import for the wasm32 ABI.
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" { fn malloc(s: usize) -> *mut u8; }

// The FFI uses 32-bit linear-memory pointers, so the native test build must
// not pretend a 64-bit host pointer can be returned through this ABI.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn malloc(_s: usize) -> *mut u8 { core::ptr::null_mut() }

static mut SLAB: Option<Vec<Option<Pos>>> = None;
static mut SEARCH: Option<Search> = None;

fn slab() -> &'static mut Vec<Option<Pos>> {
    unsafe {
        SLAB.get_or_insert_with(|| { let mut v = Vec::new(); v.push(None); v })
    }
}
fn search() -> &'static mut Search {
    unsafe { SEARCH.get_or_insert_with(Search::new) }
}

#[no_mangle]
pub extern "C" fn pxb_alloc(size: u32) -> u32 {
    let cap = (size as usize).max(16);
    let align = 8usize;
    if let Ok(layout) = std::alloc::Layout::from_size_align(cap, align) {
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if !ptr.is_null() {
            return ptr as u32;
        }
    }
    unsafe { malloc(size as usize) as u32 }
}

#[no_mangle]
pub extern "C" fn pxb_free(p: u32, s: u32) {
    if p == 0 || s == 0 {
        return;
    }
    let cap = (s as usize).max(16);
    if let Ok(layout) = std::alloc::Layout::from_size_align(cap, 8) {
        unsafe { std::alloc::dealloc(p as *mut u8, layout) };
    }
}

#[no_mangle]
pub extern "C" fn pxb_pos_create(buf_ptr: u32, buf_len: u32) -> u32 {
    let buf = unsafe { core::slice::from_raw_parts(buf_ptr as *const u8, buf_len as usize) };
    match crate::marshal::pos_from_buffer(buf) {
        Some(mut p) => {
            crate::eval::init_eval_sum(&mut p);
            p.hist_hashes.push(crate::eval::rep_hash(&p, p.stm));
            let s = slab();
            s.push(Some(p));
            (s.len() - 1) as u32
        }
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn pxb_pos_drop(h: u32) {
    let s = slab();
    if (h as usize) < s.len() { s[h as usize] = None; }
}

#[no_mangle]
pub extern "C" fn pxb_gen_moves(h: u32, seat: u8, stage: i32, out_ptr: u32, out_cap: u32) -> u32 {
    let s = slab();
    let pos = match s.get(h as usize).and_then(|o| o.as_ref()) { Some(p) => p, None => return 0 };
    let st = if stage < 0 { None } else { Some(stage as u8) };
    let mv = gen_moves(pos, seat, st);
    let n = mv.len().min(out_cap as usize);
    if n > 0 {
        let out = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut Move, n) };
        out.copy_from_slice(&mv[..n]);
    }
    n as u32
}

#[no_mangle]
pub extern "C" fn pxb_apply_move(h: u32, mv_ptr: u32) -> u32 {
    let s = slab();
    let pos = match s.get(h as usize).and_then(|o| o.as_ref()) { Some(p) => p.clone(), None => return 0 };
    let mv = unsafe { &*(mv_ptr as *const Move) };
    let player = mv.from.checked_index().and_then(|_| slot_of(&pos, mv.from))
        .map(|sl| pos.pieces[sl].player).unwrap_or(0);
    let opp = 1 - player;
    match apply_full(&pos, mv, player, opp) {
        Some(new_pos) => { s.push(Some(new_pos)); (s.len() - 1) as u32 }
        None => 0,
    }
}

trait CheckedIndex { fn checked_index(&self) -> Option<()>; }
impl CheckedIndex for u8 {
    #[inline] fn checked_index(&self) -> Option<()> { if *self == SQ_NONE { None } else { Some(()) } }
}

#[no_mangle]
pub extern "C" fn pxb_evaluate(h: u32, seat: u8) -> i32 {
    let s = slab();
    match s.get(h as usize).and_then(|o| o.as_ref()) { Some(p) => evaluate_dispatched(p, seat), None => 0 }
}

#[no_mangle]
pub extern "C" fn pxb_king_attacked(h: u32, player: u8, by: u8) -> u32 {
    let s = slab();
    match s.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => king_attacked(p, player, by, None) as u32, None => 0,
    }
}
#[no_mangle]
pub extern "C" fn pxb_gives_check(h: u32, me: u8, opp: u8) -> u32 {
    let s = slab();
    match s.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => gives_check(p, me, opp, None) as u32, None => 0,
    }
}
#[no_mangle]
pub extern "C" fn pxb_has_legal_move(h: u32, player: u8, opp: u8) -> u32 {
    let s = slab();
    match s.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => has_legal_move(p, player, opp) as u32, None => 0,
    }
}

#[no_mangle]
pub extern "C" fn pxb_perft(h: u32, depth: u8, seat: u8) -> u64 {
    let s = slab();
    let pos = match s.get(h as usize).and_then(|o| o.as_ref()) { Some(p) => p.clone(), None => return 0 };
    perft(&pos, depth, seat)
}
#[inline]
fn write_root_move(out_move_ptr: u32, mv: Move) -> bool {
    if out_move_ptr == 0 || (out_move_ptr as usize) % core::mem::align_of::<Move>() != 0 {
        return false;
    }
    unsafe {
        *(out_move_ptr as *mut Move) = mv;
    }
    true
}

fn perft(pos: &Pos, depth: u8, player: u8) -> u64 {
    if depth == 0 { return 1; }
    let opp = 1 - player;
    let moves = gen_moves(pos, player, None);
    let mut n = 0u64;
    for m in &moves {
        if let Some(p2) = apply_full(pos, m, player, opp) {
            if !king_attacked(&p2, player, opp, None) {
                n += perft(&p2, depth - 1, opp);
            }
        }
    }
    n
}

#[no_mangle]
pub extern "C" fn pxb_search_root(h: u32, seat: u8, budget_ms: u32, max_depth: u8,
                                   out_move_ptr: u32) -> i32 {
    // Never write a result through a null/misaligned pointer or derive an
    // opponent by subtracting from an invalid seat.
    if out_move_ptr == 0
        || (out_move_ptr as usize) % core::mem::align_of::<Move>() != 0
        || seat > 1
    {
        return i32::MIN;
    }
    let slab_ref = slab();
    let mut pos = match slab_ref.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => p.clone(),
        None => {
            unsafe { *(out_move_ptr as *mut Move) = Move::default(); }
            return i32::MIN;
        }
    };
    // Shared core with pxb_think: identical iterative deepening + veto
    // re-ranking, so the legacy pick can never drift from think's ranking.
    let s = search();
    let outcome = crate::think::run_ranked_root_search(s, &mut pos, seat, budget_ms as u64, max_depth);
    if outcome.no_moves {
        unsafe { *(out_move_ptr as *mut Move) = Move::default(); }
        return i32::MIN;
    }
    match outcome.legacy_pick {
        Some(mv) => {
            if !write_root_move(out_move_ptr, mv) {
                return i32::MIN;
            }
            outcome.legacy_score
        }
        None => {
            unsafe { *(out_move_ptr as *mut Move) = Move::default(); }
            i32::MIN
        }
    }
}

// ── ONE-ENGINE THINK (see docs/think-ffi.md) ─────────────────────────────
#[no_mangle]
pub extern "C" fn pxb_think(h: u32, seat: u8, clock_ms: u32, move_no: u32, max_depth: u8,
                             server_ptr: u32, server_count: u32,
                             out_ptr: u32, out_cap: u32) -> i32 {
    if out_ptr == 0
        || (out_cap as usize) < crate::think::THINK_HEADER_BYTES
        || seat > 1
    {
        return i32::MIN;
    }
    let slab_ref = slab();
    let mut pos = match slab_ref.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => p.clone(),
        None => return i32::MIN,
    };
    // Server list: null/zero-count means local mode (no filtering). A
    // supplied list filters; an empty result after filtering is a
    // server-mismatch (status NO_SERVER_LEGAL), never an invented move.
    let server: Option<Vec<Move>> = if server_ptr == 0 || server_count == 0 {
        None
    } else {
        let bytes = unsafe {
            core::slice::from_raw_parts(server_ptr as *const u8, (server_count as usize) * 16)
        };
        let mut v = Vec::with_capacity(server_count as usize);
        for chunk in bytes.chunks_exact(16) {
            match crate::think::move_from_bytes(chunk) {
                Some(mv) => v.push(mv),
                None => return i32::MIN,
            }
        }
        Some(v)
    };
    let s = search();
    // Rank 1: Clear persistent cross-turn game_history when a new game starts
    if move_no <= 2 || (s.last_game_move_no > 0 && move_no + 2 < s.last_game_move_no) {
        s.game_history.clear();
    }
    s.last_game_move_no = move_no;
    let budget = crate::think::think_budget_ms(clock_ms as u64, move_no);
    let outcome = crate::think::run_ranked_root_search_with_server(
        s,
        &mut pos,
        seat,
        budget,
        max_depth,
        server.as_deref(),
    );
    let (cands, status) =
        crate::think::think_candidates(s, &pos, seat, &outcome, server.as_deref());
    let buf = crate::think::serialize_think(
        &cands, status, outcome.depth, outcome.nodes, outcome.veto_checks,
        outcome.veto_rejects, out_cap as usize,
    );
    if !buf.is_empty() {
        let out = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut u8, buf.len()) };
        out.copy_from_slice(&buf);
    }
    if buf.len() >= crate::think::THINK_HEADER_BYTES {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as i32
    } else {
        0
    }
}

// ── Input / ability choice ────────────────────────────────────────────────
// Picks an option by engine material value (see think::choose_input).
// Returns the chosen option INDEX, or -1 when there are no options.
#[no_mangle]
pub extern "C" fn pxb_choose_input(opt_ptr: u32, opt_count: u32) -> i32 {
    if opt_ptr == 0 || opt_count == 0 {
        return -1;
    }
    let keys = unsafe { core::slice::from_raw_parts(opt_ptr as *const u16, opt_count as usize) };
    crate::think::choose_input(keys)
}

// ── NNUE weight loading (Phase 2 wiring — no longer inert) ───────────────
// pxb_nnue_load validates the blob (exact 5,244,932-byte size + finite
// weights, see src/nnue.rs) and, on success, makes the evaluator dispatch
// to the net: search leaves, think fallbacks, and pxb_evaluate all go
// through the single dispatch point. A rejected blob returns 0 and leaves
// any previously loaded weights untouched, so the engine keeps working
// on the previous scorer. With no weights loaded, behavior is EXACTLY
// the hand-written evaluator (proven by native tests + WASM probes
// against the revamp binary). A think pins its scorer at start (see
// think.rs), so a load completing mid-think cannot flip one search.
#[no_mangle]
pub extern "C" fn pxb_nnue_load(weights_ptr: u32, weights_len: u32) -> i32 {
    if weights_ptr == 0 || weights_len == 0 {
        return 0;
    }
    let bytes = unsafe {
        core::slice::from_raw_parts(weights_ptr as *const u8, weights_len as usize)
    };
    nnue_load(bytes) as i32
}

#[no_mangle]
pub extern "C" fn pxb_nnue_loaded() -> i32 {
    nnue_is_loaded() as i32
}

#[no_mangle]
pub extern "C" fn pxb_search_nodes() -> u64 {
    let s = search();
    s.nodes
}

/// Veto observability: number of root moves veto-checked in the last
/// pxb_search_root call. If this is 0, the veto never ran.
#[no_mangle]
pub extern "C" fn pxb_veto_checks() -> u64 {
    search().veto_checks
}

/// Veto observability: number of root moves REJECTED by the veto
/// (veto_score < VETO_THRESHOLD) in the last pxb_search_root call.
#[no_mangle]
pub extern "C" fn pxb_veto_rejects() -> u64 {
    search().veto_rejects
}

/// Deepest fully completed iterative-deepening depth from the last
/// pxb_search_root call. 0 means no depth completed (instant timeout).
#[no_mangle]
pub extern "C" fn pxb_search_depth() -> i32 {
    search().completed_depth
}

/// Search score of the selected root move from the last pxb_search_root call.
#[no_mangle]
pub extern "C" fn pxb_search_score() -> i32 {
    search().best_root_score
}

/// Direct veto check for a single move: returns veto_score(pos, move).
/// JS can call this to log/inspect why a move was rejected.
/// Returns i32::MIN if the position or move is invalid.
#[no_mangle]
pub extern "C" fn pxb_veto_move(h: u32, mv_ptr: u32, seat: u8) -> i32 {
    let s = slab();
    let pos = match s.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => p, None => return i32::MIN,
    };
    let mv = unsafe { &*(mv_ptr as *const Move) };
    let opp = 1 - seat;
    veto_score(pos, mv, seat, opp)
}

/// Returns VETO_THRESHOLD so JS can compare against pxb_veto_move results.
#[no_mangle]
pub extern "C" fn pxb_veto_threshold() -> i32 {
    VETO_THRESHOLD
}

/// Heartbeat / liveness probe for the supervised WASM lifecycle.
/// JS calls this before every move; if it throws or times out,
/// the instance is dead and must be re-instantiated.
/// Returns a constant so JS can verify the exports are functional.
#[no_mangle]
pub extern "C" fn pxb_ping() -> i32 {
    0x5058 // "PX" magic — proves the instance is alive and exports work
}

/// Observability: number of in-tree and root repetition hits detected in the last search.
#[no_mangle]
pub extern "C" fn pxb_rep_hits() -> u64 {
    search().rep_hits
}

/// Cold-reset all search state and cross-turn game history (e.g., on new game or test harness).
#[no_mangle]
pub extern "C" fn pxb_reset_cold() {
    search().reset_full_cold();
}

// ── SELF-PLAY TOOL EXPORTS (additive port of the proven tool-engine contract;
//    never shipped in the APK; used only by selfplay.js for training-data generation) ──
// pxb_pos_terminal: 0 = game ongoing, 1 = player 0 won terminally
// (king_thrown or gold_win), 2 = player 1 won terminally, -1 = bad handle.
#[no_mangle]
pub extern "C" fn pxb_pos_terminal(h: u32) -> i32 {
    let s = slab();
    let pos = match s.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => p,
        None => return -1,
    };
    if let Some(w) = pos.king_thrown.or(pos.gold_win) {
        return w as i32 + 1;
    }
    0
}

// pxb_pos_pieces: dumps the live board for FEN emission. Writes u32 count
// then count x 12-byte records: key u16, sub_key u16, player u8, x u8,
// y u8, alive u8, pad u16. Returns bytes written, or -1 on a bad handle
// or a buffer too small (caller must size 4 + 256*12).
// NOTE: only on-board pos.pieces are dumped; offboard pieces (Knightmare)
// are omitted, matching the old tool contract (no offboard FEN encoding).
#[no_mangle]
pub extern "C" fn pxb_pos_pieces(h: u32, out_ptr: u32, out_cap: u32) -> i32 {
    let s = slab();
    let pos = match s.get(h as usize).and_then(|o| o.as_ref()) {
        Some(p) => p,
        None => return -1,
    };
    let n = pos.pieces.len();
    let need = 4 + n * 12;
    if out_ptr == 0 || (out_cap as usize) < need {
        return -1;
    }
    let out = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut u8, need) };
    out[0..4].copy_from_slice(&(n as u32).to_le_bytes());
    for (i, p) in pos.pieces.iter().enumerate() {
        let o = 4 + i * 12;
        out[o..o + 2].copy_from_slice(&(p.key as u16).to_le_bytes());
        out[o + 2..o + 4].copy_from_slice(&(p.sub_key as u16).to_le_bytes());
        out[o + 4] = p.player;
        out[o + 5] = p.x;
        out[o + 6] = p.y;
        out[o + 7] = if p.alive { 1 } else { 0 };
    }
    need as i32
}
