# pxb_think FFI contract (revamp: one-engine call)

The wrapper is pure transport. All move generation, search, veto,
mate/hang safety, fallback, budget policy, and input/ability choice live
in the engine behind one entry point: `pxb_think`. The wrapper must
never generate, score, rank, veto, or judge a move in JS.

## Entries

### pxb_think

```c
int32_t pxb_think(
    uint32_t h,           // position handle from pxb_pos_create
    uint8_t  seat,        // 0 = White, 1 = Black; anything else -> i32::MIN
    uint32_t clock_ms,    // remaining clock for the side to move (see budget)
    uint32_t move_no,     // game move number (paces the budget)
    uint8_t  max_depth,   // iterative-deepening cap
    uint32_t server_ptr,  // pointer to server_count x 16-byte Moves, or 0
    uint32_t server_count,// number of server moves, or 0
    uint32_t out_ptr,     // output buffer (must hold >= 48 bytes)
    uint32_t out_cap      // output buffer capacity in bytes
);
```

Returns the number of candidate records written, or `i32::MIN` on
invalid arguments (null output pointer, `out_cap < 48`, bad seat, bad
position handle). On `i32::MIN` nothing is written to `out_ptr`.

### Server-list semantics (read carefully)

- `server_ptr == 0` OR `server_count == 0` means LOCAL MODE: no
  filtering; the full ranked list is returned. There is deliberately no
  way to express "the server list is empty" through this call.
- A NON-EMPTY supplied list filters the ranking to moves byte-identical
  to a server move (full 16-byte Move equality).
- If filtering leaves zero candidates, the result has status
  `THINK_NO_SERVER_LEGAL` (1) and count 0. The engine NEVER invents a
  move: on count 0 with status 1 the wrapper must RESYNC the board from
  the server and re-think once. If the resync still yields no legal
  server move, the wrapper must not send anything.
- A supplied list that is malformed (a 16-byte chunk that does not
  decode to a Move) returns `i32::MIN`.

### Budget policy (inside the engine; clock reading stays in JS)

`think_budget_ms(clock_ms, move_no)`:

- `clock_ms == 0` (blind clock): 4000 ms.
- Otherwise: `clock_ms / max(60 - move_no, 20)`, clamped to
  [400, 8000], then capped at `clock_ms / 3`.
- Tested values: (0, any) -> 4000; (473000, 0) -> 7883; (480000, 0) ->
  8000; (6000, 80) -> 400; (1000, 0) -> 333; (30000, 40) -> 1500;
  (120000, 55) -> 6000.
- Budget 0 (which forces the fallback path) arises only for tiny
  clocks: `clock_ms` 1 or 2 (cap `clock_ms / 3` = 0). `clock_ms == 0`
  means blind, not zero budget. `max_depth == 0` forces the fallback
  path at any clock.

### Output layout

Header, 48 bytes, little-endian:

| offset | type  | field                                      |
|--------|-------|--------------------------------------------|
| 0      | u32   | magic `0x5058`                             |
| 4      | u32   | version `1`                                |
| 8      | u32   | count: candidate records actually written  |
| 12     | i32   | status (see below)                         |
| 16     | i32   | search depth (deepest completed depth)     |
| 20     | i32   | best_score: first candidate's search_score, or 0 when count == 0 |
| 24     | u64   | nodes                                      |
| 32     | u64   | veto_checks                                |
| 40     | u64   | veto_rejects                               |

Then `count` records of 24 bytes each:

| offset | type | field                                                     |
|--------|------|-----------------------------------------------------------|
| 0      | Move | 16 bytes: flags u32, offboard_id u32, from u8 @8, to u8 @9, promo u8 @10, ep_victim u8 @11, special_a u8 @12, special_b u8 @13, push_dx i8 @14, push_dy i8 @15 |
| 16     | i32  | search_score (raw search score of the move)               |
| 20     | i32  | veto_score (soft penalty folded into the ranking; hard vetoes never appear) |

Ranking order is best-first by EFFECTIVE score
(`search_score + veto_score`, saturating). If `out_cap` cannot hold all
candidates, the best-first prefix is kept and `count` is the number
written. If `out_cap < 48`, the call fails with `i32::MIN` and writes
nothing (a header-only buffer is never produced).

### Status codes

| code | name                  | meaning |
|------|-----------------------|---------|
| 0    | THINK_OK              | ranked list from a completed search (possibly partial depth on timeout; the last completed depth's ranking is used) |
| 1    | THINK_NO_SERVER_LEGAL | supplied server list filtered every candidate out (resync + re-think; never invent a move) |
| 2    | THINK_NO_LEGAL_MOVES  | the position has no legal moves at all (count 0) |
| 3    | THINK_FALLBACK_ONLY   | no depth completed (zero budget or max_depth 0): a single fallback candidate — the veto-preferring legal root move, else the first legal move — statically scored through the engine evaluator and veto-scored. This is the engine's own root move, not a JS guess. |

### Search-core sharing (no second scorer)

`pxb_search_root` is refactored onto the SAME core
(`run_ranked_root_search`) that `pxb_think` uses: identical iterative
deepening, identical veto re-ranking, identical legacy pick
(`ranked[0]`; when every move is vetoed, the raw best from the last
completed depth; when no depth completed, the fallback). The legacy
getters (`pxb_search_depth`, `pxb_search_score`, `pxb_search_nodes`,
`pxb_veto_checks`, `pxb_veto_rejects`) still reflect the last search.
Cold-instance probes confirm think's top move and score equal
search_root's pick and score on all fixtures (P1, dtXQo1 turns 82/84/88);
see `probe-think-diag3.js`. (Warm-instance calls can differ between
any two searches — old engine included — because the transposition /
history tables persist across calls; that is pre-existing engine
behavior, not a think/search divergence.)

### pxb_choose_input (input / ability choice)

```c
int32_t pxb_choose_input(uint32_t opt_ptr, uint32_t opt_count);
```

`opt_ptr` points to `opt_count` little-endian u16 PieceKeys. Returns
the chosen option INDEX, or -1 when the list is null/empty. The choice
is the option with the highest engine material value
(`pixie_val` falling back to the standard value); ties keep the
earliest option. Unknown keys score 0, so a menu of only non-piece
options resolves to index 0 (the transport first-option behavior).
The wrapper sends the chosen option; it never picks one itself.

### NNUE weight loading (Phase 2 wiring — live, no longer inert)

```c
int32_t pxb_nnue_load(uint32_t weights_ptr, uint32_t weights_len);
int32_t pxb_nnue_loaded(void);
```

`pxb_nnue_load` accepts ONLY a valid model blob: exactly 5,244,932
bytes (5120x256 + 256 + 256 + 1 f32 LE, no header) with all weights
finite; anything else returns 0 and leaves previously loaded weights
untouched. On success, the evaluator dispatches to the net: search
leaves, think fallbacks, and `pxb_evaluate` all go through the single
dispatch point (`src/nnue.rs`), with probability mapped to centipawns
(see `docs/nnue-feature-schema.md`) and terminal states returning
MATE_CONST as before. With no weights loaded, behavior is EXACTLY the
hand-written evaluator — proven by native tests and by WASM probes
against the revamp binary (cold think numbers unchanged: P1 a7->a5
-383, T82 d1->b1 -1460, T84 f1->g1 -2553, T88 g1->g2 -2730). A think
pins its scorer at start (NNUE when loaded, handwritten otherwise),
so a load completing mid-think cannot flip one search. The old
inert-seam behavior (any multiple-of-4 length accepted, never
consulted) is gone: the revamp probe's 16-byte dummy load now
returns 0 by design. Unknown pieces contribute no NNUE input (see the
schema doc's fallback); the wrapper only fetches model bytes and
passes them to `pxb_nnue_load` — it never scores anything.

### Deleted exports

`pxb_evaluate_cheap` is DELETED. There is exactly one static-eval entry
point: the `Evaluator` trait (`HandwrittenEvaluator` today), used by
both search and think. No second scorer exists in the engine, and none
may exist in the wrapper.

### Wrapper emergency contract (JS side, not engine behavior)

- Think throws / WASM dead: re-instantiate the engine and retry the
  think inside the remaining budget.
- Last resort: send the first legal move from the LAST think result
  with NO scoring. If there is no last think result, send nothing.
- Count 0 with status 1: resync + re-think once (see server semantics).
- Metrics (lastOurEval, collapse detection) must be computed from WASM
  scores only, like-with-like; the collapse thresholds (400cp log,
  500cp severe-drop scan) are unchanged.
