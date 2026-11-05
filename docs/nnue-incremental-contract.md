# NNUE incremental-update contract (contract + Phase 2 implementation)

A production NNUE evaluator cannot recompute the full feature vector
on every search node (at ~400K nodes per move, a full 2*TABLE*64 x
hidden matmul per node is billions of operations). It must maintain an
accumulator updated incrementally on move apply and unwound on undo.
This document defines the CONTRACT that `apply_full` / the journal
must satisfy, and records the Phase 2 implementation status below. The
accumulator itself now EXISTS (`src/accumulator.rs`); threading it
through search recursion is the remaining follow-up (see "What exists
today").

## Delta-list contract

Every move application must be expressible as an ordered list of
square-occupancy deltas. Each delta is one of:

- `Moved { key, from_sq, to_sq }` — a piece changed squares. `key` is
  the piece's identity (Pixie identity where applicable, i.e. the
  `sub_key` when present, else the base key).
- `Captured { key, sq }` — a piece left the board from `sq` (including
  blast/area removals and doomed expirations that resolve as part of
  the move, not only direct captures).
- `Transformed { old_key, new_key, sq }` — a piece on `sq` changed
  identity without moving (promotion, ability choice, phase-state
  changes that alter the feature row).
- `SideFlip` — exactly one per move, applied last: the side to move
  changed, which swaps the stm-relative planes.

Rules:

- Pixie moves can move MULTIPLE pieces (pushes, carries, swaps,
  duplication side effects): every affected piece gets its own delta.
  A contract implementation that reports only the "main" piece is
  WRONG.
- Deltas are ordered: captures of pieces that vacate a destination
  come before the mover's arrival on that square, so the accumulator
  never transiently holds two pieces on one square.
- Undo applies the INVERSE delta list in REVERSE order and must
  restore the exact prior accumulator state (bit-for-bit, not
  approximately — accumulators must not drift over a search).
- The side-to-move flip is part of the delta list, not implicit: the
  accumulator's plane mapping depends on it, and a missed flip silently
  mirrors every feature.

## What exists today vs. what phase 2 must add

- The accumulator EXISTS now (`src/accumulator.rs`): dual-perspective
  accumulators (white-stm view + black-stm view, so SideFlip is an O(1)
  perspective swap), delta application with exact undo (inverse deltas
  in reverse order), and a parity suite proving incremental == full
  refresh (ordinary moves, captures, promotions, castles, HordeMother
  spawns, journaled moves, and a 200-case random-board fuzz).
- Delta emission EXISTS for both apply paths, diff-derived from the
  actual before/after boards (exact by construction, O(pieces)):
  `apply_full_with_deltas` for the clone-based path, and
  `deltas_for_journaled_move` for the journaled path (clones the
  post-move board, unwinds the clone to the pre-move board, diffs).
  The delta vocabulary extends the contract in two documented ways:
  `Spawned` (creation has no from-square) and a `player` field on
  every piece delta (stm-relative planes need the owning side).
- NOT yet done (follow-up, stated plainly, not silently substituted):
  threading accumulator state through the abSearch/qsearch recursion.
  The evaluator trait is stateless (`evaluate(&self, pos, me)`), and
  search mixes journaled in-place apply with clone-based apply_full,
  so per-node accumulator save/restore across both paths is a larger
  change than the wiring itself. Until that threading lands, the NNUE
  evaluator scores leaves through the sparse full forward (active
  features only, <=32 rows x 256) — already cheaper than the
  hand-written evaluator's attack-map machinery — and the accumulator
  stands proven but unused by search. The parity gate the contract
  demanded (incremental == full) is met; the search-threading gate is
  the remaining step before the accumulator pays off per node.
