# NNUE feature schema, version 1 (SPEC — implemented by Phase 2 wiring)

This document defines the feature layout the NNUE evaluator uses. The
Phase 2 trainer (~/workspace/nnue-phase2/, SPEC.md v1 frozen 2026-10-24)
trained against exactly this layout, and the engine wiring
(`src/nnue.rs`, `src/accumulator.rs`) implements it. Phase 2's verified
inventory below is the shared table; the trainer's CODES list
(`train/src/bin/pretokenize.rs`) and this table are in the same order
with the same codes, and the engine converter's parity against the real
pre-tokenizer is proven on real data lines (see below).

## Versioning

- Schema version: **1**.
- Weights are only valid for the exact schema version and piece table
  they were trained against. Any change to the table (adding, removing,
  or reordering a piece type) INVALIDATES all existing weights: the
  loader must reject weights whose recorded schema version or table
  hash does not match, once phase 2 wires version checking in.
- The model file carries NO header (raw f32 LE: W1 + B1 + W2 + B2), so
  there is no embedded version or table hash to check. The wiring
  loader enforces what the format allows: the exact byte size
  (5,244,932 = 5120x256x4 + 256x4 + 256x4 + 4, which pins the 5120-input
  / 256-hidden shape) plus finite weights. That size check is the
  enforceable proxy for schema version today; a table change still
  requires a version bump and a full retrain — never silent reuse of
  old weights against a new table.

## Fixed rules (do not change without bumping the version)

- Square indexing: `sq = (y << 3) | x`, where `x` = file 0–7 (a–h) and
  `y` = rank 0–7 (rank 1 = 0). This matches the engine wire format and
  the move encoding (`from`/`to` are squares in this scheme).
- Side encoding: STM-RELATIVE planes. There are exactly two planes:
  plane 0 = the side-to-move's own pieces, plane 1 = the opponent's
  pieces. A single shared plane (the old trainer's bug, which destroyed
  side information and uppercased every code) is FORBIDDEN.
- Feature index: `plane * (TABLE * 64) + table_idx * 64 + sq`, where
  `TABLE` = the number of piece types in the inventory table below and
  `table_idx` = the row index of the piece type in that table. Planes
  are stm-relative (plane 0 = side-to-move's own pieces, plane 1 =
  opponent's), and `sq` is stm-relative too: the raw engine square when
  stm is white, `63 - sq` when stm is black.
- Total features: `2 * TABLE * 64` = **5120** for TABLE = 40.
- The old stub in `src/nnue.rs` (`2176 -> 256 -> 1`, `pos_to_features()`
  returning all zeros) was replaced by the Phase 2 wiring: the
  converter in `src/nnue.rs` (`pos_features`, effective-key resolution,
  unknown-piece skip) and the sparse forward pass. Never train against
  the stub's layout.

## Piece inventory (SINGLE DEFINED TABLE — reconciled by Phase 2)

The inventory below is the ONE table the trainer and the engine share.
It is the trainer's verified-occurring set: exactly the pieces that
occur in the training data (`combined_raw.txt`, DP-tokenized), plus the
6 standard pieces — NOT all 67 `PieceKey` types, and NOT the old
trainer's 34-code list (which omitted PhaseRook). Verified against the
engine `PieceKey` enum VALUES in `src/types.rs`, not names alone.

Row order is part of the schema: once weights are trained, rows must
not be reordered without a version bump.

| table_idx | PieceKey (value) | code | notes |
|-----------|------------------|------|-------|
| 0 | Aristocrat (20) | Ar | |
| 1 | AntiViolence (19) | Av | |
| 2 | Basilisk (22) | Ba | basilisk keeps Ba (banker takes Bk) |
| 3 | BisHop (24) | Bh | bis-hop — NOT Bishopper (23, untrained gap) |
| 4 | Banker (21) | Bk | |
| 5 | Bouncer (26) | Bo | |
| 6 | Blueprint (48) | Bp | extended key, but TRAINED |
| 7 | BladeRunner (25) | Br | |
| 8 | Camel (27) | Ca | (*) distinctive-letter assignment |
| 9 | Cardinal (28) | Cd | (*) distinctive-letter assignment |
| 10 | Dancer (29) | Da | |
| 11 | Djinn (30) | Dj | |
| 12 | ElectroKnight (31) | Ek | |
| 13 | EpeePawn (10) | Ep | |
| 14 | Fish (32) | Fi | |
| 15 | FissionReactor (33) | Fr | |
| 16 | GoldenPawn (8) | Gp | |
| 17 | Gunslinger (34) | Gs | |
| 18 | HordeMother (35) | Hm | |
| 19 | HeroPawn (9) | Hp | |
| 20 | Icicle (36) | Ic | |
| 21 | IronPawn (12) | Ip | |
| 22 | Knightmare (37) | Km | |
| 23 | Marauder (38) | Ma | |
| 24 | Pilgrim (40) | Pg | (*) distinctive-letter assignment |
| 25 | Pinata (47) | Pi | (*) extended key, but TRAINED |
| 26 | PawnWithKnife (7) | Pk | |
| 27 | PhaseRook (39) | Pr | the piece the old 34-list omitted |
| 28 | Recruit (16) | Rc | |
| 29 | Rocketman (41) | Rm | king-slot piece in hand-written eval, own row here |
| 30 | Shrike (11) | Sh | |
| 31 | SumoRook (42) | Sr | |
| 32 | WarAutomaton (17) | Wa | |
| 33 | WarpJumper (18) | Wj | |
| 34 | Pawn (1) | P | standard |
| 35 | Rook (4) | R | standard |
| 36 | Knight (2) | N | standard |
| 37 | Bishop (3) | B | standard |
| 38 | Queen (5) | Q | standard |
| 39 | King (6) | K | standard |

(*) Name assignment by distinctive-letter logic only (pilgrim is the
only P-piece containing 'g'; camel contains no 'd', cardinal does).
Training is unaffected by these names (features are indices); the
engine side is verified here by PieceKey identity, not by the names.
Game-log codes use first-two-letters and DIFFER from data codes:
log Ph -> data Pr, Pa -> Pk, He -> Hp, El -> Ek, Su -> Sr, Kn -> Km,
Bl -> Br (bladeRunner) or Bp (blueprint), Ba -> Ba (basilisk) or Bk
(banker), Ca -> Ca (camel) or Cd (cardinal), Fi -> Fi (fish) or Fr
(fissionReactor), Pi -> Pg (pilgrim) or Pi (pinata), Bi -> NO DATA
CODE (bishopper gap).

### Pieces with NO row (unknown-piece fallback)

The following have no training code and therefore NO feature: Bishopper
(23), Hordeling (13), TranslucentPawn (14), Ooze (15), all king
variants (WarKing 53, MeditatingKing 54, PopeKing 55, IrreverentKing
56), all queen variants (VoidQueen 50, NullQueen 51, WitchQueen 52),
RookZero (46), and every other extended key (43-45, 49, 57-67)
except Blueprint (48) and Pinata (47), which are trained rows above.
The wiring fallback is: such a piece contributes NO NNUE input (it is
skipped, never mapped to a wrong trained feature), while the
hand-written evaluator still sees it. A position's NNUE view can
therefore be sparser than its hand-written view; an all-unknown board
yields the bias-only prior, never a crash. This replaces the revamp
spec's "fail loudly" rule for unmapped pieces: failing loudly at
inference time would forfeit real games containing these pieces, and
the skip is explicit, counted (`FeatureSet.skipped`), and tested —
not a silent vanishing.

### Parity proof

The engine converter was diffed against the REAL trainer
pre-tokenizer binary (`train/src/bin/pretokenize.rs --dump`) on real
`combined_raw.txt` lines: 200/200 lines identical, then 2000/2000
lines identical (feature sets AND stm-relative labels). A
self-contained parity test (independent SPEC §5 oracle in
`src/nnue.rs` tests) also runs in the native suite. Any future
converter change must re-run both before weights ship.

## Open items for phase 2 (decided by the wiring)

- Network shape: 5120 -> 256 (ReLU) -> 1 (sigmoid), per the trainer.
- Limbo/offboard/doomed-list entries get NO features — only on-board,
  alive pieces, matching the static evaluator's view. Doomed pieces
  still standing on the board DO get features (they are on the board).
- Special states (phased, carried, doomed, charged) are NOT encoded:
  the schema has no state features.
- Probability -> centipawn scale: cp = round(400 * log10(p / (1 - p))),
  p clamped to [0.001, 0.999] (cp in [-1200, 1200]), negated when the
  caller is not the side to move. Terminal mate/gold-win states bypass
  the net and return MATE_CONST, like the hand-written evaluator.
