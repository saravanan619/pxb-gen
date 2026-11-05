// Lever-3 bench harness (STAGING ONLY, never shipped).
// Fixed positions, fixed search depth, huge time budget (no time cutoff),
// deterministic node counts. Prints: best move (from/to/flags/promo),
// score, nodes, wall ms, nodes/sec. Repeated runs -> median in report.
// Usage: lever3-bench [depth] [reps]   (defaults: depth 4, reps 5)
use pxb_engine::eval::{init_eval_sum, rep_hash};
use pxb_engine::pos::{Piece, Pos};
use pxb_engine::search::{abSearch, Search};
use pxb_engine::types::*;
use std::time::Instant;

fn build(pieces: &[(PieceKey, u8, u8, u8)], stm: u8) -> Pos {
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
        ket_x2: 2,
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
    // Realistic game length: rep_count() scans the whole history every
    // node, so a 1-ply history would understate live per-node cost.
    // Dummy values are distinct from any real position hash (collision
    // odds ~80/2^32), so search behaviour and node counts are unchanged.
    for i in 0..80u32 {
        pos.hist_hashes.push(0x7f00_0000u32.wrapping_add(i * 0x0101_0101));
    }
    pos
}

fn opening() -> (Pos, u8) {
    use PieceKey::*;
    let mut v: Vec<(PieceKey, u8, u8, u8)> = Vec::new();
    let back = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook];
    for (x, &k) in back.iter().enumerate() {
        v.push((k, 0, x as u8, 0));
        v.push((Pawn, 0, x as u8, 1));
        v.push((k, 1, x as u8, 7));
        v.push((Pawn, 1, x as u8, 6));
    }
    (build(&v, 0), 0)
}

fn messy_a() -> (Pos, u8) {
    // ~26 pieces: standard core + Pilgrim, ElectroKnight, Fish, Marauder,
    // BladeRunner, SumoRook, Gunslinger. White to move, open centre.
    use PieceKey::*;
    let v = vec![
        (King, 0, 4, 0), (Queen, 0, 3, 1), (Rook, 0, 0, 0), (Rook, 0, 7, 2),
        (Bishop, 0, 2, 2), (Knight, 0, 5, 2), (Pawn, 0, 0, 1), (Pawn, 0, 1, 2),
        (Pawn, 0, 3, 3), (Pawn, 0, 4, 3), (Pawn, 0, 6, 1),
        (Pilgrim, 0, 2, 4), (ElectroKnight, 0, 5, 4), (Fish, 0, 1, 0),
        (King, 1, 4, 7), (Queen, 1, 3, 6), (Rook, 1, 0, 7), (Rook, 1, 7, 5),
        (Bishop, 1, 5, 5), (Knight, 1, 2, 5), (Pawn, 1, 0, 6), (Pawn, 1, 4, 4),
        (Pawn, 1, 5, 6), (Pawn, 1, 7, 6),
        (Marauder, 1, 3, 4), (BladeRunner, 1, 6, 4), (SumoRook, 1, 1, 7),
        (Gunslinger, 1, 6, 6),
    ];
    (build(&v, 0), 0)
}

fn messy_b() -> (Pos, u8) {
    // Tactical open position: Dancer, Shrike, Bishopper, WarAutomaton,
    // Banker, HeroPawn, AntiViolence, Basilisk (exercises AV/paralysis paths).
    use PieceKey::*;
    let v = vec![
        (King, 0, 6, 0), (Queen, 0, 3, 0), (Rook, 0, 0, 3), (Knight, 0, 4, 2),
        (Pawn, 0, 2, 1), (Pawn, 0, 5, 1), (Pawn, 0, 6, 2),
        (Dancer, 0, 3, 3), (Shrike, 0, 5, 3), (Bishopper, 0, 1, 3),
        (WarAutomaton, 0, 4, 4), (Banker, 0, 2, 0),
        (King, 1, 5, 7), (Queen, 1, 2, 6), (Rook, 1, 7, 7), (Bishop, 1, 3, 5),
        (Pawn, 1, 3, 4), (Pawn, 1, 4, 6), (Pawn, 1, 6, 5),
        (HeroPawn, 1, 1, 5), (AntiViolence, 1, 4, 5), (Basilisk, 1, 0, 4),
        (Recruit, 1, 6, 3),
    ];
    (build(&v, 0), 0)
}

fn messy_c() -> (Pos, u8) {
    // Black to move, queenless-ish scramble with HordeMother, Rocketman,
    // EpeePawn, Knightmare-adjacent specials avoided (Knightmare offboard
    // semantics differ); Djinn + Icicle present.
    use PieceKey::*;
    let v = vec![
        (King, 0, 4, 0), (Rook, 0, 0, 0), (Bishop, 0, 5, 1), (Knight, 0, 2, 2),
        (Pawn, 0, 1, 1), (Pawn, 0, 3, 2), (Pawn, 0, 5, 3),
        (Djinn, 0, 6, 2), (Icicle, 0, 2, 4),
        (King, 1, 4, 7), (Queen, 1, 3, 7), (Rook, 1, 7, 6), (Knight, 1, 5, 5),
        (Pawn, 1, 0, 6), (Pawn, 1, 2, 6), (Pawn, 1, 4, 5), (Pawn, 1, 6, 6),
        (HordeMother, 1, 3, 4), (Rocketman, 1, 1, 4), (EpeePawn, 1, 5, 4),
    ];
    (build(&v, 1), 1)
}

fn endgame() -> (Pos, u8) {
    use PieceKey::*;
    let v = vec![
        (King, 0, 4, 0), (Queen, 0, 3, 3), (Rook, 0, 0, 0), (Pawn, 0, 5, 1),
        (Pawn, 0, 6, 2),
        (King, 1, 4, 7), (Rook, 1, 0, 7), (Pawn, 1, 3, 6), (Pawn, 1, 5, 6),
    ];
    (build(&v, 0), 0)
}

fn mate_pos() -> (Pos, u8) {
    // Defended contact mate: K f6, Q d7 vs K h8, White to move.
    use PieceKey::*;
    let v = vec![
        (King, 0, 5, 5), (Queen, 0, 3, 6), (King, 1, 7, 7),
    ];
    (build(&v, 0), 0)
}

fn run_one(pos0: &Pos, seat: u8, depth: i32) -> (Option<(u8, u8, u32, u8)>, i32, u64, u128) {
    let mut s = Search::new();
    s.reset_for_think(3_600_000);
    s.evaluator = &pxb_engine::nnue::NNUE_EVALUATOR;
    let mut p = pos0.clone();
    let t = Instant::now();
    let sc = abSearch(&mut s, &mut p, depth, -1_000_000_000, 1_000_000_000, seat, 1 - seat, seat, 0, 0, None);
    let ms = t.elapsed().as_millis();
    let bm = s.best_root.map(|m| (m.from, m.to, m.flags, m.promo));
    (bm, sc, s.nodes, ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depth: i32 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let reps: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(5);
    println!("nnue_loaded={} depth={} reps={}", pxb_engine::nnue::nnue_is_loaded(), depth, reps);
    let positions: Vec<(&str, (Pos, u8))> = vec![
        ("P1-opening", opening()),
        ("P2-messy-A", messy_a()),
        ("P3-messy-B-AV-Basilisk", messy_b()),
        ("P4-messy-C-black", messy_c()),
        ("P5-endgame", endgame()),
        ("P6-mate", mate_pos()),
    ];
    for (name, (pos, seat)) in &positions {
        let mut times: Vec<u128> = Vec::new();
        let mut first: Option<(Option<(u8, u8, u32, u8)>, i32, u64)> = None;
        let mut stable = true;
        for _ in 0..reps {
            let (bm, sc, nodes, ms) = run_one(pos, *seat, depth);
            times.push(ms);
            match &first {
                None => first = Some((bm, sc, nodes)),
                Some((fb, fs, fn_)) => {
                    if *fb != bm || *fs != sc || *fn_ != nodes { stable = false; }
                }
            }
        }
        times.sort();
        let med = times[times.len() / 2];
        let (bm, sc, nodes) = first.unwrap();
        let nps = if med > 0 { nodes as f64 / (med as f64 / 1000.0) } else { 0.0 };
        println!("{} move={:?} score={} nodes={} median_ms={} nodes_per_sec={:.0} stable={} times_ms={:?}",
            name, bm, sc, nodes, med, nps, stable, times);
    }
}
