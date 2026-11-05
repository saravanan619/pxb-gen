// Lever-3 micro-profile harness (STAGING ONLY).
// Times individual hot-path functions on the P2 messy position, in isolation,
// so the optimisation target is evidence, not guesswork. Sums outputs into a
// black-box sink so the compiler cannot elide the work.
use pxb_engine::apply::make_journaled;
use pxb_engine::attacks::{gives_check, king_attacked};
use pxb_engine::eval::{init_eval_sum, rep_hash, zobrist};
use pxb_engine::journal::Journal;
use pxb_engine::movegen::gen_moves;
use pxb_engine::nnue::{nnue_evaluate, nnue_is_loaded};
use pxb_engine::pos::{occ, Piece, Pos};
use pxb_engine::search::rep_count;
use pxb_engine::types::*;
use std::hint::black_box;
use std::time::Instant;

fn build(pieces: &[(PieceKey, u8, u8, u8)], stm: u8) -> Pos {
    let mut pos = Pos { at: [0; 64], pieces: Vec::new(), limbo: Vec::new(),
        offboard: Vec::new(), doomed: Vec::new(), hist_hashes: Vec::new(),
        ep: None, halfmove: 0, eval_sum: 0, ket_x2: 2, queen_count: 0,
        king_thrown: None, gold_win: None,
        last_cap: [(PieceKey::None, PieceKey::None); 2], horde_seq: 0, id_seq: 0, stm };
    for (i, &(key, player, x, y)) in pieces.iter().enumerate() {
        pos.pieces.push(Piece { id: (i + 1) as u32, key, sub_key: key, player, x, y,
            alive: true, ..Default::default() });
        pos.at[((y as usize) << 3) | (x as usize)] = (i + 1) as u16;
    }
    init_eval_sum(&mut pos);
    pos.hist_hashes.push(rep_hash(&pos, stm));
    pos
}

fn messy_a() -> Pos {
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
    build(&v, 0)
}

fn time_it<F: FnMut() -> i64>(label: &str, iters: usize, mut f: F) {
    // Warmup
    let mut sink: i64 = 0;
    for _ in 0..iters.min(1000) { sink = sink.wrapping_add(f()); }
    let t = Instant::now();
    for _ in 0..iters { sink = sink.wrapping_add(f()); }
    let ms = t.elapsed().as_millis() as f64;
    println!("{:28} iters={:8} total_ms={:8.1} ns_per_call={:8.1} sink={}",
        label, iters, ms, ms * 1_000_000.0 / iters as f64, black_box(sink));
}

fn main() {
    println!("nnue_loaded={}", nnue_is_loaded());
    let pos = messy_a();
    let moves = gen_moves(&pos, 0, None);
    println!("root moves: {}", moves.len());
    let occ_local = occ(&pos);

    time_it("gen_moves stage None", 20_000, || {
        let m = gen_moves(&pos, 0, None);
        m.len() as i64
    });
    time_it("gen_moves stage 0", 20_000, || {
        let m = gen_moves(&pos, 0, Some(0));
        m.len() as i64
    });
    time_it("gen_moves stage 1", 20_000, || {
        let m = gen_moves(&pos, 0, Some(1));
        m.len() as i64
    });
    time_it("occ()", 200_000, || {
        let o = occ(&pos);
        o.iter().filter(|&&b| b).count() as i64
    });
    time_it("king_attacked", 20_000, || {
        king_attacked(&pos, 0, 1, Some(&occ_local)) as i64
    });
    time_it("gives_check", 20_000, || {
        gives_check(&pos, 0, 1, Some(&occ_local)) as i64
    });
    time_it("zobrist (TT key)", 200_000, || {
        zobrist(&pos, 0) as i64
    });
    time_it("rep_hash (zobrist_u64)", 200_000, || {
        rep_hash(&pos, 0) as i64
    });
    time_it("rep_count", 200_000, || {
        rep_count(&pos, 0, rep_hash(&pos, 0)) as i64
    });
    time_it("nnue_evaluate full", 20_000, || {
        nnue_evaluate(&pos, 0).unwrap_or(0) as i64
    });
    // Journaled make+unwind over all root moves, per iteration.
    time_it("make_journaled+unwind all root moves", 2_000, || {
        let mut p = pos.clone();
        let mut j = Journal::default();
        let mut n: i64 = 0;
        for mv in &moves {
            let mark = j.mark();
            if make_journaled(&mut p, mv, 0, 1, &mut j) {
                n += 1;
                j.unwind(&mut p, mark);
            } else {
                j.unwind(&mut p, mark);
            }
        }
        n
    });
    time_it("piece_attacks all own pieces", 20_000, || {
        let mut n: i64 = 0;
        for i in 0..pos.pieces.len() {
            let p = &pos.pieces[i];
            if p.alive && p.player == 0 {
                n += pxb_engine::movegen::piece_attacks(&pos, i, &occ_local, true, false).0.count_ones() as i64;
            }
        }
        n
    });
    time_it("gen_special_moves all own pieces", 20_000, || {
        let mut out: Vec<pxb_engine::types::Move> = Vec::new();
        for i in 0..pos.pieces.len() {
            let p = &pos.pieces[i];
            if p.alive && p.player == 0 {
                pxb_engine::movegen_special::gen_special_moves(&pos, i, 0, 1, &occ_local, &mut out);
            }
        }
        out.len() as i64
    });
    time_it("nnue delta unmapped", 20_000, || {
        pxb_engine::nnue::unmapped_material_delta_cp(&pos, 0) as i64
    });
    time_it("nnue delta king_exposure", 20_000, || {
        pxb_engine::nnue::king_exposure_nnue_delta(&pos, 0) as i64
    });
    time_it("nnue delta aura", 20_000, || {
        pxb_engine::nnue::pixie_aura_nnue_delta(&pos, 0) as i64
    });
    time_it("nnue delta state", 20_000, || {
        pxb_engine::nnue::pixie_state_nnue_delta(&pos, 0) as i64
    });
    time_it("apply_full all root moves", 2_000, || {
        let mut n: i64 = 0;
        for mv in &moves {
            if let Some(p2) = pxb_engine::apply_full::apply_full(&pos, mv, 0, 1) {
                n += p2.pieces.len() as i64;
            }
        }
        n
    });
    time_it("square_attacked per capture dest", 20_000, || {
        let mut n: i64 = 0;
        for mv in &moves {
            if mv.flags & pxb_engine::types::MF_CAPTURE != 0 && mv.to < 64 {
                n += pxb_engine::attacks::square_attacked(&pos, mv.to, 1, &occ_local) as i64;
            }
        }
        n
    });
    // Per-move legality pattern from ab_search_core: journaled apply +
    // occ + king_attacked + gives_check, for journalable moves only.
    time_it("per-move legality (journaled subset)", 2_000, || {
        let mut p = pos.clone();
        let mut j = Journal::default();
        let mut n: i64 = 0;
        for mv in &moves {
            let mark = j.mark();
            if make_journaled(&mut p, mv, 0, 1, &mut j) {
                let o = occ(&p);
                if !king_attacked(&p, 0, 1, Some(&o)) {
                    n += gives_check(&p, 0, 1, Some(&o)) as i64;
                }
                j.unwind(&mut p, mark);
            } else {
                j.unwind(&mut p, mark);
            }
        }
        n
    });
}
