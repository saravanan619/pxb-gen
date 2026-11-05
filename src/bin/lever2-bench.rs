//! Lever-2 fixed-position benchmark (staging only, NOT shipped).
//! Runs run_ranked_root_search at a fixed max_depth with a huge budget so
//! the node cap / depth reached, not the clock, is what stops the search
//! (except the 1.5M hard node cap). Prints: nodes, completed depth,
//! wall time, legacy pick, legacy score.
//! Usage: lever2-bench [max_depth]  (default 5)

use pxb_engine::eval::{init_eval_sum, rep_hash};
use pxb_engine::pos::{Piece, Pos};
use pxb_engine::search::Search;
use pxb_engine::think::run_ranked_root_search;
use pxb_engine::types::*;
use std::time::Instant;

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

fn sq_name(s: u8) -> String {
    if s >= 64 {
        return format!("off({s})");
    }
    let f = (s & 7) as u8;
    let r = (s >> 3) + 1;
    format!("{}{}", (b'a' + f) as char, r)
}

fn positions() -> Vec<(&'static str, Pos, u8)> {
    use PieceKey::*;
    let mut v: Vec<(&'static str, Pos, u8)> = Vec::new();
    // P0: standard opening, White to move.
    let mut std: Vec<(PieceKey, u8, u8, u8)> = Vec::new();
    let back = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook];
    for (x, k) in back.iter().enumerate() {
        std.push((*k, 0, x as u8, 0));
        std.push((Pawn, 0, x as u8, 1));
        std.push((*k, 1, x as u8, 7));
        std.push((Pawn, 1, x as u8, 6));
    }
    v.push(("P0-opening-std", build_pos(&std, 0), 0));
    // P1: repo fixture middlegame (search tests), Black to move.
    v.push((
        "P1-mid-fixture",
        build_pos(
            &[
                (King, 0, 4, 0),
                (Queen, 0, 3, 0),
                (Rook, 0, 0, 0),
                (Knight, 0, 6, 0),
                (Pawn, 0, 4, 3),
                (King, 1, 4, 7),
                (Queen, 1, 3, 7),
                (Rook, 1, 0, 7),
                (Pilgrim, 1, 2, 7),
                (Pawn, 1, 4, 6),
            ],
            1,
        ),
        1,
    ));
    // P2: messy middlegame, Pixie-heavy, White to move.
    v.push((
        "P2-mid-pixie",
        build_pos(
            &[
                (King, 0, 4, 0),
                (Queen, 0, 3, 3),
                (Rook, 0, 0, 0),
                (Knight, 0, 5, 2),
                (Bishop, 0, 2, 2),
                (ElectroKnight, 0, 6, 3),
                (Dancer, 0, 1, 3),
                (Pawn, 0, 3, 2),
                (Pawn, 0, 4, 2),
                (Pawn, 0, 5, 1),
                (King, 1, 4, 7),
                (Queen, 1, 3, 5),
                (Rook, 1, 7, 7),
                (Knight, 1, 2, 5),
                (AntiViolence, 1, 4, 4),
                (Marauder, 1, 6, 5),
                (SumoRook, 1, 1, 6),
                (Pawn, 1, 3, 5),
                (Pawn, 1, 4, 6),
                (Pawn, 1, 5, 6),
            ],
            0,
        ),
        0,
    ));
    // P3: messy middlegame, pawn tension + sliders, Black to move.
    v.push((
        "P3-mid-tension",
        build_pos(
            &[
                (King, 0, 6, 0),
                (Queen, 0, 3, 1),
                (Rook, 0, 0, 0),
                (Rook, 0, 7, 0),
                (Bishop, 0, 2, 3),
                (Bishop, 0, 5, 2),
                (Knight, 0, 3, 3),
                (Pawn, 0, 2, 2),
                (Pawn, 0, 3, 2),
                (Pawn, 0, 4, 3),
                (Pawn, 0, 5, 2),
                (Pawn, 0, 6, 1),
                (King, 1, 6, 7),
                (Queen, 1, 3, 6),
                (Rook, 1, 0, 7),
                (Rook, 1, 7, 7),
                (Bishop, 1, 5, 4),
                (Knight, 1, 4, 4),
                (Knight, 1, 2, 4),
                (Pawn, 1, 2, 5),
                (Pawn, 1, 3, 4),
                (Pawn, 1, 4, 5),
                (Pawn, 1, 5, 5),
                (Pawn, 1, 6, 6),
            ],
            1,
        ),
        1,
    ));
    // P4: endgame/mate — Kf6 Qd7 vs Kh8, White to move (mate in 1, Qg7#).
    v.push((
        "P4-endgame-mate",
        build_pos(&[(King, 0, 5, 5), (Queen, 0, 3, 6), (King, 1, 7, 7)], 0),
        0,
    ));
    // P5: endgame rook/bishop vs rook/knight, Black to move (repo warm/cold target).
    // P6-P11: UNSEEN strength-check set (not used during tuning decisions).
    v.push((
        "P6-mid-powers",
        build_pos(
            &[
                (King, 0, 4, 0),
                (Queen, 0, 3, 2),
                (Rook, 0, 7, 0),
                (HordeMother, 0, 1, 2),
                (Gunslinger, 0, 5, 3),
                (BladeRunner, 0, 2, 4),
                (Pawn, 0, 4, 2),
                (Pawn, 0, 6, 1),
                (King, 1, 4, 7),
                (Queen, 1, 3, 6),
                (Rook, 1, 0, 7),
                (AntiViolence, 1, 3, 4),
                (Knight, 1, 6, 5),
                (Pawn, 1, 4, 5),
                (Pawn, 1, 5, 6),
            ],
            0,
        ),
        0,
    ));
    v.push((
        "P7-mid-fish-rocket",
        build_pos(
            &[
                (King, 0, 6, 0),
                (Rook, 0, 3, 0),
                (Bishop, 0, 2, 2),
                (Fish, 0, 5, 3),
                (Rocketman, 0, 1, 4),
                (Pawn, 0, 4, 2),
                (King, 1, 6, 7),
                (Queen, 1, 2, 5),
                (Rook, 1, 7, 7),
                (Dancer, 1, 4, 4),
                (Pilgrim, 1, 2, 6),
                (Pawn, 1, 5, 5),
                (Pawn, 1, 3, 6),
            ],
            1,
        ),
        1,
    ));
    v.push((
        "P8-queenless",
        build_pos(
            &[
                (King, 0, 4, 0),
                (Rook, 0, 0, 0),
                (Rook, 0, 7, 1),
                (Bishop, 0, 2, 2),
                (Knight, 0, 5, 2),
                (Knight, 0, 3, 3),
                (Pawn, 0, 4, 2),
                (Pawn, 0, 6, 1),
                (King, 1, 4, 7),
                (Rook, 1, 0, 7),
                (Rook, 1, 7, 6),
                (Bishop, 1, 5, 5),
                (Knight, 1, 2, 5),
                (Pawn, 1, 3, 5),
                (Pawn, 1, 4, 6),
            ],
            0,
        ),
        0,
    ));
    v.push((
        "P9-pawn-heroes",
        build_pos(
            &[
                (King, 0, 4, 0),
                (HeroPawn, 0, 3, 4),
                (GoldenPawn, 0, 5, 3),
                (Pawn, 0, 2, 2),
                (Pawn, 0, 6, 2),
                (King, 1, 4, 7),
                (HeroPawn, 1, 4, 4),
                (EpeePawn, 1, 2, 5),
                (Pawn, 1, 5, 5),
                (Pawn, 1, 6, 6),
            ],
            1,
        ),
        1,
    ));
    v.push((
        "P10-capture-rich",
        build_pos(
            &[
                (King, 0, 6, 0),
                (Queen, 0, 3, 3),
                (Rook, 0, 0, 3),
                (Knight, 0, 4, 4),
                (Bishop, 0, 5, 2),
                (Pawn, 0, 3, 2),
                (King, 1, 6, 7),
                (Queen, 1, 3, 4),
                (Rook, 1, 7, 4),
                (Knight, 1, 4, 3),
                (Bishop, 1, 2, 5),
                (Pawn, 1, 4, 5),
            ],
            0,
        ),
        0,
    ));
    v.push((
        "P11-mid-pixie-flip",
        build_pos(
            &[
                (King, 1, 3, 7),
                (Queen, 1, 4, 4),
                (Rook, 1, 7, 7),
                (Knight, 1, 2, 5),
                (Bishop, 1, 5, 5),
                (ElectroKnight, 1, 1, 4),
                (Dancer, 1, 6, 4),
                (Pawn, 1, 4, 5),
                (Pawn, 1, 3, 5),
                (Pawn, 1, 2, 6),
                (King, 0, 3, 0),
                (Queen, 0, 4, 2),
                (Rook, 0, 0, 0),
                (Knight, 0, 5, 2),
                (AntiViolence, 0, 3, 3),
                (Marauder, 0, 1, 2),
                (SumoRook, 0, 6, 1),
                (Pawn, 0, 4, 1),
                (Pawn, 0, 3, 2),
                (Pawn, 0, 2, 1),
            ],
            1,
        ),
        1,
    ));
    v.push((
        "P5-endgame-rook",
        build_pos(
            &[
                (King, 0, 6, 0),
                (Rook, 0, 3, 0),
                (Bishop, 0, 2, 3),
                (Pawn, 0, 5, 1),
                (King, 1, 6, 7),
                (Rook, 1, 0, 7),
                (Knight, 1, 5, 5),
                (Pawn, 1, 6, 6),
            ],
            1,
        ),
        1,
    ));
    v
}

fn main() {
    let max_depth: u8 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(5);
    let cluster = std::env::args().nth(2).map_or(false, |a| a == "cluster");
    println!("lever2-bench max_depth={max_depth} budget=600000ms cluster={cluster}");
    for (name, pos0, seat) in positions() {
        let mut s = Search::new();
        s.cluster_worker = cluster;
        let mut p = pos0.clone();
        let t = Instant::now();
        let out = run_ranked_root_search(&mut s, &mut p, seat, 600_000, max_depth);
        let el = t.elapsed().as_millis();
        let pick = match out.legacy_pick {
            Some(m) => format!("{}->{} promo={}", sq_name(m.from), sq_name(m.to), m.promo),
            None => "none".to_string(),
        };
        println!(
            "{name} depth={} nodes={} time_ms={} pick={} score={} veto={}/{}",
            out.depth, out.nodes, el, pick, out.legacy_score, out.veto_checks, out.veto_rejects
        );
    }
}
