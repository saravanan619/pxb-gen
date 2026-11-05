// Dev tool: engine-side feature dump for parity checks against the
// trainer pre-tokenizer. Reads combined_raw-format lines
// ("FEN stm - - ply move | outcome") from a file argument (or stdin)
// and prints "label feat..." per valid line, in the pre-tokenizer
// --dump format (stm-relative label, sorted feature indices).
// Usage: nnue-feature-dump <file>
// Not used by any engine path; exists so the converter can be diffed
// against the real trainer binary on real data lines.

use pxb_engine::nnue::{fen_to_pos, pos_features};
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reader: Box<dyn BufRead> = if args.len() > 1 {
        let f = std::fs::File::open(&args[1]).expect("open input");
        Box::new(BufReader::new(f))
    } else {
        Box::new(BufReader::new(std::io::stdin()))
    };
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 2 {
            continue;
        }
        let left: Vec<&str> = parts[0].trim().split_whitespace().collect();
        if left.len() < 2 {
            continue;
        }
        let outcome: f32 = match parts[1].trim().parse() {
            Ok(o) => o,
            Err(_) => continue,
        };
        let (fen, stm_white) = (left[0], left[1] == "w");
        let pos = match fen_to_pos(fen, stm_white) {
            Some(p) => p,
            None => continue,
        };
        let feats = pos_features(&pos);
        let label = if stm_white { outcome } else { 1.0 - outcome };
        let mut out = format!("{label:.3}");
        for f in &feats.indices {
            out.push_str(&format!(" {f}"));
        }
        println!("{out}");
    }
}
