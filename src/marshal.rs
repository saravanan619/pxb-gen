use crate::pos::*;
use crate::types::*;

#[inline] fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o+1]])
}
#[inline] fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]])
}
#[inline] fn rd_i8 (b: &[u8], o: usize) -> i8 { b[o] as i8 }

pub const HEADER_SIZE: usize = 48;
pub const PIECE_SIZE: usize = 48;
pub const LIMBO_SIZE: usize = 24;
pub const OFFBOARD_SIZE: usize = 16;
pub const DOOMED_SIZE: usize = 8;
pub const HIST_SIZE: usize = 4;
pub const MAGIC: u16 = 0x5058;
pub const VERSION: u8 = 1;

pub fn pos_from_buffer(b: &[u8]) -> Option<Pos> {
    if b.len() < HEADER_SIZE { return None; }
    if rd_u16(b, 0) != MAGIC { return None; }
    if b[2] != VERSION { return None; }

    let flags = b[3];
    let _seat = rd_i8(b, 4);
    let stm = b[5] & 1;
    let ep_x = rd_i8(b, 6);
    let ep_y = rd_i8(b, 7);
    let ep_pawn_x = rd_i8(b, 8);
    let ep_pawn_y = rd_i8(b, 9);
    let halfmove = rd_u16(b, 10);
    let n_pieces = rd_u16(b, 12) as usize;
    let n_limbo = rd_u16(b, 14) as usize;
    let n_offboard = rd_u16(b, 16) as usize;
    let n_doomed = rd_u16(b, 18) as usize;
    let n_hist = rd_u16(b, 20) as usize;
    let kt_byte = rd_i8(b, 22);
    let gw_byte = rd_i8(b, 23);
    let horde_seq = rd_u32(b, 24);
    let id_seq = rd_u32(b, 28);
    // bytes 32..40: last_cap keys (2 players × 2 keys)
    let lc0_k = rd_u16(b, 32);
    let lc0_s = rd_u16(b, 34);
    let lc1_k = rd_u16(b, 36);
    let lc1_s = rd_u16(b, 38);
    // 40..48 reserved

    let expect = HEADER_SIZE
        + n_pieces   * PIECE_SIZE
        + n_limbo    * LIMBO_SIZE
        + n_offboard * OFFBOARD_SIZE
        + n_doomed   * DOOMED_SIZE
        + n_hist     * HIST_SIZE;
    if b.len() < expect { return None; }
    if n_hist > MAX_HIST { return None; }

    let mut pos = Pos {
        at: [0; 64],
        pieces: Vec::with_capacity(n_pieces),
        limbo: Vec::with_capacity(n_limbo),
        offboard: Vec::with_capacity(n_offboard),
        doomed: Vec::with_capacity(n_doomed),
        hist_hashes: Vec::with_capacity(n_hist),
        ep: if ep_x >= 0 {
            Some(EpInfo { x: ep_x, y: ep_y, pawn_x: ep_pawn_x, pawn_y: ep_pawn_y })
        } else { None },
        halfmove,
        eval_sum: 0,
        ket_x2: 2,
        queen_count: 0,
        king_thrown: if flags & 1 != 0 { Some(kt_byte as u8) } else { None },
        gold_win:    if flags & 2 != 0 { Some(gw_byte as u8) } else { None },
        last_cap: [
            (PieceKey::from_u16(lc0_k), PieceKey::from_u16(lc0_s)),
            (PieceKey::from_u16(lc1_k), PieceKey::from_u16(lc1_s)),
        ],
        horde_seq,
        id_seq,
        stm,
    };

    // ── Pieces ──
    let mut off = HEADER_SIZE;
    for i in 0..n_pieces {
        let mut p = Piece::default();
        p.id = rd_u32(b, off);
        p.key = PieceKey::from_u16(rd_u16(b, off + 4));
        p.sub_key = PieceKey::from_u16(rd_u16(b, off + 6));
        p.player = b[off + 8];
        p.x = b[off + 9];
        p.y = b[off + 10];
        // off+11 reserved
        p.state_flags = rd_u16(b, off + 12);
        p.odo   = b[off + 14];
        p.kills = b[off + 15];
        p.caps  = b[off + 16];
        p.quiet = b[off + 17];
        p.frost = b[off + 18];
        p.encore= b[off + 19];
        p.horde_id = rd_u32(b, off + 20);
        p.home_x = rd_i8(b, off + 24);
        p.home_y = rd_i8(b, off + 25);
        p.n_abilities = b[off + 26].min(MAX_ABILITIES as u8);
        // off+27 reserved
        let ab_off = off + 28;
        for a in 0..(p.n_abilities as usize) {
            p.abilities[a].key   = b[ab_off + a * 2];
            p.abilities[a].range = b[ab_off + a * 2 + 1];
        }
        p.alive = true;

        // Register square
        if p.x < 8 && p.y < 8 {
            let s = sq(p.x as i8, p.y as i8);
            pos.at[s as usize] = (i + 1) as u16;
        }
        pos.pieces.push(p);
        off += PIECE_SIZE;
    }

    // ── Limbo ──
    for _ in 0..n_limbo {
        let mut l = LimboPiece::default();
        l.id = rd_u32(b, off);
        l.key = PieceKey::from_u16(rd_u16(b, off + 4));
        l.sub_key = PieceKey::from_u16(rd_u16(b, off + 6));
        l.player = b[off + 8];
        // off+9 reserved
        l.home_x = rd_i8(b, off + 10);
        l.home_y = rd_i8(b, off + 11);
        pos.limbo.push(l);
        off += LIMBO_SIZE;
    }

    // ── Offboard ──
    for _ in 0..n_offboard {
        let mut o = OffboardPiece::default();
        o.id = rd_u32(b, off);
        o.key = PieceKey::from_u16(rd_u16(b, off + 4));
        o.sub_key = PieceKey::from_u16(rd_u16(b, off + 6));
        o.player = b[off + 8];
        o.off_since = b[off + 9];
        // Staging coords live in the record's spare bytes (10, 11) so the
        // 16-byte record size is unchanged. Legacy buffers decode 0,0.
        o.ox = b[off + 10] as i8;
        o.oy = b[off + 11] as i8;
        pos.offboard.push(o);
        off += OFFBOARD_SIZE;
    }

    // ── Doomed ──
    for _ in 0..n_doomed {
        let mut d = DoomedEntry::default();
        d.id = rd_u32(b, off);
        d.by = b[off + 4];
        pos.doomed.push(d);
        off += DOOMED_SIZE;
    }

    // ── Hist hashes ──
    for _ in 0..n_hist {
        pos.hist_hashes.push(rd_u32(b, off));
        off += HIST_SIZE;
    }

    Some(pos)
}

#[cfg(test)]
mod pixie_wire_identity_tests {
    use super::{pos_from_buffer, HEADER_SIZE, MAGIC, PIECE_SIZE, VERSION};
    use crate::eval::{effective_key, piece_value};
    use crate::types::PieceKey;

    #[test]
    fn phase_rook_identity_survives_base_first_wire_encoding() {
        // Match the actual little-endian ABI layout: base Rook in key,
        // PhaseRook in sub_key. The generic evaluator must not collapse it
        // to a standard Rook after the WASM marshal boundary.
        let mut b = vec![0u8; HEADER_SIZE + PIECE_SIZE];
        b[0..2].copy_from_slice(&MAGIC.to_le_bytes());
        b[2] = VERSION;
        b[5] = 0; // White to move
        b[12..14].copy_from_slice(&1u16.to_le_bytes());
        let p = HEADER_SIZE;
        b[p + 4..p + 6].copy_from_slice(&(PieceKey::Rook as u16).to_le_bytes());
        b[p + 6..p + 8].copy_from_slice(&(PieceKey::PhaseRook as u16).to_le_bytes());
        b[p + 8] = 0; // player
        b[p + 9] = 1; // b-file
        b[p + 10] = 0; // first rank

        let pos = pos_from_buffer(&b).expect("valid single-piece ABI buffer");
        let piece = &pos.pieces[0];
        assert_eq!(effective_key(piece), PieceKey::PhaseRook);
        assert_eq!(piece_value(piece), 574);
    }
}
