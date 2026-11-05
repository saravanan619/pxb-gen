#![allow(static_mut_refs)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(unused_imports)]

pub mod types;
pub mod pos;
pub mod journal;
pub mod marshal;
pub mod attacks;
pub mod movegen;
pub mod movegen_special;
pub mod movegen_castle;
pub mod apply;
pub mod apply_full;
pub mod eval;
pub mod nnue;
pub mod accumulator;
pub mod search;
pub mod ffi;
pub mod veto;
pub mod think;

// JS-provided clock on wasm32. std::time::Instant panics in wasm32-unknown-unknown,
// so WASM builds read time through this import while native builds use std::time.
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    fn pxb_now_ms() -> f64;
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub(crate) fn now_ms() -> u64 {
    unsafe { pxb_now_ms() as u64 }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub(crate) fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
