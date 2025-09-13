// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=test_working --error-format=human -Copt-level=3

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::f16;
use cuda_std::kernel;

// Direct extern declaration to test if intrinsic exists
extern "C" {
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.f16.f32"]
    fn test_wmma_16x16x16(
        a0: i16, a1: i16, a2: i16, a3: i16,
        a4: i16, a5: i16, a6: i16, a7: i16,
        a8: i16, a9: i16, a10: i16, a11: i16,
        a12: i16, a13: i16, a14: i16, a15: i16,
        b0: i16, b1: i16, b2: i16, b3: i16,
        b4: i16, b5: i16, b6: i16, b7: i16,
        b8: i16, b9: i16, b10: i16, b11: i16,
        b12: i16, b13: i16, b14: i16, b15: i16,
        c0: f32, c1: f32, c2: f32, c3: f32,
        c4: f32, c5: f32, c6: f32, c7: f32,
    ) -> [f32; 8];
}

static mut RESULT: [f32; 8] = [0.0; 8];

#[kernel]
pub unsafe fn test_working() {
    let result = test_wmma_16x16x16(
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    RESULT = result;
}