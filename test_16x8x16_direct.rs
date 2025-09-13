// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=test_direct --error-format=human -Copt-level=3

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::kernel;

// Direct extern declaration to test if intrinsic exists
extern "C" {
    #[link_name = "llvm.nvvm.wmma.m16n8k16.mma.sync.row.row.f16.f32"]
    fn test_wmma_16x8x16(
        a0: i16, a1: i16, a2: i16, a3: i16,
        a4: i16, a5: i16, a6: i16, a7: i16,
        b0: i16, b1: i16, b2: i16, b3: i16,
        b4: i16, b5: i16, b6: i16, b7: i16,
        c0: f32, c1: f32, c2: f32, c3: f32,
    ) -> [f32; 4];
}

static mut RESULT: [f32; 4] = [0.0; 4];

#[kernel]
pub unsafe fn test_direct() {
    let result = test_wmma_16x8x16(
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        0.0, 0.0, 0.0, 0.0,
    );
    RESULT = result;
}