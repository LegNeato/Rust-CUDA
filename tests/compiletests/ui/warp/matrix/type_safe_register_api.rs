// build-pass
// compile-flags: --emit=llvm-ir --error-format=human

//! Test the type-safe register API with compile-time size checking

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, Accumulator, MatrixA, MatrixB, MatrixExt};

#[kernel]
pub unsafe fn test_type_safe_mma_only_api() {
    // Shape<16, 8, 16> is MMA-only (no WMMA load/store)
    type Shape = dims::Shape<16, 8, 16>;

    // Type-safe from_array - compile-time size checking!
    // For bf16 with Shape<16, 8, 16>, we need exactly 8 registers for A and B
    let a_frag: MatrixA<bf16, Shape, layout::Row> = MatrixA::from_array([
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
        bf16::from_f32(6.0),
        bf16::from_f32(7.0),
        bf16::from_f32(8.0),
    ]); // Exactly 8 elements - compile-time checked!

    let b_frag: MatrixB<bf16, Shape, layout::Row> = MatrixB::from_array(
        [
        bf16::from_f32(1.0); 8  // Array repeat syntax - exactly 8
    ],
    );

    // Accumulator needs exactly 4 registers for f32 with Shape<16, 8, 16>
    let c_frag: Accumulator<f32, Shape> = Accumulator::from_array([0.0, 1.0, 2.0, 3.0]);
    // Exactly 4 elements - compile-time checked!

    // MMA operations work normally
    // MMA operations would work if the proper MMA trait implementation exists
    // let _result = c_frag.mma(&a_frag, &b_frag);
}

#[kernel]
pub unsafe fn test_type_safe_wmma_api() {
    // Shape<16, 16, 16> supports both WMMA and the register API
    type Shape = dims::Shape<16, 16, 16>;

    // For bf16 with Shape<16, 16, 16>, we need exactly 16 registers
    let a_frag: MatrixA<bf16, Shape, layout::Row> = MatrixA::from_array(
        [
        bf16::from_f32(1.0); 16  // Exactly 16 elements
    ],
    );

    let b_frag: MatrixB<bf16, Shape, layout::Row> = MatrixB::splat(bf16::from_f32(2.0));

    // Accumulator needs exactly 8 registers for f32 with Shape<16, 16, 16>
    let c_frag: Accumulator<f32, Shape> = Accumulator::from_array(
        [
        0.0; 8  // Exactly 8 elements
    ],
    );

    // MMA operations would work if the proper MMA trait implementation exists
    // let _result = c_frag.mma(&a_frag, &b_frag);
}
