// build-pass
// compile-flags: --emit=llvm-ir --error-format=human

//! Test that MMA-only shapes can be used with the register-based API

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, Accumulator, MatrixA, MatrixB, MatrixExt, TensorCore};

#[kernel]
pub unsafe fn test_mma_only_shape_api() {
    // Shape<16, 8, 16> is MMA-only (no WMMA load/store)
    type Shape = dims::Shape<16, 8, 16>;
    let tc = TensorCore::<bf16, Shape>::new();

    // Can't use load - this would fail to compile:
    // let mut a_frag = tc.matrix_a();
    // a_frag.load::<16>(ptr); // ERROR: requires WmmaShape

    // Instead, use register-based construction:

    // Option 1: From array values (compile-time checked)
    let a_frag: MatrixA<bf16, Shape, layout::Row> = MatrixA::from_array([bf16::from_f32(1.0); 8]);

    // Option 2: Splat a single value
    let b_frag: MatrixB<bf16, Shape, layout::Row> = MatrixB::splat(bf16::from_f32(2.0));

    // Option 3: Create accumulator
    let c_frag: Accumulator<f32, Shape> = Accumulator::splat(0.0);

    // MMA operations work normally for Shape<16, 8, 16>
    let _result = c_frag.mma(&a_frag, &b_frag);
}

#[kernel]
pub unsafe fn test_wmma_shape_can_use_both_apis() {
    // Shape<16, 16, 16> supports both WMMA and MMA
    type Shape = dims::Shape<16, 16, 16>;
    use cuda_std::f16;
    let tc = TensorCore::<f16, Shape>::new();

    // Can use traditional WMMA load
    let mut a_frag = tc.matrix_a::<layout::Row>();
    // a_frag.load::<16>(data_ptr); // Would work if we had a pointer

    // Can also use register-based API
    let b_frag: MatrixB<f16, Shape, layout::Row> = MatrixB::splat(f16::from_f32(1.0));
    let c_frag: Accumulator<f32, Shape> = Accumulator::from_array([0.0; 8]);

    // MMA works with f16 for Shape<16, 16, 16>
    let _result = c_frag.mma(&a_frag, &b_frag);
}
