// build-fail
// compile-flags: --emit=llvm-ir --error-format=human

//! Test that MMA-only shapes correctly fail WMMA global memory operations

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixA, MatrixB};

#[kernel]
pub unsafe fn test_shape_16x8x16_cannot_load_from_global() {
    // Shape<16, 8, 16> is MMA-only (no WMMA global load)
    type Shape = dims::Shape<16, 8, 16>;

    let global_data = [bf16::from_f32(1.0); 256];
    let global_ptr = global_data.as_ptr();

    let mut a_frag: MatrixA<bf16, Shape, layout::Row> = MatrixA::new();

    // This should fail - Shape<16, 8, 16> doesn't implement WmmaShape
    a_frag.load::<16>(global_ptr);
    //~^ ERROR: `dims::Shape<16, 8, 16>` does not support WMMA load/store operations
}

#[kernel]
pub unsafe fn test_shape_16x8x16_b_matrix_cannot_load_from_global() {
    type Shape = dims::Shape<16, 8, 16>;

    let global_data = [bf16::from_f32(1.0); 256];
    let global_ptr = global_data.as_ptr();

    let mut b_frag: MatrixB<bf16, Shape, layout::Row> = MatrixB::new();

    // This should also fail
    b_frag.load::<16>(global_ptr);
    //~^ ERROR: `dims::Shape<16, 8, 16>` does not support WMMA load/store operations
}
