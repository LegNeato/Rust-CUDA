// build-pass
// compile-flags: --emit=llvm-ir --error-format=human

//! Test to reproduce runtime issue with MatrixB load

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixA, MatrixB, TensorCore};

#[kernel]
pub unsafe fn attention_issue_test(data_ptr: *const bf16, out_ptr: *mut bf16) {
    // Valid shapes for bf16 according to CUDA docs: 16x16x16, 32x8x16, 8x32x16
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<bf16, Shape>::new();

    // Create MatrixB fragment with Col layout (for K^T in attention)
    let mut k_frag: MatrixB<bf16, Shape, layout::Col> = tc.matrix_b();

    // Try to load with stride of 128 (DIM from original kernel)
    // This is what's failing in the attention kernel
    k_frag.load::<128>(data_ptr);

    // Write marker to show we got past the load
    *out_ptr = bf16::from_f32(1.0);
}

#[kernel]
pub unsafe fn simpler_test(data_ptr: *const bf16, out_ptr: *mut bf16) {
    // Even simpler case - Shape<16, 16, 16> with Row layout
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<bf16, Shape>::new();

    let mut b_frag: MatrixB<bf16, Shape, layout::Row> = tc.matrix_b();

    // Try loading with stride 16
    b_frag.load::<16>(data_ptr);

    *out_ptr = bf16::from_f32(2.0);
}
