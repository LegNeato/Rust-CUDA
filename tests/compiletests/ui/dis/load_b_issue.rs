// build-fail
// compile-flags: -Cllvm-args=--disassemble-entry=test_load_b_issue --error-format=human

//! Test to investigate load_b_into codegen issue

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixB, TensorCore};
use cuda_std::{bf16, f16};

#[kernel]
pub unsafe fn test_load_b_issue() {
    type Shape = dims::Shape<16, 16, 16>;

    // Create MatrixB using TensorCore builder
    let tc = TensorCore::<f16, Shape>::new();
    let mut b_frag = tc.matrix_b::<layout::Row>();

    // Dummy data array
    let data = [f16::from_f32(0.0); 256];
    let data_ptr = data.as_ptr();

    // This load_b operation might be generating a trap
    b_frag.load::<16>(data_ptr);
}

#[kernel]
pub unsafe fn test_load_b_different_types() {
    type Shape = dims::Shape<16, 16, 16>;

    // Test f16
    let tc_f16 = TensorCore::<f16, Shape>::new();
    let mut b_f16 = tc_f16.matrix_b::<layout::Row>();
    let data_f16 = [f16::from_f32(0.0); 256];
    b_f16.load::<16>(data_f16.as_ptr());

    // Test bf16
    let tc_bf16 = TensorCore::<bf16, Shape>::new();
    let mut b_bf16 = tc_bf16.matrix_b::<layout::Row>();
    let data_bf16 = [bf16::from_f32(0.0); 256];
    b_bf16.load::<16>(data_bf16.as_ptr());

    // Test i8
    let tc_i8 = TensorCore::<i8, Shape>::new();
    let mut b_i8 = tc_i8.matrix_b::<layout::Row>();
    let data_i8 = [0i8; 256];
    b_i8.load::<16>(data_i8.as_ptr());

    // Test u8
    let tc_u8 = TensorCore::<u8, Shape>::new();
    let mut b_u8 = tc_u8.matrix_b::<layout::Row>();
    let data_u8 = [0u8; 256];
    b_u8.load::<16>(data_u8.as_ptr());
}

#[kernel]
pub unsafe fn test_tf32_load_b() {
    type Shape = dims::Shape<16, 16, 8>;

    // Test TF32 specifically since it has different shape requirements
    let tc_f32 = TensorCore::<f32, Shape>::new();
    let mut b_tf32 = tc_f32.matrix_b::<layout::Row>();
    let data_tf32 = [0.0f32; 256];
    b_tf32.load::<16>(data_tf32.as_ptr());
}
