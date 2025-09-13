//! Test for the new type-safe warp matrix API

#![cfg_attr(target_os = "cuda", no_std)]
#![feature(abi_ptx)]

use cuda_std::prelude::*;

#[kernel]
pub unsafe fn test_wmma_type_safety() {
    use cuda_std::warp::matrix::{Matrix16x16x16, Layout};
    
    // Create a builder for 16x16x16 operations
    let builder = Matrix16x16x16::new();
    
    // Create fragments with compile-time type checking
    let a = builder.a_f16(Layout::RowMajor);
    let b = builder.b_f16(Layout::ColMajor);
    let mut c = builder.accumulator_f32();
    
    // Initialize accumulator
    c.fill(0.0);
    
    // This would fail to compile if we tried:
    // - let a = builder.a_f16(Layout::RowMajor);
    // - let b = builder.b_i8(Layout::ColMajor);  // Wrong type!
    // - mma(&a, &b, &c);  // Type error!
    
    // The following would also fail:
    // - Using mismatched dimensions (e.g., 16x16x16 with 32x8x16)
    // - Using wrong accumulator types
    // - Using incompatible layouts
}