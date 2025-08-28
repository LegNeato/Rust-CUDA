// build-pass
// compile-flags: --emit=llvm-ir --error-format=human

//! Test ldmatrix functionality for loading from shared memory

#![cfg_attr(target_os = "cuda", no_std)]
#![allow(improper_ctypes)]

use cuda_std::bf16;
use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, Accumulator, MatrixA, MatrixB, MatrixExt};

// Module-level shared memory for testing
#[cuda_std::address_space(shared)]
static mut SMEM_BF16: [bf16; 256] = [bf16::ZERO; 256];

#[cuda_std::address_space(shared)]
static mut SMEM_F16: [f16; 256] = [f16::ZERO; 256];

#[kernel]
pub unsafe fn test_ldmatrix_shape_16x8x16_bf16() {
    // Shape<16, 8, 16> supports ldmatrix for shared memory loading
    type Shape = dims::Shape<16, 8, 16>;

    // Initialize shared memory
    for i in 0..256 {
        SMEM_BF16[i] = bf16::from_f32(1.0 + i as f32 * 0.1);
    }
    let shared_ptr = core::ptr::addr_of_mut!(SMEM_BF16[0]) as *const bf16;

    // Load matrix A from shared memory using ldmatrix
    let mut a_frag: MatrixA<bf16, Shape, layout::Row> = MatrixA::new();
    a_frag.load_from_shared::<16>(shared_ptr);

    // Load matrix B from shared memory using ldmatrix
    let mut b_frag: MatrixB<bf16, Shape, layout::Row> = MatrixB::new();
    b_frag.load_from_shared::<16>(shared_ptr);

    // Can still use register-based initialization
    let c_frag: Accumulator<f32, Shape> = Accumulator::splat(0.0);

    // MMA operations work normally
    let _result = c_frag.mma(&a_frag, &b_frag);
}

#[kernel]
pub unsafe fn test_ldmatrix_shape_16x8x16_f16() {
    // Shape<16, 8, 16> with f16 also supports ldmatrix
    type Shape = dims::Shape<16, 8, 16>;

    // Initialize shared memory
    for i in 0..256 {
        SMEM_F16[i] = f16::from_f32(1.0 + i as f32 * 0.1);
    }
    let shared_ptr = core::ptr::addr_of_mut!(SMEM_F16[0]) as *const f16;

    // Load with different layouts
    let mut a_row: MatrixA<f16, Shape, layout::Row> = MatrixA::new();
    a_row.load_from_shared::<16>(shared_ptr);

    let mut a_col: MatrixA<f16, Shape, layout::Col> = MatrixA::new();
    a_col.load_from_shared::<16>(shared_ptr);

    let mut b_row: MatrixB<f16, Shape, layout::Row> = MatrixB::new();
    b_row.load_from_shared::<16>(shared_ptr);

    let mut b_col: MatrixB<f16, Shape, layout::Col> = MatrixB::new();
    b_col.load_from_shared::<16>(shared_ptr);
}

#[kernel]
pub unsafe fn test_shape_16x16x16_supports_both() {
    // Shape<16, 16, 16> supports both WMMA and ldmatrix
    type Shape = dims::Shape<16, 16, 16>;

    // Global memory (stack allocated for testing)
    let global_data = [bf16::from_f32(1.0); 256];
    let global_ptr = global_data.as_ptr();

    // Use module-level shared memory
    for i in 0..256 {
        SMEM_BF16[i] = bf16::from_f32(2.0 + i as f32 * 0.1);
    }
    let shared_ptr = core::ptr::addr_of_mut!(SMEM_BF16[0]) as *const bf16;

    let mut a_frag: MatrixA<bf16, Shape, layout::Row> = MatrixA::new();

    // Can load from global memory (WMMA)
    a_frag.load::<16>(global_ptr);

    // Can also load from shared memory (ldmatrix)
    a_frag.load_from_shared::<16>(shared_ptr);
}
