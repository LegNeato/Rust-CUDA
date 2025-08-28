// Test that Shape<16, 8, 16> correctly prevents WMMA load/store operations
// compile-fail

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixB, TensorCore};

#[kernel]
pub unsafe fn test_mma_only_shape_no_wmma() {
    // Shape<16, 8, 16> is MMA-only according to LLVM spec
    type Shape = dims::Shape<16, 8, 16>;
    let tc = TensorCore::<bf16, Shape>::new();

    // Shape<16, 8, 16> doesn't support WMMA load
    let mut b_frag: MatrixB<bf16, Shape, layout::Row> = tc.matrix_b();
    b_frag.load::<16>(core::ptr::null());
}
