// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, TensorCore};

#[kernel]
pub unsafe fn test_stride_too_small() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();
    let mut a = tc.matrix_a::<layout::Row>();
    let ptr: *const f16 = core::ptr::null();

    // Stride too small for matrix dimensions
    a.load::<4>(ptr);
}
