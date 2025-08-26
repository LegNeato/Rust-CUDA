// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, TensorCore};

#[kernel]
pub unsafe fn test_invalid_stride() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();
    let mut a = tc.matrix_a::<layout::Row>();
    let ptr: *const f16 = core::ptr::null();

    // Stride must be multiple of 8 for f16
    a.load::<7>(ptr);
}
