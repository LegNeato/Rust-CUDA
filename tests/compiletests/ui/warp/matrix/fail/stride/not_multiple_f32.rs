// compile-fail

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, TensorCore};

#[kernel]
pub unsafe fn test_invalid_f32_stride() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f32, Shape>::new();
    let mut a = tc.matrix_a::<layout::Row>();
    let ptr: *const f32 = core::ptr::null();

    // Stride must be multiple of 4 for f32
    a.load::<3>(ptr);
}
