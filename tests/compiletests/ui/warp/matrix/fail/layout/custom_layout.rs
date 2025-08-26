// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_invalid_layout() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();
    let acc = tc.accumulator();
    let ptr: *mut f32 = core::ptr::null_mut();

    struct CustomLayout;

    // CustomLayout doesn't implement Layout trait
    acc.store::<CustomLayout, 16>(ptr);
}
