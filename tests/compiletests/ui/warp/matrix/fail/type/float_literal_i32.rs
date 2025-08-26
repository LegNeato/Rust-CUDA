// compile-fail

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_wrong_literal_type() {
    type Shape = dims::Shape<16, 16, 16>;

    let tc_i8 = TensorCore::<i8, Shape>::new();
    let mut c_i32 = tc_i8.accumulator();

    // Float literal incompatible with i32 accumulator
    c_i32.fill(1.5f32);
}
