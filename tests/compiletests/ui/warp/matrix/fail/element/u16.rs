// compile-fail

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_u16_element() {
    // u16 doesn't implement MatrixElement
    let _tc = TensorCore::<u16, dims::Shape<16, 16, 16>>::new();
}
