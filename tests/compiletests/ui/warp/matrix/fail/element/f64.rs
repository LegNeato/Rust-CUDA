// compile-fail

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_f64_element() {
    // f64 doesn't implement MatrixElement
    let _tc = TensorCore::<f64, dims::Shape<16, 16, 16>>::new();
}
