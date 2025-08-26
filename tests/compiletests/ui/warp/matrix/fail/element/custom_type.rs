// compile-fail

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

struct CustomType;

#[kernel]
pub unsafe fn test_custom_type() {
    // CustomType doesn't implement MatrixElement
    let _tc = TensorCore::<CustomType, dims::Shape<16, 16, 16>>::new();
}
