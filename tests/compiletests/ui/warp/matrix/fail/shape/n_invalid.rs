// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_invalid_n_dimension() {
    // N dimension must be 8, 16, or 32
    let _tc = TensorCore::<f16, dims::Shape<16, 15, 16>>::new();
}
