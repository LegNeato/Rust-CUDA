// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_invalid_k_dimension() {
    // K dimension must be 4, 8, 16, 32, 64, or 128
    let _tc = TensorCore::<f16, dims::Shape<16, 16, 3>>::new();
}
