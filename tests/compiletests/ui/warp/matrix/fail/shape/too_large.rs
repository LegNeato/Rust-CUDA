// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_dimensions_too_large() {
    // Dimensions exceed hardware limits
    let _tc = TensorCore::<f16, dims::Shape<64, 64, 64>>::new();
}
