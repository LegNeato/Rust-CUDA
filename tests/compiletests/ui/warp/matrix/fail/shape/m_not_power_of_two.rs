// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_invalid_m_dimension() {
    // M dimension must be power of 2
    let _tc = TensorCore::<f16, dims::Shape<7, 8, 16>>::new();
}
