// Test pattern creation
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_patterns() {
    let _down = patterns::Down::new(1);
    let _up = patterns::Up::new(1);
    let _xor = patterns::Xor::new(16);
    let _idx = patterns::Index::new(0);
}