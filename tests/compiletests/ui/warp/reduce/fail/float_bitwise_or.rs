// build-fail

use cuda_std::kernel;
use cuda_std::warp::reduce::Reduction;

#[kernel]
pub unsafe fn test_float_bitwise_or() {
    let reduction = Reduction::<f64>::all_threads();

    // f64 doesn't implement BitwiseReduceValue
    let _result = reduction.or(2.718f64);
}
