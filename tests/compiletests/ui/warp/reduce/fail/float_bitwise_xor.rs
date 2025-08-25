// build-fail

use cuda_std::kernel;
use cuda_std::warp::reduce::Reduction;

#[kernel]
pub unsafe fn test_float_bitwise_xor() {
    let reduction = Reduction::<f32>::all_threads();

    // f32 doesn't implement BitwiseReduceValue
    let _result = reduction.xor(1.414f32);
}
