// Minimal shuffle test to isolate issue
// build-pass

use cuda_std::kernel;
use cuda_std::warp::ShuffleValue;
use cuda_std::warp::WarpMask;

#[kernel]
pub unsafe fn test_minimal_shuffle() {
    let mask = WarpMask::all();
    let value = 42i32;
    
    // Direct call to the trait method
    let _result = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);
}