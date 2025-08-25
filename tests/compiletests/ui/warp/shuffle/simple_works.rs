// Minimal test that works
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue};

#[kernel]
pub unsafe fn test_works() {
    let mask = WarpMask::all();
    let value = 42i32;
    
    // This works
    let _result = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);
}