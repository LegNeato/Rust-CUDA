// Test if generic Result returns cause issues
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue};

// Non-generic function that returns Result
unsafe fn shuffle_i32_direct(mask: WarpMask, value: i32) -> Result<i32, cuda_std::warp::shuffle::InvalidLane> {
    <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32)
}

// Generic function that returns Result  
unsafe fn shuffle_generic<T: ShuffleValue>(mask: WarpMask, value: T) -> Result<T, cuda_std::warp::shuffle::InvalidLane> {
    T::shuffle_down(mask, value, 1, 32)
}

#[kernel]
pub unsafe fn test_direct() {
    let mask = WarpMask::all();
    let value = 42i32;
    
    // This should work - non-generic
    let _result = shuffle_i32_direct(mask, value);
}

#[kernel]
pub unsafe fn test_generic() {
    let mask = WarpMask::all();
    let value = 42i32;
    
    // This might crash - generic function
    let _result = shuffle_generic::<i32>(mask, value);
}