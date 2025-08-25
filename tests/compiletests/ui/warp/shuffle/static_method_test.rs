// Test static method vs instance method
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue, ShuffleWidth};

struct TestShuffleI32 {
    mask: WarpMask,
    width: ShuffleWidth,
}

impl TestShuffleI32 {
    // Static method
    unsafe fn do_shuffle_static(mask: WarpMask, width: ShuffleWidth, value: i32) -> Result<i32, cuda_std::warp::shuffle::InvalidLane> {
        <i32 as ShuffleValue>::shuffle_down(mask, value, 1, width.value())
    }
    
    // Instance method with self
    unsafe fn do_shuffle_self(&self, value: i32) -> Result<i32, cuda_std::warp::shuffle::InvalidLane> {
        <i32 as ShuffleValue>::shuffle_down(self.mask, value, 1, self.width.value())
    }
}

#[kernel]
pub unsafe fn test_static_method() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let value = 42i32;
    
    // Static method call
    let _result = TestShuffleI32::do_shuffle_static(mask, width, value);
}

#[kernel]
pub unsafe fn test_instance_method() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let value = 42i32;
    
    let shuffle = TestShuffleI32 { mask, width };
    
    // Instance method call - this might crash
    let _result = shuffle.do_shuffle_self(value);
}