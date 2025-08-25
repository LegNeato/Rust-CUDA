// Test if static methods work better
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleWidth, ShuffleValue};
use core::marker::PhantomData;

#[repr(C)]
struct SimpleShuffle<T> {
    mask: WarpMask,
    width: ShuffleWidth,
    _phantom: PhantomData<T>,
}

impl<T: ShuffleValue> SimpleShuffle<T> {
    // Static method - doesn't take self
    unsafe fn down_static(mask: WarpMask, width: ShuffleWidth, value: T, delta: u32) -> Result<T, cuda_std::warp::shuffle::InvalidLane> {
        T::shuffle_down(mask, value, delta, width.value())
    }
}

#[kernel]
pub unsafe fn test_static_shuffle() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    
    let value = 42i32;
    let _result = SimpleShuffle::<i32>::down_static(mask, width, value, 1);
}