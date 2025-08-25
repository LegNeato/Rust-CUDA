// Test simplified Shuffle struct
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
    fn new(mask: WarpMask, width: ShuffleWidth) -> Self {
        Self {
            mask,
            width,
            _phantom: PhantomData,
        }
    }
    
    // Simplified down method
    unsafe fn down(self, value: T, delta: u32) -> Result<T, cuda_std::warp::shuffle::InvalidLane> {
        T::shuffle_down(self.mask, value, delta, self.width.value())
    }
}

#[kernel]
pub unsafe fn test_simple_shuffle() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let shuffle = SimpleShuffle::<i32>::new(mask, width);
    
    let value = 42i32;
    let _result = shuffle.down(value, 1);
}