// Test generic struct with ShuffleValue bound
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue, ShuffleWidth};
use core::marker::PhantomData;

struct TestShuffle<T: ShuffleValue> {
    mask: WarpMask,
    width: ShuffleWidth,
    _phantom: PhantomData<T>,
}

impl<T: ShuffleValue> TestShuffle<T> {
    fn new(mask: WarpMask, width: ShuffleWidth) -> Self {
        Self {
            mask,
            width,
            _phantom: PhantomData,
        }
    }
    
    unsafe fn do_shuffle(&self, value: T) -> Result<T, cuda_std::warp::shuffle::InvalidLane> {
        T::shuffle_down(self.mask, value, 1, self.width.value())
    }
}

#[kernel]
pub unsafe fn test_generic_struct() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    
    let shuffle = TestShuffle::<i32>::new(mask, width);
    let value = 42i32;
    
    // This might crash
    let _result = shuffle.do_shuffle(value);
}