// Test non-generic struct
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue, ShuffleWidth};

struct TestShuffleI32 {
    mask: WarpMask,
    width: ShuffleWidth,
}

impl TestShuffleI32 {
    fn new(mask: WarpMask, width: ShuffleWidth) -> Self {
        Self { mask, width }
    }
    
    unsafe fn do_shuffle(&self, value: i32) -> Result<i32, cuda_std::warp::shuffle::InvalidLane> {
        <i32 as ShuffleValue>::shuffle_down(self.mask, value, 1, self.width.value())
    }
}

#[kernel]
pub unsafe fn test_non_generic() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    
    let shuffle = TestShuffleI32::new(mask, width);
    let value = 42i32;
    
    let _result = shuffle.do_shuffle(value);
}