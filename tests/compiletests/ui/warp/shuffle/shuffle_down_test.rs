// Test shuffle down with pattern
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle, ShuffleWidth};
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_shuffle_down() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let shuffle = Shuffle::<i32>::new(mask, width);
    
    let value = 42i32;
    let pattern = patterns::Down::new(1);
    
    // This is where it might crash
    let _result = shuffle.down(value, pattern);
}