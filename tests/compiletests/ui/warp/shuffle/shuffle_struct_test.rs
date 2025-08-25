// Test Shuffle struct without patterns
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle, ShuffleWidth};

#[kernel]
pub unsafe fn test_shuffle_struct() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    
    let _shuffle = Shuffle::<i32>::new(mask, width);
    // Just create the struct, don't use it yet
}