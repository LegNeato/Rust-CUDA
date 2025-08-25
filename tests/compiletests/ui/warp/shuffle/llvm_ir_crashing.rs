// build-pass
// compile-flags: --emit=llvm-ir

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle, ShuffleWidth};
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_crashing() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let shuffle = Shuffle::<i32>::new(mask, width);
    
    let value = 42i32;
    let pattern = patterns::Down::new(1);
    
    // Method call - this crashes
    let _result = shuffle.down(value, pattern);
}