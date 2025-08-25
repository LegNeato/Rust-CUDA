// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=test_crashing --error-format=human

use cuda_std::kernel;
use cuda_std::warp::shuffle::patterns;
use cuda_std::warp::{Shuffle, ShuffleWidth, WarpMask};

#[kernel]
pub unsafe fn test_crashing() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    // This is now a ZST - the parameters are ignored
    let shuffle = Shuffle::<i32>::new(mask, width);

    let value = 42i32;
    let pattern = patterns::Down::new(1);

    // Method call - should not crash now as shuffle is a ZST
    let _result = shuffle.down(value, pattern);
}
