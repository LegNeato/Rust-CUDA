#![cfg_attr(target_os = "cuda", no_std)]
#![feature(abi_ptx)]
#![feature(core_intrinsics)]

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue, Shuffle, ShuffleWidth};
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_working() {
    let mask = WarpMask::all();
    let value = 42i32;
    
    // Direct trait call - this works
    let _result = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);
}

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