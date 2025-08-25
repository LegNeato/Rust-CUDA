// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=test_working --error-format=human

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue};

#[kernel]
pub unsafe fn test_working() {
    let mask = WarpMask::all();
    let value = 42i32;
    
    // Direct trait call - this works
    let _result = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);
}