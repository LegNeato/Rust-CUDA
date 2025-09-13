// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=test_intrinsic --error-format=human

use cuda_std::kernel;
use cuda_std::warp::matrix::intrinsics::*;

static mut RESULT: [f32; 4] = [0.0; 4];

#[kernel]
pub unsafe fn test_intrinsic() {
    // Call the 16x8x16 intrinsic directly
    let result = wmma_mma_f16_f32_row_row_m16n8k16(
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        0.0, 0.0, 0.0, 0.0,
    );
    RESULT = result;
}