// Test that from_array fails with wrong element type

#![no_std]

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixA, MatrixExt};

#[kernel]
pub unsafe fn test_type_mismatch() {
    type Shape = dims::Shape<16, 8, 16>;

    let _bad = MatrixA::<bf16, Shape, layout::Row>::from_array(
        [
         // Wrong type, should be bf16
        1.0f32; 8
    ],
    );
}
