// Test that from_array fails with wrong number of elements

#![no_std]

use cuda_std::bf16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixA, MatrixExt};

#[kernel]
pub unsafe fn test_wrong_size() {
    type Shape = dims::Shape<16, 8, 16>;

    let _bad = MatrixA::<bf16, Shape, layout::Row>::from_array(
        [
         // Wrong size, should be 8
        bf16::from_f32(1.0); 16
    ],
    );
}
