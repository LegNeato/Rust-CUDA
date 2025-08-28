// Test that MatrixB::from_array fails with wrong number of elements

#![no_std]

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixB, MatrixExt};

#[kernel]
pub unsafe fn test_matrix_b_wrong_size() {
    type Shape = dims::Shape<16, 16, 16>;

    let _bad = MatrixB::<f16, Shape, layout::Row>::from_array(
        [
        // Wrong size, should be 16
        f16::from_f32(1.0); 8
    ],
    );
}
