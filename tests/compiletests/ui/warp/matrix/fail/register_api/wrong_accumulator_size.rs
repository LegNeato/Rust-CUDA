// Test that Accumulator::from_array fails with wrong number of elements

#![no_std]

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, Accumulator, MatrixExt};

#[kernel]
pub unsafe fn test_wrong_accumulator_size() {
    type Shape = dims::Shape<16, 8, 16>;

    let _bad: Accumulator<f32, Shape> = Accumulator::from_array(
        [
        // Wrong size, should be 4 for Shape<16, 8, 16>
        0.0; 8
    ],
    );
}
