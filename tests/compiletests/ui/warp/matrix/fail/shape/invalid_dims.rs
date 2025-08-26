// Test that invalid matrix dimensions cannot be created
// compile-fail

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, TensorCore};

#[kernel]
pub unsafe fn test_invalid_matrix_dimensions() {
    // This should fail - 17x17x17 is not a valid tensor core shape
    let invalid = TensorCore::<f16, dims::Shape<17, 17, 17>>::new();
    //~^ ERROR the trait bound `Shape<17, 17, 17>: TensorCoreShape` is not satisfied

    // This should fail - 64x64x64 is too large
    let too_large = TensorCore::<f16, dims::Shape<64, 64, 64>>::new();
    //~^ ERROR the trait bound `Shape<64, 64, 64>: TensorCoreShape` is not satisfied

    // This should fail - 1x1x1 is too small
    let too_small = TensorCore::<f16, dims::Shape<1, 1, 1>>::new();
    //~^ ERROR the trait bound `Shape<1, 1, 1>: TensorCoreShape` is not satisfied
}
