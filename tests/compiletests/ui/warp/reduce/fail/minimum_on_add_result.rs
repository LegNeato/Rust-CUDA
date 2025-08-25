// build-fail

use cuda_std::kernel;
use cuda_std::warp::reduce::Reduction;

#[kernel]
pub unsafe fn test_minimum_on_add_result() {
    let reduction = Reduction::<i32>::all_threads();
    let add_result = reduction.add(42);

    // minimum() method only exists for Min results, not Add
    let _wrong = add_result.minimum();
}
