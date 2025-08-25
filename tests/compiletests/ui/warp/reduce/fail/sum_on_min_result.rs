// build-fail

use cuda_std::kernel;
use cuda_std::warp::reduce::Reduction;

#[kernel]
pub unsafe fn test_sum_on_min_result() {
    let reduction = Reduction::<i32>::all_threads();
    let min_result = reduction.min(42);

    // sum() method only exists for Add results, not Min
    let _wrong = min_result.sum();
}
