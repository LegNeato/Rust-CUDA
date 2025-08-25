// build-fail

use cuda_std::kernel;
use cuda_std::warp::reduce::Reduction;

#[kernel]
pub unsafe fn test_non_copy_type() {
    struct NonCopyType {
        data: Vec<i32>,
    }

    // NonCopyType doesn't implement Copy or ReduceValue
    let _reduction = Reduction::<NonCopyType>::all_threads();
}
