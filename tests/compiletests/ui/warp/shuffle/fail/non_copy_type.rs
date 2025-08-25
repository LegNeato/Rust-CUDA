// build-fail

use cuda_std::kernel;
use cuda_std::warp::shuffle::Shuffle;

#[kernel]
pub unsafe fn test_non_copy_type() {
    struct NonCopyType {
        data: Vec<i32>,
    }

    // NonCopyType doesn't implement Copy or ShuffleValue
    let _shuffle = Shuffle::<NonCopyType>::full_warp();
}
