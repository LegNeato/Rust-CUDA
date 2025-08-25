// build-fail

use cuda_std::kernel;
use cuda_std::warp::vote::VoteEquality;
use cuda_std::warp::WarpMask;

#[kernel]
pub unsafe fn test_vote_non_copy() {
    struct NonCopyType {
        data: [u32; 3],
    }

    let mask = WarpMask::all();
    let value = NonCopyType {
        data: [1, 2, 3],
    };

    // NonCopyType doesn't implement Copy or VoteEquality
    let _result = NonCopyType::vote_all_equal(mask, value);
}
