// Test CUDA warp vote functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{self, Predicate, Vote, WarpMask};

#[kernel]
pub unsafe fn test_warp_vote_basic() {
    let mask = WarpMask::all();
    let lane_id = warp::lane_id();

    // Test using the builder pattern
    let vote = Vote::new(mask);

    // Test vote all
    let all_even = vote.all(Predicate::<()>::new((lane_id % 2) == 0));
    if all_even.is_unanimous() {
        // All threads have even lane IDs
    }

    // Test vote any
    let any_zero = vote.any(Predicate::<()>::new(lane_id == 0));
    if any_zero.has_any() {
        // At least one thread has lane ID 0
    }

    // Test ballot operation
    let ballot = vote.ballot(Predicate::<()>::new(lane_id < 16));
    let _count = ballot.count();
    let _lane0_voted = ballot.lane_voted(0);
}
