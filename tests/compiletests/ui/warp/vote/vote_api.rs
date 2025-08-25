// Test vote operation API patterns
// build-pass

use cuda_std::kernel;
use cuda_std::warp::vote::{AllVoteResult, AnyVoteResult, BallotResult, WarpMaskVoteExt};
use cuda_std::warp::{Predicate, Vote, WarpMask};

#[kernel]
pub unsafe fn test_vote_builder_pattern() {
    let mask = WarpMask::all();
    let lane_id = cuda_std::warp::lane_id();

    // Test builder pattern
    let vote = Vote::new(mask);

    // Chain multiple vote operations
    let pred_even = Predicate::<()>::new((lane_id % 2) == 0);
    let pred_small = Predicate::<()>::new(lane_id < 16);

    let all_even = vote.all(pred_even);
    let any_small = vote.any(pred_small);
    let ballot_result = vote.ballot(pred_small);

    // Use results
    if all_even.has_dissent() {
        // Not all threads are even
    }

    if any_small.has_any() {
        // At least one thread has lane_id < 16
    }

    let active_count = ballot_result.count();
    assert_eq!(active_count, 16);
}

#[kernel]
pub unsafe fn test_vote_all_threads() {
    let lane_id = cuda_std::warp::lane_id();

    // Test convenience constructor for all threads
    let vote = Vote::all_threads();

    let pred = Predicate::<()>::new(lane_id < 32);
    let all_result = vote.all(pred);
    assert!(all_result.is_unanimous());
}

#[kernel]
pub unsafe fn test_warp_mask_extension() {
    let mask = WarpMask::all();
    let lane_id = cuda_std::warp::lane_id();

    // Test extension trait methods
    let vote = mask.vote();

    // Test convenience methods
    let all_result = mask.all_satisfy(lane_id < 32);
    assert!(all_result.is_unanimous());

    let any_result = mask.any_satisfy(lane_id == 0);
    assert!(any_result.has_any());

    let ballot_result = mask.ballot_for(lane_id < 16);
    assert_eq!(ballot_result.count(), 16);
}

#[kernel]
pub unsafe fn test_vote_result_types() {
    let mask = WarpMask::all();
    let vote = Vote::new(mask);
    let lane_id = cuda_std::warp::lane_id();

    // Test AllVoteResult methods
    let all_result: AllVoteResult = vote.all(Predicate::<()>::new(lane_id < 32));
    assert!(all_result.is_unanimous());
    assert!(!all_result.has_dissent());
    assert!(all_result.as_bool());

    // Test AnyVoteResult methods
    let any_result: AnyVoteResult = vote.any(Predicate::<()>::new(lane_id == 0));
    assert!(any_result.has_any());
    assert!(!any_result.has_none());
    assert!(any_result.as_bool());

    // Test BallotResult methods
    let ballot_result: BallotResult = vote.ballot(Predicate::<()>::new(lane_id % 2 == 0));
    assert_eq!(ballot_result.count(), 16);
    assert!(ballot_result.lane_voted(0));
    assert!(!ballot_result.lane_voted(1));
    assert!(!ballot_result.none_voted());

    // Test mask extraction
    let result_mask = ballot_result.mask();
    assert_eq!(result_mask.raw(), 0x55555555); // Every other lane
}

#[kernel]
pub unsafe fn test_ballot_iteration() {
    let mask = WarpMask::all();
    let vote = Vote::new(mask);
    let lane_id = cuda_std::warp::lane_id();

    // Create a ballot with specific lanes
    let ballot = vote.ballot(Predicate::<()>::new(lane_id < 8));

    // Test iteration over true lanes
    let mut count = 0;
    for lane in ballot.true_lanes() {
        assert!(lane < 8);
        count += 1;
    }
    assert_eq!(count, 8);

    // Test all_voted method
    let lower_half = WarpMask::new(0x000000FF);
    assert!(ballot.all_voted(lower_half));

    let upper_half = WarpMask::new(0xFFFF0000);
    assert!(!ballot.all_voted(upper_half));
}
