// Test vote operations with different masks
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{Predicate, Vote, WarpMask};

#[kernel]
pub unsafe fn test_vote_with_partial_masks() {
    let lane_id = cuda_std::warp::lane_id();

    // Test with lower half mask
    let lower_mask = WarpMask::new(0x0000FFFF);
    let vote_lower = Vote::new(lower_mask);

    let pred = Predicate::<()>::new(lane_id < 8);
    let all_result = vote_lower.all(pred);
    // Only checking lanes 0-15, lanes 0-7 satisfy predicate
    assert!(!all_result.is_unanimous());

    // Test with upper half mask
    let upper_mask = WarpMask::new(0xFFFF0000);
    let vote_upper = Vote::new(upper_mask);

    let pred_upper = Predicate::<()>::new(lane_id >= 16);
    let all_upper = vote_upper.all(pred_upper);
    assert!(all_upper.is_unanimous()); // All lanes 16-31 satisfy >= 16
}

#[kernel]
pub unsafe fn test_vote_with_single_lane() {
    let lane_id = cuda_std::warp::lane_id();

    // Test with single lane masks
    for target_lane in 0..32 {
        let single_mask = WarpMask::lane(target_lane);
        let vote_single = Vote::new(single_mask);

        let pred = Predicate::<()>::new(lane_id == target_lane);
        let all_result = vote_single.all(pred);

        // When voting on a single lane, if that lane matches,
        // the vote should be unanimous
        if lane_id == target_lane {
            assert!(all_result.is_unanimous());
        }
    }
}

#[kernel]
pub unsafe fn test_vote_with_alternating_masks() {
    let lane_id = cuda_std::warp::lane_id();

    // Even lanes mask
    let even_mask = WarpMask::new(0x55555555);
    let vote_even = Vote::new(even_mask);

    let pred_even = Predicate::<()>::new((lane_id % 2) == 0);
    let all_even = vote_even.all(pred_even);
    assert!(all_even.is_unanimous()); // All even lanes are even

    // Odd lanes mask
    let odd_mask = WarpMask::new(0xAAAAAAAA);
    let vote_odd = Vote::new(odd_mask);

    let pred_odd = Predicate::<()>::new((lane_id % 2) == 1);
    let all_odd = vote_odd.all(pred_odd);
    assert!(all_odd.is_unanimous()); // All odd lanes are odd
}

#[kernel]
pub unsafe fn test_ballot_with_masks() {
    let lane_id = cuda_std::warp::lane_id();

    // Test ballot with quarter mask (lanes 0-7)
    let quarter_mask = WarpMask::new(0x000000FF);
    let vote_quarter = Vote::new(quarter_mask);

    let pred = Predicate::<()>::new(lane_id < 4);
    let ballot = vote_quarter.ballot(pred);

    // Only lanes 0-3 should be set in the ballot
    assert_eq!(ballot.count(), 4);
    assert_eq!(ballot.mask().raw() & 0x000000FF, 0x0000000F);

    // Lanes outside the vote mask shouldn't affect the result
    for lane in 8..32 {
        assert!(!ballot.lane_voted(lane));
    }
}

#[kernel]
pub unsafe fn test_vote_mask_combinations() {
    let lane_id = cuda_std::warp::lane_id();

    // Combine multiple conditions with masks
    let mask1 = WarpMask::new(0x0F0F0F0F); // Pattern of 4 on, 4 off
    let mask2 = WarpMask::new(0xF0F0F0F0); // Opposite pattern

    let vote1 = Vote::new(mask1);
    let vote2 = Vote::new(mask2);

    let pred_small = Predicate::<()>::new(lane_id < 16);

    let result1 = vote1.ballot(pred_small);
    let result2 = vote2.ballot(pred_small);

    // The union should cover lanes 0-15
    let combined = result1.mask().raw() | result2.mask().raw();
    assert_eq!(combined & 0x0000FFFF, 0x0000FFFF);
}
