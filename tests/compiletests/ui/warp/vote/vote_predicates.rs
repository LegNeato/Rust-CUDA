// Test vote operations with predicates
// build-pass

use cuda_std::kernel;
use cuda_std::warp::vote::predicates::{Comparison, Condition, PredicateExt, Validation};
use cuda_std::warp::{Predicate, Vote, WarpMask};

#[kernel]
pub unsafe fn test_predicate_types() {
    let mask = WarpMask::all();
    let vote = Vote::new(mask);
    let lane_id = cuda_std::warp::lane_id();

    // Test condition predicate
    let is_even = (lane_id % 2) == 0;
    let condition_pred = is_even.condition();
    let all_even = vote.all(condition_pred);
    let _unanimous = all_even.is_unanimous();

    // Test validation predicate
    let is_valid = lane_id < 32;
    let validation_pred = is_valid.validation();
    let all_valid = vote.all(validation_pred);
    let _result = all_valid.is_unanimous();

    // Test comparison predicate
    let is_less = lane_id < 16;
    let comparison_pred = is_less.comparison();
    let all_less = vote.all(comparison_pred);
    let _dissent = all_less.has_dissent();
}

#[kernel]
pub unsafe fn test_typed_predicates() {
    let mask = WarpMask::all();
    let vote = Vote::new(mask);
    let lane_id = cuda_std::warp::lane_id();

    // Test with different type tags
    struct TypeA;
    struct TypeB;

    let pred_a = Predicate::<TypeA>::new(lane_id == 0);
    let any_zero_a = vote.any(pred_a);
    let _has_any = any_zero_a.has_any();
    let _has_none = any_zero_a.has_none();

    let pred_b = Predicate::<TypeB>::new(lane_id < 16);
    let any_lower_b = vote.any(pred_b);
    let _result_b = any_lower_b.as_bool();

    // Test with unit type
    let unit_pred = Predicate::<()>::new(true);
    let all_true = vote.all(unit_pred);
    let _unanimous = all_true.is_unanimous();

    // Test generic helper function inside kernel
    fn test_generic_pred<T>(vote: &Vote, condition: bool) {
        let pred = Predicate::<T>::new(condition);
        let _result = unsafe { vote.any(pred) };
    }

    test_generic_pred::<TypeA>(&vote, lane_id == 0);
    test_generic_pred::<TypeB>(&vote, lane_id < 16);
    test_generic_pred::<()>(&vote, true);
}

#[kernel]
pub unsafe fn test_predicate_conversions() {
    let mask = WarpMask::all();
    let vote = Vote::new(mask);
    let lane_id = cuda_std::warp::lane_id();

    // Test converting predicates to booleans
    let pred = Predicate::<()>::new(lane_id < 8);
    let result = vote.ballot(pred);

    // Check individual lanes
    for lane in 0..32 {
        let _voted = result.lane_voted(lane);
    }

    // Check count
    let _count = result.count();
}
