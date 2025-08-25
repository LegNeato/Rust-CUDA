// Test practical reduction patterns and use cases
// build-pass

use cuda_std::kernel;
use cuda_std::warp::reduce::{Add, Max, Min, Reduction};
use cuda_std::warp::{self, WarpMask};

#[kernel]
pub unsafe fn test_sum_lane_indices() {
    let lane_id = warp::lane_id();
    let reduction = Reduction::<u32>::all_threads();

    // Sum of all lane indices (0+1+2+...+31 = 496)
    let result = reduction.add(lane_id);
    let _sum = result.sum();
}

#[kernel]
pub unsafe fn test_find_min_max_values() {
    let lane_id = warp::lane_id();
    let reduction = Reduction::<u32>::all_threads();

    // Each thread has a different value
    let value = (lane_id * 3 + 7) % 100;

    let min_result = reduction.min(value);
    let max_result = reduction.max(value);

    let _min = min_result.minimum();
    let _max = max_result.maximum();
}

#[kernel]
pub unsafe fn test_conditional_reduction() {
    let lane_id = warp::lane_id();

    // Only even lanes participate
    let mask = if lane_id % 2 == 0 {
        WarpMask::new(1 << lane_id)
    } else {
        WarpMask::new(0)
    };

    // Create reduction with partial participation
    let all_mask = WarpMask::new(0x55555555); // Even lanes only
    let reduction = Reduction::<u32>::new(all_mask);

    let value = lane_id * 10;
    let _result = reduction.add(value);
}

#[kernel]
pub unsafe fn test_floating_point_accumulation() {
    let lane_id = warp::lane_id() as f32;
    let reduction = Reduction::<f32>::all_threads();

    // Each thread contributes 1/32 of the total
    let value = 1.0f32 / 32.0;
    let result = reduction.add(value);

    // Sum should be approximately 1.0
    let _sum = result.value();
}

#[kernel]
pub unsafe fn test_bitwise_flag_aggregation() {
    let lane_id = warp::lane_id();
    let reduction = Reduction::<u32>::all_threads();

    // Each thread sets a different bit
    let flag = 1u32 << (lane_id % 32);

    // OR reduction combines all flags
    let or_result = reduction.or(flag);
    let _combined_flags = or_result.value();

    // AND reduction finds common flags (should be 0)
    let and_result = reduction.and(flag);
    let _common_flags = and_result.value();
}

#[kernel]
pub unsafe fn test_prefix_sum_style() {
    let lane_id = warp::lane_id();

    // Progressive mask for prefix-sum style operation
    let mut accumulated = lane_id as i32;

    for width in [2, 4, 8, 16, 32] {
        let mask = WarpMask::new((1u32 << width) - 1);
        let reduction = Reduction::<i32>::new(mask);

        if lane_id < width {
            let result = reduction.add(accumulated);
            accumulated = result.value();
        }
    }

    let _final = accumulated;
}

#[kernel]
pub unsafe fn test_reduction_with_divergence() {
    let lane_id = warp::lane_id();
    let reduction = Reduction::<i32>::all_threads();

    // Different values based on condition
    let value = if lane_id < 16 { 10i32 } else { 20i32 };

    let sum = reduction.add(value);
    let min = reduction.min(value);
    let max = reduction.max(value);

    // Sum should be 10*16 + 20*16 = 480
    let _s = sum.value();
    // Min should be 10
    let _m1 = min.value();
    // Max should be 20
    let _m2 = max.value();
}
