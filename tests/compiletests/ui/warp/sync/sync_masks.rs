// Test warp sync operations with different masks
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{self, WarpMask};

#[kernel]
pub unsafe fn test_sync_with_masks() {
    let lane_id = warp::lane_id();

    // Sync all threads
    warp::sync(WarpMask::all());

    // Sync with specific mask value
    warp::sync_mask(0xFFFFFFFF);

    // Sync lower half
    let lower_half = WarpMask::new(0x0000FFFF);
    warp::sync(lower_half);

    // Sync upper half
    let upper_half = WarpMask::new(0xFFFF0000);
    warp::sync(upper_half);

    // Sync even lanes
    let even_mask = WarpMask::new(0x55555555);
    warp::sync(even_mask);

    // Sync odd lanes
    let odd_mask = WarpMask::new(0xAAAAAAAA);
    warp::sync(odd_mask);

    // Sync single lane
    let single_lane = WarpMask::lane(lane_id);
    warp::sync(single_lane);

    // Sync quadrants
    let first_quarter = WarpMask::new(0x000000FF);
    let second_quarter = WarpMask::new(0x0000FF00);
    let third_quarter = WarpMask::new(0x00FF0000);
    let fourth_quarter = WarpMask::new(0xFF000000);

    warp::sync(first_quarter);
    warp::sync(second_quarter);
    warp::sync(third_quarter);
    warp::sync(fourth_quarter);
}

#[kernel]
pub unsafe fn test_sync_patterns() {
    // Pattern: sync based on lane conditions
    let lane_id = warp::lane_id();

    // Sync lanes that are multiples of 4
    let multiples_of_4 = WarpMask::new(0x11111111);
    warp::sync(multiples_of_4);

    // Sync based on dynamic condition
    let active = warp::active_mask();
    warp::sync(active);

    // Conditional sync based on lane ID
    if lane_id < 16 {
        let lower_mask = WarpMask::new(0x0000FFFF);
        warp::sync(lower_mask);
    } else {
        let upper_mask = WarpMask::new(0xFFFF0000);
        warp::sync(upper_mask);
    }
}

#[kernel]
pub unsafe fn test_sync_combinations() {
    let lane_id = warp::lane_id();

    // Combine multiple masks
    let mask1 = WarpMask::new(0x0F0F0F0F);
    let mask2 = WarpMask::new(0xF0F0F0F0);

    // Sync first pattern
    warp::sync(mask1);

    // Sync second pattern
    warp::sync(mask2);

    // Sync union of both
    let combined = WarpMask::new(mask1.raw() | mask2.raw());
    warp::sync(combined);

    // Sync intersection
    let intersection = WarpMask::new(mask1.raw() & mask2.raw());
    if intersection.raw() != 0 {
        warp::sync(intersection);
    }
}
