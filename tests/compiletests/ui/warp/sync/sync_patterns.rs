// Test various sync patterns compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{self, WarpMask};

#[kernel]
pub unsafe fn test_sync_all_patterns() {
    // Different ways to sync all threads
    warp::sync(WarpMask::all());
    warp::sync_mask(0xFFFFFFFF);

    let full_mask = WarpMask::new(0xFFFFFFFF);
    warp::sync(full_mask);
}

#[kernel]
pub unsafe fn test_sync_partial_patterns() {
    // Sync lower half
    warp::sync(WarpMask::new(0x0000FFFF));
    warp::sync_mask(0x0000FFFF);

    // Sync upper half
    warp::sync(WarpMask::new(0xFFFF0000));
    warp::sync_mask(0xFFFF0000);

    // Sync even lanes
    warp::sync(WarpMask::new(0x55555555));

    // Sync odd lanes
    warp::sync(WarpMask::new(0xAAAAAAAA));
}

#[kernel]
pub unsafe fn test_sync_single_lanes() {
    // Sync individual lanes
    for i in 0..32 {
        let lane_mask = WarpMask::lane(i);
        warp::sync(lane_mask);
    }

    // Sync using shift pattern
    let mut mask = 1u32;
    for _ in 0..32 {
        warp::sync_mask(mask);
        mask = mask.rotate_left(1);
    }
}

#[kernel]
pub unsafe fn test_sync_quadrants() {
    // Sync by quadrants
    warp::sync(WarpMask::new(0x000000FF)); // First 8 lanes
    warp::sync(WarpMask::new(0x0000FF00)); // Second 8 lanes
    warp::sync(WarpMask::new(0x00FF0000)); // Third 8 lanes
    warp::sync(WarpMask::new(0xFF000000)); // Fourth 8 lanes
}

#[kernel]
pub unsafe fn test_conditional_sync() {
    let lane_id = warp::lane_id();

    // Conditional sync based on lane ID
    if lane_id < 16 {
        warp::sync(WarpMask::new(0x0000FFFF));
    } else {
        warp::sync(WarpMask::new(0xFFFF0000));
    }

    // Sync based on even/odd
    if (lane_id % 2) == 0 {
        warp::sync(WarpMask::even_lanes());
    } else {
        warp::sync(WarpMask::odd_lanes());
    }
}

#[kernel]
pub unsafe fn test_dynamic_sync() {
    let active = warp::active_mask();
    let lane_id = warp::lane_id();

    // Sync with current active threads
    warp::sync(active);

    // Create dynamic mask based on condition
    let my_lane_mask = WarpMask::lane(lane_id);
    warp::sync(my_lane_mask);

    // Sync with computed mask
    let computed_mask = WarpMask::new(active.raw() & 0x0000FFFF);
    warp::sync(computed_mask);
}

#[kernel]
pub unsafe fn test_nested_sync() {
    // Multiple sync operations in sequence
    warp::sync(WarpMask::all());

    let lower = WarpMask::new(0x0000FFFF);
    warp::sync(lower);

    let upper = WarpMask::new(0xFFFF0000);
    warp::sync(upper);

    warp::sync(WarpMask::all());
}
