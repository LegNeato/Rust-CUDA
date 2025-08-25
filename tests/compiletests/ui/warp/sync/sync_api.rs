// Test warp sync API and mask operations compile
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{self, WarpMask};

#[kernel]
pub unsafe fn test_warp_mask_creation() {
    // Test different ways to create WarpMask
    let _all_mask = WarpMask::all();
    let _none_mask = WarpMask::none();
    let _custom_mask = WarpMask::new(0x12345678);

    // Test single lane masks
    let _lane_mask = WarpMask::lane(5);
    let _lane_mask_2 = WarpMask::lane(31);

    // Test range masks
    let _lower_16 = WarpMask::range(0, 16);
    let _middle_8 = WarpMask::range(12, 20);
}

#[kernel]
pub unsafe fn test_warp_mask_operations() {
    let mask1 = WarpMask::new(0x0F0F0F0F);
    let mask2 = WarpMask::new(0xF0F0F0F0);

    // Test mask predicates
    let _empty_check = mask1.is_empty();
    let _full_check = mask1.is_full();

    // Test popcount
    let _count1 = mask1.count();
    let _count2 = mask2.count();

    // Test contains_lane
    let _contains = mask1.contains_lane(0);
    let _contains2 = mask2.contains_lane(4);

    // Test raw access
    let _raw = mask1.raw();
}

#[kernel]
pub unsafe fn test_active_mask() {
    // Get current active mask
    let active = warp::active_mask();

    // Sync with active mask
    warp::sync(active);

    // Use active mask in operations
    let _count = active.count();
    let _contains = active.contains_lane(0);
}

#[kernel]
pub unsafe fn test_lane_operations() {
    // Get lane ID
    let lane_id = warp::lane_id();

    // Create mask for current lane
    let my_mask = WarpMask::lane(lane_id);

    // Use mask in sync
    warp::sync(my_mask);
}

#[kernel]
pub unsafe fn test_mask_builders() {
    // Test even/odd lane masks
    let even_lanes = WarpMask::even_lanes();
    let odd_lanes = WarpMask::odd_lanes();

    // Test quadrant masks
    let _q1 = WarpMask::quadrant(0);
    let _q2 = WarpMask::quadrant(1);
    let _q3 = WarpMask::quadrant(2);
    let _q4 = WarpMask::quadrant(3);

    // Use masks in sync operations
    warp::sync(even_lanes);
    warp::sync(odd_lanes);
}

#[kernel]
pub unsafe fn test_mask_combinations() {
    let mask1 = WarpMask::new(0x0F0F0F0F);
    let mask2 = WarpMask::new(0xF0F0F0F0);

    // Combine masks using bitwise operations
    let combined = WarpMask::new(mask1.raw() | mask2.raw());
    let intersection = WarpMask::new(mask1.raw() & mask2.raw());
    let xor = WarpMask::new(mask1.raw() ^ mask2.raw());

    // Use combined masks
    warp::sync(combined);
    warp::sync(intersection);
    warp::sync(xor);
}
