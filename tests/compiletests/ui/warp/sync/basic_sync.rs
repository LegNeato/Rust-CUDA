// Test CUDA warp-level functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp;

#[kernel]
pub unsafe fn test_warp_functions() {
    // Test lane ID function
    let _lane = warp::lane_id();

    // Test active mask function
    let _mask = warp::active_mask();

    // Test warp sync with full mask
    warp::sync(warp::WarpMask::all());

    // Test warp sync with partial mask
    warp::sync_mask(0x0000FFFF);
    
    // Test creating specific lane masks
    let _single_lane = warp::WarpMask::lane(5);
    let _custom_mask = warp::WarpMask::new(0xFF00FF00);
}