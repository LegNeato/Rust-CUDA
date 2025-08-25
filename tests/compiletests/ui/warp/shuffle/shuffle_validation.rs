// Test validation APIs for shuffle width and lane IDs compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::{patterns, ShuffleWidth};

#[kernel]
pub unsafe fn test_shuffle_width_validation() {
    // ShuffleWidth::new returns Option for compile-time validation

    // Invalid widths return None
    let invalid = ShuffleWidth::new(33);
    assert!(invalid.is_none());

    let not_power_of_two = ShuffleWidth::new(15);
    assert!(not_power_of_two.is_none());

    let zero = ShuffleWidth::new(0);
    assert!(zero.is_none());

    // Valid widths return Some
    assert!(ShuffleWidth::new(32).is_some());
    assert!(ShuffleWidth::new(16).is_some());
    assert!(ShuffleWidth::new(8).is_some());
    assert!(ShuffleWidth::new(4).is_some());
    assert!(ShuffleWidth::new(2).is_some());
    assert!(ShuffleWidth::new(1).is_some());
}

#[kernel]
pub unsafe fn test_lane_id_validation() {
    // patterns::Index::new returns Option for compile-time validation

    // Invalid lane IDs return None
    let invalid = patterns::Index::new(32);
    assert!(invalid.is_none());

    let way_out = patterns::Index::new(100);
    assert!(way_out.is_none());

    // Valid lane IDs return Some
    let valid_min = patterns::Index::new(0);
    assert!(valid_min.is_some());

    let valid_max = patterns::Index::new(31);
    assert!(valid_max.is_some());

    let valid_mid = patterns::Index::new(15);
    assert!(valid_mid.is_some());

    // Test unsafe unchecked constructor compiles
    let unchecked = unsafe { patterns::Index::new_unchecked(5) };
    let _ = unchecked;
}
