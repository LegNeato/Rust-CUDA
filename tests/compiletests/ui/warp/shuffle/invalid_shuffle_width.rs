// Test that invalid shuffle widths are handled safely
// build-pass

use cuda_std::kernel;
use cuda_std::warp::ShuffleWidth;

#[kernel]
pub unsafe fn test_invalid_shuffle_width() {
    // ShuffleWidth::new returns None for invalid widths
    
    // 33 is not a valid width (> 32)
    let invalid = ShuffleWidth::new(33);
    assert!(invalid.is_none());
    
    // 15 is not a power of 2
    let not_power_of_two = ShuffleWidth::new(15);
    assert!(not_power_of_two.is_none());
    
    // 0 is not valid
    let zero = ShuffleWidth::new(0);
    assert!(zero.is_none());
    
    // Valid widths work
    assert!(ShuffleWidth::new(32).is_some());
    assert!(ShuffleWidth::new(16).is_some());
    assert!(ShuffleWidth::new(8).is_some());
    assert!(ShuffleWidth::new(4).is_some());
    assert!(ShuffleWidth::new(2).is_some());
    assert!(ShuffleWidth::new(1).is_some());
}
