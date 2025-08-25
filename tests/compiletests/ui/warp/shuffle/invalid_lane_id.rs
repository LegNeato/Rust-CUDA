// Test that invalid lane IDs are handled safely
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_invalid_lane_id() {
    // patterns::Index::new returns None for invalid lane IDs
    
    // 32 is out of range (valid is 0-31)
    let invalid = patterns::Index::new(32);
    assert!(invalid.is_none());
    
    // Way out of range
    let way_out_of_range = patterns::Index::new(100);
    assert!(way_out_of_range.is_none());
    
    // Valid lane IDs work
    let valid_min = patterns::Index::new(0);
    assert!(valid_min.is_some());
    
    let valid_max = patterns::Index::new(31);
    assert!(valid_max.is_some());
    
    let valid_mid = patterns::Index::new(15);
    assert!(valid_mid.is_some());
}
