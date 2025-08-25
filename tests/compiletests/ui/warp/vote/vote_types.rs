// Test vote operations with different types
// build-pass

use cuda_std::kernel;
use cuda_std::warp::vote::{VoteEquality, WarpMaskVoteExt};
use cuda_std::warp::{Vote, WarpMask};

#[kernel]
pub unsafe fn test_vote_equality_types() {
    let mask = WarpMask::all();
    let lane_id = cuda_std::warp::lane_id();

    // Test with i32
    let val_i32: i32 = lane_id as i32;
    let eq_i32 = i32::vote_all_equal(mask, val_i32);
    let _equal = eq_i32.all_equal();

    // Test with u32
    let val_u32: u32 = lane_id;
    let eq_u32 = u32::vote_all_equal(mask, val_u32);
    let _equal = eq_u32.all_equal();

    // Test with i64
    let val_i64: i64 = lane_id as i64;
    let eq_i64 = i64::vote_all_equal(mask, val_i64);
    let _equal = eq_i64.all_equal();

    // Test with u64
    let val_u64: u64 = lane_id as u64;
    let eq_u64 = u64::vote_all_equal(mask, val_u64);
    let _equal = eq_u64.all_equal();

    // Test with f32
    let val_f32: f32 = lane_id as f32;
    let eq_f32 = f32::vote_all_equal(mask, val_f32);
    let _equal = eq_f32.all_equal();

    // Test with f64
    let val_f64: f64 = lane_id as f64;
    let eq_f64 = f64::vote_all_equal(mask, val_f64);
    let _equal = eq_f64.all_equal();

    // Test with small types
    let val_i8: i8 = (lane_id % 128) as i8;
    let eq_i8 = i8::vote_all_equal(mask, val_i8);
    let _equal = eq_i8.all_equal();

    let val_u8: u8 = (lane_id % 256) as u8;
    let eq_u8 = u8::vote_all_equal(mask, val_u8);
    let _equal = eq_u8.all_equal();

    let val_i16: i16 = lane_id as i16;
    let eq_i16 = i16::vote_all_equal(mask, val_i16);
    let _equal = eq_i16.all_equal();

    let val_u16: u16 = lane_id as u16;
    let eq_u16 = u16::vote_all_equal(mask, val_u16);
    let _equal = eq_u16.all_equal();
}

#[kernel]
pub unsafe fn test_vote_equality_with_same_values() {
    let mask = WarpMask::all();

    // All threads have the same value
    let same_val = 42i32;
    let eq_result = i32::vote_all_equal(mask, same_val);
    assert!(eq_result.all_equal());
    assert!(!eq_result.has_divergence());

    // Check match mask
    if let Some(match_mask) = eq_result.match_mask() {
        assert_eq!(match_mask.raw(), mask.raw());
    }
}
