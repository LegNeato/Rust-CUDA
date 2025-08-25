// Test ShuffleValue trait methods and static calls
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::{ShuffleExt, ShuffleValue};
use cuda_std::warp::WarpMask;

// Test static trait method calls
#[kernel]
pub unsafe fn test_static_trait_calls() {
    let mask = WarpMask::all();
    let value = 42i32;

    // Static calls with explicit type
    let r1 = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);
    let r2 = <i32 as ShuffleValue>::shuffle_up(mask, value, 1, 32);
    let r3 = <i32 as ShuffleValue>::shuffle_xor(mask, value, 16, 32);
    let r4 = <i32 as ShuffleValue>::shuffle_idx(mask, value, 0, 32);

    // Check results
    let _v1 = r1.unwrap_or(0);
    let _v2 = r2.unwrap_or(0);
    let _v3 = r3.unwrap_or(0);
    let _v4 = r4.unwrap_or(0);
}

// Test with different types
#[kernel]
pub unsafe fn test_different_type_traits() {
    let mask = WarpMask::all();

    // i32
    let i32_val = 42i32;
    let _r1 = <i32 as ShuffleValue>::shuffle_down(mask, i32_val, 1, 32);

    // u32
    let u32_val = 42u32;
    let _r2 = <u32 as ShuffleValue>::shuffle_down(mask, u32_val, 1, 32);

    // f32
    let f32_val = 3.14f32;
    let _r3 = <f32 as ShuffleValue>::shuffle_down(mask, f32_val, 1, 32);

    // i64
    let i64_val = 42i64;
    let _r4 = <i64 as ShuffleValue>::shuffle_down(mask, i64_val, 1, 32);

    // f64
    let f64_val = 3.14f64;
    let _r5 = <f64 as ShuffleValue>::shuffle_down(mask, f64_val, 1, 32);
}

// Test ShuffleExt trait methods
#[kernel]
pub unsafe fn test_shuffle_ext() {
    let mask = WarpMask::all();
    let value = 42i32;

    // Extension trait methods
    let r1 = value.shuffle_down(mask, 1, 32);
    let r2 = value.shuffle_up(mask, 1, 32);

    let _v1 = r1.unwrap_or(0);
    let _v2 = r2.unwrap_or(0);

    // Create shuffle via extension trait
    let _shuffle = i32::shuffle(mask, cuda_std::warp::ShuffleWidth::full_warp());
}

// Helper function using trait bounds
unsafe fn shuffle_helper<T: ShuffleValue>(value: T, delta: u32) -> T {
    let mask = WarpMask::all();
    let result = T::shuffle_down(mask, value, delta, 32);
    result.unwrap_or(value)
}

#[kernel]
pub unsafe fn test_trait_bound_function() {
    let i32_result = shuffle_helper(42i32, 1);
    let f32_result = shuffle_helper(3.14f32, 1);

    let _v1 = i32_result;
    let _v2 = f32_result;
}

// Test non-generic wrapper functions
unsafe fn shuffle_i32_down(value: i32, delta: u32) -> i32 {
    let mask = WarpMask::all();
    let result = <i32 as ShuffleValue>::shuffle_down(mask, value, delta, 32);
    result.unwrap_or(0)
}

unsafe fn shuffle_f32_down(value: f32, delta: u32) -> f32 {
    let mask = WarpMask::all();
    let result = <f32 as ShuffleValue>::shuffle_down(mask, value, delta, 32);
    result.unwrap_or(0.0)
}

#[kernel]
pub unsafe fn test_non_generic_wrappers() {
    let i32_result = shuffle_i32_down(42, 1);
    let f32_result = shuffle_f32_down(3.14, 1);

    let _v1 = i32_result;
    let _v2 = f32_result;
}
