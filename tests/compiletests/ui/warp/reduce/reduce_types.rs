// Test reduction with all supported types
// build-pass

use cuda_std::kernel;
use cuda_std::warp::reduce::{BitwiseReduceValue, ReduceValue, Reduction};
use cuda_std::warp::WarpMask;

// 32-bit integer types
#[kernel]
pub unsafe fn test_reduce_i32() {
    let reduction = Reduction::<i32>::all_threads();
    let value = 42i32;

    let _add = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
    let _and = reduction.and(value);
    let _or = reduction.or(value);
    let _xor = reduction.xor(value);
}

#[kernel]
pub unsafe fn test_reduce_u32() {
    let reduction = Reduction::<u32>::all_threads();
    let value = 42u32;

    let _add = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
    let _and = reduction.and(value);
    let _or = reduction.or(value);
    let _xor = reduction.xor(value);
}

// 64-bit integer types
#[kernel]
pub unsafe fn test_reduce_i64() {
    let reduction = Reduction::<i64>::all_threads();
    let value = 42i64;

    let _add = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
    let _and = reduction.and(value);
    let _or = reduction.or(value);
    let _xor = reduction.xor(value);
}

#[kernel]
pub unsafe fn test_reduce_u64() {
    let reduction = Reduction::<u64>::all_threads();
    let value = 42u64;

    let _add = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
    let _and = reduction.and(value);
    let _or = reduction.or(value);
    let _xor = reduction.xor(value);
}

// Floating point types (no bitwise operations)
#[kernel]
pub unsafe fn test_reduce_f32() {
    let reduction = Reduction::<f32>::all_threads();
    let value = 3.14f32;

    let _add = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
    // Note: f32 doesn't implement BitwiseReduceValue
}

#[kernel]
pub unsafe fn test_reduce_f64() {
    let reduction = Reduction::<f64>::all_threads();
    let value = 3.14159f64;

    let _add = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
    // Note: f64 doesn't implement BitwiseReduceValue
}

// Test trait bounds
unsafe fn generic_reduce<T: ReduceValue>(value: T) -> T {
    let reduction = Reduction::<T>::all_threads();
    let result = reduction.add(value);
    result.into_value()
}

unsafe fn generic_bitwise_reduce<T: BitwiseReduceValue>(value: T) -> T {
    let reduction = Reduction::<T>::all_threads();
    let result = reduction.and(value);
    result.into_value()
}

#[kernel]
pub unsafe fn test_generic_reduction() {
    let _i32_result = generic_reduce(42i32);
    let _f32_result = generic_reduce(3.14f32);

    let _u32_bitwise = generic_bitwise_reduce(0xFFFFu32);
    let _i64_bitwise = generic_bitwise_reduce(0x1234i64);
}
