// Test bitwise reduction operations - AND, OR, XOR
// build-pass

use cuda_std::kernel;
use cuda_std::warp::reduce::{And, BitwiseReduceValue, Or, Reduction, Xor};
use cuda_std::warp::WarpMask;

#[kernel]
pub unsafe fn test_bitwise_and_reduction() {
    let mask = WarpMask::all();
    let reduction = Reduction::<u32>::new(mask);

    let value = 0xFFFF_0000u32;
    let result = reduction.and(value);

    let _value = result.value();
    let _value2 = result.into_value();
}

#[kernel]
pub unsafe fn test_bitwise_or_reduction() {
    let mask = WarpMask::all();
    let reduction = Reduction::<u32>::new(mask);

    let value = 0x0000_FFFFu32;
    let result = reduction.or(value);

    let _value = result.value();
}

#[kernel]
pub unsafe fn test_bitwise_xor_reduction() {
    let mask = WarpMask::all();
    let reduction = Reduction::<u32>::new(mask);

    let value = 0xAAAA_AAAAu32;
    let result = reduction.xor(value);

    let _value = result.value();
}

#[kernel]
pub unsafe fn test_bitwise_on_integers() {
    let reduction = Reduction::<i32>::all_threads();

    let value = 0x12345678i32;

    // All bitwise operations on i32
    let _and = reduction.and(value);
    let _or = reduction.or(value);
    let _xor = reduction.xor(value);
}

#[kernel]
pub unsafe fn test_direct_bitwise_trait_calls() {
    let mask = WarpMask::all();
    let value = 0xDEADBEEFu32;

    // Direct trait method calls
    let and_result = <u32 as BitwiseReduceValue>::reduce_and(mask, value);
    let or_result = <u32 as BitwiseReduceValue>::reduce_or(mask, value);
    let xor_result = <u32 as BitwiseReduceValue>::reduce_xor(mask, value);

    let _a = and_result;
    let _o = or_result;
    let _x = xor_result;
}

#[kernel]
pub unsafe fn test_bitwise_with_mask() {
    // Only upper half of warp
    let mask = WarpMask::new(0xFFFF0000);
    let reduction = Reduction::<u32>::new(mask);

    let value = 0xFF00FF00u32;
    let _and = reduction.and(value);
    let _or = reduction.or(value);
    let _xor = reduction.xor(value);
}
