// Test basic reduction operations - fundamental reduction functionality
// build-pass

use cuda_std::kernel;
use cuda_std::warp::reduce::{Add, Max, Min, ReduceValue, Reduction};
use cuda_std::warp::WarpMask;

#[kernel]
pub unsafe fn test_basic_add_reduction() {
    let mask = WarpMask::all();
    let reduction = Reduction::<i32>::new(mask);

    let value = 1i32;
    let result = reduction.add(value);

    // Access sum via specific method
    let _sum = result.sum();

    // Or via generic value method
    let _value = result.value();
    let _value2 = result.into_value();
}

#[kernel]
pub unsafe fn test_basic_min_reduction() {
    let mask = WarpMask::all();
    let reduction = Reduction::<i32>::new(mask);

    let value = 42i32;
    let result = reduction.min(value);

    // Access minimum via specific method
    let _min = result.minimum();

    // Check comparison
    if result.is_less_than(100) {
        let _v = result.value();
    }
}

#[kernel]
pub unsafe fn test_basic_max_reduction() {
    let mask = WarpMask::all();
    let reduction = Reduction::<i32>::new(mask);

    let value = 42i32;
    let result = reduction.max(value);

    // Access maximum via specific method
    let _max = result.maximum();

    // Check comparison
    if result.is_greater_than(0) {
        let _v = result.value();
    }
}

#[kernel]
pub unsafe fn test_convenience_constructor() {
    // Use all_threads convenience constructor
    let reduction = Reduction::<i32>::all_threads();

    let value = 10i32;
    let _sum = reduction.add(value);
    let _min = reduction.min(value);
    let _max = reduction.max(value);
}

#[kernel]
pub unsafe fn test_direct_trait_calls() {
    let mask = WarpMask::all();
    let value = 42i32;

    // Direct trait method calls
    let sum = <i32 as ReduceValue>::reduce_add(mask, value);
    let min = <i32 as ReduceValue>::reduce_min(mask, value);
    let max = <i32 as ReduceValue>::reduce_max(mask, value);

    let _s = sum;
    let _m1 = min;
    let _m2 = max;
}

#[kernel]
pub unsafe fn test_masked_reduction() {
    // Test with custom mask (lower half of warp)
    let mask = WarpMask::new(0x0000FFFF);
    let reduction = Reduction::<i32>::new(mask);

    let value = 5i32;
    let _result = reduction.add(value);
}
