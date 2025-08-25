// Test Reduction struct API and result types
// build-pass

use cuda_std::kernel;
use cuda_std::warp::reduce::{
    Add, And, Max, Min, Or, ReduceValue, Reduction, ReductionOp, ReductionResult, Xor,
};
use cuda_std::warp::WarpMask;

#[kernel]
pub unsafe fn test_reduction_struct_creation() {
    // Different ways to create Reduction
    let mask = WarpMask::all();
    let _r1 = Reduction::<i32>::new(mask);
    let _r2 = Reduction::<f32>::new(mask);
    let _r3 = Reduction::<u64>::new(mask);

    // Convenience constructor
    let _r4 = Reduction::<i32>::all_threads();
    let _r5 = Reduction::<f64>::all_threads();
}

#[kernel]
pub unsafe fn test_reduction_result_api() {
    let reduction = Reduction::<i32>::all_threads();
    let value = 42i32;

    // Test Add result methods
    let add_result = reduction.add(value);
    let _sum = add_result.sum(); // Specific to Add
    let _value = add_result.value(); // Generic access
    let _consumed = add_result.into_value(); // Consume result

    // Test Min result methods
    let min_result = reduction.min(value);
    let _min = min_result.minimum(); // Specific to Min
    if min_result.is_less_than(100) {
        let _v = min_result.value();
    }

    // Test Max result methods
    let max_result = reduction.max(value);
    let _max = max_result.maximum(); // Specific to Max
    if max_result.is_greater_than(0) {
        let _v = max_result.value();
    }
}

#[kernel]
pub unsafe fn test_custom_masks() {
    // Test with various masks
    let full_mask = WarpMask::all();
    let lower_half = WarpMask::new(0x0000FFFF);
    let upper_half = WarpMask::new(0xFFFF0000);
    let even_lanes = WarpMask::new(0x55555555);
    let odd_lanes = WarpMask::new(0xAAAAAAAA);

    let value = 10i32;

    let r_full = Reduction::<i32>::new(full_mask);
    let r_lower = Reduction::<i32>::new(lower_half);
    let r_upper = Reduction::<i32>::new(upper_half);
    let r_even = Reduction::<i32>::new(even_lanes);
    let r_odd = Reduction::<i32>::new(odd_lanes);

    let _s1 = r_full.add(value);
    let _s2 = r_lower.add(value);
    let _s3 = r_upper.add(value);
    let _s4 = r_even.add(value);
    let _s5 = r_odd.add(value);
}

#[kernel]
pub unsafe fn test_chained_operations() {
    let reduction = Reduction::<i32>::all_threads();
    let value = 42i32;

    // Perform multiple reductions with same Reduction instance
    let sum = reduction.add(value);
    let min = reduction.min(value);
    let max = reduction.max(value);
    let and = reduction.and(value);
    let or = reduction.or(value);
    let xor = reduction.xor(value);

    // Extract all values
    let _results = (
        sum.value(),
        min.value(),
        max.value(),
        and.value(),
        or.value(),
        xor.value(),
    );
}

// Test that ReductionResult preserves operation type
fn verify_add_result(result: ReductionResult<i32, Add>) -> i32 {
    result.sum() // This method is only available for Add
}

fn verify_min_result(result: ReductionResult<i32, Min>) -> bool {
    result.is_less_than(100) // This method is only available for Min
}

fn verify_max_result(result: ReductionResult<i32, Max>) -> bool {
    result.is_greater_than(0) // This method is only available for Max
}

#[kernel]
pub unsafe fn test_result_type_safety() {
    let reduction = Reduction::<i32>::all_threads();
    let value = 42i32;

    let add_result = reduction.add(value);
    let min_result = reduction.min(value);
    let max_result = reduction.max(value);

    let _sum = verify_add_result(add_result);
    let _is_small = verify_min_result(min_result);
    let _is_large = verify_max_result(max_result);
}

// Test that operation types are distinct
fn operation_names() {
    // These are compile-time constants
    let _add_name = Add::NAME;
    let _min_name = Min::NAME;
    let _max_name = Max::NAME;
    let _and_name = And::NAME;
    let _or_name = Or::NAME;
    let _xor_name = Xor::NAME;
}
