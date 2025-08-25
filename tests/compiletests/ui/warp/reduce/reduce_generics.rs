// Test generic reduction operations and trait bounds
// build-pass

use core::marker::PhantomData;
use cuda_std::kernel;
use cuda_std::warp::reduce::{
    Add, And, BitwiseReduceValue, Max, Min, Or, ReduceValue, Reduction, ReductionResult, Xor,
};
use cuda_std::warp::WarpMask;

// Generic reduction function
unsafe fn generic_sum<T: ReduceValue>(value: T) -> ReductionResult<T, Add> {
    let reduction = Reduction::<T>::all_threads();
    reduction.add(value)
}

// Generic min/max finder
unsafe fn find_min_max<T: ReduceValue>(
    value: T,
) -> (ReductionResult<T, Min>, ReductionResult<T, Max>) {
    let reduction = Reduction::<T>::all_threads();
    (reduction.min(value), reduction.max(value))
}

// Generic bitwise reduction
unsafe fn bitwise_reduce<T: BitwiseReduceValue>(
    value: T,
) -> (
    ReductionResult<T, And>,
    ReductionResult<T, Or>,
    ReductionResult<T, Xor>,
) {
    let reduction = Reduction::<T>::all_threads();
    (
        reduction.and(value),
        reduction.or(value),
        reduction.xor(value),
    )
}

// Generic struct with reduction capability
struct GenericReducer<T: ReduceValue> {
    mask: WarpMask,
    _phantom: PhantomData<T>,
}

impl<T: ReduceValue> GenericReducer<T> {
    fn new(mask: WarpMask) -> Self {
        Self {
            mask,
            _phantom: PhantomData,
        }
    }

    unsafe fn reduce_sum(&self, value: T) -> ReductionResult<T, Add> {
        let reduction = Reduction::<T>::new(self.mask);
        reduction.add(value)
    }

    unsafe fn reduce_min(&self, value: T) -> ReductionResult<T, Min> {
        let reduction = Reduction::<T>::new(self.mask);
        reduction.min(value)
    }
}

// Struct for bitwise operations only
struct BitwiseReducer<T: BitwiseReduceValue> {
    reduction: Reduction<T>,
}

impl<T: BitwiseReduceValue> BitwiseReducer<T> {
    fn new() -> Self {
        Self {
            reduction: Reduction::all_threads(),
        }
    }

    unsafe fn reduce_all(
        &self,
        value: T,
    ) -> (
        ReductionResult<T, And>,
        ReductionResult<T, Or>,
        ReductionResult<T, Xor>,
    ) {
        (
            self.reduction.and(value),
            self.reduction.or(value),
            self.reduction.xor(value),
        )
    }
}

#[kernel]
pub unsafe fn test_generic_functions() {
    // Test with i32
    let i32_sum = generic_sum(42i32);
    let _v1 = i32_sum.value();

    // Test with f32
    let f32_sum = generic_sum(3.14f32);
    let _v2 = f32_sum.value();

    // Test min/max
    let (min, max) = find_min_max(100u32);
    let _min = min.value();
    let _max = max.value();
}

#[kernel]
pub unsafe fn test_generic_bitwise() {
    // Test bitwise with u32
    let (and, or, xor) = bitwise_reduce(0xFF00FF00u32);
    let _a = and.value();
    let _o = or.value();
    let _x = xor.value();

    // Test bitwise with i64
    let (and2, or2, xor2) = bitwise_reduce(0x123456789ABCDEF0i64);
    let _a2 = and2.value();
    let _o2 = or2.value();
    let _x2 = xor2.value();
}

#[kernel]
pub unsafe fn test_generic_structs() {
    let mask = WarpMask::all();

    // Test GenericReducer with different types
    let i32_reducer = GenericReducer::<i32>::new(mask);
    let f64_reducer = GenericReducer::<f64>::new(mask);

    let i32_result = i32_reducer.reduce_sum(10);
    let f64_result = f64_reducer.reduce_min(2.718);

    let _v1 = i32_result.value();
    let _v2 = f64_result.value();
}

#[kernel]
pub unsafe fn test_bitwise_struct() {
    let u32_bitwise = BitwiseReducer::<u32>::new();
    let i32_bitwise = BitwiseReducer::<i32>::new();

    let (a1, o1, x1) = u32_bitwise.reduce_all(0xAAAAAAAAu32);
    let (a2, o2, x2) = i32_bitwise.reduce_all(0x55555555i32);

    let _results = (
        a1.value(),
        o1.value(),
        x1.value(),
        a2.value(),
        o2.value(),
        x2.value(),
    );
}

// Test with multiple type parameters
unsafe fn dual_reduce<T: ReduceValue, U: ReduceValue>(
    value1: T,
    value2: U,
) -> (ReductionResult<T, Add>, ReductionResult<U, Add>) {
    let r1 = Reduction::<T>::all_threads();
    let r2 = Reduction::<U>::all_threads();
    (r1.add(value1), r2.add(value2))
}

#[kernel]
pub unsafe fn test_multiple_type_parameters() {
    let (r1, r2) = dual_reduce(42i32, 3.14f32);
    let _v1 = r1.value();
    let _v2 = r2.value();
}
