// Test generic shuffle operations - covers generic functions, structs, and Result conversions
// build-pass

use core::marker::PhantomData;
use cuda_std::kernel;
use cuda_std::warp::shuffle::{
    patterns, InvalidLane, Shuffle, ShuffleResult, ShuffleValue, ShuffleWidth,
};
use cuda_std::warp::WarpMask;

// Generic function that returns Result
unsafe fn shuffle_generic<T: ShuffleValue>(mask: WarpMask, value: T) -> Result<T, InvalidLane> {
    T::shuffle_down(mask, value, 1, 32).into()
}

// Non-generic function for comparison
unsafe fn shuffle_i32_direct(mask: WarpMask, value: i32) -> Result<i32, InvalidLane> {
    <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32).into()
}

// Generic struct with shuffle functionality
struct GenericShuffler<T: ShuffleValue> {
    mask: WarpMask,
    width: ShuffleWidth,
    _phantom: PhantomData<T>,
}

impl<T: ShuffleValue> GenericShuffler<T> {
    fn new(mask: WarpMask, width: ShuffleWidth) -> Self {
        Self {
            mask,
            width,
            _phantom: PhantomData,
        }
    }

    unsafe fn do_shuffle(&self, value: T) -> Result<T, InvalidLane> {
        T::shuffle_down(self.mask, value, 1, self.width.value()).into()
    }

    unsafe fn do_shuffle_raw(&self, value: T) -> ShuffleResult<T> {
        T::shuffle_down(self.mask, value, 1, self.width.value())
    }
}

// Test with concrete type parameter
struct ConcreteShuffler {
    shuffler: GenericShuffler<i32>,
}

impl ConcreteShuffler {
    fn new() -> Self {
        Self {
            shuffler: GenericShuffler::new(WarpMask::all(), ShuffleWidth::full_warp()),
        }
    }
}

#[kernel]
pub unsafe fn test_generic_function() {
    let mask = WarpMask::all();
    let value = 42i32;

    // Test non-generic version
    let _result1 = shuffle_i32_direct(mask, value);

    // Test generic version
    let _result2 = shuffle_generic::<i32>(mask, value);

    // Test with f32
    let float_value = 3.14f32;
    let _result3 = shuffle_generic::<f32>(mask, float_value);
}

#[kernel]
pub unsafe fn test_generic_struct() {
    let shuffler = GenericShuffler::<i32>::new(WarpMask::all(), ShuffleWidth::full_warp());
    let value = 42i32;

    // Test returning Result
    let _result1 = shuffler.do_shuffle(value);

    // Test returning ShuffleResult
    let result2 = shuffler.do_shuffle_raw(value);
    if result2.is_valid() {
        let _v = result2.value;
    }
}

#[kernel]
pub unsafe fn test_concrete_wrapper() {
    let concrete = ConcreteShuffler::new();
    let value = 42i32;

    let _result = concrete.shuffler.do_shuffle(value);
}

// Test that PhantomData doesn't cause issues
#[kernel]
pub unsafe fn test_phantom_data() {
    struct PhantomWrapper<T> {
        _phantom: PhantomData<T>,
    }

    let _wrapper: PhantomWrapper<i32> = PhantomWrapper {
        _phantom: PhantomData,
    };

    // Use shuffle with phantom data struct existing
    let mask = WarpMask::all();
    let value = 42i32;
    let result = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);
    let _v = result.unwrap_or(0);
}

// Test generic with multiple type parameters
unsafe fn shuffle_pair<T: ShuffleValue, U: ShuffleValue>(
    mask: WarpMask,
    value1: T,
    value2: U,
) -> (ShuffleResult<T>, ShuffleResult<U>) {
    let r1 = T::shuffle_down(mask, value1, 1, 32);
    let r2 = U::shuffle_down(mask, value2, 1, 32);
    (r1, r2)
}

#[kernel]
pub unsafe fn test_multiple_generics() {
    let mask = WarpMask::all();
    let (r1, r2) = shuffle_pair(mask, 42i32, 3.14f32);

    let _v1 = r1.unwrap_or(0);
    let _v2 = r2.unwrap_or(0.0);
}
