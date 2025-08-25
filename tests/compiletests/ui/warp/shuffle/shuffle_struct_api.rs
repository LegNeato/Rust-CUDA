// Test Shuffle struct API and patterns - covers the high-level Shuffle API
// build-pass

use core::marker::PhantomData;
use cuda_std::kernel;
use cuda_std::warp::shuffle::{patterns, Shuffle, ShuffleValue, ShuffleWidth};
use cuda_std::warp::WarpMask;

// Test creating and using Shuffle struct
#[kernel]
pub unsafe fn test_shuffle_struct_creation() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();

    // Create Shuffle for different types
    let _shuffle_i32 = Shuffle::<i32>::new(mask, width);
    let _shuffle_f32 = Shuffle::<f32>::new(mask, width);
    let _shuffle_u64 = Shuffle::<u64>::new(mask, width);

    // Use convenience constructor
    let _shuffle_full = Shuffle::<i32>::full_warp();
}

// Test all pattern types
#[kernel]
pub unsafe fn test_patterns() {
    // Create patterns
    let down_pattern = patterns::Down::new(1);
    let up_pattern = patterns::Up::new(1);
    let xor_pattern = patterns::Xor::new(16);
    let idx_pattern = patterns::Index::new(0).unwrap();

    // Use patterns with shuffle
    let shuffle = Shuffle::<i32>::full_warp();
    let value = 42i32;

    let _r1 = shuffle.down(value, down_pattern);
    let _r2 = shuffle.up(value, up_pattern);
    let _r3 = shuffle.xor(value, xor_pattern);
    let _r4 = shuffle.index(value, idx_pattern);
}

// Test pattern edge cases
#[kernel]
pub unsafe fn test_pattern_boundaries() {
    let shuffle = Shuffle::<i32>::full_warp();
    let value = 42i32;

    // Test various deltas
    let _r1 = shuffle.down(value, patterns::Down::new(0));
    let _r2 = shuffle.down(value, patterns::Down::new(31));

    // Test various XOR masks
    let _r3 = shuffle.xor(value, patterns::Xor::new(0));
    let _r4 = shuffle.xor(value, patterns::Xor::new(31));

    // Test unsafe index creation
    let idx = unsafe { patterns::Index::new_unchecked(31) };
    let _r5 = shuffle.index(value, idx);
}

// Test different shuffle widths
#[kernel]
pub unsafe fn test_different_widths() {
    let mask = WarpMask::all();
    let value = 42i32;

    // Full warp
    let shuffle_32 = Shuffle::<i32>::new(mask, ShuffleWidth::full_warp());
    let _r1 = shuffle_32.down(value, patterns::Down::new(1));

    // Half warp
    let shuffle_16 = Shuffle::<i32>::new(mask, ShuffleWidth::half_warp());
    let _r2 = shuffle_16.down(value, patterns::Down::new(1));

    // Quarter warp
    let shuffle_8 = Shuffle::<i32>::new(mask, ShuffleWidth::quarter_warp());
    let _r3 = shuffle_8.down(value, patterns::Down::new(1));
}

// Test custom shuffle struct (simplified version)
struct SimpleShuffle<T> {
    mask: WarpMask,
    width: ShuffleWidth,
    _phantom: PhantomData<T>,
}

impl<T: ShuffleValue> SimpleShuffle<T> {
    fn new(mask: WarpMask, width: ShuffleWidth) -> Self {
        Self {
            mask,
            width,
            _phantom: PhantomData,
        }
    }

    unsafe fn down(self, value: T, delta: u32) -> Result<T, cuda_std::warp::shuffle::InvalidLane> {
        T::shuffle_down(self.mask, value, delta, self.width.value()).into()
    }
}

#[kernel]
pub unsafe fn test_custom_shuffle_struct() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let shuffle = SimpleShuffle::<i32>::new(mask, width);

    let value = 42i32;
    let _result = shuffle.down(value, 1);
}
