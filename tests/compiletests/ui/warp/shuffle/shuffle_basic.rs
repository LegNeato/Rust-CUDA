// Test basic shuffle operations - covers fundamental shuffle functionality
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::{patterns, Shuffle, ShuffleResult, ShuffleValue, ShuffleWidth};
use cuda_std::warp::WarpMask;

#[kernel]
pub unsafe fn test_basic_shuffle_down() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();

    let value = 42i32;
    let shuffle = Shuffle::<i32>::new(mask, width);

    let down_pattern = patterns::Down::new(1);
    let result = shuffle.down(value, down_pattern);

    // ShuffleResult can be used directly
    if result.is_valid() {
        let _value = result.value;
    }

    // Or converted to Result if needed
    let as_result: Result<i32, _> = result.into();
    match as_result {
        Ok(v) => {
            let _value = v;
        }
        Err(_) => {
            // Handle invalid lane access
        }
    }
}

#[kernel]
pub unsafe fn test_basic_shuffle_up() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i32>::full_warp();

    let value = 42i32;
    let up_pattern = patterns::Up::new(1);
    let result = shuffle.up(value, up_pattern);

    // Use unwrap_or for default value
    let _value = result.unwrap_or(0);
}

#[kernel]
pub unsafe fn test_basic_shuffle_xor() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i32>::full_warp();

    let value = 42i32;
    let xor_pattern = patterns::Xor::new(16);
    let result = shuffle.xor(value, xor_pattern);

    // Use unwrap_or_else with closure
    let _value = result.unwrap_or_else(|| 0);
}

#[kernel]
pub unsafe fn test_basic_shuffle_idx() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i32>::full_warp();

    let value = 42i32;
    if let Some(idx_pattern) = patterns::Index::new(0) {
        let broadcast = shuffle.index(value, idx_pattern);
        let _broadcast_value = broadcast.unwrap_or(value);
    }
}

// Test direct trait method calls (simpler cases that work)
#[kernel]
pub unsafe fn test_direct_trait_calls() {
    let mask = WarpMask::all();
    let value = 42i32;

    // Direct trait call - returns ShuffleResult
    let result = <i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32);

    if result.is_valid() {
        let _v = result.value;
    }
}

// Test minimal working case
#[kernel]
pub unsafe fn test_minimal() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let _shuffle = Shuffle::<i32>::new(mask, width);

    // Just creating the Shuffle struct works
}
