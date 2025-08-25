// Test basic shuffle operations
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle, ShuffleWidth};
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_basic_shuffle_down() {
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    
    let value = 42i32;
    let shuffle = Shuffle::<i32>::new(mask, width);
    
    // Shuffle down with type-safe pattern
    let down_pattern = patterns::Down::new(1);
    let result = shuffle.down(value, down_pattern);
    match result {
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
    if let Ok(v) = result {
        let _value = v;
    }
}

#[kernel]
pub unsafe fn test_basic_shuffle_xor() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i32>::full_warp();
    
    let value = 42i32;
    let xor_pattern = patterns::Xor::new(16);
    let _result = shuffle.xor(value, xor_pattern);
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