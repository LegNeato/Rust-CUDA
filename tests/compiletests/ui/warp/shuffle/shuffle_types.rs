// Test shuffle with different types
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle};
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_shuffle_i32() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i32>::full_warp();
    let _ = shuffle.down(42i32, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_u32() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<u32>::full_warp();
    let _ = shuffle.down(42u32, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_f32() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<f32>::full_warp();
    let _ = shuffle.down(42.0f32, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_i64() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i64>::full_warp();
    let _ = shuffle.down(42i64, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_u64() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<u64>::full_warp();
    let _ = shuffle.down(42u64, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_f64() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<f64>::full_warp();
    let _ = shuffle.down(42.0f64, patterns::Down::new(1));
}