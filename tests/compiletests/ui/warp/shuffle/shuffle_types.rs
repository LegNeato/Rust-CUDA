// Test shuffle with all supported types
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::{patterns, Shuffle};
use cuda_std::warp::WarpMask;

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

// Small types (8-bit and 16-bit)
#[kernel]
pub unsafe fn test_shuffle_i8() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i8>::full_warp();
    let _ = shuffle.down(42i8, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_i16() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<i16>::full_warp();
    let _ = shuffle.down(42i16, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_u8() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<u8>::full_warp();
    let _ = shuffle.down(42u8, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_shuffle_u16() {
    let mask = WarpMask::all();
    let shuffle = Shuffle::<u16>::full_warp();
    let _ = shuffle.down(42u16, patterns::Down::new(1));
}
