// Test shuffle with small types (i8, i16, u8, u16)
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle};
use cuda_std::warp::shuffle::patterns;

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