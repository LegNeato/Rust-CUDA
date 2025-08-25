// Test different shuffle widths for sub-warp operations
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, Shuffle, ShuffleWidth};
use cuda_std::warp::shuffle::patterns;

#[kernel]
pub unsafe fn test_full_warp_shuffle() {
    let mask = WarpMask::all();
    let full = ShuffleWidth::full_warp();   // 32 threads
    
    let value = 123.0f32;
    let shuffle_full = Shuffle::<f32>::new(mask, full);
    let _ = shuffle_full.down(value, patterns::Down::new(1));
}

#[kernel]
pub unsafe fn test_half_warp_shuffle() {
    let mask = WarpMask::all();
    let half = ShuffleWidth::half_warp();   // 16 threads
    
    let value = 123.0f32;
    // Half warp shuffle (two independent groups of 16)
    let shuffle_half = Shuffle::<f32>::new(mask, half);
    let _ = shuffle_half.up(value, patterns::Up::new(1));
}

#[kernel]
pub unsafe fn test_quarter_warp_shuffle() {
    let mask = WarpMask::all();
    let quarter = ShuffleWidth::quarter_warp(); // 8 threads
    
    let value = 123.0f32;
    // Quarter warp shuffle (four independent groups of 8)
    let shuffle_quarter = Shuffle::<f32>::new(mask, quarter);
    let _ = shuffle_quarter.xor(value, patterns::Xor::new(4));
}