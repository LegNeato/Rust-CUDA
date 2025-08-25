// build-fail

use cuda_std::kernel;
use cuda_std::warp::shuffle::{patterns, Shuffle};

#[kernel]
pub unsafe fn test_wrong_pattern_type() {
    let shuffle = Shuffle::<i32>::full_warp();
    let down_pattern = patterns::Down::new(1);

    // up() expects patterns::Up, not patterns::Down
    let _result = shuffle.up(42, down_pattern);
}
