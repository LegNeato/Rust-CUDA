// Test practical shuffle patterns like reduction
// build-pass

use cuda_std::kernel;
use cuda_std::warp::shuffle::{patterns, Shuffle};
use cuda_std::warp::{self, WarpMask};

#[kernel]
pub unsafe fn test_butterfly_reduction() {
    let lane_id = warp::lane_id();
    let mask = WarpMask::all();

    // Butterfly reduction using XOR shuffles
    let mut value = lane_id as i32;
    let shuffle = Shuffle::<i32>::full_warp();

    for distance in [16, 8, 4, 2, 1] {
        let pattern = patterns::Xor::new(distance);
        let result: Result<i32, _> = shuffle.xor(value, pattern).into();
        if let Ok(shuffled) = result {
            value += shuffled;
        }
    }
    // Now value contains the sum of all lane IDs
}

#[kernel]
pub unsafe fn test_broadcast_from_lane0() {
    let lane_id = warp::lane_id();
    let mask = WarpMask::all();

    // Broadcast from lane 0
    let my_value = (lane_id * 10) as i32;
    let shuffle = Shuffle::<i32>::full_warp();

    if let Some(broadcast_pattern) = patterns::Index::new(0) {
        let broadcast_result: Result<i32, _> = shuffle.index(my_value, broadcast_pattern).into();
        if let Ok(broadcast_value) = broadcast_result {
            // All threads now have the value from lane 0
            let _shared = broadcast_value;
        }
    }
}

#[kernel]
pub unsafe fn test_shift_pattern() {
    let lane_id = warp::lane_id();
    let mask = WarpMask::all();

    // Shift pattern for prefix sum
    let mut prefix_sum = lane_id as i32;
    let shuffle = Shuffle::<i32>::full_warp();

    for offset in 1..32 {
        let pattern = patterns::Up::new(offset);
        let result: Result<i32, _> = shuffle.up(prefix_sum, pattern).into();
        if let Ok(prev_value) = result {
            prefix_sum += prev_value;
        }
    }
}
