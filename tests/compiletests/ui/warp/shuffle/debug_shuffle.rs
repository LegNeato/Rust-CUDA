// Debug shuffle to find the issue
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{WarpMask, ShuffleValue, ShuffleWidth};
use cuda_std::warp::shuffle::{patterns, ShuffleDelta};

#[kernel]
pub unsafe fn test_mimick_shuffle_down() {
    // This mimics what Shuffle::down does
    let mask = WarpMask::all();
    let width = ShuffleWidth::full_warp();
    let value = 42i32;
    let pattern = patterns::Down::new(1);
    
    // This is exactly what shuffle.down() does internally
    let _result = <i32 as ShuffleValue>::shuffle_down(
        mask, 
        value, 
        pattern.delta().value(), 
        width.value()
    );
}

#[kernel]
pub unsafe fn test_with_struct_field_access() {
    // Test accessing fields from a struct
    struct MyShuffleData {
        mask: WarpMask,
        width: ShuffleWidth,
    }
    
    let data = MyShuffleData {
        mask: WarpMask::all(),
        width: ShuffleWidth::full_warp(),
    };
    
    let value = 42i32;
    
    // Use struct fields
    let _result = <i32 as ShuffleValue>::shuffle_down(
        data.mask,
        value,
        1,
        data.width.value()
    );
}