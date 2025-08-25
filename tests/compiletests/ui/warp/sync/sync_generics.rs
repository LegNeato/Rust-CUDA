// Test sync operations with generic functions
// build-pass

use cuda_std::kernel;
use cuda_std::warp::{self, WarpMask};

#[kernel]
pub unsafe fn test_sync_with_generics() {
    // Generic function that performs sync
    fn generic_sync<T>() {
        unsafe {
            warp::sync(WarpMask::all());
            let mask = WarpMask::new(0xFFFFFFFF);
            warp::sync(mask);
        }
    }

    generic_sync::<i32>();
    generic_sync::<f32>();
    generic_sync::<()>();
}

#[kernel]
pub unsafe fn test_generic_mask_ops() {
    let mask1 = WarpMask::all();
    let mask2 = WarpMask::none();

    // Generic function using masks
    fn sync_with_mask<T>(_phantom: T, mask: WarpMask) {
        unsafe {
            warp::sync(mask);
        }
    }

    sync_with_mask(42i32, mask1);
    sync_with_mask(3.14f32, mask2);
    sync_with_mask((), mask1);
}

#[kernel]
pub unsafe fn test_generic_lane_ops() {
    // Generic function that uses lane operations
    fn get_lane_mask<T>() -> WarpMask {
        let lane_id = unsafe { warp::lane_id() };
        WarpMask::lane(lane_id)
    }

    let mask_i32 = get_lane_mask::<i32>();
    warp::sync(mask_i32);

    let mask_f64 = get_lane_mask::<f64>();
    warp::sync(mask_f64);

    let mask_unit = get_lane_mask::<()>();
    warp::sync(mask_unit);
}

#[kernel]
pub unsafe fn test_generic_active_mask() {
    // Get active mask in generic context
    let active = warp::active_mask();

    // Use in generic sync
    fn sync_active<T>() {
        unsafe {
            let mask = warp::active_mask();
            warp::sync(mask);
        }
    }

    sync_active::<i32>();
    sync_active::<u64>();
    sync_active::<()>();

    // Generic function with multiple type params
    fn sync_with_types<T, U>() {
        unsafe {
            warp::sync(WarpMask::all());
        }
    }

    sync_with_types::<i32, f32>();
    sync_with_types::<u8, u16>();
}
