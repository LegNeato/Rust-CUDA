// Test bool matrix type works correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, TensorCore};

#[kernel]
pub unsafe fn test_bool_matrices() {
    type Shape = dims::Shape<16, 16, 16>;

    // Create tensor core for bool elements
    let tc = TensorCore::<bool, Shape>::new();

    // Create matrix fragments
    let mut a = tc.matrix_a::<layout::Row>();
    let mut b = tc.matrix_b::<layout::Col>();
    let mut acc = tc.accumulator(); // Returns i32 accumulator

    // Initialize accumulator
    acc.fill(0i32);

    // Load with bool-compatible strides
    let a_ptr: *const bool = core::ptr::null();
    let b_ptr: *const bool = core::ptr::null();

    a.load::<16>(a_ptr);
    a.load::<32>(a_ptr);
    a.load::<48>(a_ptr);
    a.load::<64>(a_ptr);

    b.load::<16>(b_ptr);
    b.load::<32>(b_ptr);
    b.load::<48>(b_ptr);
    b.load::<64>(b_ptr);

    // Store result
    let c_ptr: *mut i32 = core::ptr::null_mut();
    acc.store::<layout::Row, 16>(c_ptr);
}

#[kernel]
pub unsafe fn test_bool_with_different_shapes() {
    // Test bool with various valid shapes
    {
        let tc = TensorCore::<bool, dims::Shape<16, 16, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    {
        let tc = TensorCore::<bool, dims::Shape<8, 8, 32>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    {
        let tc = TensorCore::<bool, dims::Shape<8, 8, 4>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }
}
