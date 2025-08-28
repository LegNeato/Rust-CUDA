// Test CUDA warp matrix functions (tensor core) compile correctly
// build-pass

use cuda_std::f16;
use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MatrixElement, TensorCore};

#[kernel]
pub unsafe fn test_warp_matrix_type_safe() {
    // Create a tensor core operation with compile-time validated dimensions
    // Type-driven API: specify element type directly
    let tc_f16 = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();

    // Create matrix fragments with type-safe API
    let mut a_fragment = tc_f16.matrix_a::<layout::Row>();
    let mut b_fragment = tc_f16.matrix_b::<layout::Col>();
    let mut c_fragment = tc_f16.accumulator(); // Returns f32 accumulator by default

    // Initialize accumulator
    c_fragment.fill(0.0);

    // Create mock pointers for demonstration
    let a_matrix: *const f16 = core::ptr::null();
    let b_matrix: *const f16 = core::ptr::null();
    let c_matrix: *mut f32 = core::ptr::null_mut();

    // Load operations would use compile-time stride validation
    a_fragment.load::<16>(a_matrix);  // STRIDE validated at compile time  
    b_fragment.load::<16>(b_matrix);

    // Perform matrix multiply-accumulate
    c_fragment.mma_inplace(&a_fragment, &b_fragment);

    // Store result (f32 needs stride multiple of 4)
    c_fragment.store::<layout::Row, 16>(c_matrix);
}

#[kernel]
pub unsafe fn test_different_tensor_shapes() {
    // 16x16x16 - Most common configuration
    let tc_16x16 = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();
    let _a = tc_16x16.matrix_a::<layout::Row>();

    // 32x8x16 - Tall and skinny
    let tc_32x8 = TensorCore::<f16, dims::Shape<32, 8, 16>>::new();
    let _a = tc_32x8.matrix_a::<layout::Row>();

    // 8x32x16 - Short and wide
    let tc_8x32 = TensorCore::<f16, dims::Shape<8, 32, 16>>::new();
    let _a = tc_8x32.matrix_a::<layout::Row>();
}

#[kernel]
pub unsafe fn test_different_element_types() {
    // f16 input, f32 accumulator (most common)
    {
        let tc = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator(); // f32 by default for f16
    }

    // f16 input, f16 accumulator
    {
        let tc = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator_f16(); // explicitly use f16 accumulator
    }

    // i8 input, i32 accumulator
    {
        let tc = TensorCore::<i8, dims::Shape<16, 16, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator(); // i32 for i8
    }
}

#[kernel]
pub unsafe fn test_layout_combinations() {
    let tc = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();

    // All valid layout combinations
    let _a_row = tc.matrix_a::<layout::Row>();
    let _a_col = tc.matrix_a::<layout::Col>();
    let _b_row = tc.matrix_b::<layout::Row>();
    let _b_col = tc.matrix_b::<layout::Col>();

    // Accumulator can have different layouts for storage
    let acc = tc.accumulator();
    let output: *mut f32 = core::ptr::null_mut();
    acc.store::<layout::Row, 16>(output);
    acc.store::<layout::Col, 16>(output);
}
