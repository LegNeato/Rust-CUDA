// Test matrix operations and MMA compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, TensorCore};
use cuda_std::{bf16, f16};

#[kernel]
pub unsafe fn test_mma_operations() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();

    let a = tc.matrix_a::<layout::Row>();
    let b = tc.matrix_b::<layout::Col>();
    let mut c = tc.accumulator();

    // Test fill operation
    c.fill(0.0f32);
    c.fill(1.0f32);
    c.fill(-1.0f32);

    // Test MMA extension trait methods
    // In-place MMA: c = a × b + c
    c.mma_inplace(&a, &b);

    // Functional MMA: new_c = a × b + c
    let _new_c = c.mma(&a, &b);
}

#[kernel]
pub unsafe fn test_chained_mma_operations() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();

    let a1 = tc.matrix_a::<layout::Row>();
    let b1 = tc.matrix_b::<layout::Col>();
    let a2 = tc.matrix_a::<layout::Row>();
    let b2 = tc.matrix_b::<layout::Col>();

    let mut acc = tc.accumulator();
    acc.fill(0.0);

    // Chain multiple MMA operations
    acc.mma_inplace(&a1, &b1);
    acc.mma_inplace(&a2, &b2);

    // Or use functional style
    let _result = acc.mma(&a1, &b1).mma(&a2, &b2);
}

#[kernel]
pub unsafe fn test_different_accumulator_types() {
    type Shape = dims::Shape<16, 16, 16>;

    // f16 -> f32 accumulator (default)
    {
        let tc = TensorCore::<f16, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Col>();
        let mut c_f32 = tc.accumulator();

        c_f32.fill(1.0f32);
        c_f32.mma_inplace(&a, &b);
    }

    // f16 -> f16 accumulator (lower precision)
    {
        let tc = TensorCore::<f16, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Col>();
        let mut c_f16 = tc.accumulator_f16();

        c_f16.fill(f16::from_f32(1.0));
        c_f16.mma_inplace(&a, &b);
    }

    // i8 -> i32 accumulator
    {
        let tc = TensorCore::<i8, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Col>();
        let mut c_i32 = tc.accumulator();

        c_i32.fill(0i32);
        c_i32.mma_inplace(&a, &b);
    }
}

#[kernel]
pub unsafe fn test_mixed_layouts() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();

    // Test all layout combinations
    let a_row = tc.matrix_a::<layout::Row>();
    let a_col = tc.matrix_a::<layout::Col>();
    let b_row = tc.matrix_b::<layout::Row>();
    let b_col = tc.matrix_b::<layout::Col>();

    let mut acc = tc.accumulator();

    // Row-Row
    acc.mma_inplace(&a_row, &b_row);

    // Row-Col
    acc.mma_inplace(&a_row, &b_col);

    // Col-Row
    acc.mma_inplace(&a_col, &b_row);

    // Col-Col
    acc.mma_inplace(&a_col, &b_col);
}

#[kernel]
pub unsafe fn test_load_store_operations() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();

    let mut a = tc.matrix_a::<layout::Row>();
    let mut b = tc.matrix_b::<layout::Col>();
    let acc = tc.accumulator();

    // Mock pointers for demonstration (would be real in practice)
    let a_ptr: *const f16 = core::ptr::null();
    let b_ptr: *const f16 = core::ptr::null();
    let c_ptr: *mut f32 = core::ptr::null_mut();

    // Load with compile-time validated strides
    // These would actually load data if pointers were valid
    a.load::<16>(a_ptr);
    a.load::<32>(a_ptr);
    a.load::<64>(a_ptr);

    b.load::<16>(b_ptr);
    b.load::<32>(b_ptr);
    b.load::<64>(b_ptr);

    // Store with layout and stride
    acc.store::<layout::Row, 16>(c_ptr);
    acc.store::<layout::Col, 16>(c_ptr);
    acc.store::<layout::Row, 32>(c_ptr);
    acc.store::<layout::Col, 32>(c_ptr);
}

#[kernel]
pub unsafe fn test_accumulator_initialization() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();

    let mut acc = tc.accumulator();

    // Different ways to initialize accumulator
    acc.fill(0.0f32); // Zero
    acc.fill(1.0f32); // Identity-like
    acc.fill(2.5f32); // Custom value
    acc.fill(-0.5f32); // Negative value

    // For integer accumulators
    let tc_i8 = TensorCore::<i8, Shape>::new();
    let mut acc_i32 = tc_i8.accumulator();

    acc_i32.fill(0i32);
    acc_i32.fill(1i32);
    acc_i32.fill(-1i32);
    acc_i32.fill(100i32);
}

#[kernel]
pub unsafe fn test_16x8x16_shape_operations() {
    // Test the new 16x8x16 shape
    type Shape = dims::Shape<16, 8, 16>;

    // Test with f16
    {
        let tc = TensorCore::<f16, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Row>();
        let mut acc = tc.accumulator();

        acc.fill(0.0f32);
        acc.mma_inplace(&a, &b);
    }

    // Test with bf16
    {
        let tc = TensorCore::<bf16, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Row>();
        let mut acc = tc.accumulator();

        acc.fill(1.0f32);
        let _result = acc.mma(&a, &b);
    }

    // Test with i8
    {
        let tc = TensorCore::<i8, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Row>();
        let mut acc = tc.accumulator();

        acc.fill(0i32);
        acc.mma_inplace(&a, &b);
    }

    // Test with u8
    {
        let tc = TensorCore::<u8, Shape>::new();
        let a = tc.matrix_a::<layout::Row>();
        let b = tc.matrix_b::<layout::Row>();
        let mut acc = tc.accumulator();

        acc.fill(0i32);
        acc.mma_inplace(&a, &b);
    }
}
