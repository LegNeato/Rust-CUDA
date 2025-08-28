// Test matrix types and element types compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp::matrix::{
    dims, layout, Accumulator, MatrixA, MatrixB, MatrixElement, TensorCore, TensorCoreShape,
};
use cuda_std::{bf16, f16};

#[kernel]
pub unsafe fn test_all_element_types() {
    // Test f16 element type
    {
        let tc = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();
        let _a: MatrixA<f16, dims::Shape<16, 16, 16>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<f16, dims::Shape<16, 16, 16>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<f32, dims::Shape<16, 16, 16>> = tc.accumulator();
        let _c_f16: Accumulator<f16, dims::Shape<16, 16, 16>> = tc.accumulator_f16();
    }

    // Test bf16 element type
    {
        let tc = TensorCore::<bf16, dims::Shape<16, 16, 16>>::new();
        let _a: MatrixA<bf16, dims::Shape<16, 16, 16>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<bf16, dims::Shape<16, 16, 16>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<f32, dims::Shape<16, 16, 16>> = tc.accumulator();
    }

    // Test f32 element type (TF32)
    {
        let tc = TensorCore::<f32, dims::Shape<16, 16, 8>>::new();
        let _a: MatrixA<f32, dims::Shape<16, 16, 8>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<f32, dims::Shape<16, 16, 8>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<f32, dims::Shape<16, 16, 8>> = tc.accumulator();
    }

    // Test i8 element type
    {
        let tc = TensorCore::<i8, dims::Shape<16, 16, 16>>::new();
        let _a: MatrixA<i8, dims::Shape<16, 16, 16>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<i8, dims::Shape<16, 16, 16>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<i32, dims::Shape<16, 16, 16>> = tc.accumulator();
    }

    // Test u8 element type
    {
        let tc = TensorCore::<u8, dims::Shape<16, 16, 16>>::new();
        let _a: MatrixA<u8, dims::Shape<16, 16, 16>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<u8, dims::Shape<16, 16, 16>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<i32, dims::Shape<16, 16, 16>> = tc.accumulator();
    }

    // Test i32 element type
    {
        let tc = TensorCore::<i32, dims::Shape<16, 16, 16>>::new();
        let _a: MatrixA<i32, dims::Shape<16, 16, 16>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<i32, dims::Shape<16, 16, 16>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<i32, dims::Shape<16, 16, 16>> = tc.accumulator();
    }

    // Test bool element type (uses u8 under the hood)
    {
        let tc = TensorCore::<bool, dims::Shape<16, 16, 16>>::new();
        let _a: MatrixA<bool, dims::Shape<16, 16, 16>, layout::Row> = tc.matrix_a();
        let _b: MatrixB<bool, dims::Shape<16, 16, 16>, layout::Col> = tc.matrix_b();
        let _c: Accumulator<i32, dims::Shape<16, 16, 16>> = tc.accumulator();
    }
}

#[kernel]
pub unsafe fn test_all_valid_shapes() {
    // 16x16x16 - Square configuration
    {
        let tc = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    // 32x8x16 - Tall and skinny
    {
        let tc = TensorCore::<f16, dims::Shape<32, 8, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    // 8x32x16 - Short and wide
    {
        let tc = TensorCore::<f16, dims::Shape<8, 32, 16>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    // 16x16x8 - Reduced K dimension
    {
        let tc = TensorCore::<f16, dims::Shape<16, 16, 8>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    // 8x8x32 - Small MxN, large K
    {
        let tc = TensorCore::<f16, dims::Shape<8, 8, 32>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    // 8x8x128 - Small MxN, very large K
    {
        let tc = TensorCore::<f16, dims::Shape<8, 8, 128>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }

    // 8x8x4 - Minimal configuration
    {
        let tc = TensorCore::<f16, dims::Shape<8, 8, 4>>::new();
        let _a = tc.matrix_a::<layout::Row>();
        let _b = tc.matrix_b::<layout::Col>();
        let _c = tc.accumulator();
    }
}

#[kernel]
pub unsafe fn test_layout_combinations() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();

    // Row-major matrix A
    let a_row: MatrixA<f16, Shape, layout::Row> = tc.matrix_a();

    // Column-major matrix A
    let a_col: MatrixA<f16, Shape, layout::Col> = tc.matrix_a();

    // Row-major matrix B
    let b_row: MatrixB<f16, Shape, layout::Row> = tc.matrix_b();

    // Column-major matrix B
    let b_col: MatrixB<f16, Shape, layout::Col> = tc.matrix_b();

    // All combinations are valid for MMA
    let mut acc = tc.accumulator();
    // Row-Row combination
    let _result1 = acc.mma(&a_row, &b_row);
    // Row-Col combination
    let _result2 = acc.mma(&a_row, &b_col);
    // Col-Row combination
    let _result3 = acc.mma(&a_col, &b_row);
    // Col-Col combination
    let _result4 = acc.mma(&a_col, &b_col);
}

// Helper generic function (not a kernel)
fn test_generic_operations_helper<T>()
where
    T: MatrixElement,
{
    // Generic function with matrix element constraint
    fn create_matrix_a<E, S, L>() -> MatrixA<E, S, L>
    where
        E: MatrixElement,
        S: TensorCoreShape,
        L: layout::Layout,
    {
        MatrixA::new()
    }

    // Can be called with any valid element type
    let _a_f16 = create_matrix_a::<f16, dims::Shape<16, 16, 16>, layout::Row>();
    let _a_i8 = create_matrix_a::<i8, dims::Shape<16, 16, 16>, layout::Col>();
}

#[kernel]
pub unsafe fn test_generic_matrix_operations() {
    // Kernel functions can't be generic, but they can call generic functions
    test_generic_operations_helper::<f16>();
    test_generic_operations_helper::<i8>();
}

#[kernel]
pub unsafe fn test_default_implementations() {
    type Shape = dims::Shape<16, 16, 16>;

    // Test Default trait implementations
    let _a: MatrixA<f16, Shape, layout::Row> = Default::default();
    let _b: MatrixB<f16, Shape, layout::Col> = Default::default();
    let _c: Accumulator<f32, Shape> = Default::default();

    // Test for different types
    let _a_i8: MatrixA<i8, Shape, layout::Row> = Default::default();
    let _b_u8: MatrixB<u8, Shape, layout::Col> = Default::default();
    let _c_i32: Accumulator<i32, Shape> = Default::default();
}
