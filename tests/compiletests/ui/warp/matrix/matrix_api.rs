// Test matrix API patterns and builder interface
// build-pass

use cuda_std::kernel;
use cuda_std::warp::matrix::{
    dims, layout, Accumulator, MatrixA, MatrixB, MatrixElement, StrideValidator, 
    TensorCore, TensorCoreShape, ValidStride,
};
use cuda_std::{bf16, f16};

#[kernel]
pub unsafe fn test_tensor_core_builder() {
    const TC: TensorCore<f16, dims::Shape<16, 16, 16>> =
        TensorCore::<f16, dims::Shape<16, 16, 16>>::new();

    let _a = TC.matrix_a::<layout::Row>();
    let _b = TC.matrix_b::<layout::Col>();
    let _c = TC.accumulator();
}

#[kernel]
pub unsafe fn test_stride_validation() {
    type Shape = dims::Shape<16, 16, 16>;
    let tc = TensorCore::<f16, Shape>::new();
    let mut a = tc.matrix_a::<layout::Row>();

    let ptr: *const f16 = core::ptr::null();

    a.load::<8>(ptr);
    a.load::<16>(ptr);
    a.load::<24>(ptr);
    a.load::<32>(ptr);
    a.load::<40>(ptr);
    a.load::<48>(ptr);
    a.load::<56>(ptr);
    a.load::<64>(ptr);

    let tc_f32 = TensorCore::<f32, Shape>::new();
    let mut a_f32 = tc_f32.matrix_a::<layout::Row>();
    let ptr_f32: *const f32 = core::ptr::null();

    a_f32.load::<4>(ptr_f32);
    a_f32.load::<8>(ptr_f32);
    a_f32.load::<12>(ptr_f32);
    a_f32.load::<16>(ptr_f32);
    a_f32.load::<20>(ptr_f32);
    a_f32.load::<24>(ptr_f32);
    a_f32.load::<28>(ptr_f32);
    a_f32.load::<32>(ptr_f32);
}

#[kernel]
pub unsafe fn test_matrix_creation_patterns() {
    type Shape = dims::Shape<16, 16, 16>;

    let a1: MatrixA<f16, Shape, layout::Row> = MatrixA::new();
    let b1: MatrixB<f16, Shape, layout::Col> = MatrixB::new();
    let c1: Accumulator<f32, Shape> = Accumulator::new();

    let tc = TensorCore::<f16, Shape>::new();
    let a2 = tc.matrix_a::<layout::Row>();
    let b2 = tc.matrix_b::<layout::Col>();
    let c2 = tc.accumulator();

    let a3: MatrixA<f16, Shape, layout::Row> = Default::default();
    let b3: MatrixB<f16, Shape, layout::Col> = Default::default();
    let c3: Accumulator<f32, Shape> = Default::default();
}

#[kernel]
pub unsafe fn test_type_inference() {
    let tc = TensorCore::<f16, dims::Shape<16, 16, 16>>::new();

    let a = tc.matrix_a::<layout::Row>();
    let b = tc.matrix_b::<layout::Col>();
    let c = tc.accumulator();

    let tc_i8 = TensorCore::<i8, dims::Shape<16, 16, 16>>::new();
    let c_i8 = tc_i8.accumulator();

    let tc_bf16 = TensorCore::<bf16, dims::Shape<16, 16, 16>>::new();
    let c_bf16 = tc_bf16.accumulator();
}

#[kernel]
pub unsafe fn test_shape_as_type_parameter() {
    type SmallShape = dims::Shape<8, 8, 4>;
    type StandardShape = dims::Shape<16, 16, 16>;
    type LargeKShape = dims::Shape<8, 8, 128>;

    let tc_small = TensorCore::<f16, SmallShape>::new();
    let tc_standard = TensorCore::<f16, StandardShape>::new();
    let tc_large_k = TensorCore::<f16, LargeKShape>::new();

    let _a_small = tc_small.matrix_a::<layout::Row>();
    let _a_standard = tc_standard.matrix_a::<layout::Row>();
    let _a_large_k = tc_large_k.matrix_a::<layout::Row>();
}

#[kernel]
pub unsafe fn test_generic_functions_with_constraints() {
    fn create_tensor_core<S>() -> TensorCore<f16, S>
    where
        S: TensorCoreShape,
    {
        TensorCore::new()
    }

    fn create_matrix_a<T, S>() -> MatrixA<T, S, layout::Row>
    where
        T: MatrixElement,
        S: TensorCoreShape,
    {
        MatrixA::new()
    }

    let _tc1 = create_tensor_core::<dims::Shape<16, 16, 16>>();
    let _tc2 = create_tensor_core::<dims::Shape<8, 8, 32>>();

    let _a1 = create_matrix_a::<f16, dims::Shape<16, 16, 16>>();
    let _a2 = create_matrix_a::<i8, dims::Shape<8, 8, 4>>();
}

#[kernel]
pub unsafe fn test_accumulator_specialization() {
    type Shape = dims::Shape<16, 16, 16>;

    {
        let tc = TensorCore::<f16, Shape>::new();
        let _acc_f32 = tc.accumulator();
        let _acc_f16 = tc.accumulator_f16();
    }

    {
        let tc_bf16 = TensorCore::<bf16, Shape>::new();
        let _acc = tc_bf16.accumulator();

        let tc_i8 = TensorCore::<i8, Shape>::new();
        let _acc = tc_i8.accumulator();

        let tc_u8 = TensorCore::<u8, Shape>::new();
        let _acc = tc_u8.accumulator();

        let tc_f32 = TensorCore::<f32, Shape>::new();
        let _acc = tc_f32.accumulator();

        let tc_i32 = TensorCore::<i32, Shape>::new();
        let _acc = tc_i32.accumulator();
    }
}
