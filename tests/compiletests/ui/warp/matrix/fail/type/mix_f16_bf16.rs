// compile-fail

use cuda_std::kernel;
use cuda_std::warp::matrix::{dims, layout, MmaExt, TensorCore};
use cuda_std::{bf16, f16};

#[kernel]
pub unsafe fn test_mixed_types() {
    type Shape = dims::Shape<16, 16, 16>;

    let tc_f16 = TensorCore::<f16, Shape>::new();
    let a_f16 = tc_f16.matrix_a::<layout::Row>();

    let tc_bf16 = TensorCore::<bf16, Shape>::new();
    let b_bf16 = tc_bf16.matrix_b::<layout::Col>();
    let mut c_bf16 = tc_bf16.accumulator();

    // Cannot mix f16 and bf16 matrices
    c_bf16.mma_inplace(&a_f16, &b_bf16);
}
