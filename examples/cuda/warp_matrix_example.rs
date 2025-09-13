use cuda_std::warp::*;
use cuda_std::*;
use glam::Vec4;
use half::f16;

/// Example kernel demonstrating tensor core usage with the new warp matrix API
#[kernel]
pub unsafe fn wmma_example_kernel(
    a_ptr: *const f16,
    b_ptr: *const f16,
    c_ptr: *const f32,
    d_ptr: *mut f32,
) {
    // Load matrix fragments from memory
    let frag_a: FragmentA16x16x16F16ColMajor = load_matrix_sync(a_ptr, 16);
    let frag_b: FragmentB16x16x16F16RowMajor = load_matrix_sync(b_ptr, 16);
    let frag_c: FragmentC16x16x16F32 = load_matrix_sync(c_ptr, 16);

    // Perform matrix multiply-accumulate: D = A * B + C
    let frag_d: FragmentC16x16x16F32 = mma_sync::<_, _, _, MMA16x16x16F16F16F32ColRow>(&frag_a, &frag_b, &frag_c);

    // Store result back to memory
    store_matrix_sync(d_ptr, &frag_d, 16, MatrixLayout::RowMajor);
}

/// Example showing fill_fragment usage
#[kernel] 
pub unsafe fn wmma_fill_example() {
    let mut frag_c: FragmentC16x16x16F32 = FragmentC16x16x16F32 { x: [0.0; 8] };
    
    // Fill fragment with constant value
    fill_fragment(&mut frag_c, 1.0f32);
    
    // Fragment is now filled with 1.0 values
}

/// Example showing glam interop
#[kernel]
pub unsafe fn wmma_glam_example() {
    // Convert from glam Vec4 to fragment
    let vec = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let frag_a: FragmentA16x16x16F16ColMajor = vec.into();
    
    // Create fragment from array of Vec4s
    let vecs = [Vec4::new(1.0, 2.0, 3.0, 4.0), Vec4::new(5.0, 6.0, 7.0, 8.0)];
    let frag_c: FragmentC16x16x16F32 = vecs.into();
    
    // Convert fragment back to array of Vec4s
    let result_vecs: [Vec4; 2] = frag_c.into();
}