// compile-fail

use cuda_std::kernel;

// Error: Parameter 'wrong_name' not found in function signature
#[cuda_std::required_address_space(wrong_name = shared)]
unsafe fn load_from_shared(ptr: *const f32) -> f32 {
    *ptr
}

#[kernel]
pub unsafe fn test_kernel() {}