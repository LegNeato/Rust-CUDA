// compile-fail

use cuda_std::kernel;

// Error: required_address_space requires explicit parameter names
#[cuda_std::required_address_space(shared)]
unsafe fn load_from_shared(ptr: *const f32) -> f32 {
    *ptr
}

#[kernel]
pub unsafe fn test_kernel() {}