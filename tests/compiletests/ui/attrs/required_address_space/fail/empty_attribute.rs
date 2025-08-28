// compile-fail

use cuda_std::kernel;

// Error: Empty attribute
#[cuda_std::required_address_space()]
unsafe fn load_from_shared(ptr: *const f32) -> f32 {
    *ptr
}

#[kernel]
pub unsafe fn test_kernel() {}
