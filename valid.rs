// build-pass

use cuda_std::kernel;

// Test with single pointer parameter - shared memory
#[cuda_std::required_address_space(ptr = shared)]
unsafe fn load_from_shared(ptr: *const f32) -> f32 {
    *ptr
}

// Test with any(global, constant) syntax
#[cuda_std::required_address_space(ptr = any(global, constant))]
unsafe fn load_from_global_or_constant(ptr: *const f32) -> f32 {
    *ptr
}

// Test with multiple pointer parameters
#[cuda_std::required_address_space(src = shared, dst = global)]
unsafe fn copy_shared_to_global(src: *const f32, dst: *mut f32) {
    *dst = *src;
}

// Test with mutable pointer
#[cuda_std::required_address_space(ptr = shared)]
unsafe fn store_to_shared(ptr: *mut f32, value: f32) {
    *ptr = value;
}

// Test with multiple address spaces in any()
#[cuda_std::required_address_space(ptr = any(global, shared, constant))]
unsafe fn load_from_any(ptr: *const f32) -> f32 {
    *ptr
}

#[kernel]
pub unsafe fn test_kernel() {
    // The test validates that the macro parses and applies attributes correctly.
    // We don't need to actually call the functions for the compile-time validation to work.
}