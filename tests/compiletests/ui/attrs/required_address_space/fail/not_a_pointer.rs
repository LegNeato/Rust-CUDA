// compile-fail

use cuda_std::kernel;

// Error: Parameter 'value' must be a pointer type
#[cuda_std::required_address_space(value = shared)]
unsafe fn process_value(value: f32) -> f32 {
    value + 1.0
}

#[kernel]
pub unsafe fn test_kernel() {}
