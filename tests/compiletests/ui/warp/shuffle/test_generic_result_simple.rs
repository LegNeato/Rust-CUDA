// Test if generic Result with simple error type works
// build-pass

use cuda_std::kernel;

// Simple error type
#[derive(Clone, Copy)]
struct SimpleError;

// Generic function with Result<T, SimpleError>
unsafe fn result_simple_generic<T: Copy>(value: T) -> Result<T, SimpleError> {
    Ok(value)
}

// Generic function with Result<T, ()> - unit error
unsafe fn result_unit_generic<T: Copy>(value: T) -> Result<T, ()> {
    Ok(value)
}

#[kernel]
pub unsafe fn test_generic_result_simple() {
    let value = 42i32;
    
    // Generic function returning Result<T, SimpleError>
    let _result1 = result_simple_generic::<i32>(value);
    
    // Generic function returning Result<T, ()>
    let _result2 = result_unit_generic::<i32>(value);
}