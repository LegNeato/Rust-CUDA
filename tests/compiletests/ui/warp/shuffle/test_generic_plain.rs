// Test if generic functions without Result work
// build-pass

use cuda_std::kernel;

// Generic function that returns T directly (no Result)
unsafe fn identity_generic<T: Copy>(value: T) -> T {
    value
}

// Generic function with Option return
unsafe fn option_generic<T: Copy>(value: T) -> Option<T> {
    Some(value)
}

#[kernel]
pub unsafe fn test_generic_plain() {
    let value = 42i32;
    
    // Generic function returning T
    let _result1 = identity_generic::<i32>(value);
    
    // Generic function returning Option<T>
    let _result2 = option_generic::<i32>(value);
}