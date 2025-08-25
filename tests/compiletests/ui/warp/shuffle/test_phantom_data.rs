// Test if PhantomData in generic structs causes issues
// build-pass

use cuda_std::kernel;
use core::marker::PhantomData;

#[repr(C)]
struct TestStruct<T> {
    value: u32,
    _phantom: PhantomData<T>,
}

impl<T> TestStruct<T> {
    fn new(value: u32) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
    
    // Takes self by value like Shuffle does
    fn test_method(self) -> u32 {
        self.value
    }
}

#[kernel]
pub unsafe fn test_phantom_data() {
    let test = TestStruct::<i32>::new(42);
    let _result = test.test_method();
}