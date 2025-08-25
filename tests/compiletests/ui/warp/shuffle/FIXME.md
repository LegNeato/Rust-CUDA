# Shuffle Test Issues

## Current Status
- Direct trait calls work: `<i32 as ShuffleValue>::shuffle_down(mask, value, 1, 32)`
- Creating Shuffle struct works: `Shuffle::<i32>::new(mask, width)`
- Creating patterns works: `patterns::Down::new(1)`
- **CRASHES**: Calling methods on Shuffle struct: `shuffle.down(value, pattern)`

## Issue
The NVVM backend segfaults when compiling code that uses the Shuffle struct methods with patterns.
This appears to be a compiler bug related to how the generic bounds and phantom types are handled.

## Workaround
Use the trait methods directly instead of the Shuffle struct for now.

## Tests Status
- ✅ minimal_shuffle.rs - Direct trait call
- ✅ shuffle_struct_test.rs - Creating Shuffle struct
- ✅ pattern_test.rs - Creating patterns
- ❌ shuffle_down_test.rs - Using Shuffle methods (segfault)
- ❌ basic_shuffle.rs - Uses Shuffle methods
- ❌ shuffle_types.rs - Uses Shuffle methods
- ❌ shuffle_small_types.rs - Uses Shuffle methods
- ❌ shuffle_widths.rs - Uses Shuffle methods
- ❌ shuffle_patterns.rs - Uses Shuffle methods