# Rust-CUDA Bug: `shared_array!` Macro Doesn't Emit Shared Memory

## Summary
The `shared_array!` macro fails to create actual shared memory. The static it creates inside a function is optimized away or treated as local memory, causing `InvalidValue` errors at runtime.

## The Problem

### What Works ✅
Module-level shared memory using `#[cuda_std::address_space(shared)]`:
```rust
#[cuda_std::address_space(shared)]
static mut GLOBAL_SHARED: [f32; 256] = [0.0; 256];

#[kernel]
pub unsafe fn my_kernel() {
    GLOBAL_SHARED[tid] = value;  // Works!
    thread::sync_threads();
    let val = GLOBAL_SHARED[tid];  // Works!
}
```

This generates correct PTX:
```ptx
.shared .align 2 .b8 _ZN12attention_v113GLOBAL_SHARED17h8e04c87fec73dcb0E[1024];
st.shared.f32 [%rd6], %f1;
ld.shared.f32 %f2, [%rd6];
```

### What Doesn't Work ❌
The `shared_array!` macro:
```rust
#[kernel]
pub unsafe fn my_kernel() {
    let smem = shared_array![f32; 256];
    *smem.add(tid) = value;  // Writes to local memory
    thread::sync_threads();
    let val = *smem.add(tid);  // InvalidValue!
}
```

This generates NO shared memory declaration in PTX!

## Root Cause

The `shared_array!` macro creates a function with a static inside:
```rust
fn shared_array() -> *mut T {
    #[address_space(shared)]
    static SHARED: ... = ...;
    SHARED.0.get() as *mut T
}
```

Because the function is `#[inline(always)]` and the static is inside a function scope, the compiler:
1. Inlines the function
2. Treats the "static" as a local allocation
3. Never emits `.shared` declarations in PTX
4. At runtime, memory accesses go to local/register memory instead of shared

## Impact
- The `shared_array!` macro is completely broken
- Users must use module-level statics instead
- This breaks compatibility with CUDA patterns
- Flash Attention and other algorithms can't be implemented idiomatically

## Workaround
Use module-level shared memory statics:
```rust
#[cuda_std::address_space(shared)]
static mut SHARED: [T; SIZE] = [initial; SIZE];
```

Limitations of workaround:
- Can't have dynamic sizes
- Must be declared at module level
- Can't have multiple different shared allocations per kernel
- Breaks encapsulation

## Fix Needed
The `shared_array!` macro needs to be redesigned. Options:
1. Generate module-level statics with unique names
2. Use a different mechanism that preserves shared memory through inlining
3. Add compiler support for function-scoped shared statics
4. Use extern shared memory like `dynamic_shared_mem` does

## Verification
Compile any kernel using `shared_array!` and check the PTX - there will be no `.shared` declarations.