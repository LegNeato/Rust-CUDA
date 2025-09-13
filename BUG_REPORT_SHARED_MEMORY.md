# Rust-CUDA Critical Bug: Shared Memory Not Emitted to PTX

## Summary
The `shared_array!` macro in Rust-CUDA does not emit any shared memory declarations in the generated PTX code, causing runtime `InvalidValue` errors when trying to access the "shared" memory.

## The Bug
When using `shared_array!` macro:
```rust
let smem = shared_array![bf16; 16384];
```

Expected PTX output (like CUDA C++):
```ptx
.extern .shared .align 16 .b8 smem[];
```

Actual PTX output:
**No shared memory declaration at all**

## Root Cause Analysis

### 1. PTX Generation Issue
Comparing Rust-CUDA vs CUDA C++ PTX output:

**CUDA C++ (working):**
```ptx
.extern .shared .align 16 .b8 smem[];
// Later in code:
cp.async.cg.shared.global [%r355], [%rd59], 16;
```

**Rust-CUDA (broken):**
- No `.shared` declaration anywhere in PTX
- The macro creates a static but it's being treated as local memory
- No `addrspace(3)` annotations in LLVM IR

### 2. Macro Implementation Problem
The `shared_array!` macro (in `/Users/legnitto/src/Rust-CUDA/crates/cuda_std/src/shared.rs`) creates:
```rust
#[address_space(shared)]
static SHARED: SyncWrapper = SyncWrapper(UnsafeCell::new(MaybeUninit::uninit()));
```

But this `#[address_space(shared)]` attribute is not being properly translated through the compilation pipeline:
1. LLVM IR doesn't contain `addrspace(3)` annotations
2. PTX doesn't contain `.shared` declarations
3. At runtime, the memory is actually local/register memory, not shared

### 3. Why It Causes InvalidValue
When the kernel tries to:
1. Write to what it thinks is shared memory (actually local)
2. Call `thread::sync_threads()`
3. Read from that "shared" memory

The CUDA runtime detects invalid memory access patterns and returns `InvalidValue`.

## Impact
This bug makes it **impossible** to implement any optimized GPU kernels in Rust-CUDA, including:
- Flash Attention
- Matrix multiplication (GEMM)
- Convolutions
- Any kernel requiring thread cooperation

## Reproduction
```rust
#[kernel]
pub unsafe fn test_shared() {
    let smem = shared_array![f32; 256];
    *smem.add(0) = 1.0;
    thread::sync_threads();
    let val = *smem.add(0); // InvalidValue here
}
```

## Verification
The same pattern works perfectly in CUDA C++, confirming this is a Rust-CUDA codegen bug.

## Suggested Fix
The `rustc_codegen_nvvm` backend needs to:
1. Recognize `#[address_space(shared)]` attributes
2. Emit proper `addrspace(3)` in LLVM IR
3. Generate `.shared` declarations in PTX

## Workaround
None. Shared memory is fundamental to GPU programming and cannot be worked around efficiently.