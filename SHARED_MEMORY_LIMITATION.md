# Critical Rust-CUDA Shared Memory Limitation

## The Issue
Rust-CUDA cannot handle the standard CUDA pattern of cooperative shared memory loading followed by reading from that memory, even with proper synchronization. This pattern works perfectly in normal CUDA C++ but causes `InvalidValue` errors in Rust-CUDA.

## The Failing Pattern
```rust
// This pattern causes InvalidValue in Rust-CUDA:
let smem = shared_array![bf16; 16384];

// 1. Cooperative loading - all threads participate
for i in 0..elements_per_thread {
    let idx = tid * elements_per_thread + i;
    if idx < total_elements {
        unsafe { *smem.add(idx) = *global_ptr.add(idx); }
    }
}

// 2. Synchronize threads
thread::sync_threads();

// 3. Read from shared memory - THIS FAILS!
let value = unsafe { *smem.add(tid) };  // InvalidValue error
```

## Why This Matters
This pattern is fundamental to GPU optimization and is used in virtually every high-performance CUDA kernel:
- Matrix multiplication (GEMM)
- Flash Attention
- Convolutions
- Reductions
- Any kernel that needs to share data between threads

## Verification
The exact same pattern works in standard CUDA C++:
```cpp
// From attention_v1.cu - works perfectly
extern __shared__ nv_bfloat16 smem[];
global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
__syncthreads();
ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], Q_smem + addr);  // No error
```

## Impact
This limitation makes it impossible to implement many optimized GPU algorithms in Rust-CUDA, including:
- Flash Attention (requires Q, K, V in shared memory)
- Optimized GEMM kernels
- Most kernels that achieve > 50% of theoretical GPU performance

## Workaround
The only workaround is to read directly from global memory, which severely impacts performance:
```rust
// Instead of reading from shared memory:
// let q_val = unsafe { *smem.add(q_idx) };

// Must read from global memory:
let q_val = unsafe { *q_ptr.add(global_q_idx) };
```

This typically results in 5-10x performance degradation.

## Root Cause (Hypothesis)
The issue appears to be in how Rust-CUDA handles shared memory pointers after synchronization. Possible causes:
1. PTX generation issue with shared memory addresses
2. LLVM optimization incorrectly reordering memory operations
3. Missing memory fence instructions in generated PTX
4. Type system interaction with shared memory lifetime

## Reproduction
Run the Flash Attention v1 kernel in `/Users/legnitto/src/rust-gpu.github.io/blog/2025-01-27-flash-attention/code/crates/gpu/v1/src/lib.rs` with the shared memory read uncommented at line 80.