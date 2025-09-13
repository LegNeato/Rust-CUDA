# SHA2 on Rust-CUDA: Debugging and Fix Documentation

## Problem Statement
The SHA2 crate compiles successfully with rust-cuda but fails at runtime with `LaunchFailed` errors when executing certain kernels (specifically SHA512).

## Investigation Process

### 1. Initial Symptom Discovery
- **Tool Used**: `deploy-cuda.sh` script to deploy and run examples on remote CUDA machine
- **Finding**: SHA256 kernels work, but SHA512 kernels fail with `Error: LaunchFailed`

### 2. PTX Analysis
- **Tools Used**: 
  - Custom bash scripts to extract and analyze PTX
  - `grep` to search for function definitions
  - `sed` and `awk` to extract specific kernel code
  
- **Key Findings**:
  - All 4 kernels (sha256_oneshot, sha256_incremental, sha512_oneshot, sha512_incremental) are present in PTX
  - SHA256 kernel has 9679 registers and 256 bytes of local memory
  - SHA512 kernel has only 6 registers and contains `trap` instructions

### 3. Root Cause Identification
- **Location**: `/crates/rustc_codegen_nvvm/src/builder.rs`
- **Problem**: The `atomic_load` and `atomic_store` functions immediately insert `trap` instructions
- **Code Pattern**:
  ```rust
  fn atomic_load(...) {
      let (ty, f) = self.get_intrinsic("llvm.trap");
      self.call(ty, None, None, f, &[], None, None);
      // ... unreachable load
  }
  ```

### 4. Why SHA512 Triggers This
- SHA512 uses 64-bit operations more extensively than SHA256
- Some operation in SHA512 (likely array access or integer operations) gets compiled to atomic loads/stores
- NVVM IR doesn't support native atomic loads/stores, causing the codegen to insert trap instructions

## The Fix

### Fix 1: Atomic Operations (Partial Success)
We fixed the atomic load/store operations that were generating trap instructions by emulating them with:
1. **Volatile loads/stores** - Prevents optimization and ensures memory access
2. **Memory barriers** (`llvm.nvvm.membar.sys`) - Ensures proper memory ordering

This fixed *some* trap instructions but not all.

### Root Cause 2: 128-bit Integer Operations
After fixing atomic operations, we discovered that rust-cuda also generates trap instructions for **any 128-bit integer intrinsics**, including:
- `ctlz`, `cttz`, `ctpop` (bit counting operations)
- `bswap`, `bitreverse` (byte/bit reversal)
- `rotate_left`, `rotate_right` (bit rotation)

SHA512 appears to use some operations that result in 128-bit intermediate values, triggering these traps.

### Implementation

#### For atomic_load:
```rust
fn atomic_load(...) -> &'ll Value {
    // Add memory fence before for acquire semantics
    match order {
        Acquire | AcqRel | SequentiallyConsistent => {
            self.call(membar_sys, ...);
        }
    }
    
    // Volatile load
    let load = LLVMBuildLoad(...);
    LLVMSetVolatile(load, True);
    
    // Add fence after for sequential consistency
    if order == SequentiallyConsistent {
        self.call(membar_sys, ...);
    }
    
    load
}
```

#### For atomic_store:
Similar pattern with fences for release semantics and volatile stores.

## Testing Strategy

### Test Files
1. `examples/cuda/sha2_crates_io/` - Contains comprehensive tests for both SHA256 and SHA512 using the sha2 crate from crates.io:
   - One-shot API (`Sha256::digest()`)
   - Incremental API (`hasher.update()`)

### Deployment Process
1. Use `run-example.sh` for quick deployment and testing
2. Verifies GPU results against CPU implementation
3. All 4 test cases should show "✅ Results match!"

## Current Investigation (2025-08-08)

### Root Cause Identified: SHA512 Uses u128 for Block Length Tracking

After extensive debugging and tracing, we've identified the exact source of the SHA512 failure:

#### The Smoking Gun
In `sha2-0.10.8/src/core_api.rs`, the `Sha512VarCore` struct uses a u128 field:
```rust
pub struct Sha512VarCore {
    state: consts::State512,
    block_len: u128,  // <-- This is the problem!
}

impl UpdateCore for Sha512VarCore {
    fn update_blocks(&mut self, blocks: &[Block<Self>]) {
        self.block_len += blocks.len() as u128;  // Line 108: u128 arithmetic
        compress512(&mut self.state, blocks);
    }
}
```

Compare this with `Sha256VarCore` which uses `block_len: u64` - this is why SHA256 works but SHA512 doesn't!

#### What Happens During Compilation

When SHA512::digest() is called:
1. Creates a `Sha512VarCore` with `block_len: u128 = 0`
2. Calls `update()` which performs `self.block_len += blocks.len() as u128`
3. This triggers a cascade of u128 operations

Our debug output shows hundreds of u128 operations being generated, including:
- Basic arithmetic (add, sub, mul) - ✅ Successfully emulated
- Bitwise operations (and, or, xor, shifts) - ✅ Successfully emulated  
- Comparisons (IntEQ, IntULT, IntUGE) - ✅ Work with LLVM
- **Division and remainder (udiv, urem)** - ❌ NOT IMPLEMENTED - causes traps!

#### The Missing Operations

The critical missing operations that cause the failure:
```
WARNING: Unimplemented i128 operation: udiv with 2 args, falling back to trap
WARNING: Unimplemented i128 operation: urem with 2 args, falling back to trap
```

These division operations are likely generated for:
- Calculating how many 128-byte blocks fit in the input (`length / 128`)
- Getting the remaining bytes (`length % 128`)

### Solution Implemented: Partial u128 Arithmetic Emulation

We've implemented emulation of most 128-bit integer operations using pairs of 64-bit values:

1. **Basic arithmetic**: add, sub, mul with proper carry/borrow handling
2. **Bitwise operations**: and, or, xor, not
3. **Shift operations**: shl, lshr, ashr with full support for shifts >= 64
4. **Unary operations**: neg (two's complement)

The implementation intercepts these operations in the LLVM builder macro and replaces them with emulated versions using 64-bit operations that NVVM can handle.

### Still Missing: Division and Remainder

To fully fix SHA512, we need to implement:
- `emulate_i128_udiv`: 128-bit unsigned division
- `emulate_i128_urem`: 128-bit unsigned remainder
- `emulate_i128_sdiv`: 128-bit signed division (if needed)
- `emulate_i128_srem`: 128-bit signed remainder (if needed)

These are complex operations requiring long division algorithms using 64-bit primitives.

## Current Investigation (2025-08-07)

### New Finding: 128-bit Integer Intrinsics Generate Traps

After analyzing the PTX output and tracing through the codegen, we've discovered that **any 128-bit integer intrinsics** are generating trap instructions in rust-cuda. This is happening in `/crates/rustc_codegen_nvvm/src/intrinsic.rs`.

#### Code Path:
1. SHA512 operations trigger certain intrinsics (likely in internal operations)
2. These hit the `codegen_intrinsic_call` function in `intrinsic.rs`
3. For operations like `ctlz`, `cttz`, `ctpop`, `bswap`, `bitreverse`, `rotate_left`, `rotate_right`:
   - If the width is 128 bits, it calls `handle_128_bit_intrinsic`
   - This function attempts to use LLVM intrinsics like `llvm.ctpop.i128`
   - However, NVVM doesn't properly support these, leading to trap generation

#### Evidence from PTX:
```ptx
// SHA512 kernel contains multiple trap instructions
$L__BB2_6:
    setp.lt.u64 	%p5, %rd2, 129;
    @%p5 bra 	$L__BB2_8;
    trap;

$L__BB2_8:
    trap;
```

#### The Problem Code (intrinsic.rs:27-67):
```rust
fn handle_128_bit_intrinsic<'ll>(
    b: &mut Builder<'_, 'll, '_>,
    name: Symbol,
    args: &[OperandRef<'_, &'ll Value>],
) -> &'ll Value {
    // CUDA 12+ has native __int128 support, so we can use LLVM intrinsics directly
    match name {
        sym::ctlz | sym::cttz => {
            // Tries to use llvm.ctlz.i128, etc.
            b.call_intrinsic(&llvm_name, &[args[0].immediate(), y])
        }
        // ... other 128-bit operations
        _ => {
            // Falls back to abort for unsupported ops
            b.abort_and_ret_i128()
        }
    }
}
```

The comment claims "CUDA 12+ supports 128-bit integers natively" but this is clearly not working with NVVM.

## Next Steps

### To Fix the Issue:
1. ⏳ Implement proper 128-bit intrinsic emulation (similar to atomic operations)
2. ⏳ Test with SHA512 to ensure it works
3. ⏳ Verify no performance regression in SHA256

### Potential Solutions:
1. **Emulate 128-bit operations using 64-bit operations** (like LLVM does for targets without native 128-bit support)
2. **Use software implementations** of these intrinsics
3. **Detect and warn** when 128-bit operations are used

### Future Improvements:
1. Add proper atomic operation support for common cases (atomicAdd, atomicCAS)
2. Improve error messages when CUDA features are unsupported
3. Add compile-time warnings for potentially problematic operations
4. Consider whether CUDA 12's __int128 support can be leveraged differently

## Technical Notes

### NVVM Limitations
- NVVM IR lacks native atomic load/store instructions
- Only has atomic RMW operations (add, sub, exchange, CAS)
- Memory ordering must be emulated with explicit barriers

### Memory Barrier Types in NVVM
- `llvm.nvvm.membar.sys` - System-wide memory fence
- `llvm.nvvm.membar.gl` - Global memory fence
- `llvm.nvvm.membar.cta` - Thread block level fence

### Volatile vs Atomic
- Volatile: Prevents compiler optimizations, ensures memory access happens
- Atomic: Provides indivisible operations with memory ordering guarantees
- Our solution: Volatile + barriers ≈ Atomic (good enough for most cases)

## Tools and Scripts Created

1. **run-example.sh** - Simplified deployment script
2. **check_ptx.sh** - Analyzes PTX function signatures
3. **analyze_ptx.sh** - Compares PTX between working/failing kernels
4. **extract_sha512.sh** - Extracts specific kernel code
5. **debug_launch.sh** - Runs with debug environment variables

## Lessons Learned

1. **PTX inspection is crucial** - The `trap` instructions were the smoking gun
2. **Register count differences** - Can indicate optimization/compilation issues
3. **Atomic operations in CUDA** - Are more limited than CPU architectures
4. **Emulation is acceptable** - Perfect atomics aren't always needed; volatile + barriers work for many cases