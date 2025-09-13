#!/bin/bash
set -e

# Remote connection details  
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Analyzing i128 implementation in PTX..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== Creating test program with i128 operations ==="
cat > /tmp/test_i128.rs << 'RUSTEOF'
#![no_std]
#![no_main]
#![feature(abi_ptx)]

use core::panic::PanicInfo;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_i128_ops(a: *mut i128, b: *const i128, c: *const i128) {
    let b_val = *b;
    let c_val = *c;
    
    // Test overflow operations
    let (add_result, add_overflow) = b_val.overflowing_add(c_val);
    let (sub_result, sub_overflow) = b_val.overflowing_sub(c_val);
    let (mul_result, mul_overflow) = b_val.overflowing_mul(c_val);
    
    *a = if add_overflow { 
        add_result 
    } else if sub_overflow { 
        sub_result 
    } else if mul_overflow { 
        mul_result 
    } else { 
        add_result.wrapping_add(sub_result).wrapping_add(mul_result)
    };
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}
RUSTEOF

echo -e "\n=== Compiling test with Rust-CUDA ==="
export CUDA_PATH=/usr/local/cuda
export RUST_BACKTRACE=1

# Try to compile with rustc_codegen_nvvm
cd /root/rust-cuda
if [ -f "crates/rustc_codegen_nvvm/Cargo.toml" ]; then
    echo "Building with rustc_codegen_nvvm..."
    cd crates/rustc_codegen_nvvm
    cargo build --release 2>&1 | tail -5 || echo "Build had issues"
    
    # Find the codegen library
    CODEGEN_LIB=$(find target -name "librustc_codegen_nvvm*.so" 2>/dev/null | head -1)
    if [ -n "$CODEGEN_LIB" ]; then
        echo "Found codegen at: $CODEGEN_LIB"
    fi
fi

echo -e "\n=== Checking how SHA2 example handles i128 ==="
cd /root/rust-cuda/examples/cuda/cpu/sha2_crates_io

# Look for u128/i128 usage in source
echo "Searching for 128-bit integer usage in SHA2 source:"
grep -r "u128\|i128" src/ 2>/dev/null | head -5 || echo "No direct u128/i128 usage found"

# Check the kernel PTX
if [ -f /tmp/sha2_kernels.ptx ]; then
    echo -e "\n=== Analyzing PTX for 128-bit operations ==="
    echo "Looking for functions that might be i128 operations:"
    grep -E "\.func.*128|call.*128" /tmp/sha2_kernels.ptx | head -10 || echo "No 128-bit function calls"
    
    echo -e "\nLooking for multi-precision arithmetic patterns:"
    grep -B2 -A2 "add\.cc\|addc\|sub\.cc\|subc" /tmp/sha2_kernels.ptx | head -20 || echo "No carry-chain arithmetic found"
    
    echo -e "\nChecking for 64-bit pair operations (how i128 is often implemented):"
    grep -E "\.b64.*\.b64|ld\.param\.b64.*ld\.param\.b64" /tmp/sha2_kernels.ptx | head -10 || echo "No paired 64-bit ops"
fi

echo -e "\n=== Summary ==="
echo "i128 operations in NVVM/PTX are typically lowered to:"
echo "1. Pairs of 64-bit operations with carry/borrow chains"
echo "2. Function calls to runtime library implementations"
echo "3. Direct multi-precision arithmetic instructions"
EOF