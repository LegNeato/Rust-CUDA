#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Testing i128 overflow operations..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== Creating test program with i128 overflow checks ==="
mkdir -p /tmp/test_i128
cat > /tmp/test_i128/main.rs << 'RUSTEOF'
#![no_std]
#![no_main]
#![feature(abi_ptx)]

use core::panic::PanicInfo;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_i128_overflow(
    results: *mut u8,
    a: i128,
    b: i128,
    c: u128,
    d: u128,
) {
    // Test signed overflow operations
    let (add_res, add_overflow) = a.overflowing_add(b);
    let (sub_res, sub_overflow) = a.overflowing_sub(b);
    let (mul_res, mul_overflow) = a.overflowing_mul(b);
    
    // Test unsigned overflow operations  
    let (uadd_res, uadd_overflow) = c.overflowing_add(d);
    let (usub_res, usub_overflow) = c.overflowing_sub(d);
    let (umul_res, umul_overflow) = c.overflowing_mul(d);
    
    // Store overflow flags
    *results.offset(0) = add_overflow as u8;
    *results.offset(1) = sub_overflow as u8;
    *results.offset(2) = mul_overflow as u8;
    *results.offset(3) = uadd_overflow as u8;
    *results.offset(4) = usub_overflow as u8;
    *results.offset(5) = umul_overflow as u8;
    
    // Store some results to prevent optimization
    *results.offset(6) = (add_res & 0xFF) as u8;
    *results.offset(7) = (uadd_res & 0xFF) as u8;
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}
RUSTEOF

echo -e "\n=== Building test with rustc_codegen_nvvm ==="
cd /root/rust-cuda
export CUDA_PATH=/usr/local/cuda

# Build the kernel
rustc +nightly \
    --target nvptx64-nvidia-cuda \
    -C llvm-args=--nvvm-arch=sm_75 \
    -C target-cpu=sm_75 \
    -C panic=abort \
    -Z codegen-backend=/root/rust-cuda/target/release/librustc_codegen_nvvm.so \
    --crate-type cdylib \
    -O \
    -o /tmp/test_i128/kernel.ptx \
    /tmp/test_i128/main.rs 2>&1 | tail -20 || true

echo -e "\n=== Checking generated PTX for overflow functions ==="
if [ -f /tmp/test_i128/kernel.ptx ]; then
    echo "Searching for i128 overflow function calls:"
    grep -E "call.*__rust.*128.*o|call.*__nvvm.*128.*o" /tmp/test_i128/kernel.ptx | head -10 || echo "No overflow function calls found"
    
    echo -e "\nSearching for overflow detection patterns:"
    grep -E "setp.*overflow|bra.*overflow" /tmp/test_i128/kernel.ptx | head -10 || echo "No overflow detection patterns"
else
    echo "PTX generation failed, checking for errors..."
    
    # Try with debug output
    echo -e "\n=== Retrying with verbose output ==="
    RUST_BACKTRACE=1 rustc +nightly \
        --target nvptx64-nvidia-cuda \
        -C llvm-args=--nvvm-arch=sm_75 \
        -C target-cpu=sm_75 \
        -C panic=abort \
        -Z codegen-backend=/root/rust-cuda/target/release/librustc_codegen_nvvm.so \
        --crate-type cdylib \
        -O \
        --emit=llvm-ir \
        -o /tmp/test_i128/kernel.ll \
        /tmp/test_i128/main.rs 2>&1 | tail -30 || true
    
    if [ -f /tmp/test_i128/kernel.ll ]; then
        echo -e "\n=== LLVM IR generated, checking for i128 functions ==="
        grep -E "__rust.*128|__nvvm.*128" /tmp/test_i128/kernel.ll | head -20 || echo "No i128 functions in LLVM IR"
    fi
fi

echo -e "\n=== Checking if compiler-builtins is available ==="
find /root/rust-cuda -name "*compiler*builtin*" -type f 2>/dev/null | head -5 || echo "No compiler-builtins files found"

# Check in the Rust toolchain
ls -la ~/.rustup/toolchains/nightly-*/lib/rustlib/nvptx64-nvidia-cuda/lib/*compiler* 2>/dev/null || echo "No compiler-builtins in rustlib"
EOF