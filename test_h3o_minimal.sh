#!/bin/bash

# Create a minimal test case that uses h3o without CUDA to understand what it does

cat > /tmp/test_h3o.rs << 'EOF'
#![no_std]
#![no_main]

use h3o::Resolution;

#[no_mangle]
pub extern "C" fn test_resolution() -> u8 {
    let res = Resolution::Zero;
    res as u8
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
EOF

echo "Compiling minimal h3o test to LLVM IR..."
cd /tmp

rustc +nightly \
    --target nvptx64-nvidia-cuda \
    --emit=llvm-ir \
    -C opt-level=0 \
    -C debuginfo=0 \
    -L dependency=/root/.cargo/registry \
    test_h3o.rs \
    2>&1 | tee h3o_compile.log

if [ -f test_h3o.ll ]; then
    echo "LLVM IR generated. Checking for GEP instructions..."
    grep -n "getelementptr" test_h3o.ll | head -20
fi