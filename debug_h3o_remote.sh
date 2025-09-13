#!/bin/bash

# Debug script to run on remote machine
echo "=== Debugging h3o compilation issue ==="

# First, let's try to find the h3o source
echo "Finding h3o source in cargo registry..."
find ~/.cargo/registry/src -name "h3o-*" -type d 2>/dev/null | head -5

# Try to compile with LLVM IR output
echo "Attempting to compile h3o with LLVM IR emission..."
cd ~/rust-cuda/examples/cuda/h3o_crates_io/kernels

# Export environment for detailed output
export RUST_BACKTRACE=full
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG=true

# Try building with emit llvm-bc
cargo rustc --release \
    --target nvptx64-nvidia-cuda \
    -- --emit=llvm-bc \
    2>&1 | tee /tmp/h3o_llvm_build.log || true

echo "Build log saved to /tmp/h3o_llvm_build.log"

# Look for any LLVM files generated
echo "Looking for generated LLVM files..."
find ~/rust-cuda/target -name "*.bc" -o -name "*.ll" 2>/dev/null | grep h3o | head -10

# Check if there's an ICE report
if ls /root/.cargo/registry/src/index.crates.io-*/h3o-*/rustc-ice-*.txt 2>/dev/null; then
    echo "Found ICE report, displaying last one:"
    ls -t /root/.cargo/registry/src/index.crates.io-*/h3o-*/rustc-ice-*.txt | head -1 | xargs tail -100
fi