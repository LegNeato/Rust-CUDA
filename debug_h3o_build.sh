#!/bin/bash

# Debug script to extract LLVM IR from h3o compilation
set -e

echo "Attempting to build h3o with LLVM IR output..."

cd examples/cuda/h3o_crates_io/kernels

# Try to build with verbose output and emit LLVM IR
RUST_BACKTRACE=full \
CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG=true \
cargo rustc \
    --release \
    --target nvptx64-nvidia-cuda \
    -Z build-std=core,alloc \
    -Z build-std-features=compiler-builtins-mem \
    -- \
    --emit=llvm-ir \
    -C llvm-args=-arch=compute_61 \
    -Z codegen-backend=$(pwd)/../../../../target/release/deps/librustc_codegen_nvvm.so \
    2>&1 | tee /tmp/h3o_build.log || true

echo "Build log saved to /tmp/h3o_build.log"

# Look for any generated LLVM IR files
find target -name "*.ll" -type f 2>/dev/null | head -10