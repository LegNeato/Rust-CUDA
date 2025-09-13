#!/bin/bash

# Sync to remote
rsync -avz \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    -e "ssh -o StrictHostKeyChecking=no -p 35598" \
    . root@ssh5.vast.ai:/root/rust-cuda/

# Build with PTX disassembly output
ssh -o StrictHostKeyChecking=no root@ssh5.vast.ai -p 35598 << 'EOF'
cd /root/rust-cuda
export RUST_BACKTRACE=1

# Try to build with disassembly flags to see what functions exist
cd examples/cuda/h3o_crates_io

# Clean previous builds
cargo clean

# Build with verbose output and save intermediate files
CARGO_ENCODED_RUSTFLAGS="-Zcodegen-backend=/root/rust-cuda/target/release/deps/librustc_codegen_nvvm.so -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(nvvm_internal) -Zcrate-attr=no_std -Zsaturating_float_casts=false -Cllvm-args=-arch=compute_61 -Cllvm-args=--override-libm -Cllvm-args=--disassemble --emit=llvm-ir" \
    cargo build --release 2>&1 | tee /tmp/h3o_build.log

# Also try with just emit llvm-ir to see the LLVM IR before PTX conversion
CARGO_ENCODED_RUSTFLAGS="-Zcodegen-backend=/root/rust-cuda/target/release/deps/librustc_codegen_nvvm.so -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(nvvm_internal) -Zcrate-attr=no_std -Zsaturating_float_casts=false -Cllvm-args=-arch=compute_61 -Cllvm-args=--override-libm --emit=llvm-ir" \
    cargo build --release 2>&1 | tee /tmp/h3o_build_llvm.log

echo "=== Build logs saved to /tmp/h3o_build*.log ==="
EOF