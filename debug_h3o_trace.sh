#!/bin/bash

echo "=== Building h3o with trace logging enabled ==="

# Run on remote with RUST_LOG enabled for tracing
ssh -o StrictHostKeyChecking=no root@ssh5.vast.ai -p 42034 << 'EOF'
cd /root/rust-cuda/examples/cuda/h3o_crates_io

# Clean previous builds
cargo clean

# Build with trace logging to capture GEP instructions
export RUST_LOG=rustc_codegen_nvvm=trace
export RUST_BACKTRACE=full

echo "Building with trace logging..."
cargo build --release 2>&1 | grep -E "(gep:|GEP|getelementptr)" | head -100 > /tmp/gep_trace.log

echo "=== GEP instructions logged ==="
cat /tmp/gep_trace.log | head -50

echo -e "\n=== Looking for the problematic GEP ==="
# The error mentions type mismatch, so look for any warnings or unusual patterns
cargo build --release 2>&1 | grep -B5 -A5 "Explicit gep type" | head -50
EOF