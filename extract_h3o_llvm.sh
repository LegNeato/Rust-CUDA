#!/bin/bash

# Sync and extract LLVM IR even if build fails
rsync -avz \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    -e "ssh -o StrictHostKeyChecking=no -p 10839" \
    . root@ssh1.vast.ai:/root/rust-cuda/

ssh -o StrictHostKeyChecking=no root@ssh1.vast.ai -p 10839 << 'EOF'
cd /root/rust-cuda
export RUST_BACKTRACE=1

# Clean and try to build
cd examples/cuda/h3o_crates_io
cargo clean

# Build and ignore errors
cargo build --release 2>&1 || true

# Find and display any LLVM IR files
echo "=== Looking for LLVM IR files ==="
find /root/rust-cuda/target -name "*.ll" -type f 2>/dev/null | while read f; do
    echo "Found: $f"
    echo "Size: $(wc -l "$f")"
done

# Check for final_module specifically
if [ -f /root/rust-cuda/target/release/build/*/out/final_module.ll ]; then
    echo "=== Found final_module.ll ==="
    ls -la /root/rust-cuda/target/release/build/*/out/final_module.ll
    echo "First 100 lines:"
    head -100 /root/rust-cuda/target/release/build/*/out/final_module.ll
fi

# Also check the kernel directory
echo "=== Checking kernel build directory ==="
find /root/rust-cuda/target/cuda-builder -name "*.ll" -type f 2>/dev/null | while read f; do
    echo "Found: $f"
done
EOF