#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Checking i128 implementation in compiled code..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== Building SHA2 example with debug symbols ==="
cd examples/cuda/cpu/sha2_crates_io
cargo build --release 2>&1 | tail -5

echo -e "\n=== Checking for i128 symbols in the binary ==="
nm target/release/sha2_crates_io 2>/dev/null | grep -E "__rust.*128.*addo|__rust.*128.*subo|__rust.*128.*mulo" || echo "No __rust_*128_*o symbols found"

echo -e "\n=== Checking compiler-builtins linkage ==="
ldd target/release/sha2_crates_io | grep -E "libgcc|compiler" || echo "No external compiler-builtins found"

echo -e "\n=== Checking if Rust's compiler-builtins is linked ==="
nm target/release/sha2_crates_io 2>/dev/null | grep -E "compiler_builtins" | head -5 || echo "No compiler_builtins symbols"

echo -e "\n=== Looking for i128 operations in the binary ==="
objdump -d target/release/sha2_crates_io 2>/dev/null | grep -E "call.*__rust.*128" | head -10 || echo "No calls to __rust*128 functions found"

echo -e "\n=== Checking PTX for i128 operations ==="
cd /root/rust-cuda
if [ -f /tmp/sha2_kernels.ptx ]; then
    echo "Searching for 128-bit operations in PTX..."
    grep -E "\.b128|\.u128|\.s128" /tmp/sha2_kernels.ptx | head -10 || echo "No explicit 128-bit PTX operations found"
    
    echo -e "\nSearching for multi-word arithmetic (carry operations)..."
    grep -E "add\.cc|addc|sub\.cc|subc|mad\.lo|mad\.hi" /tmp/sha2_kernels.ptx | head -10 || echo "No carry-chain operations found"
else
    echo "PTX file not found, generating it..."
    cd examples/cuda/cpu/sha2_crates_io
    CUDA_PATH=/usr/local/cuda CARGO_TARGET_DIR=/tmp/target cargo build --release 2>&1 | tail -5
    find /tmp/target -name "*.ptx" -exec cp {} /tmp/sha2_kernels.ptx \; 2>/dev/null || echo "PTX generation failed"
fi

echo -e "\n=== Checking Rust std implementation ==="
rustc --version
echo "Rust std likely contains the i128 implementations internally"
EOF