#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Running with CUDA debug..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda
source $HOME/.cargo/env

export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:$LD_LIBRARY_PATH"
export LLVM_LINK_STATIC=1
export RUST_LOG=debug
export RUST_BACKTRACE=full
export CUDA_LAUNCH_BLOCKING=1

# Try running with cuda-memcheck if available
if command -v cuda-memcheck &> /dev/null; then
    echo "Running with cuda-memcheck..."
    cuda-memcheck --tool memcheck target/debug/sha2_example 2>&1 | head -200
else
    echo "Running with debug output..."
    target/debug/sha2_example 2>&1 | head -200
fi
EOF