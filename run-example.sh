#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh4.vast.ai"
REMOTE_PORT="31305"
REMOTE_DIR="/root/rust-cuda"
SSH_OPTS="-o StrictHostKeyChecking=no"

EXAMPLE="${1:-vecadd}"

echo "🚀 Running example: $EXAMPLE"

# Sync files
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    --exclude '.claude/' \
    --exclude '*.sh' \
    -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    ./ $REMOTE_HOST:$REMOTE_DIR/

# Run the example
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << EOF
cd /root/rust-cuda
source \$HOME/.cargo/env

# Set up CUDA paths
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:\$LD_LIBRARY_PATH"
export LLVM_LINK_STATIC=1
export RUST_LOG=info
export RUST_BACKTRACE=1

echo "Building example $EXAMPLE (RELEASE mode to avoid overflow checks)..."
cargo build --release -p $EXAMPLE 2>&1

echo -e "\n==========================="
echo "Running example $EXAMPLE..."
echo "==========================="
cargo run --release -p $EXAMPLE 2>&1
EOF