#!/bin/bash

# Modified run-example script with debug tracing
set -e

EXAMPLE="h3o_crates_io"

echo "🚀 Running example with debug tracing: $EXAMPLE"

# Get instance details
INSTANCE_ID=$(cat .vast_instance_id 2>/dev/null || echo "")
if [ -z "$INSTANCE_ID" ]; then
    echo "Error: No instance ID found. Please run deploy-cuda.sh first."
    exit 1
fi

# Get SSH details
SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null | grep -o "ssh://[^']*" | sed 's|ssh://||')
if [ -z "$SSH_INFO" ]; then
    echo "Error: Could not get SSH connection info"
    exit 1
fi

SSH_USER=$(echo "$SSH_INFO" | cut -d@ -f1)
SSH_HOST=$(echo "$SSH_INFO" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f3)

# Sync files
rsync -avz --exclude target --exclude .git \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    ./ "$SSH_USER@$SSH_HOST:~/rust-cuda/"

# Build and run with debug tracing
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_USER@$SSH_HOST" << EOF
cd ~/rust-cuda

# Clean and build with tracing
cargo clean -p h3o_crates_io_kernels
export RUST_LOG=rustc_codegen_nvvm::builder=trace
export RUST_BACKTRACE=1

echo "Building with GEP tracing enabled..."
cargo build --release --package h3o_crates_io 2>&1 | tee /tmp/h3o_build_trace.log

# Extract relevant GEP traces
echo -e "\n=== GEP Instructions ==="
grep "gep:" /tmp/h3o_build_trace.log | tail -20

echo -e "\n=== Error context ==="
grep -B10 -A10 "Explicit gep type" /tmp/h3o_build_trace.log | head -30
EOF