#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Checking SHA512 kernel..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== Finding SHA512 oneshot kernel ==="
# Get line number where sha512_oneshot starts
START=$(grep -n "\.entry sha512_oneshot" /tmp/sha2_kernels.ptx | cut -d: -f1)
echo "SHA512 oneshot starts at line: $START"

# Get line number where next entry starts (or end of file)
NEXT=$(grep -n "\.entry sha512_incremental" /tmp/sha2_kernels.ptx | cut -d: -f1)
echo "Next entry at line: $NEXT"

# Extract the kernel
echo -e "\n=== SHA512 oneshot kernel (first 150 lines) ==="
sed -n "${START},$((NEXT-1))p" /tmp/sha2_kernels.ptx | head -150

echo -e "\n=== Checking for trap instruction ==="
sed -n "${START},$((NEXT-1))p" /tmp/sha2_kernels.ptx | grep -c "trap" || echo "No trap instructions"

echo -e "\n=== Checking for function calls ==="
sed -n "${START},$((NEXT-1))p" /tmp/sha2_kernels.ptx | grep -E "call|bra" | head -20
EOF