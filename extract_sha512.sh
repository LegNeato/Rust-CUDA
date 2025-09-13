#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Extracting SHA512 kernel code..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== Extracting full SHA512 oneshot kernel ==="
awk '/\.entry sha512_oneshot/,/^\.visible \.entry|^$/{print}' /tmp/sha2_kernels.ptx | head -100 > /tmp/sha512_kernel.ptx

echo "First 100 lines of SHA512 kernel:"
cat /tmp/sha512_kernel.ptx

echo -e "\n=== Checking for trap or error instructions ==="
grep -E "trap|exit|ret" /tmp/sha512_kernel.ptx | head -10

echo -e "\n=== Checking if kernel body is complete ==="
tail -20 /tmp/sha512_kernel.ptx
EOF