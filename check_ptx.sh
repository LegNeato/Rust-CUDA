#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Checking PTX file..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== PTX Functions ==="
grep -E "\.entry|\.func" /tmp/sha2_kernels.ptx | head -20

echo -e "\n=== Checking for SHA512 functions ==="
grep -c "sha512" /tmp/sha2_kernels.ptx || echo "No sha512 functions found"

echo -e "\n=== Checking for SHA256 functions ==="
grep -c "sha256" /tmp/sha2_kernels.ptx || echo "No sha256 functions found"

echo -e "\n=== Entry points ==="
grep "\.entry" /tmp/sha2_kernels.ptx

echo -e "\n=== PTX file size ==="
ls -lh /tmp/sha2_kernels.ptx
EOF