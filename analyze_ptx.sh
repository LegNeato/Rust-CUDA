#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Analyzing PTX differences between SHA256 and SHA512..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== SHA256 oneshot function header ==="
grep -A 20 "\.entry sha256_oneshot" /tmp/sha2_kernels.ptx | head -25

echo -e "\n=== SHA512 oneshot function header ==="
grep -A 20 "\.entry sha512_oneshot" /tmp/sha2_kernels.ptx | head -25

echo -e "\n=== Checking for stack/local memory usage ==="
echo "SHA256 local memory:"
grep -A 5 "\.entry sha256_oneshot" /tmp/sha2_kernels.ptx | grep -E "\.local|\.shared|\.reg" || echo "No local/shared memory declarations"

echo -e "\nSHA512 local memory:"
grep -A 5 "\.entry sha512_oneshot" /tmp/sha2_kernels.ptx | grep -E "\.local|\.shared|\.reg" || echo "No local/shared memory declarations"

echo -e "\n=== Checking for illegal instructions in SHA512 ==="
# Look for any CUDA version specific features
grep -A 50 "\.entry sha512_oneshot" /tmp/sha2_kernels.ptx | grep -E "\.target|\.address_size|\.version" | head -10

echo -e "\n=== Checking parameter declarations ==="
echo "SHA256 params:"
grep -A 10 "\.entry sha256_oneshot" /tmp/sha2_kernels.ptx | grep "\.param"

echo -e "\nSHA512 params:"
grep -A 10 "\.entry sha512_oneshot" /tmp/sha2_kernels.ptx | grep "\.param"
EOF