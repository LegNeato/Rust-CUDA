#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "Checking for trap instructions in PTX..."

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda

echo "=== Trap instructions in entire PTX ==="
grep -c "trap" /tmp/sha2_kernels.ptx || echo "No trap instructions found"

echo -e "\n=== Trap instructions per kernel ==="
for kernel in sha256_oneshot sha256_incremental sha512_oneshot sha512_incremental; do
    START=$(grep -n "\.entry $kernel" /tmp/sha2_kernels.ptx | cut -d: -f1)
    if [ -n "$START" ]; then
        echo -n "$kernel: "
        # Find next entry or end of file
        NEXT=$(grep -n "\.entry" /tmp/sha2_kernels.ptx | grep -A1 "\.entry $kernel" | tail -1 | cut -d: -f1)
        if [ "$NEXT" == "$START" ]; then
            # This is the last kernel
            sed -n "${START},\$p" /tmp/sha2_kernels.ptx | grep -c "trap" || echo "0"
        else
            sed -n "${START},$((NEXT-1))p" /tmp/sha2_kernels.ptx | grep -c "trap" || echo "0"
        fi
    fi
done

echo -e "\n=== Showing trap context in SHA512 oneshot ==="
START=$(grep -n "\.entry sha512_oneshot" /tmp/sha2_kernels.ptx | cut -d: -f1)
NEXT=$(grep -n "\.entry sha512_incremental" /tmp/sha2_kernels.ptx | cut -d: -f1)
sed -n "${START},$((NEXT-1))p" /tmp/sha2_kernels.ptx | grep -B2 -A2 "trap" | head -20
EOF