#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
REMOTE_DIR="/root/rust-cuda"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "🔧 Updating libintrinsics.bc with LLVM 7..."

# First sync the updated files
echo "📤 Syncing files to remote..."
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    --exclude '.claude/' \
    --exclude '*.sh' \
    -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    ./ $REMOTE_HOST:$REMOTE_DIR/

# Compile libintrinsics.ll to .bc on remote
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /root/rust-cuda/crates/rustc_codegen_nvvm

echo "🔍 Checking LLVM version..."
llvm-as --version || echo "llvm-as not found in PATH"

# Try to find LLVM 7 specifically
if command -v llvm-as-7 &> /dev/null; then
    echo "✅ Found llvm-as-7, using it..."
    LLVM_AS=llvm-as-7
elif command -v llvm-as &> /dev/null; then
    echo "⚠️  Using default llvm-as (may not be version 7)..."
    LLVM_AS=llvm-as
else
    echo "❌ No llvm-as found! Trying to install LLVM 7..."
    apt-get update && apt-get install -y llvm-7 llvm-7-dev
    LLVM_AS=llvm-as-7
fi

echo "📦 Compiling libintrinsics.ll to libintrinsics.bc..."
$LLVM_AS libintrinsics.ll -o libintrinsics.bc

if [ -f libintrinsics.bc ]; then
    echo "✅ Successfully created libintrinsics.bc"
    ls -la libintrinsics.bc
else
    echo "❌ Failed to create libintrinsics.bc"
    exit 1
fi

echo "🔍 Verifying the new intrinsics are present..."
llvm-dis libintrinsics.bc -o - 2>/dev/null | grep -E "__nvvm_(div|udiv|mod|umod|multi|ashl|ashr|lshr)ti3" | head -5 || echo "Could not verify intrinsics"

echo "✅ Done!"
EOF

# Copy the compiled .bc file back to local
echo "📥 Copying libintrinsics.bc back to local..."
scp -P $REMOTE_PORT $SSH_OPTS $REMOTE_HOST:$REMOTE_DIR/crates/rustc_codegen_nvvm/libintrinsics.bc \
    ./crates/rustc_codegen_nvvm/libintrinsics.bc

echo "✅ libintrinsics.bc has been updated!"