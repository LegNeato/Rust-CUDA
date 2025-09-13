#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
REMOTE_DIR="/root/rust-cuda"
SSH_OPTS="-o StrictHostKeyChecking=no"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Running cargo clippy on vast.ai${NC}"
echo "========================================"

# First sync the files
echo -e "\n${GREEN}Step 1: Syncing project files...${NC}"
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    --exclude '.claude/' \
    --exclude 'deploy-*.sh' \
    --exclude 'sync-to-vast.sh' \
    --exclude 'test-dis.sh' \
    --exclude 'run-clippy.sh' \
    -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    ./ $REMOTE_HOST:$REMOTE_DIR/

echo "Files synced successfully!"

# Run clippy
echo -e "\n${GREEN}Step 2: Running cargo clippy...${NC}"
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'CLIPPY_EOF'
cd /root/rust-cuda
source $HOME/.cargo/env
source /tmp/cuda_env.sh 2>/dev/null || true
export LLVM_LINK_STATIC=1

# Install clippy if not present
echo "Installing/updating clippy..."
rustup component add clippy

# Run clippy on all packages
echo -e "\n🔍 Running clippy on all packages..."
cargo clippy --all --all-targets --all-features -- -D warnings 2>&1 || true
CLIPPY_EOF

echo -e "\n${GREEN}✅ Clippy check complete!${NC}"