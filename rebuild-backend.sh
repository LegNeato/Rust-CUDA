#!/bin/bash

# Script to rebuild the codegen backend on remote
echo "🔨 Rebuilding codegen backend with GEP fix..."

# Use run-example infrastructure to access remote
./run-example.sh h3o_crates_io 2>&1 | head -1

# Now SSH to rebuild
INSTANCE_ID=$(cat .vast_instance_id 2>/dev/null || echo "")
if [ -n "$INSTANCE_ID" ]; then
    SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null | grep -o "ssh://[^']*" | sed 's|ssh://||')
    SSH_USER=$(echo "$SSH_INFO" | cut -d@ -f1)
    SSH_HOST=$(echo "$SSH_INFO" | cut -d@ -f2 | cut -d: -f1)
    SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f3)
    
    echo "Rebuilding codegen backend..."
    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_USER@$SSH_HOST" << 'EOF'
cd ~/rust-cuda
echo "Building rustc_codegen_nvvm..."
cargo build --release -p rustc_codegen_nvvm
echo "Build complete. Now testing h3o example..."
cargo build --release -p h3o_crates_io 2>&1 | head -50
EOF
fi