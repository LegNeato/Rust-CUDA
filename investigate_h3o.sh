#!/bin/bash

echo "=== Investigating h3o compilation issue ==="

# Find h3o in cargo registry on remote
echo "Looking for h3o source..."
ssh -o StrictHostKeyChecking=no root@ssh5.vast.ai -p 42034 << 'EOF'
cd /root/.cargo/registry/src
H3O_DIR=$(find . -name "h3o-0.8.0" -type d | head -1)
if [ -z "$H3O_DIR" ]; then
    echo "h3o source not found, downloading..."
    cd /tmp
    curl -L https://crates.io/api/v1/crates/h3o/0.8.0/download | tar xz
    H3O_DIR="/tmp/h3o-0.8.0"
fi

echo "Found h3o at: $H3O_DIR"
cd "$H3O_DIR"

# Look for potentially problematic patterns
echo -e "\n=== Checking for unsafe code ==="
grep -r "unsafe" src/ | head -10

echo -e "\n=== Checking for const fn with complex operations ==="
grep -r "const fn" src/ | head -10

echo -e "\n=== Checking for static arrays or lookup tables ==="
grep -r "static\|const.*\[" src/ | head -10

echo -e "\n=== Looking at lib.rs structure ==="
head -50 src/lib.rs
EOF