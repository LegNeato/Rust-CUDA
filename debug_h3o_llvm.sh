#!/bin/bash

echo "=== Capturing h3o LLVM IR ==="

ssh -o StrictHostKeyChecking=no -p 10839 root@ssh1.vast.ai << 'EOF'
cd ~/rust-cuda

# Clean h3o build
cargo clean -p h3o

# Try to build h3o and save LLVM IR
export RUST_LOG=rustc_codegen_nvvm=debug
export RUSTFLAGS="--emit=llvm-ir"

echo "Building h3o with LLVM IR emission..."
cd examples/cuda/h3o_crates_io/kernels

# Build with verbose output
cargo rustc --release \
    --target nvptx64-nvidia-cuda \
    -- --emit=llvm-bc,llvm-ir \
    2>&1 | tee /tmp/h3o_build_verbose.log || true

echo -e "\n=== Looking for generated LLVM files ==="
find /root/rust-cuda/target -name "*h3o*.ll" -o -name "*h3o*.bc" 2>/dev/null | head -10

echo -e "\n=== Checking h3o source for const/static data ==="
H3O_SRC=$(find ~/.cargo/registry/src -name "h3o-0.8.0" -type d | head -1)
if [ -n "$H3O_SRC" ]; then
    echo "Found h3o source at: $H3O_SRC"
    
    echo -e "\n=== Static/const declarations in h3o ==="
    grep -r "static\|const " "$H3O_SRC/src" | grep -E "^\s*(pub\s+)?(static|const)" | head -20
    
    echo -e "\n=== Const fn declarations in h3o ==="
    grep -r "const fn" "$H3O_SRC/src" | head -20
fi

echo -e "\n=== Error details from build log ==="
grep -A5 -B5 "Explicit gep type" /tmp/h3o_build_verbose.log | head -30 || echo "No error found in log"
EOF