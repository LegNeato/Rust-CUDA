#!/bin/bash
set -e

# Script to sync Rust-CUDA changes to vast.ai machine and run builds/tests

# Configuration
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
REMOTE_DIR="/root/Rust-CUDA"
SSH_OPTS="-o StrictHostKeyChecking=no"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Deploying Rust-CUDA to vast.ai${NC}"
echo "========================================"

# Function to run commands on remote machine
remote_exec() {
    ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST "$1"
}

# Run cargo command with proper environment
cargo_cmd() {
    local cmd="$1"
    local packages="${2:-cust -p rustc_codegen_nvvm -p nvvm -p cust_raw -p cust_core}"
    remote_exec "export PATH=\"\$HOME/.cargo/bin:\$PATH\" && export LD_LIBRARY_PATH='/usr/local/cuda/nvvm/lib64:\$LD_LIBRARY_PATH' && export LLVM_LINK_STATIC=1 && cd $REMOTE_DIR && cargo $cmd -p $packages"
}

# Step 1: Create remote directory
echo -e "\n${GREEN}Step 1: Creating remote directory...${NC}"
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Step 2: Sync files using rsync
echo -e "\n${GREEN}Step 2: Syncing project files...${NC}"
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    --exclude '.claude/' \
    --exclude 'sync-to-vast.sh' \
    -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    ./ $REMOTE_HOST:$REMOTE_DIR/

echo "Files synced successfully!"

# Step 3: Set up environment
echo -e "\n${GREEN}Step 3: Setting up on vast.ai machine...${NC}"

# Install dependencies if not already installed
echo -e "\n${GREEN}Installing system dependencies...${NC}"
echo "Updating package lists..."
remote_exec "sudo apt-get update"

echo "Installing build essentials..."
remote_exec "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq -y install build-essential pkg-config libssl-dev"

echo "Installing additional dependencies..."
remote_exec "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq -y install clang curl libtinfo-dev xz-utils zlib1g-dev"

echo "Installing GUI dependencies..."
remote_exec "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq -y install cmake libfontconfig-dev libx11-xcb-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev"

echo "Installing LLVM build dependencies..."
remote_exec "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq -y install libffi-dev libedit-dev libncurses5-dev libxml2-dev python3 ninja-build"

# Verify critical tools are available
remote_exec "which pkg-config || (echo 'ERROR: pkg-config not found after installation' && exit 1)"
remote_exec "pkg-config --version && echo 'pkg-config is working'"
remote_exec "pkg-config --modversion openssl || echo 'OpenSSL not detected by pkg-config'"

# Install LLVM 7.1 if not already installed
echo -e "\n${GREEN}Checking for LLVM 7.1...${NC}"
remote_exec "if ! command -v llvm-config-7 &> /dev/null; then \
    echo 'Installing LLVM 7.1...'; \
    cd /tmp && \
    curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz && \
    tar -xf llvm-7.1.0.src.tar.xz && \
    cd llvm-7.1.0.src && \
    mkdir build && cd build && \
    ARCH=\$(dpkg --print-architecture) && \
    if [ \"\$ARCH\" = \"amd64\" ]; then \
        TARGETS=\"X86;NVPTX\"; \
    else \
        TARGETS=\"AArch64;NVPTX\"; \
    fi && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD=\"\$TARGETS\" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_ENABLE_ZLIB=ON \
        -DLLVM_ENABLE_TERMINFO=ON \
        -DCMAKE_INSTALL_PREFIX=/usr \
        .. && \
    sudo ninja -j\$(nproc) && \
    sudo ninja install && \
    sudo ln -s /usr/bin/llvm-config /usr/bin/llvm-config-7 && \
    cd / && \
    rm -rf /tmp/llvm-7.1.0.src*; \
else \
    echo 'LLVM 7.1 already installed'; \
fi"

# Install rustup if not already installed
echo -e "\n${GREEN}Installing Rust toolchain...${NC}"
remote_exec "if ! command -v rustup &> /dev/null; then echo 'Installing rustup...'; curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; fi"

# Source cargo env and install toolchain
remote_exec "export PATH=\"\$HOME/.cargo/bin:\$PATH\" && cd $REMOTE_DIR && cargo version || true"

# Step 4: Run build commands
echo -e "\n${GREEN}Step 4: Running build commands...${NC}"

case "${1:-all}" in
    "build")
        echo "Running cargo build..."
        cargo_cmd "build"
        ;;
    "check")
        echo "Running cargo check..."
        cargo_cmd "check"
        ;;
    "test")
        echo "Running cargo test..."
        cargo_cmd "test"
        ;;
    "examples")
        echo "Testing CUDA examples..."
        remote_exec "export PATH=\"\$HOME/.cargo/bin:\$PATH\" && export LD_LIBRARY_PATH='/usr/local/cuda/nvvm/lib64:\$LD_LIBRARY_PATH' && export LLVM_LINK_STATIC=1 && cd $REMOTE_DIR && echo 'Testing vecadd example...' && cd examples/cuda/vecadd && cargo build --release && echo 'Build successful! Running vecadd...' && cargo run --release && echo -e '\n\nTesting gemm example...' && cd ../gemm && cargo build --release && echo 'Build successful! Running gemm...' && cargo run --release && echo -e '\n\nAll examples tested successfully!'"
        ;;
    "clippy")
        echo "Running cargo clippy..."
        cargo_cmd "clippy --workspace --exclude \"optix*\" --exclude \"path-tracer\" --exclude \"denoiser\" --exclude \"ex*\" --exclude \"cudnn*\" -- -D warnings"
        ;;
    "clean-clippy")
        echo "Cleaning and running cargo clippy..."
        remote_exec "export PATH=\"\$HOME/.cargo/bin:\$PATH\" && export LD_LIBRARY_PATH='/usr/local/cuda/nvvm/lib64:\$LD_LIBRARY_PATH' && export LLVM_LINK_STATIC=1 && cd $REMOTE_DIR && cargo clean && cargo clippy --workspace --exclude \"optix*\" --exclude \"path-tracer\" --exclude \"denoiser\" --exclude \"ex*\" --exclude \"cudnn*\" -- -D warnings"
        ;;
    "all")
        echo "Running all commands..."
        remote_exec "export PATH=\"\$HOME/.cargo/bin:\$PATH\" && export LD_LIBRARY_PATH='/usr/local/cuda/nvvm/lib64:\$LD_LIBRARY_PATH' && export LLVM_LINK_STATIC=1 && cd $REMOTE_DIR && echo 'Running cargo check...' && cargo check -p cust -p rustc_codegen_nvvm -p nvvm -p cust_raw -p cust_core && echo -e '\nRunning cargo build...' && cargo build -p cust -p rustc_codegen_nvvm -p nvvm -p cust_raw -p cust_core && echo -e '\nRunning cargo test...' && cargo test -p cust -p rustc_codegen_nvvm -p nvvm -p cust_raw -p cust_core"
        ;;
    "interactive")
        echo "Opening interactive SSH session..."
        ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST -L 8080:localhost:8080 -t "cd $REMOTE_DIR && bash"
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown command: ${1}${NC}"
        echo "Usage: $0 [build|check|test|examples|clippy|clean-clippy|all|interactive]"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✅ Done!${NC}"
echo ""
echo "Useful commands:"
echo "  - SSH: ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST -L 8080:localhost:8080"
echo "  - Interactive session: $0 interactive"