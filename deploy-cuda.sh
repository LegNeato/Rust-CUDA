#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh7.vast.ai"
REMOTE_PORT="20213"
REMOTE_DIR="/root/rust-cuda"
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

# Step 1: Create remote directory
echo -e "\n${GREEN}Step 1: Creating remote directory...${NC}"
remote_exec "mkdir -p $REMOTE_DIR"

# Step 2: Sync files using rsync
echo -e "\n${GREEN}Step 2: Syncing project files...${NC}"
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    --exclude '.claude/' \
    --exclude 'deploy-*.sh' \
    --exclude 'sync-to-vast.sh' \
    --exclude 'test-dis.sh' \
    -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    ./ $REMOTE_HOST:$REMOTE_DIR/

echo "Files synced successfully!"

# Step 3: Set up environment and dependencies
echo -e "\n${GREEN}Step 3: Setting up environment...${NC}"

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'SETUP_EOF'
set -e

# Clear MOTD
echo "" > /etc/motd

cd /root/rust-cuda

# Check GPU and CUDA
echo -e "\n🎮 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo -e "\n🚀 CUDA Version:"
nvcc --version 2>/dev/null || echo "NVCC not found"

# Find CUDA installation directory
CUDA_DIR=""
if [ -d "/usr/local/cuda" ]; then
    CUDA_DIR="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.8" ]; then
    CUDA_DIR="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_DIR="/usr/local/cuda-12.6"
else
    echo "Searching for CUDA installation..."
    CUDA_DIR=$(find /usr/local -maxdepth 1 -name "cuda*" -type d | head -1)
fi

if [ -z "$CUDA_DIR" ]; then
    echo "ERROR: Could not find CUDA installation!"
    exit 1
fi

echo "Using CUDA installation at: $CUDA_DIR"

# Find libnvvm.so and create symlink if needed
echo -e "\n📦 Setting up NVVM libraries..."
NVVM_LIB_DIR="$CUDA_DIR/nvvm/lib64"
if [ ! -d "$NVVM_LIB_DIR" ]; then
    echo "ERROR: NVVM lib directory not found at $NVVM_LIB_DIR"
    exit 1
fi

# Check for libnvvm.so files
echo "NVVM libraries found:"
ls -la $NVVM_LIB_DIR/libnvvm.so* 2>/dev/null || echo "No libnvvm.so files found"

# Create symlink for libnvvm.so.4 if it doesn't exist
if [ ! -f "$NVVM_LIB_DIR/libnvvm.so.4" ]; then
    if [ -f "$NVVM_LIB_DIR/libnvvm.so" ]; then
        echo "Creating symlink: libnvvm.so.4 -> libnvvm.so"
        ln -sf $NVVM_LIB_DIR/libnvvm.so $NVVM_LIB_DIR/libnvvm.so.4
    else
        echo "WARNING: libnvvm.so not found, build may fail"
    fi
fi

# Update LD_LIBRARY_PATH in bashrc
echo "export LD_LIBRARY_PATH=\"$NVVM_LIB_DIR:\$LD_LIBRARY_PATH\"" > /tmp/cuda_env.sh

# Install system dependencies
echo -e "\n📦 Installing system dependencies..."
apt-get update > /dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    clang \
    curl \
    libssl-dev \
    libtinfo-dev \
    pkg-config \
    xz-utils \
    zlib1g-dev \
    libffi-dev \
    libedit-dev \
    libncurses5-dev \
    libxml2-dev \
    python3 \
    ninja-build \
    cmake > /dev/null 2>&1

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo -e "\n🦀 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# Source cargo env
source $HOME/.cargo/env

# Show Rust version
echo -e "\n🦀 Rust version:"
cargo --version
rustc --version

# Check for LLVM 7
if ! command -v llvm-config-7 &> /dev/null && ! command -v llvm-config &> /dev/null; then
    echo -e "\n⚠️  LLVM 7 not found. Installing..."
    
    mkdir -p /tmp/llvm7
    cd /tmp/llvm7
    
    echo "Downloading LLVM 7.1.0..."
    curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz
    tar -xf llvm-7.1.0.src.tar.xz
    cd llvm-7.1.0.src
    mkdir build && cd build
    
    echo "Building LLVM 7.1.0 (this will take a while)..."
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
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
        ..
    
    ninja -j$(nproc)
    ninja install
    
    # Create symlink
    ln -sf /usr/bin/llvm-config /usr/bin/llvm-config-7
    
    # Cleanup
    cd /
    rm -rf /tmp/llvm7
    
    echo "LLVM 7 installed successfully!"
fi

echo -e "\n✅ Environment setup complete!"
echo "CUDA directory: $CUDA_DIR"
echo "NVVM library path: $NVVM_LIB_DIR"
SETUP_EOF

# Step 4: Build and run based on arguments
case "${1:-build}" in
    "build")
        echo -e "\n${GREEN}Step 4: Building project...${NC}"
        ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'BUILD_EOF'
cd /root/rust-cuda
source $HOME/.cargo/env
source /tmp/cuda_env.sh
export LLVM_LINK_STATIC=1
export RUST_LOG=info

echo "Building rustc_codegen_nvvm..."
cargo build -p rustc_codegen_nvvm

echo -e "\nRunning tests..."
cargo test -p rustc_codegen_nvvm
BUILD_EOF
        ;;
    
    "example")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Example name required${NC}"
            echo "Usage: $0 example <example_name>"
            exit 1
        fi
        
        echo -e "\n${GREEN}Step 4: Running example $2...${NC}"
        ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << EOF
cd /root/rust-cuda
source \$HOME/.cargo/env
source /tmp/cuda_env.sh
export LLVM_LINK_STATIC=1
export RUST_LOG=info
export RUST_BACKTRACE=1

echo "Building example $2..."
if cargo build -p $2 2>&1; then
    echo -e "\nRunning example $2..."
    cargo run -p $2 2>&1
else
    echo "Build failed. Checking environment..."
    echo "LD_LIBRARY_PATH: \$LD_LIBRARY_PATH"
    echo -e "\nSearching for libnvvm.so:"
    find /usr/local -name "libnvvm.so*" 2>/dev/null
    echo -e "\nLLVM config:"
    llvm-config --version 2>/dev/null || echo "llvm-config not found"
fi
EOF
        ;;
    
    "all-examples")
        echo -e "\n${GREEN}Step 4: Running all examples...${NC}"
        ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'ALL_EXAMPLES_EOF'
cd /root/rust-cuda
source $HOME/.cargo/env
source /tmp/cuda_env.sh
export LLVM_LINK_STATIC=1
export RUST_LOG=info

for example_dir in examples/cuda/*/; do
    if [ -f "$example_dir/Cargo.toml" ]; then
        example=$(grep "^name = " "$example_dir/Cargo.toml" | head -1 | cut -d'"' -f2)
        if [ -n "$example" ] && [ "$example" != "*-kernels" ]; then
            echo -e "\n========================================="
            echo "Building and running $example..."
            echo "========================================="
            if cargo build -p $example 2>&1; then
                cargo run -p $example 2>&1
            else
                echo "Failed to build $example"
            fi
        fi
    fi
done
ALL_EXAMPLES_EOF
        ;;
    
    "test")
        echo -e "\n${GREEN}Step 4: Running compiletests...${NC}"
        ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'TEST_EOF'
cd /root/rust-cuda
source $HOME/.cargo/env
source /tmp/cuda_env.sh
export LLVM_LINK_STATIC=1
export RUST_LOG=info
export RUST_BACKTRACE=1

echo "Running compiletests..."
TARGET_ARCHS=$(grep -oP '(?<=--target-arch )[^"]+' .github/workflows/ci_linux.yml | head -1 || echo "compute_70")
echo "Using target architectures: $TARGET_ARCHS"
cargo run -p compiletests --release --no-default-features -- --target-arch "$TARGET_ARCHS"
TEST_EOF
        ;;
    
    "interactive")
        echo -e "\n${GREEN}Opening interactive SSH session...${NC}"
        ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST -t "cd /root/rust-cuda && source /tmp/cuda_env.sh 2>/dev/null; bash"
        exit 0
        ;;
    
    *)
        echo -e "${RED}Unknown command: ${1}${NC}"
        echo "Usage: $0 [build|example <name>|all-examples|test|interactive]"
        echo "Examples:"
        echo "  $0 build              # Build the project"
        echo "  $0 example sha2_example  # Run specific example"
        echo "  $0 all-examples       # Run all examples"
        echo "  $0 test               # Run compiletests"
        echo "  $0 interactive        # Open SSH session"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✅ Deployment complete!${NC}"
