#!/bin/bash

PORT=15617
HOST=ssh5.vast.ai
USER=root

pushd .;
cd ~/src/myvanity;
cargo update && git add Cargo.lock && git commit -m "update lockfile" && git push;
popd;

# generate key
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if [[ ! -f ~/.ssh/id_rsa ]]
then
  ssh-keygen -t rsa -b 3072 -f ~/.ssh/id_rsa -N ""
  chmod 600 ~/.ssh/id_rsa
fi
EOF

# copy it
scp -P $PORT $USER@$HOST:~/.ssh/id_rsa.pub /tmp

# add it to github
gh ssh-key add /tmp/id_rsa.pub

# install dependencies
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
if ! which pkg-config >/dev/null 2>&1; then
  apt update
  apt install -y pkg-config libssl-dev zlib1g-dev clang
fi

# llvm-7
if [[ ! -f /usr/bin/llvm-config-7 ]]
then
  curl -sSf -L -O http://security.ubuntu.com/ubuntu/pool/universe/libf/libffi7/libffi7_3.3-5ubuntu1_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7_7.0.1-12_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-dev_7.0.1-12_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/libllvm7_7.0.1-12_amd64.deb && \
  curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-runtime_7.0.1-12_amd64.deb && \
  apt-get install -y ./*.deb && \
  ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config && \
  rm ./*.deb
fi

# rust
if [[ ! -f ~/.cargo/env ]]
then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-03-02
fi
EOF

# build
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
# clone
if [[ ! -d ed25519-vanity-rs ]]
then
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:LegNeato/ed25519-vanity-rs.git
fi

# checkout
pushd ed25519-vanity-rs
git fetch
git checkout --force master
git reset --hard origin/master

# build
. $HOME/.cargo/env
cargo clean
RUSTFLAGS="--emit=llvm-ir" NVVM_LOG=trace NVVM_DUMP_BITCODE=1 cargo build --release 2>&1 | tee build.log

# Check if module 3 files were created
echo "=== Checking for debug files ==="
ls -la /tmp/nvvm_module_*.bc 2>/dev/null || echo "No nvvm_module files found"

# If module 3 exists, analyze it
if [[ -f /tmp/nvvm_module_3.bc ]]; then
  echo "=== Analyzing module 3 with llvm-bcanalyzer ==="
  llvm-bcanalyzer-7 /tmp/nvvm_module_3.bc 2>&1 | head -50
  
  echo "=== Trying to disassemble module 3 ==="
  llvm-dis-7 /tmp/nvvm_module_3.bc -o /tmp/nvvm_module_3.bc.ll 2>&1
  
  echo "=== Trying opt to print module 3 ==="
  opt-7 -S /tmp/nvvm_module_3.bc -o /tmp/nvvm_module_3_opt.ll 2>&1
  
  if [[ -f /tmp/nvvm_module_3_opt.ll ]]; then
    echo "=== First 200 lines of module 3 LLVM IR (via opt) ==="
    head -200 /tmp/nvvm_module_3_opt.ll
  fi
fi
EOF

# Transfer debug files if needed
echo "=== Transferring debug files ==="
scp -P $PORT $USER@$HOST:/tmp/nvvm_module_3.bc . 2>/dev/null && echo "Downloaded nvvm_module_3.bc" || echo "No bitcode file to download"
scp -P $PORT $USER@$HOST:/tmp/nvvm_module_3.bc.ll . 2>/dev/null && echo "Downloaded nvvm_module_3.bc.ll" || echo "No LLVM IR file to download"
scp -P $PORT $USER@$HOST:~/ed25519-vanity-rs/build.log . 2>/dev/null && echo "Downloaded build.log" || echo "No build log to download"

# run
ssh -o StrictHostKeyChecking=no -p $PORT $USER@$HOST <<'EOF'
pushd ed25519-vanity-rs
./target/release/ed25519_vanity aaa 16384 256
EOF
