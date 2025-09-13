#!/bin/bash

# Use the existing run-example infrastructure with tracing
export RUST_LOG=rustc_codegen_nvvm::builder=trace
export EXAMPLE_BUILD_FLAGS="2>&1 | tee /tmp/h3o_trace.log"

./run-example.sh h3o_crates_io 2>&1 | head -200

echo "=== Extracting GEP traces ==="
ssh -o StrictHostKeyChecking=no root@ssh5.vast.ai -p 42034 "grep 'gep:' /tmp/h3o_trace.log | tail -20" 2>/dev/null || true