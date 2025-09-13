#!/bin/bash

# Build h3o with disassembly output
export RUSTFLAGS="-Cllvm-args=--disassemble"
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG=true

./run-example.sh h3o_crates_io 2>&1 | tee h3o_disasm.log