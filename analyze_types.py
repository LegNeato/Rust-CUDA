#!/usr/bin/env python3

import subprocess
import re
import sys

def analyze_llvm_ir(filename):
    # Read the LLVM IR
    result = subprocess.run(['llvm-dis', filename, '-o', '-'], 
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to disassemble: {result.stderr}")
        return
    
    lines = result.stdout.split('\n')
    
    # Track allocas and their types
    allocas = {}
    
    # Pattern to match alloca
    alloca_pattern = re.compile(r'%(\d+) = alloca \[(\d+) x i8\]')
    # Pattern to match bitcast
    bitcast_pattern = re.compile(r'%(\d+) = bitcast ptr %(\d+) to ptr')
    # Pattern to match store
    store_pattern = re.compile(r'store (\w+) %.*, ptr %(\d+)')
    
    print("=== Analyzing LLVM IR ===")
    
    for i, line in enumerate(lines):
        # Track allocas
        alloca_match = alloca_pattern.search(line)
        if alloca_match:
            reg = alloca_match.group(1)
            size = alloca_match.group(2)
            allocas[reg] = f"[{size} x i8]"
            print(f"Line {i+1}: Alloca %{reg} = [{size} x i8]")
        
        # Track bitcasts
        bitcast_match = bitcast_pattern.search(line)
        if bitcast_match:
            dst = bitcast_match.group(1)
            src = bitcast_match.group(2)
            if src in allocas:
                allocas[dst] = allocas[src] + " (bitcast)"
                print(f"Line {i+1}: Bitcast %{dst} from %{src} ({allocas[src]})")
        
        # Check stores
        store_match = store_pattern.search(line)
        if store_match:
            val_type = store_match.group(1)
            ptr_reg = store_match.group(2)
            
            if ptr_reg in allocas:
                alloca_type = allocas[ptr_reg]
                if val_type != 'i8' and 'i8]' in alloca_type:
                    print(f"\n*** PROBLEMATIC STORE at line {i+1} ***")
                    print(f"  Storing {val_type} through pointer %{ptr_reg}")
                    print(f"  Pointer originates from alloca: {alloca_type}")
                    print(f"  Full line: {line.strip()}")
                    
                    # Show context
                    start = max(0, i-3)
                    end = min(len(lines), i+3)
                    print("  Context:")
                    for j in range(start, end+1):
                        if j < len(lines):
                            marker = ">>>" if j == i else "   "
                            print(f"    {marker} {lines[j]}")
                    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: analyze_types.py <bitcode_file>")
        sys.exit(1)
    
    analyze_llvm_ir(sys.argv[1])