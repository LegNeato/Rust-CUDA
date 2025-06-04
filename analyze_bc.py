#!/usr/bin/env python3

import subprocess
import sys

def analyze_bitcode(bc_file):
    # Try different LLVM versions
    for llvm_ver in ['', '-7', '-14', '-15', '-16', '-17', '-18']:
        dis_cmd = f'llvm-dis{llvm_ver}'
        opt_cmd = f'opt{llvm_ver}'
        
        try:
            # Try llvm-dis
            print(f"\n=== Trying {dis_cmd} ===")
            result = subprocess.run([dis_cmd, bc_file, '-o', '-'], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Success with {dis_cmd}!")
                lines = result.stdout.split('\n')
                print(f"Total lines: {len(lines)}")
                
                # Look for problematic patterns
                for i, line in enumerate(lines):
                    if 'store' in line or 'load' in line:
                        # Show context
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        print(f"\n--- Found load/store at line {i+1} ---")
                        for j in range(start, end):
                            marker = ">>>" if j == i else "   "
                            print(f"{marker} {j+1}: {lines[j]}")
                        
                        if i > 10:  # Only show first few
                            break
                return
            else:
                print(f"Failed: {result.stderr.strip()}")
                
            # Try opt
            print(f"\n=== Trying {opt_cmd} ===")
            result = subprocess.run([opt_cmd, '-S', bc_file, '-o', '-'], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Success with {opt_cmd}!")
                return
            else:
                print(f"Failed: {result.stderr.strip()}")
                
        except FileNotFoundError:
            continue
    
    print("\nNo LLVM version could disassemble the file")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: analyze_bc.py <bitcode_file>")
        sys.exit(1)
    
    analyze_bitcode(sys.argv[1])