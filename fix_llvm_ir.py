#!/usr/bin/env python3

import sys
import re

def fix_llvm_ir(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Track allocas and their original types
    allocas = {}  # reg -> original_type
    
    # Patterns
    alloca_pattern = re.compile(r'(%\d+) = alloca \[(\d+) x i8\]')
    bitcast_from_alloca = re.compile(r'(%\d+) = bitcast ptr (%\d+) to ptr')
    store_pattern = re.compile(r'store (i\d+|ptr) (.*), ptr (%\d+)')
    gep_pattern = re.compile(r'(%\d+) = getelementptr inbounds i8, ptr (%\d+), i64 (\d+)')
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Track allocas
        alloca_match = alloca_pattern.search(line)
        if alloca_match:
            reg = alloca_match.group(1)
            size = int(alloca_match.group(2))
            allocas[reg] = ('alloca', size)
            fixed_lines.append(line)
            continue
        
        # Track bitcasts from allocas
        bitcast_match = bitcast_from_alloca.search(line)
        if bitcast_match:
            dst_reg = bitcast_match.group(1)
            src_reg = bitcast_match.group(2)
            if src_reg in allocas:
                allocas[dst_reg] = ('bitcast_from_alloca', src_reg)
            # Skip redundant bitcasts
            if line.strip().endswith('bitcast ptr %s to ptr' % src_reg):
                fixed_lines.append(f'  {dst_reg} = bitcast ptr {src_reg} to i8*\n')
                continue
        
        # Track GEPs
        gep_match = gep_pattern.search(line)
        if gep_match:
            dst_reg = gep_match.group(1)
            src_reg = gep_match.group(2)
            offset = gep_match.group(3)
            if src_reg in allocas:
                allocas[dst_reg] = ('gep', src_reg, int(offset))
        
        # Fix stores
        store_match = store_pattern.search(line)
        if store_match:
            val_type = store_match.group(1)
            val_expr = store_match.group(2)
            ptr_reg = store_match.group(3)
            
            # Check if we're storing through an alloca-derived pointer
            if ptr_reg in allocas:
                alloca_info = allocas[ptr_reg]
                
                # Determine the correct pointer type
                if val_type == 'ptr':
                    ptr_type = 'i8**'
                    val_type_fixed = 'i8*'
                elif val_type == 'i64':
                    ptr_type = 'i64*'
                    val_type_fixed = 'i64'
                elif val_type == 'i32':
                    ptr_type = 'i32*'
                    val_type_fixed = 'i32'
                elif val_type == 'i16':
                    ptr_type = 'i16*'
                    val_type_fixed = 'i16'
                elif val_type == 'i8':
                    ptr_type = 'i8*'
                    val_type_fixed = 'i8'
                else:
                    ptr_type = val_type + '*'
                    val_type_fixed = val_type
                
                # Insert a bitcast before the store
                indent = '  ' if line.startswith('  ') else ''
                new_reg = f'%bitcast.{i}'
                fixed_lines.append(f'{indent}{new_reg} = bitcast i8* {ptr_reg} to {ptr_type}\n')
                
                # Fix the store to use typed pointer and value
                if 'ptr' in line:
                    # Replace ptr with i8* in value expression
                    val_expr_fixed = val_expr.replace('ptr', 'i8*').strip()
                    fixed_line = f'{indent}store {val_type_fixed} {val_expr_fixed}, {ptr_type} {new_reg}'
                else:
                    fixed_line = f'{indent}store {val_type_fixed} {val_expr}, {ptr_type} {new_reg}'
                
                # Copy over alignment and metadata
                if 'align' in line:
                    align_match = re.search(r'(, align \d+)', line)
                    if align_match:
                        fixed_line += align_match.group(1)
                if '!' in line:
                    metadata_match = re.search(r'(, !.*)', line)
                    if metadata_match:
                        fixed_line += metadata_match.group(1)
                
                fixed_line += '\n'
                fixed_lines.append(fixed_line)
                continue
        
        # Fix opaque pointer types in function signatures
        line = re.sub(r'\bptr\b(?!\s*%)', 'i8*', line)
        
        fixed_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {len(lines)} lines of LLVM IR")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: fix_llvm_ir.py <input.ll> <output.ll>")
        sys.exit(1)
    
    fix_llvm_ir(sys.argv[1], sys.argv[2])