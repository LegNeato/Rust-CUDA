//! Extremely type-safe and ergonomic warp matrix (tensor core) operations.
//!
//! This module makes tensor core operations foolproof through the type system.
//! Invalid operations won't compile, and valid operations are intuitive.

use crate::gpu_only;
use core::marker::PhantomData;
use half::{bf16, f16};

// Re-export Layout trait for public use
pub use self::layout::Layout;

// Module for trait-based dispatch of matrix operations
mod ops;

// WMMA intrinsic declarations
#[cfg(target_arch = "nvptx64")]
extern "C" {
    // ============= 16x16x16 intrinsics =============
    // f16
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.f16"]
    fn wmma_load_a_f16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.f16"]
    fn wmma_load_a_f16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.f16"]
    fn wmma_load_b_f16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.f16"]
    fn wmma_load_b_f16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    // bf16
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.bf16"]
    fn wmma_load_a_bf16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.bf16"]
    fn wmma_load_a_bf16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.bf16"]
    fn wmma_load_b_bf16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.bf16"]
    fn wmma_load_b_bf16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.c.sync.row.stride.f32"]
    fn wmma_load_c_f32_row_m16n16k16(ptr: *const u8, stride: i32) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.row.stride.f32"]
    fn wmma_store_d_f32_row_m16n16k16(
        ptr: *mut u8,
        d0: f32, d1: f32, d2: f32, d3: f32,
        d4: f32, d5: f32, d6: f32, d7: f32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.col.stride.f32"]
    fn wmma_store_d_f32_col_m16n16k16(
        ptr: *mut u8,
        d0: f32, d1: f32, d2: f32, d3: f32,
        d4: f32, d5: f32, d6: f32, d7: f32,
        stride: i32
    );
    
    // i8/u8 load intrinsics
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.s8"]
    fn wmma_load_a_s8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.s8"]
    fn wmma_load_a_s8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.u8"]
    fn wmma_load_a_u8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.u8"]
    fn wmma_load_a_u8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.s8"]
    fn wmma_load_b_s8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.s8"]
    fn wmma_load_b_s8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.u8"]
    fn wmma_load_b_u8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.u8"]
    fn wmma_load_b_u8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.c.sync.row.stride.s32"]
    fn wmma_load_c_s32_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.row.stride.s32"]
    fn wmma_store_d_s32_row_m16n16k16(
        ptr: *mut u8,
        d0: i32, d1: i32, d2: i32, d3: i32,
        d4: i32, d5: i32, d6: i32, d7: i32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.col.stride.s32"]
    fn wmma_store_d_s32_col_m16n16k16(
        ptr: *mut u8,
        d0: i32, d1: i32, d2: i32, d3: i32,
        d4: i32, d5: i32, d6: i32, d7: i32,
        stride: i32
    );
    
    // MMA intrinsics for f16 -> f32
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.f16.f32"]
    fn wmma_mma_f16_f32_row_row_m16n16k16(
        a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16,
        a8: i16, a9: i16, a10: i16, a11: i16, a12: i16, a13: i16, a14: i16, a15: i16,
        b0: i16, b1: i16, b2: i16, b3: i16, b4: i16, b5: i16, b6: i16, b7: i16,
        b8: i16, b9: i16, b10: i16, b11: i16, b12: i16, b13: i16, b14: i16, b15: i16,
        c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32
    ) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.f16.f32"]
    fn wmma_mma_f16_f32_row_col_m16n16k16(
        a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16,
        a8: i16, a9: i16, a10: i16, a11: i16, a12: i16, a13: i16, a14: i16, a15: i16,
        b0: i16, b1: i16, b2: i16, b3: i16, b4: i16, b5: i16, b6: i16, b7: i16,
        b8: i16, b9: i16, b10: i16, b11: i16, b12: i16, b13: i16, b14: i16, b15: i16,
        c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32
    ) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.col.row.f16.f32"]
    fn wmma_mma_f16_f32_col_row_m16n16k16(
        a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16,
        a8: i16, a9: i16, a10: i16, a11: i16, a12: i16, a13: i16, a14: i16, a15: i16,
        b0: i16, b1: i16, b2: i16, b3: i16, b4: i16, b5: i16, b6: i16, b7: i16,
        b8: i16, b9: i16, b10: i16, b11: i16, b12: i16, b13: i16, b14: i16, b15: i16,
        c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32
    ) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.col.col.f16.f32"]
    fn wmma_mma_f16_f32_col_col_m16n16k16(
        a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16,
        a8: i16, a9: i16, a10: i16, a11: i16, a12: i16, a13: i16, a14: i16, a15: i16,
        b0: i16, b1: i16, b2: i16, b3: i16, b4: i16, b5: i16, b6: i16, b7: i16,
        b8: i16, b9: i16, b10: i16, b11: i16, b12: i16, b13: i16, b14: i16, b15: i16,
        c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32
    ) -> [f32; 8];
    
    // MMA intrinsics for i8/u8 -> i32
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.s8.s8.s32"]
    fn wmma_mma_s8_s32_row_row_m16n16k16(
        a0: i32, a1: i32, a2: i32, a3: i32,
        b0: i32, b1: i32, b2: i32, b3: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.s8.s8.s32"]
    fn wmma_mma_s8_s32_row_col_m16n16k16(
        a0: i32, a1: i32, a2: i32, a3: i32,
        b0: i32, b1: i32, b2: i32, b3: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.u8.u8.s32"]
    fn wmma_mma_u8_s32_row_row_m16n16k16(
        a0: i32, a1: i32, a2: i32, a3: i32,
        b0: i32, b1: i32, b2: i32, b3: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.u8.u8.s32"]
    fn wmma_mma_u8_s32_row_col_m16n16k16(
        a0: i32, a1: i32, a2: i32, a3: i32,
        b0: i32, b1: i32, b2: i32, b3: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    // ============= 32x8x16 intrinsics =============
    // f16
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.f16"]
    fn wmma_load_a_f16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.f16"]
    fn wmma_load_a_f16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.f16"]
    fn wmma_load_b_f16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.f16"]
    fn wmma_load_b_f16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    // bf16
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.bf16"]
    fn wmma_load_a_bf16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.bf16"]
    fn wmma_load_a_bf16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.bf16"]
    fn wmma_load_b_bf16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.bf16"]
    fn wmma_load_b_bf16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.row.stride.f32"]
    fn wmma_load_c_f32_row_m32n8k16(ptr: *const u8, stride: i32) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.col.stride.f32"]
    fn wmma_load_c_f32_col_m32n8k16(ptr: *const u8, stride: i32) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.row.stride.f32"]
    fn wmma_store_d_f32_row_m32n8k16(
        ptr: *mut u8,
        d0: f32, d1: f32, d2: f32, d3: f32,
        d4: f32, d5: f32, d6: f32, d7: f32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.col.stride.f32"]
    fn wmma_store_d_f32_col_m32n8k16(
        ptr: *mut u8,
        d0: f32, d1: f32, d2: f32, d3: f32,
        d4: f32, d5: f32, d6: f32, d7: f32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.row.stride.s32"]
    fn wmma_load_c_s32_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.col.stride.s32"]
    fn wmma_load_c_s32_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.row.stride.s32"]
    fn wmma_store_d_s32_row_m32n8k16(
        ptr: *mut u8,
        d0: i32, d1: i32, d2: i32, d3: i32,
        d4: i32, d5: i32, d6: i32, d7: i32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.col.stride.s32"]
    fn wmma_store_d_s32_col_m32n8k16(
        ptr: *mut u8,
        d0: i32, d1: i32, d2: i32, d3: i32,
        d4: i32, d5: i32, d6: i32, d7: i32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.mma.sync.row.row.f16.f32"]
    fn wmma_mma_f16_f32_row_row_m32n8k16(
        a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16,
        a8: i16, a9: i16, a10: i16, a11: i16, a12: i16, a13: i16, a14: i16, a15: i16,
        b0: i16, b1: i16, b2: i16, b3: i16, b4: i16, b5: i16, b6: i16, b7: i16,
        c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32
    ) -> [f32; 8];
    
    // i8/u8
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.s8"]
    fn wmma_load_a_s8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.s8"]
    fn wmma_load_a_s8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.u8"]
    fn wmma_load_a_u8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.u8"]
    fn wmma_load_a_u8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.s8"]
    fn wmma_load_b_s8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.s8"]
    fn wmma_load_b_s8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.u8"]
    fn wmma_load_b_u8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.u8"]
    fn wmma_load_b_u8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.mma.sync.row.row.s8.s8.s32"]
    fn wmma_mma_s8_s32_row_row_m32n8k16(
        a0: i32, a1: i32, a2: i32, a3: i32,
        b0: i32, b1: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m32n8k16.mma.sync.row.row.u8.u8.s32"]
    fn wmma_mma_u8_s32_row_row_m32n8k16(
        a0: i32, a1: i32, a2: i32, a3: i32,
        b0: i32, b1: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    // ============= 8x32x16 intrinsics =============
    // f16
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.f16"]
    fn wmma_load_a_f16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.f16"]
    fn wmma_load_a_f16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.f16"]
    fn wmma_load_b_f16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.f16"]
    fn wmma_load_b_f16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    // bf16
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.bf16"]
    fn wmma_load_a_bf16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.bf16"]
    fn wmma_load_a_bf16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.bf16"]
    fn wmma_load_b_bf16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.bf16"]
    fn wmma_load_b_bf16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.row.stride.f32"]
    fn wmma_load_c_f32_row_m8n32k16(ptr: *const u8, stride: i32) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.col.stride.f32"]
    fn wmma_load_c_f32_col_m8n32k16(ptr: *const u8, stride: i32) -> [f32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.row.stride.f32"]
    fn wmma_store_d_f32_row_m8n32k16(
        ptr: *mut u8,
        d0: f32, d1: f32, d2: f32, d3: f32,
        d4: f32, d5: f32, d6: f32, d7: f32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.col.stride.f32"]
    fn wmma_store_d_f32_col_m8n32k16(
        ptr: *mut u8,
        d0: f32, d1: f32, d2: f32, d3: f32,
        d4: f32, d5: f32, d6: f32, d7: f32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.row.stride.s32"]
    fn wmma_load_c_s32_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.col.stride.s32"]
    fn wmma_load_c_s32_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.row.stride.s32"]
    fn wmma_store_d_s32_row_m8n32k16(
        ptr: *mut u8,
        d0: i32, d1: i32, d2: i32, d3: i32,
        d4: i32, d5: i32, d6: i32, d7: i32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.col.stride.s32"]
    fn wmma_store_d_s32_col_m8n32k16(
        ptr: *mut u8,
        d0: i32, d1: i32, d2: i32, d3: i32,
        d4: i32, d5: i32, d6: i32, d7: i32,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.mma.sync.row.row.f16.f32"]
    fn wmma_mma_f16_f32_row_row_m8n32k16(
        a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16,
        b0: i16, b1: i16, b2: i16, b3: i16, b4: i16, b5: i16, b6: i16, b7: i16,
        b8: i16, b9: i16, b10: i16, b11: i16, b12: i16, b13: i16, b14: i16, b15: i16,
        c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32
    ) -> [f32; 8];
    
    // i8/u8
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.s8"]
    fn wmma_load_a_s8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.s8"]
    fn wmma_load_a_s8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.u8"]
    fn wmma_load_a_u8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.u8"]
    fn wmma_load_a_u8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.s8"]
    fn wmma_load_b_s8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.s8"]
    fn wmma_load_b_s8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.u8"]
    fn wmma_load_b_u8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.u8"]
    fn wmma_load_b_u8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.mma.sync.row.row.s8.s8.s32"]
    fn wmma_mma_s8_s32_row_row_m8n32k16(
        a0: i32, a1: i32,
        b0: i32, b1: i32, b2: i32, b3: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    #[link_name = "llvm.nvvm.wmma.m8n32k16.mma.sync.row.row.u8.u8.s32"]
    fn wmma_mma_u8_s32_row_row_m8n32k16(
        a0: i32, a1: i32,
        b0: i32, b1: i32, b2: i32, b3: i32,
        c0: i32, c1: i32, c2: i32, c3: i32, c4: i32, c5: i32, c6: i32, c7: i32
    ) -> [i32; 8];
    
    // ============= 8x8x4 intrinsics (f64) =============
    #[link_name = "llvm.nvvm.wmma.m8n8k4.load.a.sync.row.stride.f64"]
    fn wmma_load_a_f64_row_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.load.a.sync.col.stride.f64"]
    fn wmma_load_a_f64_col_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.load.b.sync.row.stride.f64"]
    fn wmma_load_b_f64_row_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.load.b.sync.col.stride.f64"]
    fn wmma_load_b_f64_col_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.load.c.sync.row.stride.f64"]
    fn wmma_load_c_f64_row_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.load.c.sync.col.stride.f64"]
    fn wmma_load_c_f64_col_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.store.d.sync.row.stride.f64"]
    fn wmma_store_d_f64_row_m8n8k4(
        ptr: *mut u8,
        d0: f64, d1: f64,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.store.d.sync.col.stride.f64"]
    fn wmma_store_d_f64_col_m8n8k4(
        ptr: *mut u8,
        d0: f64, d1: f64,
        stride: i32
    );
    
    #[link_name = "llvm.nvvm.wmma.m8n8k4.mma.sync.row.row.f64"]
    fn wmma_mma_f64_row_row_m8n8k4(
        a0: f64, a1: f64,
        b0: f64, b1: f64,
        c0: f64, c1: f64
    ) -> [f64; 2];
}

// ============================================================================
// Shape Types with Compile-Time Validation
// ============================================================================

/// Type-level dimensions for matrix operations
pub mod dims {
    /// Complete shape specification
    pub struct Shape<const M: usize, const N: usize, const K: usize>;
}

/// Valid tensor core shape combinations
pub trait TensorCoreShape: sealed::Sealed {
    const M: usize;
    const N: usize;
    const K: usize;
}

// Only these exact combinations are valid for tensor cores
impl TensorCoreShape for dims::Shape<16, 16, 16> {
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 16;
}

impl TensorCoreShape for dims::Shape<32, 8, 16> {
    const M: usize = 32;
    const N: usize = 8;
    const K: usize = 16;
}

impl TensorCoreShape for dims::Shape<8, 32, 16> {
    const M: usize = 8;
    const N: usize = 32;
    const K: usize = 16;
}

impl TensorCoreShape for dims::Shape<16, 16, 8> {
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 8;
}

impl TensorCoreShape for dims::Shape<8, 8, 32> {
    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 32;
}

impl TensorCoreShape for dims::Shape<8, 8, 128> {
    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 128;
}

impl TensorCoreShape for dims::Shape<8, 8, 4> {
    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 4;
}

// ============================================================================
// Layout Types
// ============================================================================

/// Type-level layout specification
pub mod layout {
    use super::sealed;
    
    /// Row-major layout
    pub struct Row;

    /// Column-major layout
    pub struct Col;
    
    /// Trait for valid layouts
    pub trait Layout: sealed::Sealed {
        const IS_ROW_MAJOR: bool;
    }

    impl Layout for Row {
        const IS_ROW_MAJOR: bool = true;
    }

    impl Layout for Col {
        const IS_ROW_MAJOR: bool = false;
    }
}

// ============================================================================
// Element Types with Compatibility Rules
// ============================================================================

/// Trait for types that can be matrix elements
pub trait MatrixElement: Copy + sealed::Sealed {
    /// The accumulator type this element requires
    type Accumulator: AccumulatorElement;

    /// Storage type in fragment
    type Storage: Copy;

    /// Number of elements per thread
    const ELEMENTS_PER_THREAD: usize;
}

/// Trait for types that can be accumulator elements
pub trait AccumulatorElement: Copy + sealed::Sealed {
    type Storage: Copy;
    const ELEMENTS_PER_THREAD: usize;
}

// f16 matrices accumulate to f32 or f16
impl MatrixElement for f16 {
    type Accumulator = f32; // Default to f32 for better precision
    type Storage = f16;
    const ELEMENTS_PER_THREAD: usize = 16;
}

// bf16 matrices accumulate to f32
impl MatrixElement for bf16 {
    type Accumulator = f32;
    type Storage = bf16;
    const ELEMENTS_PER_THREAD: usize = 16;
}

// i8 matrices accumulate to i32
impl MatrixElement for i8 {
    type Accumulator = i32;
    type Storage = i32; // Packed
    const ELEMENTS_PER_THREAD: usize = 4;
}

// u8 matrices accumulate to i32
impl MatrixElement for u8 {
    type Accumulator = i32;
    type Storage = i32; // Packed
    const ELEMENTS_PER_THREAD: usize = 4;
}

// TODO: TF32 support (16x16x8 shape) - not currently implemented
// f32 matrices would use TF32 on capable hardware
impl MatrixElement for f32 {
    type Accumulator = f32;
    type Storage = f32;
    const ELEMENTS_PER_THREAD: usize = 8;
}

// f64 matrices (for 8x8x4 shape)
impl MatrixElement for f64 {
    type Accumulator = f64;
    type Storage = f64;
    const ELEMENTS_PER_THREAD: usize = 2;
}

// i32 matrices accumulate to i32 (for integer tensor cores)
impl MatrixElement for i32 {
    type Accumulator = i32;
    type Storage = i32; // Packed
    const ELEMENTS_PER_THREAD: usize = 4;
}

// bool matrices use u8 under the hood, accumulate to i32
impl MatrixElement for bool {
    type Accumulator = i32;
    type Storage = u8; // Packed like u8
    const ELEMENTS_PER_THREAD: usize = 4;
}

impl AccumulatorElement for f32 {
    type Storage = f32;
    const ELEMENTS_PER_THREAD: usize = 8;
}

impl AccumulatorElement for f16 {
    type Storage = f16;
    const ELEMENTS_PER_THREAD: usize = 8;
}

impl AccumulatorElement for i32 {
    type Storage = i32;
    const ELEMENTS_PER_THREAD: usize = 8;
}

impl AccumulatorElement for f64 {
    type Storage = f64;
    const ELEMENTS_PER_THREAD: usize = 8;
}

// ============================================================================
// Matrix Fragments with Role-Specific Types
// ============================================================================

/// Matrix A fragment (left operand)
#[repr(C)]
pub struct MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    data: [T::Storage; 32], // Max size
    _phantom: PhantomData<(T, Shape, L)>,
}

/// Matrix B fragment (right operand)
#[repr(C)]
pub struct MatrixB<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    data: [T::Storage; 32], // Max size
    _phantom: PhantomData<(T, Shape, L)>,
}

/// Accumulator matrix fragment
#[repr(C)]
pub struct Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    data: [T::Storage; 32], // Max size
    _phantom: PhantomData<(T, Shape)>,
}

// ============================================================================
// Safe Constructors and Operations
// ============================================================================

impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    /// Create a new matrix A fragment
    #[inline]
    pub fn new() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
            _phantom: PhantomData,
        }
    }

    /// Load from memory with compile-time stride validation
    #[gpu_only]
    pub unsafe fn load<const STRIDE: usize>(&mut self, ptr: *const T)
    where
        StrideValidator<T, STRIDE>: ValidStride,
        T: ops::LoadMatrixA<Shape, L>,
    {
        T::load_a_into(ptr as *const u8, STRIDE as i32, &mut self.data);
    }
}

impl<T, Shape, L> Default for MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Shape, L> MatrixB<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    /// Create a new matrix B fragment
    #[inline]
    pub fn new() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
            _phantom: PhantomData,
        }
    }

    /// Load from memory with compile-time stride validation
    #[gpu_only]
    pub unsafe fn load<const STRIDE: usize>(&mut self, ptr: *const T)
    where
        StrideValidator<T, STRIDE>: ValidStride,
        T: ops::LoadMatrixB<Shape, L>,
    {
        T::load_b_into(ptr as *const u8, STRIDE as i32, &mut self.data);
    }
}

impl<T, Shape, L> Default for MatrixB<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Shape> Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    /// Create a new accumulator fragment
    #[inline]
    pub fn new() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
            _phantom: PhantomData,
        }
    }

    /// Load from memory with compile-time stride validation
    #[gpu_only]
    pub unsafe fn load<L, const STRIDE: usize>(&mut self, ptr: *const T)
    where
        L: Layout,
        StrideValidator<T, STRIDE>: ValidStride,
        T: ops::LoadMatrixC<Shape, L>,
    {
        T::load_c_into(ptr as *const u8, STRIDE as i32, &mut self.data);
    }

    /// Fill with a constant value
    #[gpu_only]
    pub fn fill(&mut self, value: T) {
        for i in 0..T::ELEMENTS_PER_THREAD {
            self.data[i] = unsafe { core::mem::transmute_copy(&value) };
        }
    }

    /// Store to memory with layout specification
    #[gpu_only]
    pub unsafe fn store<L, const STRIDE: usize>(&self, ptr: *mut T)
    where
        L: Layout,
        StrideValidator<T, STRIDE>: ValidStride,
        T: ops::StoreMatrixD<Shape, L>,
    {
        T::store_d_from(ptr as *mut u8, &self.data, STRIDE as i32);
    }
}

impl<T, Shape> Default for Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type-Safe Matrix Multiply-Accumulate
// ============================================================================

/// Trait for valid MMA operations - only implemented for valid combinations
pub trait Mma<A, B, C>: sealed::Sealed
where
    A: MatrixElement,
    B: MatrixElement,
    C: AccumulatorElement,
{
    type Output: AccumulatorElement;

    /// Perform the matrix multiply-accumulate operation
    fn mma<S, LA, LB>(
        a: &MatrixA<A, S, LA>,
        b: &MatrixB<B, S, LB>,
        c: &Accumulator<C, S>,
    ) -> Accumulator<Self::Output, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout;
}

/// Implementation for f16 × f16 + f32 → f32
impl Mma<f16, f16, f32> for f32 {
    type Output = f32;

    #[gpu_only]
    fn mma<S, LA, LB>(
        a: &MatrixA<f16, S, LA>,
        b: &MatrixB<f16, S, LB>,
        c: &Accumulator<f32, S>,
    ) -> Accumulator<f32, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        let mut result = Accumulator::new();
        
        // Only 16x16x16 shape is currently implemented
        if S::M == 16 && S::N == 16 && S::K == 16 {
            // Extract f16 values from matrix fragments
            let a_vals = unsafe { core::mem::transmute::<[<f16 as MatrixElement>::Storage; 16], [i16; 16]>(
                *(&a.data[..16] as *const [<f16 as MatrixElement>::Storage] as *const [<f16 as MatrixElement>::Storage; 16])
            )};
            let b_vals = unsafe { core::mem::transmute::<[<f16 as MatrixElement>::Storage; 16], [i16; 16]>(
                *(&b.data[..16] as *const [<f16 as MatrixElement>::Storage] as *const [<f16 as MatrixElement>::Storage; 16])
            )};
            let c_vals = unsafe { core::mem::transmute::<[<f32 as AccumulatorElement>::Storage; 8], [f32; 8]>(
                *(&c.data[..8] as *const [<f32 as AccumulatorElement>::Storage] as *const [<f32 as AccumulatorElement>::Storage; 8])
            )};
            
            // Call the appropriate intrinsic based on layouts
            let result_vals = unsafe {
                if LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
                    wmma_mma_f16_f32_row_row_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3], a_vals[4], a_vals[5], a_vals[6], a_vals[7],
                        a_vals[8], a_vals[9], a_vals[10], a_vals[11], a_vals[12], a_vals[13], a_vals[14], a_vals[15],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3], b_vals[4], b_vals[5], b_vals[6], b_vals[7],
                        b_vals[8], b_vals[9], b_vals[10], b_vals[11], b_vals[12], b_vals[13], b_vals[14], b_vals[15],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                } else if LA::IS_ROW_MAJOR && !LB::IS_ROW_MAJOR {
                    wmma_mma_f16_f32_row_col_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3], a_vals[4], a_vals[5], a_vals[6], a_vals[7],
                        a_vals[8], a_vals[9], a_vals[10], a_vals[11], a_vals[12], a_vals[13], a_vals[14], a_vals[15],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3], b_vals[4], b_vals[5], b_vals[6], b_vals[7],
                        b_vals[8], b_vals[9], b_vals[10], b_vals[11], b_vals[12], b_vals[13], b_vals[14], b_vals[15],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                } else if !LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
                    wmma_mma_f16_f32_col_row_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3], a_vals[4], a_vals[5], a_vals[6], a_vals[7],
                        a_vals[8], a_vals[9], a_vals[10], a_vals[11], a_vals[12], a_vals[13], a_vals[14], a_vals[15],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3], b_vals[4], b_vals[5], b_vals[6], b_vals[7],
                        b_vals[8], b_vals[9], b_vals[10], b_vals[11], b_vals[12], b_vals[13], b_vals[14], b_vals[15],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                } else {
                    wmma_mma_f16_f32_col_col_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3], a_vals[4], a_vals[5], a_vals[6], a_vals[7],
                        a_vals[8], a_vals[9], a_vals[10], a_vals[11], a_vals[12], a_vals[13], a_vals[14], a_vals[15],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3], b_vals[4], b_vals[5], b_vals[6], b_vals[7],
                        b_vals[8], b_vals[9], b_vals[10], b_vals[11], b_vals[12], b_vals[13], b_vals[14], b_vals[15],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                }
            };
            
            // Store result
            result.data[..8].copy_from_slice(&unsafe { core::mem::transmute::<[f32; 8], [<f32 as AccumulatorElement>::Storage; 8]>(result_vals) });
        }
        
        result
    }
}

/// Implementation for f16 × f16 + f16 → f16
impl Mma<f16, f16, f16> for f16 {
    type Output = f16;

    #[gpu_only]
    fn mma<S, LA, LB>(
        _a: &MatrixA<f16, S, LA>,
        _b: &MatrixB<f16, S, LB>,
        c: &Accumulator<f16, S>,
    ) -> Accumulator<f16, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        Accumulator {
            data: c.data,
            _phantom: PhantomData,
        }
    }
}

/// Implementation for i8 × i8 + i32 → i32
impl Mma<i8, i8, i32> for i32 {
    type Output = i32;

    #[gpu_only]
    fn mma<S, LA, LB>(
        a: &MatrixA<i8, S, LA>,
        b: &MatrixB<i8, S, LB>,
        c: &Accumulator<i32, S>,
    ) -> Accumulator<i32, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        let mut result = Accumulator::new();
        
        // Only 16x16x16 shape is currently implemented
        if S::M == 16 && S::N == 16 && S::K == 16 {
            let a_vals = unsafe { core::mem::transmute::<[<i8 as MatrixElement>::Storage; 4], [i32; 4]>(
                *(&a.data[..4] as *const [<i8 as MatrixElement>::Storage] as *const [<i8 as MatrixElement>::Storage; 4])
            )};
            let b_vals = unsafe { core::mem::transmute::<[<i8 as MatrixElement>::Storage; 4], [i32; 4]>(
                *(&b.data[..4] as *const [<i8 as MatrixElement>::Storage] as *const [<i8 as MatrixElement>::Storage; 4])
            )};
            let c_vals = unsafe { core::mem::transmute::<[<i32 as AccumulatorElement>::Storage; 8], [i32; 8]>(
                *(&c.data[..8] as *const [<i32 as AccumulatorElement>::Storage] as *const [<i32 as AccumulatorElement>::Storage; 8])
            )};
            
            let result_vals = unsafe {
                if LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
                    wmma_mma_s8_s32_row_row_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                } else {
                    wmma_mma_s8_s32_row_col_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                }
            };
            
            result.data[..8].copy_from_slice(&unsafe { core::mem::transmute::<[i32; 8], [<i32 as AccumulatorElement>::Storage; 8]>(result_vals) });
        }
        
        result
    }
}

/// Implementation for u8 × u8 + i32 → i32
impl Mma<u8, u8, i32> for i32 {
    type Output = i32;

    #[gpu_only]
    fn mma<S, LA, LB>(
        a: &MatrixA<u8, S, LA>,
        b: &MatrixB<u8, S, LB>,
        c: &Accumulator<i32, S>,
    ) -> Accumulator<i32, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        let mut result = Accumulator::new();
        
        // Only 16x16x16 shape is currently implemented
        if S::M == 16 && S::N == 16 && S::K == 16 {
            let a_vals = unsafe { core::mem::transmute::<[<u8 as MatrixElement>::Storage; 4], [i32; 4]>(
                *(&a.data[..4] as *const [<u8 as MatrixElement>::Storage] as *const [<u8 as MatrixElement>::Storage; 4])
            )};
            let b_vals = unsafe { core::mem::transmute::<[<u8 as MatrixElement>::Storage; 4], [i32; 4]>(
                *(&b.data[..4] as *const [<u8 as MatrixElement>::Storage] as *const [<u8 as MatrixElement>::Storage; 4])
            )};
            let c_vals = unsafe { core::mem::transmute::<[<i32 as AccumulatorElement>::Storage; 8], [i32; 8]>(
                *(&c.data[..8] as *const [<i32 as AccumulatorElement>::Storage] as *const [<i32 as AccumulatorElement>::Storage; 8])
            )};
            
            let result_vals = unsafe {
                if LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
                    wmma_mma_u8_s32_row_row_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                } else {
                    wmma_mma_u8_s32_row_col_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                }
            };
            
            result.data[..8].copy_from_slice(&unsafe { core::mem::transmute::<[i32; 8], [<i32 as AccumulatorElement>::Storage; 8]>(result_vals) });
        }
        
        result
    }
}

/// Implementation for bool × bool + i32 → i32 (uses u8 under the hood)
impl Mma<bool, bool, i32> for i32 {
    type Output = i32;

    #[gpu_only]
    fn mma<S, LA, LB>(
        a: &MatrixA<bool, S, LA>,
        b: &MatrixB<bool, S, LB>,
        c: &Accumulator<i32, S>,
    ) -> Accumulator<i32, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        let mut result = Accumulator::new();
        
        // Only 16x16x16 shape is currently implemented
        if S::M == 16 && S::N == 16 && S::K == 16 {
            // Bool uses same storage as u8 but packed into i32 arrays
            // The data is already stored as u8[16] but we need to transmute it to i32[4]
            let a_bytes = unsafe { *(&a.data[..16] as *const [u8] as *const [u8; 16]) };
            let a_vals = unsafe { core::mem::transmute::<[u8; 16], [i32; 4]>(a_bytes) };
            
            let b_bytes = unsafe { *(&b.data[..16] as *const [u8] as *const [u8; 16]) };
            let b_vals = unsafe { core::mem::transmute::<[u8; 16], [i32; 4]>(b_bytes) };
            
            let c_vals = unsafe { *(&c.data[..8] as *const [i32] as *const [i32; 8]) };
            
            let result_vals = unsafe {
                if LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
                    wmma_mma_u8_s32_row_row_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                } else {
                    wmma_mma_u8_s32_row_col_m16n16k16(
                        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                        b_vals[0], b_vals[1], b_vals[2], b_vals[3],
                        c_vals[0], c_vals[1], c_vals[2], c_vals[3], c_vals[4], c_vals[5], c_vals[6], c_vals[7]
                    )
                }
            };
            
            result.data[..8].copy_from_slice(&unsafe { core::mem::transmute::<[i32; 8], [<i32 as AccumulatorElement>::Storage; 8]>(result_vals) });
        }
        
        result
    }
}

// ============================================================================
// Stride Validation
// ============================================================================

/// Compile-time stride validation
pub struct StrideValidator<T, const STRIDE: usize>(PhantomData<T>);

/// Trait for valid strides
pub trait ValidStride: sealed::Sealed {}

// f16 requires stride to be multiple of 8
impl ValidStride for StrideValidator<f16, 8> {}
impl ValidStride for StrideValidator<f16, 16> {}
impl ValidStride for StrideValidator<f16, 24> {}
impl ValidStride for StrideValidator<f16, 32> {}
impl ValidStride for StrideValidator<f16, 40> {}
impl ValidStride for StrideValidator<f16, 48> {}
impl ValidStride for StrideValidator<f16, 56> {}
impl ValidStride for StrideValidator<f16, 64> {}

// f32 requires stride to be multiple of 4
impl ValidStride for StrideValidator<f32, 4> {}
impl ValidStride for StrideValidator<f32, 8> {}
impl ValidStride for StrideValidator<f32, 12> {}
impl ValidStride for StrideValidator<f32, 16> {}
impl ValidStride for StrideValidator<f32, 20> {}
impl ValidStride for StrideValidator<f32, 24> {}
impl ValidStride for StrideValidator<f32, 28> {}
impl ValidStride for StrideValidator<f32, 32> {}

// bf16 requires stride to be multiple of 8 (same as f16)
impl ValidStride for StrideValidator<bf16, 8> {}
impl ValidStride for StrideValidator<bf16, 16> {}
impl ValidStride for StrideValidator<bf16, 24> {}
impl ValidStride for StrideValidator<bf16, 32> {}
impl ValidStride for StrideValidator<bf16, 40> {}
impl ValidStride for StrideValidator<bf16, 48> {}
impl ValidStride for StrideValidator<bf16, 56> {}
impl ValidStride for StrideValidator<bf16, 64> {}

// i8/u8 require stride to be multiple of 16
impl ValidStride for StrideValidator<i8, 16> {}
impl ValidStride for StrideValidator<i8, 32> {}
impl ValidStride for StrideValidator<i8, 48> {}
impl ValidStride for StrideValidator<i8, 64> {}

impl ValidStride for StrideValidator<u8, 16> {}
impl ValidStride for StrideValidator<u8, 32> {}
impl ValidStride for StrideValidator<u8, 48> {}
impl ValidStride for StrideValidator<u8, 64> {}

// i32 requires stride to be multiple of 4 (same as f32)
impl ValidStride for StrideValidator<i32, 4> {}
impl ValidStride for StrideValidator<i32, 8> {}
impl ValidStride for StrideValidator<i32, 12> {}
impl ValidStride for StrideValidator<i32, 16> {}
impl ValidStride for StrideValidator<i32, 20> {}
impl ValidStride for StrideValidator<i32, 24> {}
impl ValidStride for StrideValidator<i32, 28> {}
impl ValidStride for StrideValidator<i32, 32> {}

// bool uses u8 stride requirements (multiple of 16)
impl ValidStride for StrideValidator<bool, 16> {}
impl ValidStride for StrideValidator<bool, 32> {}
impl ValidStride for StrideValidator<bool, 48> {}
impl ValidStride for StrideValidator<bool, 64> {}

// ============================================================================
// Ergonomic Builder API
// ============================================================================

/// Entry point for tensor core operations with type-driven API
pub struct TensorCore<T: MatrixElement, Shape: TensorCoreShape> {
    _phantom: PhantomData<(T, Shape)>,
}

impl<T: MatrixElement, Shape: TensorCoreShape> TensorCore<T, Shape> {
    /// Create a new tensor core operation for the given element type and shape
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Create matrix A with specified layout
    pub fn matrix_a<L: Layout>(&self) -> MatrixA<T, Shape, L> {
        MatrixA::new()
    }

    /// Create matrix B with specified layout
    pub fn matrix_b<L: Layout>(&self) -> MatrixB<T, Shape, L> {
        MatrixB::new()
    }
}

// Specialized implementations for different element types
impl<Shape: TensorCoreShape> TensorCore<f16, Shape> {
    /// Create an f32 accumulator (default precision for f16 inputs)
    pub fn accumulator(&self) -> Accumulator<f32, Shape> {
        Accumulator::new()
    }

    /// Create an f16 accumulator (lower precision, potentially faster)
    pub fn accumulator_f16(&self) -> Accumulator<f16, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<bf16, Shape> {
    /// Create an f32 accumulator for bf16 operations
    pub fn accumulator(&self) -> Accumulator<f32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<f32, Shape> {
    /// Create an f32 accumulator for TF32 operations
    pub fn accumulator(&self) -> Accumulator<f32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<i8, Shape> {
    /// Create an i32 accumulator (required for i8 operations)
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<u8, Shape> {
    /// Create an i32 accumulator (required for u8 operations)
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<i32, Shape> {
    /// Create an i32 accumulator for integer tensor cores
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<bool, Shape> {
    /// Create an i32 accumulator (bool uses u8 under the hood)
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

// Builder structs are no longer needed with the type-driven API
// The TensorCore type itself now provides all the necessary methods

// ============================================================================
// Extension trait for ergonomic MMA operations
// ============================================================================

/// Extension trait for accumulator to provide ergonomic MMA syntax
pub trait MmaExt<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    /// Perform MMA: self = a × b + self
    fn mma_inplace<E, LA, LB>(&mut self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>)
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>;

    /// Perform MMA: result = a × b + self
    fn mma<E, LA, LB>(self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>) -> Self
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>;
}

impl<T, Shape> MmaExt<T, Shape> for Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    fn mma_inplace<E, LA, LB>(&mut self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>)
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>,
    {
        *self = T::mma(a, b, self);
    }

    fn mma<E, LA, LB>(self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>) -> Self
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>,
    {
        T::mma(a, b, &self)
    }
}

// ============================================================================
// Convenient type aliases
// ============================================================================

// Type aliases removed - TensorCore now requires element type parameter
// Users should use TensorCore::<T, dims::Shape<M, N, K>>::new() directly

// ============================================================================
// Private sealing
// ============================================================================

mod sealed {
    use super::*;

    pub trait Sealed {}

    // Seal shapes
    impl Sealed for dims::Shape<16, 16, 16> {}
    impl Sealed for dims::Shape<32, 8, 16> {}
    impl Sealed for dims::Shape<8, 32, 16> {}
    impl Sealed for dims::Shape<16, 16, 8> {}
    impl Sealed for dims::Shape<8, 8, 32> {}
    impl Sealed for dims::Shape<8, 8, 128> {}
    impl Sealed for dims::Shape<8, 8, 4> {}

    // Seal layouts
    impl Sealed for layout::Row {}
    impl Sealed for layout::Col {}

    // Seal elements
    impl Sealed for f16 {}
    impl Sealed for bf16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i32 {}
    impl Sealed for bool {}

    // Seal stride validators
    impl<T, const S: usize> Sealed for StrideValidator<T, S> {}
}
