// WMMA intrinsic declarations
#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(improper_ctypes)]

// ============= 16x16x16 shape =============
mod m16n16k16 {
    // Load operations
    pub(crate) mod load {
        extern "C" {
            // f16 loads
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.f16"]
            pub(crate) fn wmma_load_a_f16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.f16"]
            pub(crate) fn wmma_load_a_f16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.f16"]
            pub(crate) fn wmma_load_b_f16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.f16"]
            pub(crate) fn wmma_load_b_f16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];

            // bf16 loads
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.bf16"]
            pub(crate) fn wmma_load_a_bf16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.bf16"]
            pub(crate) fn wmma_load_a_bf16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.bf16"]
            pub(crate) fn wmma_load_b_bf16_row_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.bf16"]
            pub(crate) fn wmma_load_b_bf16_col_m16n16k16(ptr: *const u8, stride: i32) -> [i16; 16];

            // i8/u8 loads
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.s8"]
            pub(crate) fn wmma_load_a_s8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.s8"]
            pub(crate) fn wmma_load_a_s8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.row.stride.u8"]
            pub(crate) fn wmma_load_a_u8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.sync.col.stride.u8"]
            pub(crate) fn wmma_load_a_u8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.s8"]
            pub(crate) fn wmma_load_b_s8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.s8"]
            pub(crate) fn wmma_load_b_s8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.row.stride.u8"]
            pub(crate) fn wmma_load_b_u8_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.sync.col.stride.u8"]
            pub(crate) fn wmma_load_b_u8_col_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 4];

            // Accumulator loads
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.c.sync.row.stride.f32"]
            pub(crate) fn wmma_load_c_f32_row_m16n16k16(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m16n16k16.load.c.sync.row.stride.s32"]
            pub(crate) fn wmma_load_c_s32_row_m16n16k16(ptr: *const u8, stride: i32) -> [i32; 8];
        }
    }

    // Store operations
    pub(crate) mod store {
        extern "C" {
            // f32 stores
            #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.row.stride.f32"]
            pub(crate) fn wmma_store_d_f32_row_m16n16k16(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.col.stride.f32"]
            pub(crate) fn wmma_store_d_f32_col_m16n16k16(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );

            // i32 stores
            #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.row.stride.s32"]
            pub(crate) fn wmma_store_d_s32_row_m16n16k16(
                ptr: *mut u8,
                d0: i32, d1: i32, d2: i32, d3: i32,
                d4: i32, d5: i32, d6: i32, d7: i32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.sync.col.stride.s32"]
            pub(crate) fn wmma_store_d_s32_col_m16n16k16(
                ptr: *mut u8,
                d0: i32, d1: i32, d2: i32, d3: i32,
                d4: i32, d5: i32, d6: i32, d7: i32,
                stride: i32,
            );
        }
    }

    // MMA operations
    pub(crate) mod mma {
        extern "C" {
            // f16 -> f16 operations
            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.f16.f16"]
            pub(crate) fn wmma_mma_f16_f16_row_row_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: i16, c1: i16, c2: i16, c3: i16,
                c4: i16, c5: i16, c6: i16, c7: i16,
            ) -> [i16; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.f16.f16"]
            pub(crate) fn wmma_mma_f16_f16_row_col_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: i16, c1: i16, c2: i16, c3: i16,
                c4: i16, c5: i16, c6: i16, c7: i16,
            ) -> [i16; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.col.row.f16.f16"]
            pub(crate) fn wmma_mma_f16_f16_col_row_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: i16, c1: i16, c2: i16, c3: i16,
                c4: i16, c5: i16, c6: i16, c7: i16,
            ) -> [i16; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.col.col.f16.f16"]
            pub(crate) fn wmma_mma_f16_f16_col_col_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: i16, c1: i16, c2: i16, c3: i16,
                c4: i16, c5: i16, c6: i16, c7: i16,
            ) -> [i16; 8];

            // f16 -> f32 operations
            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_row_row_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_row_col_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.col.row.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_col_row_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.col.col.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_col_col_m16n16k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            // i8/u8 -> i32 operations
            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.s8.s8.s32"]
            pub(crate) fn wmma_mma_s8_s32_row_row_m16n16k16(
                a0: i32, a1: i32, a2: i32, a3: i32,
                b0: i32, b1: i32, b2: i32, b3: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.s8.s8.s32"]
            pub(crate) fn wmma_mma_s8_s32_row_col_m16n16k16(
                a0: i32, a1: i32, a2: i32, a3: i32,
                b0: i32, b1: i32, b2: i32, b3: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.row.u8.u8.s32"]
            pub(crate) fn wmma_mma_u8_s32_row_row_m16n16k16(
                a0: i32, a1: i32, a2: i32, a3: i32,
                b0: i32, b1: i32, b2: i32, b3: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.sync.row.col.u8.u8.s32"]
            pub(crate) fn wmma_mma_u8_s32_row_col_m16n16k16(
                a0: i32, a1: i32, a2: i32, a3: i32,
                b0: i32, b1: i32, b2: i32, b3: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];
        }
    }
}

// ============= 16x8x16 shape =============
// NOTE: m16n8k16 only supports MMA operations, not WMMA load/store
mod m16n8k16 {
    // No WMMA load operations - this shape only supports MMA
    // The LLVM spec only defines MMA intrinsics for m16n8k16
    
    // No WMMA store operations - this shape only supports MMA

    // MMA operations
    pub(crate) mod mma {
        extern "C" {
            // f16 -> f32 operations
            #[link_name = "llvm.nvvm.wmma.m16n8k16.mma.sync.row.row.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_row_row_m16n8k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
            ) -> [f32; 4];

            // bf16 -> f32 operations
            #[link_name = "llvm.nvvm.wmma.m16n8k16.mma.sync.row.row.bf16.f32"]
            pub(crate) fn wmma_mma_bf16_f32_row_row_m16n8k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
            ) -> [f32; 4];

            // i8/u8 -> i32 operations
            #[link_name = "llvm.nvvm.wmma.m16n8k16.mma.sync.row.row.s8.s8.s32"]
            pub(crate) fn wmma_mma_s8_s32_row_row_m16n8k16(
                a0: i32, a1: i32,
                b0: i32, b1: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
            ) -> [i32; 4];

            #[link_name = "llvm.nvvm.wmma.m16n8k16.mma.sync.row.row.u8.u8.s32"]
            pub(crate) fn wmma_mma_u8_s32_row_row_m16n8k16(
                a0: i32, a1: i32,
                b0: i32, b1: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
            ) -> [i32; 4];
        }
    }
}

// ============= 32x8x16 shape =============
mod m32n8k16 {
    // Load operations
    pub(crate) mod load {
        extern "C" {
            // f16 loads
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.f16"]
            pub(crate) fn wmma_load_a_f16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.f16"]
            pub(crate) fn wmma_load_a_f16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.f16"]
            pub(crate) fn wmma_load_b_f16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.f16"]
            pub(crate) fn wmma_load_b_f16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];

            // bf16 loads
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.bf16"]
            pub(crate) fn wmma_load_a_bf16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.bf16"]
            pub(crate) fn wmma_load_a_bf16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.bf16"]
            pub(crate) fn wmma_load_b_bf16_row_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.bf16"]
            pub(crate) fn wmma_load_b_bf16_col_m32n8k16(ptr: *const u8, stride: i32) -> [i16; 8];

            // i8/u8 loads
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.s8"]
            pub(crate) fn wmma_load_a_s8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.s8"]
            pub(crate) fn wmma_load_a_s8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.row.stride.u8"]
            pub(crate) fn wmma_load_a_u8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.a.sync.col.stride.u8"]
            pub(crate) fn wmma_load_a_u8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.s8"]
            pub(crate) fn wmma_load_b_s8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.s8"]
            pub(crate) fn wmma_load_b_s8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.row.stride.u8"]
            pub(crate) fn wmma_load_b_u8_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.b.sync.col.stride.u8"]
            pub(crate) fn wmma_load_b_u8_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 2];

            // Accumulator loads
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.row.stride.f32"]
            pub(crate) fn wmma_load_c_f32_row_m32n8k16(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.col.stride.f32"]
            pub(crate) fn wmma_load_c_f32_col_m32n8k16(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.row.stride.s32"]
            pub(crate) fn wmma_load_c_s32_row_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 8];
            #[link_name = "llvm.nvvm.wmma.m32n8k16.load.c.sync.col.stride.s32"]
            pub(crate) fn wmma_load_c_s32_col_m32n8k16(ptr: *const u8, stride: i32) -> [i32; 8];
        }
    }

    // Store operations
    pub(crate) mod store {
        extern "C" {
            // f32 stores
            #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.row.stride.f32"]
            pub(crate) fn wmma_store_d_f32_row_m32n8k16(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.col.stride.f32"]
            pub(crate) fn wmma_store_d_f32_col_m32n8k16(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );

            // i32 stores
            #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.row.stride.s32"]
            pub(crate) fn wmma_store_d_s32_row_m32n8k16(
                ptr: *mut u8,
                d0: i32, d1: i32, d2: i32, d3: i32,
                d4: i32, d5: i32, d6: i32, d7: i32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m32n8k16.store.d.sync.col.stride.s32"]
            pub(crate) fn wmma_store_d_s32_col_m32n8k16(
                ptr: *mut u8,
                d0: i32, d1: i32, d2: i32, d3: i32,
                d4: i32, d5: i32, d6: i32, d7: i32,
                stride: i32,
            );
        }
    }

    // MMA operations
    pub(crate) mod mma {
        extern "C" {
            // f16 -> f32 operations
            #[link_name = "llvm.nvvm.wmma.m32n8k16.mma.sync.row.row.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_row_row_m32n8k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                a8: i16, a9: i16, a10: i16, a11: i16,
                a12: i16, a13: i16, a14: i16, a15: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            // i8/u8 -> i32 operations
            #[link_name = "llvm.nvvm.wmma.m32n8k16.mma.sync.row.row.s8.s8.s32"]
            pub(crate) fn wmma_mma_s8_s32_row_row_m32n8k16(
                a0: i32, a1: i32, a2: i32, a3: i32,
                b0: i32, b1: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];

            #[link_name = "llvm.nvvm.wmma.m32n8k16.mma.sync.row.row.u8.u8.s32"]
            pub(crate) fn wmma_mma_u8_s32_row_row_m32n8k16(
                a0: i32, a1: i32, a2: i32, a3: i32,
                b0: i32, b1: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];
        }
    }
}

// ============= 8x32x16 shape =============
mod m8n32k16 {
    // Load operations
    pub(crate) mod load {
        extern "C" {
            // f16 loads
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.f16"]
            pub(crate) fn wmma_load_a_f16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.f16"]
            pub(crate) fn wmma_load_a_f16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.f16"]
            pub(crate) fn wmma_load_b_f16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.f16"]
            pub(crate) fn wmma_load_b_f16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];

            // bf16 loads
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.bf16"]
            pub(crate) fn wmma_load_a_bf16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.bf16"]
            pub(crate) fn wmma_load_a_bf16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.bf16"]
            pub(crate) fn wmma_load_b_bf16_row_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.bf16"]
            pub(crate) fn wmma_load_b_bf16_col_m8n32k16(ptr: *const u8, stride: i32) -> [i16; 16];

            // i8/u8 loads
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.s8"]
            pub(crate) fn wmma_load_a_s8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.s8"]
            pub(crate) fn wmma_load_a_s8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.row.stride.u8"]
            pub(crate) fn wmma_load_a_u8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.a.sync.col.stride.u8"]
            pub(crate) fn wmma_load_a_u8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 2];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.s8"]
            pub(crate) fn wmma_load_b_s8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.s8"]
            pub(crate) fn wmma_load_b_s8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.row.stride.u8"]
            pub(crate) fn wmma_load_b_u8_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.b.sync.col.stride.u8"]
            pub(crate) fn wmma_load_b_u8_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 4];

            // Accumulator loads
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.row.stride.f32"]
            pub(crate) fn wmma_load_c_f32_row_m8n32k16(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.col.stride.f32"]
            pub(crate) fn wmma_load_c_f32_col_m8n32k16(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.row.stride.s32"]
            pub(crate) fn wmma_load_c_s32_row_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 8];
            #[link_name = "llvm.nvvm.wmma.m8n32k16.load.c.sync.col.stride.s32"]
            pub(crate) fn wmma_load_c_s32_col_m8n32k16(ptr: *const u8, stride: i32) -> [i32; 8];
        }
    }

    // Store operations
    pub(crate) mod store {
        extern "C" {
            // f32 stores
            #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.row.stride.f32"]
            pub(crate) fn wmma_store_d_f32_row_m8n32k16(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.col.stride.f32"]
            pub(crate) fn wmma_store_d_f32_col_m8n32k16(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );

            // i32 stores
            #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.row.stride.s32"]
            pub(crate) fn wmma_store_d_s32_row_m8n32k16(
                ptr: *mut u8,
                d0: i32, d1: i32, d2: i32, d3: i32,
                d4: i32, d5: i32, d6: i32, d7: i32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m8n32k16.store.d.sync.col.stride.s32"]
            pub(crate) fn wmma_store_d_s32_col_m8n32k16(
                ptr: *mut u8,
                d0: i32, d1: i32, d2: i32, d3: i32,
                d4: i32, d5: i32, d6: i32, d7: i32,
                stride: i32,
            );
        }
    }

    // MMA operations
    pub(crate) mod mma {
        extern "C" {
            // f16 -> f32 operations
            #[link_name = "llvm.nvvm.wmma.m8n32k16.mma.sync.row.row.f16.f32"]
            pub(crate) fn wmma_mma_f16_f32_row_row_m8n32k16(
                a0: i16, a1: i16, a2: i16, a3: i16,
                a4: i16, a5: i16, a6: i16, a7: i16,
                b0: i16, b1: i16, b2: i16, b3: i16,
                b4: i16, b5: i16, b6: i16, b7: i16,
                b8: i16, b9: i16, b10: i16, b11: i16,
                b12: i16, b13: i16, b14: i16, b15: i16,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            // i8/u8 -> i32 operations
            #[link_name = "llvm.nvvm.wmma.m8n32k16.mma.sync.row.row.s8.s8.s32"]
            pub(crate) fn wmma_mma_s8_s32_row_row_m8n32k16(
                a0: i32, a1: i32,
                b0: i32, b1: i32, b2: i32, b3: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];

            #[link_name = "llvm.nvvm.wmma.m8n32k16.mma.sync.row.row.u8.u8.s32"]
            pub(crate) fn wmma_mma_u8_s32_row_row_m8n32k16(
                a0: i32, a1: i32,
                b0: i32, b1: i32, b2: i32, b3: i32,
                c0: i32, c1: i32, c2: i32, c3: i32,
                c4: i32, c5: i32, c6: i32, c7: i32,
            ) -> [i32; 8];
        }
    }
}

// ============= 8x8x4 shape (f64) =============
mod m8n8k4 {
    // Load operations
    pub(crate) mod load {
        extern "C" {
            // f64 loads
            #[link_name = "llvm.nvvm.wmma.m8n8k4.load.a.sync.row.stride.f64"]
            pub(crate) fn wmma_load_a_f64_row_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
            #[link_name = "llvm.nvvm.wmma.m8n8k4.load.a.sync.col.stride.f64"]
            pub(crate) fn wmma_load_a_f64_col_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
            #[link_name = "llvm.nvvm.wmma.m8n8k4.load.b.sync.row.stride.f64"]
            pub(crate) fn wmma_load_b_f64_row_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
            #[link_name = "llvm.nvvm.wmma.m8n8k4.load.b.sync.col.stride.f64"]
            pub(crate) fn wmma_load_b_f64_col_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];

            // Accumulator loads
            #[link_name = "llvm.nvvm.wmma.m8n8k4.load.c.sync.row.stride.f64"]
            pub(crate) fn wmma_load_c_f64_row_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
            #[link_name = "llvm.nvvm.wmma.m8n8k4.load.c.sync.col.stride.f64"]
            pub(crate) fn wmma_load_c_f64_col_m8n8k4(ptr: *const u8, stride: i32) -> [f64; 2];
        }
    }

    // Store operations
    pub(crate) mod store {
        extern "C" {
            // f64 stores
            #[link_name = "llvm.nvvm.wmma.m8n8k4.store.d.sync.row.stride.f64"]
            pub(crate) fn wmma_store_d_f64_row_m8n8k4(
                ptr: *mut u8, d0: f64, d1: f64, stride: i32
            );
            #[link_name = "llvm.nvvm.wmma.m8n8k4.store.d.sync.col.stride.f64"]
            pub(crate) fn wmma_store_d_f64_col_m8n8k4(
                ptr: *mut u8, d0: f64, d1: f64, stride: i32
            );
        }
    }

    // MMA operations
    pub(crate) mod mma {
        extern "C" {
            // f64 operations
            #[link_name = "llvm.nvvm.wmma.m8n8k4.mma.sync.row.row.f64"]
            pub(crate) fn wmma_mma_f64_row_row_m8n8k4(
                a0: f64, a1: f64,
                b0: f64, b1: f64,
                c0: f64, c1: f64,
            ) -> [f64; 2];
        }
    }
}

// ============= 16x16x8 shape (TF32) =============
mod m16n16k8 {
    // Conversion operations
    pub(crate) mod convert {
        extern "C" {
            #[link_name = "llvm.nvvm.f2tf32.rna.f32"]
            pub(crate) fn float_to_tf32(x: f32) -> f32;
        }
    }

    // Load operations
    pub(crate) mod load {
        extern "C" {
            // tf32 loads
            #[link_name = "llvm.nvvm.wmma.m16n16k8.load.a.sync.row.stride.tf32"]
            pub(crate) fn wmma_load_a_tf32_row_m16n16k8(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m16n16k8.load.a.sync.col.stride.tf32"]
            pub(crate) fn wmma_load_a_tf32_col_m16n16k8(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m16n16k8.load.b.sync.row.stride.tf32"]
            pub(crate) fn wmma_load_b_tf32_row_m16n16k8(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m16n16k8.load.b.sync.col.stride.tf32"]
            pub(crate) fn wmma_load_b_tf32_col_m16n16k8(ptr: *const u8, stride: i32) -> [f32; 8];

            // Accumulator loads (f32)
            #[link_name = "llvm.nvvm.wmma.m16n16k8.load.c.sync.row.stride.f32"]
            pub(crate) fn wmma_load_c_f32_row_m16n16k8(ptr: *const u8, stride: i32) -> [f32; 8];
            #[link_name = "llvm.nvvm.wmma.m16n16k8.load.c.sync.col.stride.f32"]
            pub(crate) fn wmma_load_c_f32_col_m16n16k8(ptr: *const u8, stride: i32) -> [f32; 8];
        }
    }

    // Store operations
    pub(crate) mod store {
        extern "C" {
            // f32 stores
            #[link_name = "llvm.nvvm.wmma.m16n16k8.store.d.sync.row.stride.f32"]
            pub(crate) fn wmma_store_d_f32_row_m16n16k8(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );
            #[link_name = "llvm.nvvm.wmma.m16n16k8.store.d.sync.col.stride.f32"]
            pub(crate) fn wmma_store_d_f32_col_m16n16k8(
                ptr: *mut u8,
                d0: f32, d1: f32, d2: f32, d3: f32,
                d4: f32, d5: f32, d6: f32, d7: f32,
                stride: i32,
            );
        }
    }

    // MMA operations
    pub(crate) mod mma {
        extern "C" {
            // tf32 -> f32 operations
            #[link_name = "llvm.nvvm.wmma.m16n16k8.mma.sync.row.row.tf32.f32"]
            pub(crate) fn wmma_mma_tf32_f32_row_row_m16n16k8(
                a0: f32, a1: f32, a2: f32, a3: f32,
                a4: f32, a5: f32, a6: f32, a7: f32,
                b0: f32, b1: f32, b2: f32, b3: f32,
                b4: f32, b5: f32, b6: f32, b7: f32,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k8.mma.sync.row.col.tf32.f32"]
            pub(crate) fn wmma_mma_tf32_f32_row_col_m16n16k8(
                a0: f32, a1: f32, a2: f32, a3: f32,
                a4: f32, a5: f32, a6: f32, a7: f32,
                b0: f32, b1: f32, b2: f32, b3: f32,
                b4: f32, b5: f32, b6: f32, b7: f32,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k8.mma.sync.col.row.tf32.f32"]
            pub(crate) fn wmma_mma_tf32_f32_col_row_m16n16k8(
                a0: f32, a1: f32, a2: f32, a3: f32,
                a4: f32, a5: f32, a6: f32, a7: f32,
                b0: f32, b1: f32, b2: f32, b3: f32,
                b4: f32, b5: f32, b6: f32, b7: f32,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];

            #[link_name = "llvm.nvvm.wmma.m16n16k8.mma.sync.col.col.tf32.f32"]
            pub(crate) fn wmma_mma_tf32_f32_col_col_m16n16k8(
                a0: f32, a1: f32, a2: f32, a3: f32,
                a4: f32, a5: f32, a6: f32, a7: f32,
                b0: f32, b1: f32, b2: f32, b3: f32,
                b4: f32, b5: f32, b6: f32, b7: f32,
                c0: f32, c1: f32, c2: f32, c3: f32,
                c4: f32, c5: f32, c6: f32, c7: f32,
            ) -> [f32; 8];
        }
    }
}

// Re-export all intrinsics flat
pub(crate) use m16n16k16::load::*;
pub(crate) use m16n16k16::store::*;
pub(crate) use m16n16k16::mma::*;

// m16n8k16 has no load/store operations (MMA-only)
pub(crate) use m16n8k16::mma::*;

pub(crate) use m32n8k16::load::*;
pub(crate) use m32n8k16::store::*;
pub(crate) use m32n8k16::mma::*;

pub(crate) use m8n32k16::load::*;
pub(crate) use m8n32k16::store::*;
pub(crate) use m8n32k16::mma::*;

pub(crate) use m8n8k4::load::*;
pub(crate) use m8n8k4::store::*;
pub(crate) use m8n8k4::mma::*;

pub(crate) use m16n16k8::convert::*;
pub(crate) use m16n16k8::load::*;
pub(crate) use m16n16k8::store::*;
pub(crate) use m16n16k8::mma::*;

// ============= ldmatrix intrinsics =============
// These load matrix fragments from shared memory for MMA operations
pub(crate) mod ldmatrix {
    #[allow(dead_code)]
    extern "C" {
        // 8x8 matrix with 16-bit elements (bf16/f16)
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16"]
        pub(crate) fn ldmatrix_m8n8_x1_b16(ptr: *const u8) -> i32;
        
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16"]
        pub(crate) fn ldmatrix_m8n8_x2_b16(ptr: *const u8) -> [i32; 2];
        
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16"]
        pub(crate) fn ldmatrix_m8n8_x4_b16(ptr: *const u8) -> [i32; 4];
        
        // With transpose
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16"]
        pub(crate) fn ldmatrix_m8n8_x1_trans_b16(ptr: *const u8) -> i32;
        
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16"]
        pub(crate) fn ldmatrix_m8n8_x2_trans_b16(ptr: *const u8) -> [i32; 2];
        
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16"]
        pub(crate) fn ldmatrix_m8n8_x4_trans_b16(ptr: *const u8) -> [i32; 4];
        
        // 16x16 matrix with 8-bit elements
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.b8"]
        pub(crate) fn ldmatrix_m16n16_x1_b8(ptr: *const u8) -> [i32; 2];
        
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.b8"]
        pub(crate) fn ldmatrix_m16n16_x2_b8(ptr: *const u8) -> [i32; 4];
        
        // 16x16 with transpose (mandatory for 16x16)
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8"]
        pub(crate) fn ldmatrix_m16n16_x1_trans_b8(ptr: *const u8) -> [i32; 2];
        
        #[link_name = "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8"]
        pub(crate) fn ldmatrix_m16n16_x2_trans_b8(ptr: *const u8) -> [i32; 4];
    }
}

pub(crate) use ldmatrix::*;
