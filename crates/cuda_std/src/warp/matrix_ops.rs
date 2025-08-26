//! Type-safe matrix operations implementation using trait-based dispatch

use super::*;

// Trait for loading matrix A fragments
pub trait LoadMatrixA<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32];
}

// Trait for loading matrix B fragments  
pub trait LoadMatrixB<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32];
}

// Trait for MMA operations
pub trait MatrixMMA<Shape: TensorCoreShape, LA: Layout, LB: Layout>: MatrixElement {
    unsafe fn mma(
        a: &[Self::Storage],
        b: &[Self::Storage], 
        c: &[<Self::Accumulator as AccumulatorElement>::Storage],
    ) -> [<Self::Accumulator as AccumulatorElement>::Storage; 8];
}

// ============= 16x16x16 implementations =============

// f16 for 16x16x16
impl<L: Layout> LoadMatrixA<dims::Shape<16, 16, 16>, L> for f16 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [f16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_a_f16_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_f16_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..16].copy_from_slice(&core::mem::transmute::<[i16; 16], [f16; 16]>(raw));
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<16, 16, 16>, L> for f16 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [f16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_b_f16_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_f16_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..16].copy_from_slice(&core::mem::transmute::<[i16; 16], [f16; 16]>(raw));
        result
    }
}

// bf16 for 16x16x16 (uses f16 intrinsics)
impl<L: Layout> LoadMatrixA<dims::Shape<16, 16, 16>, L> for bf16 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [bf16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_a_f16_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_f16_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..16].copy_from_slice(&core::mem::transmute::<[i16; 16], [bf16; 16]>(raw));
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<16, 16, 16>, L> for bf16 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [bf16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_b_f16_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_f16_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..16].copy_from_slice(&core::mem::transmute::<[i16; 16], [bf16; 16]>(raw));
        result
    }
}

// i8 for 16x16x16
impl<L: Layout> LoadMatrixA<dims::Shape<16, 16, 16>, L> for i8 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_a_s8_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_s8_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<16, 16, 16>, L> for i8 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_b_s8_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_s8_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

// u8 for 16x16x16
impl<L: Layout> LoadMatrixA<dims::Shape<16, 16, 16>, L> for u8 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_a_u8_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_u8_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<16, 16, 16>, L> for u8 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_b_u8_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_u8_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

// bool for 16x16x16 (uses u8 intrinsics)
impl<L: Layout> LoadMatrixA<dims::Shape<16, 16, 16>, L> for bool {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_a_u8_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_u8_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<16, 16, 16>, L> for bool {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_b_u8_row_m16n16k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_u8_col_m16n16k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

// ============= 32x8x16 implementations =============

// f16 for 32x8x16
impl<L: Layout> LoadMatrixA<dims::Shape<32, 8, 16>, L> for f16 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [f16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_a_f16_row_m32n8k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_f16_col_m32n8k16(ptr as *const u8, stride)
        };
        result[..16].copy_from_slice(&core::mem::transmute::<[i16; 16], [f16; 16]>(raw));
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<32, 8, 16>, L> for f16 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [f16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_b_f16_row_m32n8k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_f16_col_m32n8k16(ptr as *const u8, stride)
        };
        result[..8].copy_from_slice(&core::mem::transmute::<[i16; 8], [f16; 8]>(raw));
        result
    }
}

// i8 for 32x8x16
impl<L: Layout> LoadMatrixA<dims::Shape<32, 8, 16>, L> for i8 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_a_s8_row_m32n8k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_s8_col_m32n8k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<32, 8, 16>, L> for i8 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_b_s8_row_m32n8k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_s8_col_m32n8k16(ptr as *const u8, stride)
        };
        result[..2].copy_from_slice(&raw);
        result
    }
}

// ============= 8x32x16 implementations =============

// f16 for 8x32x16
impl<L: Layout> LoadMatrixA<dims::Shape<8, 32, 16>, L> for f16 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [f16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_a_f16_row_m8n32k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_f16_col_m8n32k16(ptr as *const u8, stride)
        };
        result[..8].copy_from_slice(&core::mem::transmute::<[i16; 8], [f16; 8]>(raw));
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<8, 32, 16>, L> for f16 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [f16::from_f32(0.0); 32];
        let raw = if is_row_major {
            wmma_load_b_f16_row_m8n32k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_f16_col_m8n32k16(ptr as *const u8, stride)
        };
        result[..16].copy_from_slice(&core::mem::transmute::<[i16; 16], [f16; 16]>(raw));
        result
    }
}

// i8 for 8x32x16
impl<L: Layout> LoadMatrixA<dims::Shape<8, 32, 16>, L> for i8 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_a_s8_row_m8n32k16(ptr as *const u8, stride)
        } else {
            wmma_load_a_s8_col_m8n32k16(ptr as *const u8, stride)
        };
        result[..2].copy_from_slice(&raw);
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<8, 32, 16>, L> for i8 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0i32; 32];
        let raw = if is_row_major {
            wmma_load_b_s8_row_m8n32k16(ptr as *const u8, stride)
        } else {
            wmma_load_b_s8_col_m8n32k16(ptr as *const u8, stride)
        };
        result[..4].copy_from_slice(&raw);
        result
    }
}

// ============= 8x8x4 implementations (f64) =============

impl<L: Layout> LoadMatrixA<dims::Shape<8, 8, 4>, L> for f64 {
    #[gpu_only]
    unsafe fn load_a(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0.0f64; 32];
        let raw = if is_row_major {
            wmma_load_a_f64_row_m8n8k4(ptr as *const u8, stride)
        } else {
            wmma_load_a_f64_col_m8n8k4(ptr as *const u8, stride)
        };
        result[..2].copy_from_slice(&raw);
        result
    }
}

impl<L: Layout> LoadMatrixB<dims::Shape<8, 8, 4>, L> for f64 {
    #[gpu_only]
    unsafe fn load_b(ptr: *const Self, stride: i32, is_row_major: bool) -> [Self::Storage; 32] {
        let mut result = [0.0f64; 32];
        let raw = if is_row_major {
            wmma_load_b_f64_row_m8n8k4(ptr as *const u8, stride)
        } else {
            wmma_load_b_f64_col_m8n8k4(ptr as *const u8, stride)
        };
        result[..2].copy_from_slice(&raw);
        result
    }
}

// MMA operations for f16 -> f32 (16x16x16)
impl<LA: Layout, LB: Layout> MatrixMMA<dims::Shape<16, 16, 16>, LA, LB> for f16 {
    #[gpu_only]
    unsafe fn mma(
        a: &[Self::Storage],
        b: &[Self::Storage],
        c: &[<Self::Accumulator as AccumulatorElement>::Storage],
    ) -> [<Self::Accumulator as AccumulatorElement>::Storage; 8] {
        let a_data = core::mem::transmute::<&[f16], &[i16]>(&a[..16]);
        let b_data = core::mem::transmute::<&[f16], &[i16]>(&b[..16]);
        
        if LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
            wmma_mma_f16_f32_row_row_m16n16k16(
                a_data[0], a_data[1], a_data[2], a_data[3], a_data[4], a_data[5], a_data[6], a_data[7],
                a_data[8], a_data[9], a_data[10], a_data[11], a_data[12], a_data[13], a_data[14], a_data[15],
                b_data[0], b_data[1], b_data[2], b_data[3], b_data[4], b_data[5], b_data[6], b_data[7],
                b_data[8], b_data[9], b_data[10], b_data[11], b_data[12], b_data[13], b_data[14], b_data[15],
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            )
        } else if LA::IS_ROW_MAJOR && !LB::IS_ROW_MAJOR {
            wmma_mma_f16_f32_row_col_m16n16k16(
                a_data[0], a_data[1], a_data[2], a_data[3], a_data[4], a_data[5], a_data[6], a_data[7],
                a_data[8], a_data[9], a_data[10], a_data[11], a_data[12], a_data[13], a_data[14], a_data[15],
                b_data[0], b_data[1], b_data[2], b_data[3], b_data[4], b_data[5], b_data[6], b_data[7],
                b_data[8], b_data[9], b_data[10], b_data[11], b_data[12], b_data[13], b_data[14], b_data[15],
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            )
        } else if !LA::IS_ROW_MAJOR && LB::IS_ROW_MAJOR {
            wmma_mma_f16_f32_col_row_m16n16k16(
                a_data[0], a_data[1], a_data[2], a_data[3], a_data[4], a_data[5], a_data[6], a_data[7],
                a_data[8], a_data[9], a_data[10], a_data[11], a_data[12], a_data[13], a_data[14], a_data[15],
                b_data[0], b_data[1], b_data[2], b_data[3], b_data[4], b_data[5], b_data[6], b_data[7],
                b_data[8], b_data[9], b_data[10], b_data[11], b_data[12], b_data[13], b_data[14], b_data[15],
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            )
        } else {
            wmma_mma_f16_f32_col_col_m16n16k16(
                a_data[0], a_data[1], a_data[2], a_data[3], a_data[4], a_data[5], a_data[6], a_data[7],
                a_data[8], a_data[9], a_data[10], a_data[11], a_data[12], a_data[13], a_data[14], a_data[15],
                b_data[0], b_data[1], b_data[2], b_data[3], b_data[4], b_data[5], b_data[6], b_data[7],
                b_data[8], b_data[9], b_data[10], b_data[11], b_data[12], b_data[13], b_data[14], b_data[15],
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            )
        }
    }
}

// Add similar MMA implementations for other types and shapes...