//! Trait-based dispatch for matrix operations
//! This module provides compile-time dispatch for WMMA operations
//! based on element type and matrix shape combinations.

use super::*;
use super::layout::{Row, Col, Layout};

// Import all the intrinsic functions from the parent module
#[cfg(target_arch = "nvptx64")]
use super::{
    // 16x16x16 intrinsics
    wmma_load_a_f16_row_m16n16k16, wmma_load_a_f16_col_m16n16k16,
    wmma_load_b_f16_row_m16n16k16, wmma_load_b_f16_col_m16n16k16,
    wmma_load_a_bf16_row_m16n16k16, wmma_load_a_bf16_col_m16n16k16,
    wmma_load_b_bf16_row_m16n16k16, wmma_load_b_bf16_col_m16n16k16,
    wmma_load_a_s8_row_m16n16k16, wmma_load_a_s8_col_m16n16k16,
    wmma_load_b_s8_row_m16n16k16, wmma_load_b_s8_col_m16n16k16,
    wmma_load_a_u8_row_m16n16k16, wmma_load_a_u8_col_m16n16k16,
    wmma_load_b_u8_row_m16n16k16, wmma_load_b_u8_col_m16n16k16,
    wmma_load_c_f32_row_m16n16k16, wmma_load_c_s32_row_m16n16k16,
    wmma_store_d_f32_row_m16n16k16, wmma_store_d_f32_col_m16n16k16,
    wmma_store_d_s32_row_m16n16k16, wmma_store_d_s32_col_m16n16k16,
    
    // 32x8x16 intrinsics
    wmma_load_a_f16_row_m32n8k16, wmma_load_a_f16_col_m32n8k16,
    wmma_load_b_f16_row_m32n8k16, wmma_load_b_f16_col_m32n8k16,
    wmma_load_a_bf16_row_m32n8k16, wmma_load_a_bf16_col_m32n8k16,
    wmma_load_b_bf16_row_m32n8k16, wmma_load_b_bf16_col_m32n8k16,
    wmma_load_a_s8_row_m32n8k16, wmma_load_a_s8_col_m32n8k16,
    wmma_load_b_s8_row_m32n8k16, wmma_load_b_s8_col_m32n8k16,
    wmma_load_a_u8_row_m32n8k16, wmma_load_a_u8_col_m32n8k16,
    wmma_load_b_u8_row_m32n8k16, wmma_load_b_u8_col_m32n8k16,
    wmma_load_c_f32_row_m32n8k16, wmma_load_c_f32_col_m32n8k16,
    wmma_load_c_s32_row_m32n8k16, wmma_load_c_s32_col_m32n8k16,
    wmma_store_d_f32_row_m32n8k16, wmma_store_d_f32_col_m32n8k16,
    wmma_store_d_s32_row_m32n8k16, wmma_store_d_s32_col_m32n8k16,
    
    // 8x32x16 intrinsics
    wmma_load_a_f16_row_m8n32k16, wmma_load_a_f16_col_m8n32k16,
    wmma_load_b_f16_row_m8n32k16, wmma_load_b_f16_col_m8n32k16,
    wmma_load_a_bf16_row_m8n32k16, wmma_load_a_bf16_col_m8n32k16,
    wmma_load_b_bf16_row_m8n32k16, wmma_load_b_bf16_col_m8n32k16,
    wmma_load_a_s8_row_m8n32k16, wmma_load_a_s8_col_m8n32k16,
    wmma_load_b_s8_row_m8n32k16, wmma_load_b_s8_col_m8n32k16,
    wmma_load_a_u8_row_m8n32k16, wmma_load_a_u8_col_m8n32k16,
    wmma_load_b_u8_row_m8n32k16, wmma_load_b_u8_col_m8n32k16,
    wmma_load_c_f32_row_m8n32k16, wmma_load_c_f32_col_m8n32k16,
    wmma_load_c_s32_row_m8n32k16, wmma_load_c_s32_col_m8n32k16,
    wmma_store_d_f32_row_m8n32k16, wmma_store_d_f32_col_m8n32k16,
    wmma_store_d_s32_row_m8n32k16, wmma_store_d_s32_col_m8n32k16,
    
    // 8x8x4 intrinsics (f64)
    wmma_load_a_f64_row_m8n8k4, wmma_load_a_f64_col_m8n8k4,
    wmma_load_b_f64_row_m8n8k4, wmma_load_b_f64_col_m8n8k4,
    wmma_load_c_f64_row_m8n8k4, wmma_load_c_f64_col_m8n8k4,
    wmma_store_d_f64_row_m8n8k4, wmma_store_d_f64_col_m8n8k4,
};

// Import shape types
type Shape16x16x16 = super::dims::Shape<16, 16, 16>;
type Shape32x8x16 = super::dims::Shape<32, 8, 16>;
type Shape8x32x16 = super::dims::Shape<8, 32, 16>;
type Shape8x8x4 = super::dims::Shape<8, 8, 4>;

// Trait for loading matrix A fragments with layout in type system
pub trait LoadMatrixA<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for loading matrix B fragments with layout in type system
pub trait LoadMatrixB<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for loading accumulator matrices with layout in type system
pub trait LoadMatrixC<Shape: TensorCoreShape, L: Layout>: AccumulatorElement {
    unsafe fn load_c_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for storing accumulator matrices with layout in type system
pub trait StoreMatrixD<Shape: TensorCoreShape, L: Layout>: AccumulatorElement {
    unsafe fn store_d_from(ptr: *mut u8, data: &[Self::Storage; 32], stride: i32);
}

// Macro to implement LoadMatrixA for both layouts
macro_rules! impl_load_a {
    ($type:ty, $shape:ty, $count:literal, $row_fn:ident, $col_fn:ident) => {
        impl LoadMatrixA<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [<$type as MatrixElement>::Storage; 32]) {
                let raw = $row_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] = core::mem::transmute(raw);
                out[..$count].copy_from_slice(&storage);
            }
        }
        
        impl LoadMatrixA<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [<$type as MatrixElement>::Storage; 32]) {
                let raw = $col_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] = core::mem::transmute(raw);
                out[..$count].copy_from_slice(&storage);
            }
        }
    };
}

// Macro to implement LoadMatrixB for both layouts
macro_rules! impl_load_b {
    ($type:ty, $shape:ty, $count:literal, $row_fn:ident, $col_fn:ident) => {
        impl LoadMatrixB<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [<$type as MatrixElement>::Storage; 32]) {
                let raw = $row_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] = core::mem::transmute(raw);
                out[..$count].copy_from_slice(&storage);
            }
        }
        
        impl LoadMatrixB<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [<$type as MatrixElement>::Storage; 32]) {
                let raw = $col_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] = core::mem::transmute(raw);
                out[..$count].copy_from_slice(&storage);
            }
        }
    };
}

// Macro to implement LoadMatrixC for both layouts
macro_rules! impl_load_c {
    ($type:ty, $shape:ty, $count:literal, $row_fn:ident, $col_fn:ident) => {
        impl LoadMatrixC<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn load_c_into(ptr: *const u8, stride: i32, out: &mut [<$type as AccumulatorElement>::Storage; 32]) {
                let result = $row_fn(ptr, stride);
                out[..$count].copy_from_slice(&result);
            }
        }
        
        impl LoadMatrixC<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_c_into(ptr: *const u8, stride: i32, out: &mut [<$type as AccumulatorElement>::Storage; 32]) {
                let result = $col_fn(ptr, stride);
                out[..$count].copy_from_slice(&result);
            }
        }
    };
    // Handle case where col function doesn't exist (uses row)
    ($type:ty, $shape:ty, $count:literal, $row_fn:ident) => {
        impl LoadMatrixC<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn load_c_into(ptr: *const u8, stride: i32, out: &mut [<$type as AccumulatorElement>::Storage; 32]) {
                let result = $row_fn(ptr, stride);
                out[..$count].copy_from_slice(&result);
            }
        }
        
        impl LoadMatrixC<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_c_into(ptr: *const u8, stride: i32, out: &mut [<$type as AccumulatorElement>::Storage; 32]) {
                let result = $row_fn(ptr, stride); // Use row function for col as well
                out[..$count].copy_from_slice(&result);
            }
        }
    };
}

// Macro to implement StoreMatrixD for both layouts
macro_rules! impl_store_d {
    ($type:ty, $shape:ty, $row_fn:ident, $col_fn:ident) => {
        impl StoreMatrixD<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn store_d_from(ptr: *mut u8, data: &[<$type as AccumulatorElement>::Storage; 32], stride: i32) {
                $row_fn(
                    ptr,
                    data[0], data[1], data[2], data[3],
                    data[4], data[5], data[6], data[7],
                    stride
                );
            }
        }
        
        impl StoreMatrixD<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn store_d_from(ptr: *mut u8, data: &[<$type as AccumulatorElement>::Storage; 32], stride: i32) {
                $col_fn(
                    ptr,
                    data[0], data[1], data[2], data[3],
                    data[4], data[5], data[6], data[7],
                    stride
                );
            }
        }
    };
    // Special case for f64 which only uses 2 values
    ($type:ty, $shape:ty, $row_fn:ident, $col_fn:ident, f64) => {
        impl StoreMatrixD<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn store_d_from(ptr: *mut u8, data: &[<$type as AccumulatorElement>::Storage; 32], stride: i32) {
                $row_fn(ptr, data[0], data[1], stride);
            }
        }
        
        impl StoreMatrixD<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn store_d_from(ptr: *mut u8, data: &[<$type as AccumulatorElement>::Storage; 32], stride: i32) {
                $col_fn(ptr, data[0], data[1], stride);
            }
        }
    };
}

// ============= f16 16x16x16 implementations =============
impl_load_a!(f16, Shape16x16x16, 16, wmma_load_a_f16_row_m16n16k16, wmma_load_a_f16_col_m16n16k16);
impl_load_b!(f16, Shape16x16x16, 16, wmma_load_b_f16_row_m16n16k16, wmma_load_b_f16_col_m16n16k16);

// ============= bf16 16x16x16 implementations =============
impl_load_a!(bf16, Shape16x16x16, 16, wmma_load_a_bf16_row_m16n16k16, wmma_load_a_bf16_col_m16n16k16);
impl_load_b!(bf16, Shape16x16x16, 16, wmma_load_b_bf16_row_m16n16k16, wmma_load_b_bf16_col_m16n16k16);

// ============= i8 16x16x16 implementations =============
impl_load_a!(i8, Shape16x16x16, 4, wmma_load_a_s8_row_m16n16k16, wmma_load_a_s8_col_m16n16k16);
impl_load_b!(i8, Shape16x16x16, 4, wmma_load_b_s8_row_m16n16k16, wmma_load_b_s8_col_m16n16k16);

// ============= u8 16x16x16 implementations =============
impl_load_a!(u8, Shape16x16x16, 4, wmma_load_a_u8_row_m16n16k16, wmma_load_a_u8_col_m16n16k16);
impl_load_b!(u8, Shape16x16x16, 4, wmma_load_b_u8_row_m16n16k16, wmma_load_b_u8_col_m16n16k16);

// ============= bool 16x16x16 implementations =============
impl LoadMatrixA<Shape16x16x16, Row> for bool {
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_a_u8_row_m16n16k16(ptr, stride);
        // bool uses u8 storage but u8 intrinsics return i32 array
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

impl LoadMatrixA<Shape16x16x16, Col> for bool {
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_a_u8_col_m16n16k16(ptr, stride);
        // bool uses u8 storage but u8 intrinsics return i32 array
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

impl LoadMatrixB<Shape16x16x16, Row> for bool {
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_b_u8_row_m16n16k16(ptr, stride);
        // bool uses u8 storage but u8 intrinsics return i32 array
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

impl LoadMatrixB<Shape16x16x16, Col> for bool {
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_b_u8_col_m16n16k16(ptr, stride);
        // bool uses u8 storage but u8 intrinsics return i32 array
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

// ============= f32 accumulator 16x16x16 implementations =============
impl_load_c!(f32, Shape16x16x16, 8, wmma_load_c_f32_row_m16n16k16); // No col variant in intrinsics
impl_store_d!(f32, Shape16x16x16, wmma_store_d_f32_row_m16n16k16, wmma_store_d_f32_col_m16n16k16);

// ============= i32 accumulator 16x16x16 implementations =============
impl_load_c!(i32, Shape16x16x16, 8, wmma_load_c_s32_row_m16n16k16); // No col variant in intrinsics
impl_store_d!(i32, Shape16x16x16, wmma_store_d_s32_row_m16n16k16, wmma_store_d_s32_col_m16n16k16);

// ============= 32x8x16 shape implementations =============
// f16
impl_load_a!(f16, Shape32x8x16, 16, wmma_load_a_f16_row_m32n8k16, wmma_load_a_f16_col_m32n8k16);
impl_load_b!(f16, Shape32x8x16, 8, wmma_load_b_f16_row_m32n8k16, wmma_load_b_f16_col_m32n8k16);

// bf16
impl_load_a!(bf16, Shape32x8x16, 16, wmma_load_a_bf16_row_m32n8k16, wmma_load_a_bf16_col_m32n8k16);
impl_load_b!(bf16, Shape32x8x16, 8, wmma_load_b_bf16_row_m32n8k16, wmma_load_b_bf16_col_m32n8k16);

// i8
impl_load_a!(i8, Shape32x8x16, 4, wmma_load_a_s8_row_m32n8k16, wmma_load_a_s8_col_m32n8k16);
impl_load_b!(i8, Shape32x8x16, 2, wmma_load_b_s8_row_m32n8k16, wmma_load_b_s8_col_m32n8k16);

// u8
impl_load_a!(u8, Shape32x8x16, 4, wmma_load_a_u8_row_m32n8k16, wmma_load_a_u8_col_m32n8k16);
impl_load_b!(u8, Shape32x8x16, 2, wmma_load_b_u8_row_m32n8k16, wmma_load_b_u8_col_m32n8k16);

// bool
impl LoadMatrixA<Shape32x8x16, Row> for bool {
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_a_u8_row_m32n8k16(ptr, stride);
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

impl LoadMatrixA<Shape32x8x16, Col> for bool {
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_a_u8_col_m32n8k16(ptr, stride);
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

impl LoadMatrixB<Shape32x8x16, Row> for bool {
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_b_u8_row_m32n8k16(ptr, stride);
        let storage: [u8; 8] = core::mem::transmute(raw);
        out[..8].copy_from_slice(&storage);
    }
}

impl LoadMatrixB<Shape32x8x16, Col> for bool {
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_b_u8_col_m32n8k16(ptr, stride);
        let storage: [u8; 8] = core::mem::transmute(raw);
        out[..8].copy_from_slice(&storage);
    }
}

// f32 accumulator for 32x8x16
impl_load_c!(f32, Shape32x8x16, 8, wmma_load_c_f32_row_m32n8k16, wmma_load_c_f32_col_m32n8k16);
impl_store_d!(f32, Shape32x8x16, wmma_store_d_f32_row_m32n8k16, wmma_store_d_f32_col_m32n8k16);

// i32 accumulator for 32x8x16
impl_load_c!(i32, Shape32x8x16, 8, wmma_load_c_s32_row_m32n8k16, wmma_load_c_s32_col_m32n8k16);
impl_store_d!(i32, Shape32x8x16, wmma_store_d_s32_row_m32n8k16, wmma_store_d_s32_col_m32n8k16);

// ============= 8x32x16 shape implementations =============
// f16
impl_load_a!(f16, Shape8x32x16, 8, wmma_load_a_f16_row_m8n32k16, wmma_load_a_f16_col_m8n32k16);
impl_load_b!(f16, Shape8x32x16, 16, wmma_load_b_f16_row_m8n32k16, wmma_load_b_f16_col_m8n32k16);

// bf16
impl_load_a!(bf16, Shape8x32x16, 8, wmma_load_a_bf16_row_m8n32k16, wmma_load_a_bf16_col_m8n32k16);
impl_load_b!(bf16, Shape8x32x16, 16, wmma_load_b_bf16_row_m8n32k16, wmma_load_b_bf16_col_m8n32k16);

// i8
impl_load_a!(i8, Shape8x32x16, 2, wmma_load_a_s8_row_m8n32k16, wmma_load_a_s8_col_m8n32k16);
impl_load_b!(i8, Shape8x32x16, 4, wmma_load_b_s8_row_m8n32k16, wmma_load_b_s8_col_m8n32k16);

// u8
impl_load_a!(u8, Shape8x32x16, 2, wmma_load_a_u8_row_m8n32k16, wmma_load_a_u8_col_m8n32k16);
impl_load_b!(u8, Shape8x32x16, 4, wmma_load_b_u8_row_m8n32k16, wmma_load_b_u8_col_m8n32k16);

// bool
impl LoadMatrixA<Shape8x32x16, Row> for bool {
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_a_u8_row_m8n32k16(ptr, stride);
        let storage: [u8; 8] = core::mem::transmute(raw);
        out[..8].copy_from_slice(&storage);
    }
}

impl LoadMatrixA<Shape8x32x16, Col> for bool {
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_a_u8_col_m8n32k16(ptr, stride);
        let storage: [u8; 8] = core::mem::transmute(raw);
        out[..8].copy_from_slice(&storage);
    }
}

impl LoadMatrixB<Shape8x32x16, Row> for bool {
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_b_u8_row_m8n32k16(ptr, stride);
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

impl LoadMatrixB<Shape8x32x16, Col> for bool {
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [u8; 32]) {
        let raw = wmma_load_b_u8_col_m8n32k16(ptr, stride);
        let storage: [u8; 16] = core::mem::transmute(raw);
        out[..16].copy_from_slice(&storage);
    }
}

// f32 accumulator for 8x32x16
impl_load_c!(f32, Shape8x32x16, 8, wmma_load_c_f32_row_m8n32k16, wmma_load_c_f32_col_m8n32k16);
impl_store_d!(f32, Shape8x32x16, wmma_store_d_f32_row_m8n32k16, wmma_store_d_f32_col_m8n32k16);

// i32 accumulator for 8x32x16
impl_load_c!(i32, Shape8x32x16, 8, wmma_load_c_s32_row_m8n32k16, wmma_load_c_s32_col_m8n32k16);
impl_store_d!(i32, Shape8x32x16, wmma_store_d_s32_row_m8n32k16, wmma_store_d_s32_col_m8n32k16);

// ============= 8x8x4 shape implementations (f64) =============
impl_load_a!(f64, Shape8x8x4, 2, wmma_load_a_f64_row_m8n8k4, wmma_load_a_f64_col_m8n8k4);
impl_load_b!(f64, Shape8x8x4, 2, wmma_load_b_f64_row_m8n8k4, wmma_load_b_f64_col_m8n8k4);
impl_load_c!(f64, Shape8x8x4, 2, wmma_load_c_f64_row_m8n8k4, wmma_load_c_f64_col_m8n8k4);
impl_store_d!(f64, Shape8x8x4, wmma_store_d_f64_row_m8n8k4, wmma_store_d_f64_col_m8n8k4, f64);