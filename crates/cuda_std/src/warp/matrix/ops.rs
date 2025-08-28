//! Trait-based dispatch for matrix operations
//! This module provides compile-time dispatch for WMMA operations
//! based on element type and matrix shape combinations.

use super::layout::{Col, Layout, Row};
use super::*;

// Import all the intrinsic functions from the parent module
use super::{
    wmma_load_a_bf16_col_m16n16k16,
    wmma_load_a_bf16_col_m32n8k16,
    wmma_load_a_bf16_col_m8n32k16,
    wmma_load_a_bf16_row_m16n16k16,
    wmma_load_a_bf16_row_m32n8k16,
    wmma_load_a_bf16_row_m8n32k16,
    wmma_load_a_f16_col_m16n16k16,
    wmma_load_a_f16_col_m32n8k16,
    wmma_load_a_f16_col_m8n32k16,
    // 16x16x16 intrinsics
    wmma_load_a_f16_row_m16n16k16,
    // 16x8x16 intrinsics
    // 32x8x16 intrinsics
    wmma_load_a_f16_row_m32n8k16,
    // 8x32x16 intrinsics
    wmma_load_a_f16_row_m8n32k16,
    wmma_load_a_f64_col_m8n8k4,
    // 8x8x4 intrinsics (f64)
    wmma_load_a_f64_row_m8n8k4,
    wmma_load_a_s8_col_m16n16k16,
    wmma_load_a_s8_col_m32n8k16,
    wmma_load_a_s8_col_m8n32k16,
    wmma_load_a_s8_row_m16n16k16,
    wmma_load_a_s8_row_m32n8k16,
    wmma_load_a_s8_row_m8n32k16,
    wmma_load_a_u8_col_m16n16k16,
    wmma_load_a_u8_col_m32n8k16,
    wmma_load_a_u8_col_m8n32k16,
    wmma_load_a_u8_row_m16n16k16,
    wmma_load_a_u8_row_m32n8k16,
    wmma_load_a_u8_row_m8n32k16,
    wmma_load_b_bf16_col_m16n16k16,
    wmma_load_b_bf16_col_m32n8k16,
    wmma_load_b_bf16_col_m8n32k16,
    wmma_load_b_bf16_row_m16n16k16,
    wmma_load_b_bf16_row_m32n8k16,
    wmma_load_b_bf16_row_m8n32k16,
    wmma_load_b_f16_col_m16n16k16,
    wmma_load_b_f16_col_m32n8k16,
    wmma_load_b_f16_col_m8n32k16,
    wmma_load_b_f16_row_m16n16k16,
    wmma_load_b_f16_row_m32n8k16,
    wmma_load_b_f16_row_m8n32k16,
    wmma_load_b_f64_col_m8n8k4,
    wmma_load_b_f64_row_m8n8k4,
    wmma_load_b_s8_col_m16n16k16,
    wmma_load_b_s8_col_m32n8k16,
    wmma_load_b_s8_col_m8n32k16,
    wmma_load_b_s8_row_m16n16k16,
    wmma_load_b_s8_row_m32n8k16,
    wmma_load_b_s8_row_m8n32k16,
    wmma_load_b_u8_col_m16n16k16,
    wmma_load_b_u8_col_m32n8k16,
    wmma_load_b_u8_col_m8n32k16,
    wmma_load_b_u8_row_m16n16k16,
    wmma_load_b_u8_row_m32n8k16,
    wmma_load_b_u8_row_m8n32k16,
    wmma_load_c_f32_col_m32n8k16,
    wmma_load_c_f32_col_m8n32k16,
    wmma_load_c_f32_row_m16n16k16,
    wmma_load_c_f32_row_m32n8k16,
    wmma_load_c_f32_row_m8n32k16,
    wmma_load_c_f64_col_m8n8k4,
    wmma_load_c_f64_row_m8n8k4,
    wmma_load_c_s32_col_m32n8k16,
    wmma_load_c_s32_col_m8n32k16,
    wmma_load_c_s32_row_m16n16k16,
    wmma_load_c_s32_row_m32n8k16,
    wmma_load_c_s32_row_m8n32k16,
    wmma_store_d_f32_col_m16n16k16,
    wmma_store_d_f32_col_m32n8k16,
    wmma_store_d_f32_col_m8n32k16,
    wmma_store_d_f32_row_m16n16k16,
    wmma_store_d_f32_row_m32n8k16,
    wmma_store_d_f32_row_m8n32k16,
    wmma_store_d_f64_col_m8n8k4,
    wmma_store_d_f64_row_m8n8k4,
    wmma_store_d_s32_col_m16n16k16,
    wmma_store_d_s32_col_m32n8k16,
    wmma_store_d_s32_col_m8n32k16,
    wmma_store_d_s32_row_m16n16k16,
    wmma_store_d_s32_row_m32n8k16,
    wmma_store_d_s32_row_m8n32k16,
};

// Import shape types
type Shape16x16x16 = super::dims::Shape<16, 16, 16>;
type Shape16x16x8 = super::dims::Shape<16, 16, 8>;
type Shape16x8x16 = super::dims::Shape<16, 8, 16>;
type Shape32x8x16 = super::dims::Shape<32, 8, 16>;
type Shape8x32x16 = super::dims::Shape<8, 32, 16>;
type Shape8x8x4 = super::dims::Shape<8, 8, 4>;

// Trait for loading matrix A fragments with layout in type system
#[diagnostic::on_unimplemented(
    message = "Cannot load matrix A: `{Self}` with shape `{Shape}` and layout `{L}` is not supported",
    label = "unsupported matrix A load configuration",
    note = "This combination of element type, shape, and layout doesn't have a corresponding WMMA load implementation"
)]
pub trait LoadMatrixA<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for loading matrix B fragments with layout in type system
#[diagnostic::on_unimplemented(
    message = "Cannot load matrix B: `{Self}` with shape `{Shape}` and layout `{L}` is not supported",
    label = "unsupported matrix B load configuration",
    note = "This combination of element type, shape, and layout doesn't have a corresponding WMMA load implementation"
)]
pub trait LoadMatrixB<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for loading matrix A fragments from shared memory using ldmatrix
#[diagnostic::on_unimplemented(
    message = "Cannot load matrix A from shared memory: `{Self}` with shape `{Shape}` and layout `{L}` is not supported",
    label = "unsupported shared memory matrix A configuration",
    note = "This combination doesn't have a corresponding ldmatrix implementation"
)]
pub trait LoadMatrixAShared<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_a_shared_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for loading matrix B fragments from shared memory using ldmatrix
#[diagnostic::on_unimplemented(
    message = "Cannot load matrix B from shared memory: `{Self}` with shape `{Shape}` and layout `{L}` is not supported",
    label = "unsupported shared memory matrix B configuration",
    note = "This combination doesn't have a corresponding ldmatrix implementation"
)]
pub trait LoadMatrixBShared<Shape: TensorCoreShape, L: Layout>: MatrixElement {
    unsafe fn load_b_shared_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for loading accumulator matrices with layout in type system
#[diagnostic::on_unimplemented(
    message = "Cannot load accumulator: `{Self}` with shape `{Shape}` and layout `{L}` is not supported",
    label = "unsupported accumulator load configuration",
    note = "This combination of accumulator type, shape, and layout doesn't have a corresponding WMMA load implementation"
)]
pub trait LoadMatrixC<Shape: TensorCoreShape, L: Layout>: AccumulatorElement {
    unsafe fn load_c_into(ptr: *const u8, stride: i32, out: &mut [Self::Storage; 32]);
}

// Trait for storing accumulator matrices with layout in type system
#[diagnostic::on_unimplemented(
    message = "Cannot store accumulator: `{Self}` with shape `{Shape}` and layout `{L}` is not supported",
    label = "unsupported accumulator store configuration",
    note = "This combination of accumulator type, shape, and layout doesn't have a corresponding WMMA store implementation"
)]
pub trait StoreMatrixD<Shape: TensorCoreShape, L: Layout>: AccumulatorElement {
    unsafe fn store_d_from(ptr: *mut u8, data: &[Self::Storage; 32], stride: i32);
}

// Macro to implement LoadMatrixA for both layouts
macro_rules! impl_load_a {
    ($type:ty, $shape:ty, $count:literal, $row_fn:ident, $col_fn:ident) => {
        impl LoadMatrixA<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn load_a_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as MatrixElement>::Storage; 32],
            ) {
                let raw = $row_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] =
                    core::mem::transmute(raw);
                out[..$count].copy_from_slice(&storage);
            }
        }

        impl LoadMatrixA<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_a_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as MatrixElement>::Storage; 32],
            ) {
                let raw = $col_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] =
                    core::mem::transmute(raw);
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
            unsafe fn load_b_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as MatrixElement>::Storage; 32],
            ) {
                let raw = $row_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] =
                    core::mem::transmute(raw);
                out[..$count].copy_from_slice(&storage);
            }
        }

        impl LoadMatrixB<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_b_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as MatrixElement>::Storage; 32],
            ) {
                let raw = $col_fn(ptr, stride);
                let storage: [<$type as MatrixElement>::Storage; $count] =
                    core::mem::transmute(raw);
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
            unsafe fn load_c_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as AccumulatorElement>::Storage; 32],
            ) {
                let result = $row_fn(ptr, stride);
                out[..$count].copy_from_slice(&result);
            }
        }

        impl LoadMatrixC<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_c_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as AccumulatorElement>::Storage; 32],
            ) {
                let result = $col_fn(ptr, stride);
                out[..$count].copy_from_slice(&result);
            }
        }
    };
    // Handle case where col function doesn't exist (uses row)
    ($type:ty, $shape:ty, $count:literal, $row_fn:ident) => {
        impl LoadMatrixC<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn load_c_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as AccumulatorElement>::Storage; 32],
            ) {
                let result = $row_fn(ptr, stride);
                out[..$count].copy_from_slice(&result);
            }
        }

        impl LoadMatrixC<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn load_c_into(
                ptr: *const u8,
                stride: i32,
                out: &mut [<$type as AccumulatorElement>::Storage; 32],
            ) {
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
            unsafe fn store_d_from(
                ptr: *mut u8,
                data: &[<$type as AccumulatorElement>::Storage; 32],
                stride: i32,
            ) {
                $row_fn(
                    ptr, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                    stride,
                );
            }
        }

        impl StoreMatrixD<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn store_d_from(
                ptr: *mut u8,
                data: &[<$type as AccumulatorElement>::Storage; 32],
                stride: i32,
            ) {
                $col_fn(
                    ptr, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                    stride,
                );
            }
        }
    };
    // m16n8k16 shape removed - MMA-only, no WMMA store operations
    // Special case for f64 which only uses 2 values
    ($type:ty, $shape:ty, $row_fn:ident, $col_fn:ident, f64) => {
        impl StoreMatrixD<$shape, Row> for $type {
            #[inline(always)]
            unsafe fn store_d_from(
                ptr: *mut u8,
                data: &[<$type as AccumulatorElement>::Storage; 32],
                stride: i32,
            ) {
                $row_fn(ptr, data[0], data[1], stride);
            }
        }

        impl StoreMatrixD<$shape, Col> for $type {
            #[inline(always)]
            unsafe fn store_d_from(
                ptr: *mut u8,
                data: &[<$type as AccumulatorElement>::Storage; 32],
                stride: i32,
            ) {
                $col_fn(ptr, data[0], data[1], stride);
            }
        }
    };
}

// ============= f16 16x16x16 implementations =============
impl_load_a!(
    f16,
    Shape16x16x16,
    16,
    wmma_load_a_f16_row_m16n16k16,
    wmma_load_a_f16_col_m16n16k16
);
impl_load_b!(
    f16,
    Shape16x16x16,
    16,
    wmma_load_b_f16_row_m16n16k16,
    wmma_load_b_f16_col_m16n16k16
);

// ============= bf16 16x16x16 implementations =============
impl_load_a!(
    bf16,
    Shape16x16x16,
    16,
    wmma_load_a_bf16_row_m16n16k16,
    wmma_load_a_bf16_col_m16n16k16
);
impl_load_b!(
    bf16,
    Shape16x16x16,
    16,
    wmma_load_b_bf16_row_m16n16k16,
    wmma_load_b_bf16_col_m16n16k16
);

// ============= i8 16x16x16 implementations =============
impl_load_a!(
    i8,
    Shape16x16x16,
    4,
    wmma_load_a_s8_row_m16n16k16,
    wmma_load_a_s8_col_m16n16k16
);
impl_load_b!(
    i8,
    Shape16x16x16,
    4,
    wmma_load_b_s8_row_m16n16k16,
    wmma_load_b_s8_col_m16n16k16
);

// ============= u8 16x16x16 implementations =============
impl_load_a!(
    u8,
    Shape16x16x16,
    4,
    wmma_load_a_u8_row_m16n16k16,
    wmma_load_a_u8_col_m16n16k16
);
impl_load_b!(
    u8,
    Shape16x16x16,
    4,
    wmma_load_b_u8_row_m16n16k16,
    wmma_load_b_u8_col_m16n16k16
);

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
impl_store_d!(
    f32,
    Shape16x16x16,
    wmma_store_d_f32_row_m16n16k16,
    wmma_store_d_f32_col_m16n16k16
);

// ============= i32 accumulator 16x16x16 implementations =============
impl_load_c!(i32, Shape16x16x16, 8, wmma_load_c_s32_row_m16n16k16); // No col variant in intrinsics
impl_store_d!(
    i32,
    Shape16x16x16,
    wmma_store_d_s32_row_m16n16k16,
    wmma_store_d_s32_col_m16n16k16
);

// ============= 32x8x16 shape implementations =============
// f16
impl_load_a!(
    f16,
    Shape32x8x16,
    16,
    wmma_load_a_f16_row_m32n8k16,
    wmma_load_a_f16_col_m32n8k16
);
impl_load_b!(
    f16,
    Shape32x8x16,
    8,
    wmma_load_b_f16_row_m32n8k16,
    wmma_load_b_f16_col_m32n8k16
);

// bf16
impl_load_a!(
    bf16,
    Shape32x8x16,
    16,
    wmma_load_a_bf16_row_m32n8k16,
    wmma_load_a_bf16_col_m32n8k16
);
impl_load_b!(
    bf16,
    Shape32x8x16,
    8,
    wmma_load_b_bf16_row_m32n8k16,
    wmma_load_b_bf16_col_m32n8k16
);

// i8
impl_load_a!(
    i8,
    Shape32x8x16,
    4,
    wmma_load_a_s8_row_m32n8k16,
    wmma_load_a_s8_col_m32n8k16
);
impl_load_b!(
    i8,
    Shape32x8x16,
    2,
    wmma_load_b_s8_row_m32n8k16,
    wmma_load_b_s8_col_m32n8k16
);

// u8
impl_load_a!(
    u8,
    Shape32x8x16,
    4,
    wmma_load_a_u8_row_m32n8k16,
    wmma_load_a_u8_col_m32n8k16
);
impl_load_b!(
    u8,
    Shape32x8x16,
    2,
    wmma_load_b_u8_row_m32n8k16,
    wmma_load_b_u8_col_m32n8k16
);

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
impl_load_c!(
    f32,
    Shape32x8x16,
    8,
    wmma_load_c_f32_row_m32n8k16,
    wmma_load_c_f32_col_m32n8k16
);
impl_store_d!(
    f32,
    Shape32x8x16,
    wmma_store_d_f32_row_m32n8k16,
    wmma_store_d_f32_col_m32n8k16
);

// i32 accumulator for 32x8x16
impl_load_c!(
    i32,
    Shape32x8x16,
    8,
    wmma_load_c_s32_row_m32n8k16,
    wmma_load_c_s32_col_m32n8k16
);
impl_store_d!(
    i32,
    Shape32x8x16,
    wmma_store_d_s32_row_m32n8k16,
    wmma_store_d_s32_col_m32n8k16
);

// ============= 16x8x16 shape implementations =============
// f16 - Shape16x8x16 is MMA-only, no WMMA load/store

// bf16 - Shape16x8x16 is MMA-only, no WMMA load/store

// i8 - Shape16x8x16 is MMA-only, no WMMA load/store

// u8 - Shape16x8x16 is MMA-only, no WMMA load/store

// f32 accumulator for 16x8x16 - MMA-only, no WMMA load/store

// i32 accumulator for 16x8x16 - MMA-only, no WMMA load/store

// ============= 8x32x16 shape implementations =============
// f16
impl_load_a!(
    f16,
    Shape8x32x16,
    8,
    wmma_load_a_f16_row_m8n32k16,
    wmma_load_a_f16_col_m8n32k16
);
impl_load_b!(
    f16,
    Shape8x32x16,
    16,
    wmma_load_b_f16_row_m8n32k16,
    wmma_load_b_f16_col_m8n32k16
);

// bf16
impl_load_a!(
    bf16,
    Shape8x32x16,
    8,
    wmma_load_a_bf16_row_m8n32k16,
    wmma_load_a_bf16_col_m8n32k16
);
impl_load_b!(
    bf16,
    Shape8x32x16,
    16,
    wmma_load_b_bf16_row_m8n32k16,
    wmma_load_b_bf16_col_m8n32k16
);

// i8
impl_load_a!(
    i8,
    Shape8x32x16,
    2,
    wmma_load_a_s8_row_m8n32k16,
    wmma_load_a_s8_col_m8n32k16
);
impl_load_b!(
    i8,
    Shape8x32x16,
    4,
    wmma_load_b_s8_row_m8n32k16,
    wmma_load_b_s8_col_m8n32k16
);

// u8
impl_load_a!(
    u8,
    Shape8x32x16,
    2,
    wmma_load_a_u8_row_m8n32k16,
    wmma_load_a_u8_col_m8n32k16
);
impl_load_b!(
    u8,
    Shape8x32x16,
    4,
    wmma_load_b_u8_row_m8n32k16,
    wmma_load_b_u8_col_m8n32k16
);

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
impl_load_c!(
    f32,
    Shape8x32x16,
    8,
    wmma_load_c_f32_row_m8n32k16,
    wmma_load_c_f32_col_m8n32k16
);
impl_store_d!(
    f32,
    Shape8x32x16,
    wmma_store_d_f32_row_m8n32k16,
    wmma_store_d_f32_col_m8n32k16
);

// i32 accumulator for 8x32x16
impl_load_c!(
    i32,
    Shape8x32x16,
    8,
    wmma_load_c_s32_row_m8n32k16,
    wmma_load_c_s32_col_m8n32k16
);
impl_store_d!(
    i32,
    Shape8x32x16,
    wmma_store_d_s32_row_m8n32k16,
    wmma_store_d_s32_col_m8n32k16
);

// ============= 8x8x4 shape implementations (f64) =============
impl_load_a!(
    f64,
    Shape8x8x4,
    2,
    wmma_load_a_f64_row_m8n8k4,
    wmma_load_a_f64_col_m8n8k4
);
impl_load_b!(
    f64,
    Shape8x8x4,
    2,
    wmma_load_b_f64_row_m8n8k4,
    wmma_load_b_f64_col_m8n8k4
);
impl_load_c!(
    f64,
    Shape8x8x4,
    2,
    wmma_load_c_f64_row_m8n8k4,
    wmma_load_c_f64_col_m8n8k4
);
impl_store_d!(
    f64,
    Shape8x8x4,
    wmma_store_d_f64_row_m8n8k4,
    wmma_store_d_f64_col_m8n8k4,
    f64
);

// ============= 16x16x8 shape implementations (TF32) =============
// f32 uses TF32 with 16x16x8 shape - special handling required
impl LoadMatrixA<Shape16x16x8, Row> for f32 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [f32; 32]) {
        let raw = wmma_load_a_tf32_row_m16n16k8(ptr, stride);
        // The intrinsic returns 8 f32 values for TF32
        // We need to pad to 32 for consistency with the storage size
        out[0..8].copy_from_slice(&raw);
        // Zero out the rest
        for i in 8..32 {
            out[i] = 0.0;
        }
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_into(_ptr: *const u8, _stride: i32, _out: &mut [f32; 32]) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixA<Shape16x16x8, Col> for f32 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_into(ptr: *const u8, stride: i32, out: &mut [f32; 32]) {
        let raw = wmma_load_a_tf32_col_m16n16k8(ptr, stride);
        out[0..8].copy_from_slice(&raw);
        for i in 8..32 {
            out[i] = 0.0;
        }
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_into(_ptr: *const u8, _stride: i32, _out: &mut [f32; 32]) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixB<Shape16x16x8, Row> for f32 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [f32; 32]) {
        let raw = wmma_load_b_tf32_row_m16n16k8(ptr, stride);
        out[0..8].copy_from_slice(&raw);
        for i in 8..32 {
            out[i] = 0.0;
        }
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_into(_ptr: *const u8, _stride: i32, _out: &mut [f32; 32]) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixB<Shape16x16x8, Col> for f32 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_into(ptr: *const u8, stride: i32, out: &mut [f32; 32]) {
        let raw = wmma_load_b_tf32_col_m16n16k8(ptr, stride);
        out[0..8].copy_from_slice(&raw);
        for i in 8..32 {
            out[i] = 0.0;
        }
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_into(_ptr: *const u8, _stride: i32, _out: &mut [f32; 32]) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

// f32 accumulator for 16x16x8
impl_load_c!(
    f32,
    Shape16x16x8,
    8,
    wmma_load_c_f32_row_m16n16k8,
    wmma_load_c_f32_col_m16n16k8
);
impl_store_d!(
    f32,
    Shape16x16x8,
    wmma_store_d_f32_row_m16n16k8,
    wmma_store_d_f32_col_m16n16k8
);

// ============================================================================
// Shared Memory Load Implementations (ldmatrix)
// ============================================================================

use super::{ldmatrix_m8n8_x2_b16, ldmatrix_m8n8_x2_trans_b16};

// Shape<16, 8, 16> with bf16 - uses 2x 8x8 matrices (x2)
impl LoadMatrixAShared<Shape16x8x16, Row> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        // Load 2 matrices of 8x8 for the 16x8 shape
        let result = ldmatrix_m8n8_x2_b16(ptr);
        // Result is [i32; 2] which contains 4 i16s packed
        let storage: [bf16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [bf16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixAShared<Shape16x8x16, Col> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        // Use transpose version for column layout
        let result = ldmatrix_m8n8_x2_trans_b16(ptr);
        let storage: [bf16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [bf16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x8x16, Row> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        let result = ldmatrix_m8n8_x2_b16(ptr);
        let storage: [bf16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [bf16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x8x16, Col> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        let result = ldmatrix_m8n8_x2_trans_b16(ptr);
        let storage: [bf16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [bf16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

// Shape<16, 8, 16> with f16 - uses 2x 8x8 matrices (x2)
impl LoadMatrixAShared<Shape16x8x16, Row> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x2_b16(ptr);
        // Transmute to f16 array
        let storage: [f16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [f16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixAShared<Shape16x8x16, Col> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x2_trans_b16(ptr);
        let storage: [f16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [f16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x8x16, Row> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x2_b16(ptr);
        let storage: [f16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [f16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x8x16, Col> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x2_trans_b16(ptr);
        let storage: [f16; 4] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [f16::ZERO; 32];
        // Then copy the actual data
        out[..4].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

// ============= Shape<16, 16, 16> shared memory loading implementations =============
// Shape<16, 16, 16> uses 4x 8x8 matrices (x4) = 16 registers for bf16/f16

impl LoadMatrixAShared<Shape16x16x16, Row> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        let result = ldmatrix_m8n8_x4_b16(ptr);
        let storage: [bf16; 8] = core::mem::transmute(result);
        // Zero out the entire output first
        *out = [bf16::ZERO; 32];
        // Then copy the actual data to the first 8 elements (16 registers, but bf16 is 2 per register)
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixAShared<Shape16x16x16, Col> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        let result = ldmatrix_m8n8_x4_trans_b16(ptr);
        let storage: [bf16; 8] = core::mem::transmute(result);
        *out = [bf16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x16x16, Row> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        let result = ldmatrix_m8n8_x4_b16(ptr);
        let storage: [bf16; 8] = core::mem::transmute(result);
        *out = [bf16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x16x16, Col> for bf16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [bf16; 32]) {
        let result = ldmatrix_m8n8_x4_trans_b16(ptr);
        let storage: [bf16; 8] = core::mem::transmute(result);
        *out = [bf16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<bf16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

// Shape<16, 16, 16> with f16
impl LoadMatrixAShared<Shape16x16x16, Row> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x4_b16(ptr);
        let storage: [f16; 8] = core::mem::transmute(result);
        *out = [f16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixAShared<Shape16x16x16, Col> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_a_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x4_trans_b16(ptr);
        let storage: [f16; 8] = core::mem::transmute(result);
        *out = [f16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_a_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x16x16, Row> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x4_b16(ptr);
        let storage: [f16; 8] = core::mem::transmute(result);
        *out = [f16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}

impl LoadMatrixBShared<Shape16x16x16, Col> for f16 {
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn load_b_shared_into(ptr: *const u8, _stride: i32, out: &mut [f16; 32]) {
        let result = ldmatrix_m8n8_x4_trans_b16(ptr);
        let storage: [f16; 8] = core::mem::transmute(result);
        *out = [f16::ZERO; 32];
        out[..8].copy_from_slice(&storage);
    }

    #[cfg(not(target_arch = "nvptx64"))]
    unsafe fn load_b_shared_into(
        _ptr: *const u8,
        _stride: i32,
        _out: &mut [<f16 as MatrixElement>::Storage; 32],
    ) {
        unimplemented!("Matrix operations are only supported on NVPTX64")
    }
}
