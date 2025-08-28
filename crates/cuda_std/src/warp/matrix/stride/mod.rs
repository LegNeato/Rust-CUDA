//! Stride validation for WMMA operations
//!
//! This module provides compile-time validation of stride parameters for tensor core operations.
//! Different element types have different stride requirements based on their alignment needs:
//!
//! - f16/bf16/i16/u16: multiple of 8 (16 bytes for 8 elements)
//! - f32/i32/u32: multiple of 4 (16 bytes for 4 elements)  
//! - f64/i64/u64: multiple of 2 (16 bytes for 2 elements)
//! - i8/u8/bool: multiple of 16 (16 bytes for 16 elements)
//!
//! The maximum supported stride is 8192.

use core::marker::PhantomData;

/// Maximum supported stride value
pub const MAX_STRIDE: usize = 8192;

// ============================================================================
// Stride Validation Types
// ============================================================================

/// Phantom type for compile-time stride validation
pub struct StrideValidator<T, const STRIDE: usize> {
    _phantom: PhantomData<T>,
}

// Sealed trait pattern to prevent external implementation
mod sealed {
    pub trait Sealed {}
    // Seal all StrideValidator types
    impl<T, const STRIDE: usize> Sealed for super::StrideValidator<T, STRIDE> {}
}

/// Trait for valid stride configurations
/// 
/// This trait is implemented for type-stride combinations that satisfy
/// the alignment requirements for tensor core operations.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid stride configuration",
    label = "invalid stride for tensor core operations",
    note = "f16/bf16/i16/u16 require stride to be a multiple of 8",
    note = "f32/i32/u32 require stride to be a multiple of 4",
    note = "f64/i64/u64 require stride to be a multiple of 2",
    note = "i8/u8/bool require stride to be a multiple of 16"
)]
pub trait ValidStride: sealed::Sealed {}

// Type-specific stride implementation modules
// Each module includes auto-generated implementations from build.rs
mod f16;
mod bf16;
mod f32;
mod f64;
mod i8;
mod u8;
mod i16;
mod u16;
mod i32;
mod u32;
mod i64;
mod u64;
mod bool;