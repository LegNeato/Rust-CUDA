//! Warp reduction operations with extreme type safety.
//!
//! This module provides type-safe abstractions for CUDA warp reduction operations,
//! making invalid states unrepresentable and ensuring all validation happens
//! at compile time.

use super::shuffle::ShuffleValue;
use super::sync::WarpMask;
use crate::gpu_only;
#[cfg(target_os = "cuda")]
use core::arch::asm;
use core::marker::PhantomData;

// ============================================================================
// Reduction Operations - Type-safe specification of reduction types
// ============================================================================

/// Marker trait for reduction operations
pub trait ReductionOp: sealed::Sealed {
    /// The name of this reduction operation
    const NAME: &'static str;
}

/// Private module to seal the ReductionOp trait
mod sealed {
    pub trait Sealed {}
}

/// Addition reduction
#[derive(Debug, Clone, Copy)]
pub struct Add;
impl sealed::Sealed for Add {}
impl ReductionOp for Add {
    const NAME: &'static str = "add";
}

/// Minimum reduction
#[derive(Debug, Clone, Copy)]
pub struct Min;
impl sealed::Sealed for Min {}
impl ReductionOp for Min {
    const NAME: &'static str = "min";
}

/// Maximum reduction
#[derive(Debug, Clone, Copy)]
pub struct Max;
impl sealed::Sealed for Max {}
impl ReductionOp for Max {
    const NAME: &'static str = "max";
}

/// Bitwise AND reduction
#[derive(Debug, Clone, Copy)]
pub struct And;
impl sealed::Sealed for And {}
impl ReductionOp for And {
    const NAME: &'static str = "and";
}

/// Bitwise OR reduction
#[derive(Debug, Clone, Copy)]
pub struct Or;
impl sealed::Sealed for Or {}
impl ReductionOp for Or {
    const NAME: &'static str = "or";
}

/// Bitwise XOR reduction
#[derive(Debug, Clone, Copy)]
pub struct Xor;
impl sealed::Sealed for Xor {}
impl ReductionOp for Xor {
    const NAME: &'static str = "xor";
}

// ============================================================================
// Reduction Result - Type-safe result with semantic meaning
// ============================================================================

/// Result of a reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReductionResult<T, Op: ReductionOp> {
    value: T,
    _op: PhantomData<Op>,
}

impl<T, Op: ReductionOp> ReductionResult<T, Op> {
    #[inline(always)]
    const fn new(value: T) -> Self {
        Self {
            value,
            _op: PhantomData,
        }
    }

    /// Get the reduced value
    #[inline(always)]
    pub const fn value(&self) -> T
    where
        T: Copy,
    {
        self.value
    }

    /// Consume the result and return the value
    #[inline(always)]
    pub fn into_value(self) -> T {
        self.value
    }
}

// Special methods for specific reduction types
impl<T: Copy> ReductionResult<T, Add> {
    /// Get the sum from an addition reduction
    #[inline(always)]
    pub const fn sum(&self) -> T {
        self.value
    }
}

impl<T: Copy + Ord> ReductionResult<T, Min> {
    /// Get the minimum value from a min reduction
    #[inline(always)]
    pub const fn minimum(&self) -> T {
        self.value
    }

    /// Check if this value is less than another
    #[inline(always)]
    pub fn is_less_than(&self, other: T) -> bool {
        self.value < other
    }
}

impl<T: Copy + Ord> ReductionResult<T, Max> {
    /// Get the maximum value from a max reduction
    #[inline(always)]
    pub const fn maximum(&self) -> T {
        self.value
    }

    /// Check if this value is greater than another
    #[inline(always)]
    pub fn is_greater_than(&self, other: T) -> bool {
        self.value > other
    }
}

// ============================================================================
// Reduction Builder - Type-safe reduction operations
// ============================================================================

/// A reduction operation builder
pub struct Reduction<T: ReduceValue> {
    mask: WarpMask,
    _phantom: PhantomData<T>,
}

impl<T: ReduceValue> Reduction<T> {
    /// Create a new reduction operation
    #[inline(always)]
    pub const fn new(mask: WarpMask) -> Self {
        Self {
            mask,
            _phantom: PhantomData,
        }
    }

    /// Create a reduction for all threads
    #[inline(always)]
    pub const fn all_threads() -> Self {
        Self::new(WarpMask::all())
    }

    /// Perform an addition reduction
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn add(&self, value: T) -> ReductionResult<T, Add> {
        ReductionResult::new(T::reduce_add(self.mask, value))
    }

    /// Perform a minimum reduction
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn min(&self, value: T) -> ReductionResult<T, Min> {
        ReductionResult::new(T::reduce_min(self.mask, value))
    }

    /// Perform a maximum reduction
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn max(&self, value: T) -> ReductionResult<T, Max> {
        ReductionResult::new(T::reduce_max(self.mask, value))
    }

    /// Perform a bitwise AND reduction
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn and(&self, value: T) -> ReductionResult<T, And>
    where
        T: BitwiseReduceValue,
    {
        ReductionResult::new(T::reduce_and(self.mask, value))
    }

    /// Perform a bitwise OR reduction
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn or(&self, value: T) -> ReductionResult<T, Or>
    where
        T: BitwiseReduceValue,
    {
        ReductionResult::new(T::reduce_or(self.mask, value))
    }

    /// Perform a bitwise XOR reduction
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn xor(&self, value: T) -> ReductionResult<T, Xor>
    where
        T: BitwiseReduceValue,
    {
        ReductionResult::new(T::reduce_xor(self.mask, value))
    }
}

// ============================================================================
// Reduction Traits - Type-safe specification of reducible types
// ============================================================================

/// Trait for types that support basic reduction operations
pub trait ReduceValue: Copy {
    /// Addition reduction
    unsafe fn reduce_add(mask: WarpMask, value: Self) -> Self;

    /// Minimum reduction
    unsafe fn reduce_min(mask: WarpMask, value: Self) -> Self;

    /// Maximum reduction
    unsafe fn reduce_max(mask: WarpMask, value: Self) -> Self;
}

/// Trait for types that support bitwise reduction operations
pub trait BitwiseReduceValue: ReduceValue {
    /// Bitwise AND reduction
    unsafe fn reduce_and(mask: WarpMask, value: Self) -> Self;

    /// Bitwise OR reduction
    unsafe fn reduce_or(mask: WarpMask, value: Self) -> Self;

    /// Bitwise XOR reduction
    unsafe fn reduce_xor(mask: WarpMask, value: Self) -> Self;
}

// ============================================================================
// Implementations for integer types using native PTX instructions
// ============================================================================

impl ReduceValue for i32 {
    #[gpu_only]
    unsafe fn reduce_add(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.add.s32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_min(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.min.s32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_max(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.max.s32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }
}

impl BitwiseReduceValue for i32 {
    #[gpu_only]
    unsafe fn reduce_and(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.and.b32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_or(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.or.b32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_xor(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.xor.b32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }
}

impl ReduceValue for u32 {
    #[gpu_only]
    unsafe fn reduce_add(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.add.u32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_min(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.min.u32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_max(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.max.u32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }
}

impl BitwiseReduceValue for u32 {
    #[gpu_only]
    unsafe fn reduce_and(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.and.b32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_or(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.or.b32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }

    #[gpu_only]
    unsafe fn reduce_xor(mask: WarpMask, value: Self) -> Self {
        let out;
        asm!(
            "redux.sync.xor.b32 {}, {}, {};",
            out(reg32) out,
            in(reg32) value,
            in(reg32) mask.raw()
        );
        out
    }
}

// ============================================================================
// Implementations for 64-bit types using shuffle operations
// ============================================================================

impl ReduceValue for i64 {
    #[gpu_only]
    unsafe fn reduce_add(mask: WarpMask, mut value: Self) -> Self {
        // Implement using shuffle operations in a tree reduction pattern
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.wrapping_add(shuffled);
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_min(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.min(shuffled);
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_max(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.max(shuffled);
        }
        value
    }
}

impl BitwiseReduceValue for i64 {
    #[gpu_only]
    unsafe fn reduce_and(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value &= shuffled;
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_or(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value |= shuffled;
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_xor(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value ^= shuffled;
        }
        value
    }
}

impl ReduceValue for u64 {
    #[gpu_only]
    unsafe fn reduce_add(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.wrapping_add(shuffled);
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_min(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.min(shuffled);
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_max(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.max(shuffled);
        }
        value
    }
}

impl BitwiseReduceValue for u64 {
    #[gpu_only]
    unsafe fn reduce_and(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value &= shuffled;
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_or(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value |= shuffled;
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_xor(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value ^= shuffled;
        }
        value
    }
}

// ============================================================================
// Implementations for floating-point types
// ============================================================================

impl ReduceValue for f32 {
    #[gpu_only]
    unsafe fn reduce_add(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value += shuffled;
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_min(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.min(shuffled);
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_max(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.max(shuffled);
        }
        value
    }
}

impl ReduceValue for f64 {
    #[gpu_only]
    unsafe fn reduce_add(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value += shuffled;
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_min(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.min(shuffled);
        }
        value
    }

    #[gpu_only]
    unsafe fn reduce_max(mask: WarpMask, mut value: Self) -> Self {
        for offset in [16, 8, 4, 2, 1] {
            let shuffled =
                <Self as ShuffleValue>::shuffle_down(mask, value, offset, 32).unwrap_or(value);
            value = value.max(shuffled);
        }
        value
    }
}

// ============================================================================
// Extension Traits - Making the API intuitive and composable
// ============================================================================

/// Extension trait for WarpMask to enable fluent reduction API
pub trait WarpMaskReduceExt {
    /// Start a reduction operation on this mask
    fn reduce<T: ReduceValue>(&self) -> Reduction<T>;
}

impl WarpMaskReduceExt for WarpMask {
    #[inline(always)]
    fn reduce<T: ReduceValue>(&self) -> Reduction<T> {
        Reduction::new(*self)
    }
}

/// Extension trait for values that can be reduced
pub trait ReduceExt: ReduceValue {
    /// Reduce this value across the warp using addition
    unsafe fn reduce_sum(self, mask: WarpMask) -> ReductionResult<Self, Add>;

    /// Find the minimum value across the warp
    unsafe fn reduce_minimum(self, mask: WarpMask) -> ReductionResult<Self, Min>;

    /// Find the maximum value across the warp
    unsafe fn reduce_maximum(self, mask: WarpMask) -> ReductionResult<Self, Max>;
}

impl<T: ReduceValue> ReduceExt for T {
    #[gpu_only]
    #[inline(always)]
    unsafe fn reduce_sum(self, mask: WarpMask) -> ReductionResult<Self, Add> {
        mask.reduce().add(self)
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn reduce_minimum(self, mask: WarpMask) -> ReductionResult<Self, Min> {
        mask.reduce().min(self)
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn reduce_maximum(self, mask: WarpMask) -> ReductionResult<Self, Max> {
        mask.reduce().max(self)
    }
}
