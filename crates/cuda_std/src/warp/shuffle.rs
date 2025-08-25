//! Warp shuffle operations with extreme type safety.
//!
//! This module provides type-safe abstractions for CUDA warp shuffle operations,
//! making invalid states unrepresentable and ensuring all validation happens
//! at compile time.

use super::sync::WarpMask;
use crate::gpu_only;
#[cfg(target_os = "cuda")]
use core::arch::asm;
use core::marker::PhantomData;

// ============================================================================
// Core Types - Making invalid states unrepresentable
// ============================================================================

/// Width of the shuffle operation - must be a power of 2 and <= 32
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShuffleWidth(u32);

impl ShuffleWidth {
    /// Create a new shuffle width, validated at compile time when possible
    #[inline(always)]
    pub const fn new(width: u32) -> Option<Self> {
        if width.is_power_of_two() && width <= 32 {
            Some(Self(width))
        } else {
            None
        }
    }

    /// Create from raw value (internal use only)
    #[inline(always)]
    const fn from_raw(width: u32) -> Self {
        Self(width)
    }

    /// Full warp width (32 threads)
    #[inline(always)]
    pub const fn full_warp() -> Self {
        Self(32)
    }

    /// Half warp width (16 threads)
    #[inline(always)]
    pub const fn half_warp() -> Self {
        Self(16)
    }

    /// Quarter warp width (8 threads)
    #[inline(always)]
    pub const fn quarter_warp() -> Self {
        Self(8)
    }

    /// Get the raw width value
    #[inline(always)]
    pub const fn value(&self) -> u32 {
        self.0
    }
}

/// Represents a lane ID with compile-time bounds checking where possible
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneId(u32);

impl LaneId {
    /// Create a new lane ID
    #[inline(always)]
    pub const fn new(id: u32) -> Option<Self> {
        if id < 32 {
            Some(Self(id))
        } else {
            None
        }
    }

    /// Create a lane ID without bounds checking (unsafe)
    #[inline(always)]
    pub const unsafe fn new_unchecked(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw lane ID
    #[inline(always)]
    pub const fn value(&self) -> u32 {
        self.0
    }
}

/// Delta for shuffle up/down operations
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShuffleDelta(u32);

impl ShuffleDelta {
    /// Create a new shuffle delta
    #[inline(always)]
    pub const fn new(delta: u32) -> Self {
        Self(delta)
    }

    /// Get the raw delta value
    #[inline(always)]
    pub const fn value(&self) -> u32 {
        self.0
    }
}

// ============================================================================
// Shuffle Errors - Type-safe error handling
// ============================================================================

/// Error type for shuffle operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidLane;

impl core::fmt::Display for InvalidLane {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Shuffle operation accessed an invalid lane")
    }
}

// ============================================================================
// Shuffle Patterns - Type-safe shuffle pattern specifications
// ============================================================================

/// Represents different shuffle patterns
pub mod patterns {
    use super::*;

    /// Shuffle down pattern - get value from lane at current_lane + delta
    #[repr(transparent)]
    pub struct Down {
        delta: ShuffleDelta,
    }

    impl Down {
        #[inline(always)]
        pub const fn new(delta: u32) -> Self {
            Self {
                delta: ShuffleDelta::new(delta),
            }
        }

        #[inline(always)]
        pub const fn delta(&self) -> ShuffleDelta {
            self.delta
        }
    }

    /// Shuffle up pattern - get value from lane at current_lane - delta
    #[repr(transparent)]
    pub struct Up {
        delta: ShuffleDelta,
    }

    impl Up {
        #[inline(always)]
        pub const fn new(delta: u32) -> Self {
            Self {
                delta: ShuffleDelta::new(delta),
            }
        }

        #[inline(always)]
        pub const fn delta(&self) -> ShuffleDelta {
            self.delta
        }
    }

    /// Shuffle XOR pattern - get value from lane at current_lane ^ mask
    #[repr(transparent)]
    pub struct Xor {
        lane_mask: u32,
    }

    impl Xor {
        #[inline(always)]
        pub const fn new(lane_mask: u32) -> Self {
            Self { lane_mask }
        }

        #[inline(always)]
        pub const fn mask(&self) -> u32 {
            self.lane_mask
        }
    }

    /// Shuffle index pattern - get value from specific lane
    #[repr(transparent)]
    pub struct Index {
        source_lane: LaneId,
    }

    impl Index {
        #[inline(always)]
        pub const fn new(source_lane: u32) -> Option<Self> {
            if let Some(lane) = LaneId::new(source_lane) {
                Some(Self { source_lane: lane })
            } else {
                None
            }
        }

        /// Create without bounds checking (unsafe)
        #[inline(always)]
        pub const unsafe fn new_unchecked(source_lane: u32) -> Self {
            Self {
                source_lane: LaneId::new_unchecked(source_lane),
            }
        }

        #[inline(always)]
        pub const fn source_lane(&self) -> LaneId {
            self.source_lane
        }
    }
}

// ============================================================================
// Shuffle Operations - Type-safe warp shuffling
// ============================================================================

/// A shuffle operation builder for type-safe shuffling
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Shuffle<T: ShuffleValue> {
    mask: WarpMask,
    width: ShuffleWidth,
    _phantom: PhantomData<T>,
}

impl<T: ShuffleValue> Shuffle<T> {
    /// Create a new shuffle operation
    #[inline(always)]
    pub const fn new(mask: WarpMask, width: ShuffleWidth) -> Self {
        Self {
            mask,
            width,
            _phantom: PhantomData,
        }
    }

    /// Create a shuffle for the full warp
    #[inline(always)]
    pub const fn full_warp() -> Self {
        Self::new(WarpMask::all(), ShuffleWidth::full_warp())
    }

    /// Shuffle down by delta
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn down(self, value: T, pattern: patterns::Down) -> Result<T, InvalidLane> {
        T::shuffle_down(
            self.mask,
            value,
            pattern.delta().value(),
            self.width.value(),
        )
    }

    /// Shuffle up by delta
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn up(self, value: T, pattern: patterns::Up) -> Result<T, InvalidLane> {
        T::shuffle_up(
            self.mask,
            value,
            pattern.delta().value(),
            self.width.value(),
        )
    }

    /// Shuffle with XOR pattern
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn xor(self, value: T, pattern: patterns::Xor) -> Result<T, InvalidLane> {
        T::shuffle_xor(self.mask, value, pattern.mask(), self.width.value())
    }

    /// Shuffle from specific lane
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn index(self, value: T, pattern: patterns::Index) -> Result<T, InvalidLane> {
        T::shuffle_idx(
            self.mask,
            value,
            pattern.source_lane().value(),
            self.width.value(),
        )
    }
}

// ============================================================================
// ShuffleValue Trait - Types that can be shuffled
// ============================================================================

/// Trait for types that support warp shuffle
pub trait ShuffleValue: Copy {
    /// Shuffle down implementation
    unsafe fn shuffle_down(
        mask: WarpMask,
        value: Self,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane>;

    /// Shuffle up implementation
    unsafe fn shuffle_up(
        mask: WarpMask,
        value: Self,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane>;

    /// Shuffle XOR implementation
    unsafe fn shuffle_xor(
        mask: WarpMask,
        value: Self,
        lane_mask: u32,
        width: u32,
    ) -> Result<Self, InvalidLane>;

    /// Shuffle index implementation
    unsafe fn shuffle_idx(
        mask: WarpMask,
        value: Self,
        src_lane: u32,
        width: u32,
    ) -> Result<Self, InvalidLane>;
}

// C-compatible struct to match LLVM IR's {i32, i8} return type
// This fixes an ABI mismatch where Rust would represent (u32, bool) as [2 x i32]
// but the LLVM intrinsic returns {i32, i8} (a struct, not an array)
#[doc(hidden)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct WarpShuffleResult {
    value: u32,
    predicate: u8,
}

// Helper function using the intrinsic from libintrinsics.ll
#[gpu_only]
unsafe fn warp_shuffle_32(
    mode: u32,
    mask: u32,
    value: u32,
    b: u32,
    width: u32,
) -> Result<u32, InvalidLane> {
    extern "C" {
        // see libintrinsics.ll - returns {i32, i8} in LLVM IR, which maps to our WarpShuffleResult struct
        fn __nvvm_warp_shuffle(mask: u32, mode: u32, a: u32, b: u32, c: u32) -> WarpShuffleResult;
    }

    // width must be a power of 2 and <= 32
    debug_assert!(width.is_power_of_two() && width <= 32);

    // Encode width parameter as nvcc does
    let mut c = 0u32;
    c |= 0b11111; // clamp value
    c |= (32 - width) << 8; // width encoding

    let result = __nvvm_warp_shuffle(mask, mode, value, b, c);
    if result.predicate != 0 {
        Ok(result.value)
    } else {
        Err(InvalidLane)
    }
}

// Shuffle modes for the intrinsic
const SHUFFLE_MODE_DOWN: u32 = 0;
const SHUFFLE_MODE_UP: u32 = 1;
const SHUFFLE_MODE_XOR: u32 = 2;
const SHUFFLE_MODE_IDX: u32 = 3;

// Macro to implement shuffle for 32-bit types
macro_rules! impl_shuffle_32 {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ShuffleValue for $ty {
                #[gpu_only]
                unsafe fn shuffle_down(mask: WarpMask, value: Self, delta: u32, width: u32) -> Result<Self, InvalidLane> {
                    warp_shuffle_32(
                        SHUFFLE_MODE_DOWN,
                        mask.raw(),
                        value as u32,
                        delta,
                        width
                    ).map(|v| v as Self)
                }

                #[gpu_only]
                unsafe fn shuffle_up(mask: WarpMask, value: Self, delta: u32, width: u32) -> Result<Self, InvalidLane> {
                    warp_shuffle_32(
                        SHUFFLE_MODE_UP,
                        mask.raw(),
                        value as u32,
                        delta,
                        width
                    ).map(|v| v as Self)
                }

                #[gpu_only]
                unsafe fn shuffle_xor(mask: WarpMask, value: Self, lane_mask: u32, width: u32) -> Result<Self, InvalidLane> {
                    warp_shuffle_32(
                        SHUFFLE_MODE_XOR,
                        mask.raw(),
                        value as u32,
                        lane_mask,
                        width
                    ).map(|v| v as Self)
                }

                #[gpu_only]
                unsafe fn shuffle_idx(mask: WarpMask, value: Self, src_lane: u32, width: u32) -> Result<Self, InvalidLane> {
                    warp_shuffle_32(
                        SHUFFLE_MODE_IDX,
                        mask.raw(),
                        value as u32,
                        src_lane,
                        width
                    ).map(|v| v as Self)
                }
            }
        )*
    };
}

impl_shuffle_32! {
    i32, u32,
}

// Special case for f32 to preserve bit pattern
impl ShuffleValue for f32 {
    #[gpu_only]
    unsafe fn shuffle_down(
        mask: WarpMask,
        value: Self,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u32 as ShuffleValue>::shuffle_down(mask, bits, delta, width)?;
        Ok(f32::from_bits(result))
    }

    #[gpu_only]
    unsafe fn shuffle_up(
        mask: WarpMask,
        value: Self,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u32 as ShuffleValue>::shuffle_up(mask, bits, delta, width)?;
        Ok(f32::from_bits(result))
    }

    #[gpu_only]
    unsafe fn shuffle_xor(
        mask: WarpMask,
        value: Self,
        lane_mask: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u32 as ShuffleValue>::shuffle_xor(mask, bits, lane_mask, width)?;
        Ok(f32::from_bits(result))
    }

    #[gpu_only]
    unsafe fn shuffle_idx(
        mask: WarpMask,
        value: Self,
        src_lane: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u32 as ShuffleValue>::shuffle_idx(mask, bits, src_lane, width)?;
        Ok(f32::from_bits(result))
    }
}

// For 64-bit types, we shuffle high and low parts separately
macro_rules! impl_shuffle_64 {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ShuffleValue for $ty {
                #[gpu_only]
                unsafe fn shuffle_down(mask: WarpMask, value: Self, delta: u32, width: u32) -> Result<Self, InvalidLane> {
                    let lo = (value & 0xFFFFFFFF) as u32;
                    let hi = (value >> 32) as u32;
                    let lo_result = <u32 as ShuffleValue>::shuffle_down(mask, lo, delta, width)?;
                    let hi_result = <u32 as ShuffleValue>::shuffle_down(mask, hi, delta, width)?;
                    Ok(((hi_result as $ty) << 32) | (lo_result as $ty))
                }

                #[gpu_only]
                unsafe fn shuffle_up(mask: WarpMask, value: Self, delta: u32, width: u32) -> Result<Self, InvalidLane> {
                    let lo = (value & 0xFFFFFFFF) as u32;
                    let hi = (value >> 32) as u32;
                    let lo_result = <u32 as ShuffleValue>::shuffle_up(mask, lo, delta, width)?;
                    let hi_result = <u32 as ShuffleValue>::shuffle_up(mask, hi, delta, width)?;
                    Ok(((hi_result as $ty) << 32) | (lo_result as $ty))
                }

                #[gpu_only]
                unsafe fn shuffle_xor(mask: WarpMask, value: Self, lane_mask: u32, width: u32) -> Result<Self, InvalidLane> {
                    let lo = (value & 0xFFFFFFFF) as u32;
                    let hi = (value >> 32) as u32;
                    let lo_result = <u32 as ShuffleValue>::shuffle_xor(mask, lo, lane_mask, width)?;
                    let hi_result = <u32 as ShuffleValue>::shuffle_xor(mask, hi, lane_mask, width)?;
                    Ok(((hi_result as $ty) << 32) | (lo_result as $ty))
                }

                #[gpu_only]
                unsafe fn shuffle_idx(mask: WarpMask, value: Self, src_lane: u32, width: u32) -> Result<Self, InvalidLane> {
                    let lo = (value & 0xFFFFFFFF) as u32;
                    let hi = (value >> 32) as u32;
                    let lo_result = <u32 as ShuffleValue>::shuffle_idx(mask, lo, src_lane, width)?;
                    let hi_result = <u32 as ShuffleValue>::shuffle_idx(mask, hi, src_lane, width)?;
                    Ok(((hi_result as $ty) << 32) | (lo_result as $ty))
                }
            }
        )*
    };
}

impl_shuffle_64! {
    i64, u64,
}

// For f64, we need to handle the bit pattern correctly
impl ShuffleValue for f64 {
    #[gpu_only]
    unsafe fn shuffle_down(
        mask: WarpMask,
        value: Self,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u64 as ShuffleValue>::shuffle_down(mask, bits, delta, width)?;
        Ok(f64::from_bits(result))
    }

    #[gpu_only]
    unsafe fn shuffle_up(
        mask: WarpMask,
        value: Self,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u64 as ShuffleValue>::shuffle_up(mask, bits, delta, width)?;
        Ok(f64::from_bits(result))
    }

    #[gpu_only]
    unsafe fn shuffle_xor(
        mask: WarpMask,
        value: Self,
        lane_mask: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u64 as ShuffleValue>::shuffle_xor(mask, bits, lane_mask, width)?;
        Ok(f64::from_bits(result))
    }

    #[gpu_only]
    unsafe fn shuffle_idx(
        mask: WarpMask,
        value: Self,
        src_lane: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        let bits = value.to_bits();
        let result = <u64 as ShuffleValue>::shuffle_idx(mask, bits, src_lane, width)?;
        Ok(f64::from_bits(result))
    }
}

// For smaller types, we pack them into 32-bit values
macro_rules! impl_shuffle_small {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ShuffleValue for $ty {
                #[gpu_only]
                unsafe fn shuffle_down(mask: WarpMask, value: Self, delta: u32, width: u32) -> Result<Self, InvalidLane> {
                    let result = <u32 as ShuffleValue>::shuffle_down(mask, value as u32, delta, width)?;
                    Ok(result as Self)
                }

                #[gpu_only]
                unsafe fn shuffle_up(mask: WarpMask, value: Self, delta: u32, width: u32) -> Result<Self, InvalidLane> {
                    let result = <u32 as ShuffleValue>::shuffle_up(mask, value as u32, delta, width)?;
                    Ok(result as Self)
                }

                #[gpu_only]
                unsafe fn shuffle_xor(mask: WarpMask, value: Self, lane_mask: u32, width: u32) -> Result<Self, InvalidLane> {
                    let result = <u32 as ShuffleValue>::shuffle_xor(mask, value as u32, lane_mask, width)?;
                    Ok(result as Self)
                }

                #[gpu_only]
                unsafe fn shuffle_idx(mask: WarpMask, value: Self, src_lane: u32, width: u32) -> Result<Self, InvalidLane> {
                    let result = <u32 as ShuffleValue>::shuffle_idx(mask, value as u32, src_lane, width)?;
                    Ok(result as Self)
                }
            }
        )*
    };
}

impl_shuffle_small! {
    i8, i16,
    u8, u16,
}

// ============================================================================
// Extension Traits - Making the API intuitive and composable
// ============================================================================

/// Extension trait for values that can be shuffled
pub trait ShuffleExt: ShuffleValue {
    /// Create a shuffle operation for this type
    fn shuffle(mask: WarpMask, width: ShuffleWidth) -> Shuffle<Self>;

    /// Shuffle this value down
    unsafe fn shuffle_down(
        self,
        mask: WarpMask,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane>;

    /// Shuffle this value up
    unsafe fn shuffle_up(self, mask: WarpMask, delta: u32, width: u32)
        -> Result<Self, InvalidLane>;
}

impl<T: ShuffleValue> ShuffleExt for T {
    #[inline(always)]
    fn shuffle(mask: WarpMask, width: ShuffleWidth) -> Shuffle<Self> {
        Shuffle::new(mask, width)
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn shuffle_down(
        self,
        mask: WarpMask,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        ShuffleValue::shuffle_down(mask, self, delta, width)
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn shuffle_up(
        self,
        mask: WarpMask,
        delta: u32,
        width: u32,
    ) -> Result<Self, InvalidLane> {
        ShuffleValue::shuffle_up(mask, self, delta, width)
    }
}
