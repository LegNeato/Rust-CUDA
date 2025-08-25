//! Warp synchronization primitives.

use crate::gpu_only;
#[cfg(target_os = "cuda")]
use core::arch::asm;

/// A type-safe wrapper for warp masks.
///
/// This type ensures that warp masks are used correctly and provides
/// convenient methods for mask manipulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct WarpMask(u32);

impl WarpMask {
    /// Create a new warp mask from a raw u32.
    #[inline]
    pub const fn new(mask: u32) -> Self {
        Self(mask)
    }

    /// Create a mask for all threads in the warp.
    #[inline]
    pub const fn all() -> Self {
        Self(super::FULL_MASK)
    }

    /// Create an empty mask.
    #[inline]
    pub const fn none() -> Self {
        Self(0)
    }

    /// Create a mask for a single lane.
    #[inline]
    pub const fn lane(lane_id: u32) -> Self {
        debug_assert!(lane_id < super::WARP_SIZE);
        Self(1 << lane_id)
    }

    /// Create a mask for a range of lanes.
    #[inline]
    pub const fn lanes(start: u32, count: u32) -> Self {
        debug_assert!(start < super::WARP_SIZE);
        debug_assert!(start + count <= super::WARP_SIZE);
        let mask = if count == 32 {
            !0u32
        } else {
            ((1u64 << count) - 1) as u32
        };
        Self(mask << start)
    }

    /// Get the raw mask value.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Check if a lane is included in this mask.
    #[inline]
    pub const fn contains_lane(self, lane_id: u32) -> bool {
        (self.0 & (1 << lane_id)) != 0
    }

    /// Count the number of active lanes in this mask.
    #[inline]
    pub const fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    /// Union with another mask.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection with another mask.
    #[inline]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Difference with another mask (lanes in self but not in other).
    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Symmetric difference with another mask.
    #[inline]
    pub const fn symmetric_difference(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Complement of this mask.
    #[inline]
    pub const fn complement(self) -> Self {
        Self(!self.0)
    }

    /// Create a mask for even-numbered lanes (0, 2, 4, ...).
    #[inline]
    pub const fn even_lanes() -> Self {
        Self(0x55555555)
    }

    /// Create a mask for odd-numbered lanes (1, 3, 5, ...).
    #[inline]
    pub const fn odd_lanes() -> Self {
        Self(0xAAAAAAAA)
    }

    /// Create a mask for a specific quadrant (0-3).
    #[inline]
    pub const fn quadrant(quad: u32) -> Self {
        debug_assert!(quad < 4);
        Self(0xFF << (quad * 8))
    }

    /// Create a mask for a range of lanes.
    #[inline]
    pub const fn range(start: u32, end: u32) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= super::WARP_SIZE);
        if start == end {
            return Self(0);
        }
        let count = end - start;
        Self::lanes(start, count)
    }

    /// Check if the mask is empty (no lanes set).
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if the mask is full (all lanes set).
    #[inline]
    pub const fn is_full(self) -> bool {
        self.0 == super::FULL_MASK
    }

    /// Count the number of set lanes in the mask.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl Default for WarpMask {
    fn default() -> Self {
        Self::all()
    }
}

impl From<u32> for WarpMask {
    fn from(mask: u32) -> Self {
        Self::new(mask)
    }
}

impl From<WarpMask> for u32 {
    fn from(mask: WarpMask) -> Self {
        mask.raw()
    }
}

/// Synchronizes threads in a warp according to a mask.
///
/// # Safety
///
/// The behavior of this function is undefined if:
/// - Any thread inside `mask` has exited.
/// - The executing thread is not inside of `mask`.
/// - On compute_62 and below, all threads in `mask` must call with the same mask.
#[gpu_only]
#[inline(always)]
pub unsafe fn sync(mask: WarpMask) {
    extern "C" {
        #[link_name = "llvm.nvvm.bar.warp.sync"]
        fn sync_impl(mask: u32);
    }
    sync_impl(mask.raw());
}

/// Synchronizes threads in a warp according to a raw mask value.
///
/// Prefer using [`sync`] with a [`WarpMask`] for better type safety.
///
/// # Safety
///
/// Same safety requirements as [`sync`].
#[gpu_only]
#[inline(always)]
pub unsafe fn sync_mask(mask: u32) {
    sync(WarpMask::new(mask))
}

/// Returns the current thread's lane ID within the warp (0-31).
#[gpu_only]
#[inline(always)]
pub fn lane_id() -> u32 {
    let mut out;
    unsafe {
        asm!(
            "mov.u32 {}, %laneid;",
            out(reg32) out
        );
    }
    out
}

/// Returns a mask of currently active threads in the warp.
#[gpu_only]
#[inline(always)]
pub fn active_mask() -> WarpMask {
    let mut out;
    unsafe {
        asm!(
            "activemask.b32 {};",
            out(reg32) out
        );
    }
    WarpMask::new(out)
}
