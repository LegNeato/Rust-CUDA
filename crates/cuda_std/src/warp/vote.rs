//! Warp voting and ballot operations with extreme type safety.
//!
//! This module provides type-safe abstractions for CUDA warp vote operations,
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

/// A predicate that can be evaluated by threads in a warp.
/// This wrapper ensures type safety and prevents mixing of incompatible predicates.
#[derive(Clone, Copy)]
pub struct Predicate<T = ()> {
    value: bool,
    _phantom: PhantomData<T>,
}

impl<T> Predicate<T> {
    /// Create a new predicate with a type tag for semantic meaning
    #[inline(always)]
    pub const fn new(value: bool) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    /// Get the raw boolean value
    #[inline(always)]
    pub const fn value(&self) -> bool {
        self.value
    }
}

/// Semantic predicate types for better API clarity
pub mod predicates {
    use super::*;

    /// A condition that should be checked across the warp
    pub struct Condition;

    /// A validation predicate
    pub struct Validation;

    /// A comparison result
    pub struct Comparison;

    /// Extension trait for creating semantic predicates
    pub trait PredicateExt {
        /// Create a condition predicate
        fn condition(self) -> Predicate<Condition>;

        /// Create a validation predicate
        fn validation(self) -> Predicate<Validation>;

        /// Create a comparison predicate
        fn comparison(self) -> Predicate<Comparison>;
    }

    impl PredicateExt for bool {
        #[inline(always)]
        fn condition(self) -> Predicate<Condition> {
            Predicate::new(self)
        }

        #[inline(always)]
        fn validation(self) -> Predicate<Validation> {
            Predicate::new(self)
        }

        #[inline(always)]
        fn comparison(self) -> Predicate<Comparison> {
            Predicate::new(self)
        }
    }
}

// ============================================================================
// Vote Results - Type-safe result types with semantic meaning
// ============================================================================

/// Result of an `all` vote operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllVoteResult {
    unanimous: bool,
}

impl AllVoteResult {
    #[inline(always)]
    const fn new(unanimous: bool) -> Self {
        Self { unanimous }
    }

    /// Returns true if all threads in the mask voted true
    #[inline(always)]
    pub const fn is_unanimous(&self) -> bool {
        self.unanimous
    }

    /// Returns true if at least one thread voted false
    #[inline(always)]
    pub const fn has_dissent(&self) -> bool {
        !self.unanimous
    }

    /// Convert to a simple boolean
    #[inline(always)]
    pub const fn as_bool(&self) -> bool {
        self.unanimous
    }
}

/// Result of an `any` vote operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyVoteResult {
    has_any: bool,
}

impl AnyVoteResult {
    #[inline(always)]
    const fn new(has_any: bool) -> Self {
        Self { has_any }
    }

    /// Returns true if at least one thread voted true
    #[inline(always)]
    pub const fn has_any(&self) -> bool {
        self.has_any
    }

    /// Returns true if no threads voted true
    #[inline(always)]
    pub const fn has_none(&self) -> bool {
        !self.has_any
    }

    /// Convert to a simple boolean
    #[inline(always)]
    pub const fn as_bool(&self) -> bool {
        self.has_any
    }
}

/// Result of a ballot operation - a bitmask of voting threads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BallotResult {
    mask: WarpMask,
}

impl BallotResult {
    #[inline(always)]
    const fn new(mask: WarpMask) -> Self {
        Self { mask }
    }

    /// Get the underlying mask
    #[inline(always)]
    pub const fn mask(&self) -> WarpMask {
        self.mask
    }

    /// Count the number of threads that voted true
    #[inline(always)]
    pub const fn count(&self) -> u32 {
        self.mask.raw().count_ones()
    }

    /// Check if a specific lane voted true
    #[inline(always)]
    pub const fn lane_voted(&self, lane_id: u32) -> bool {
        (self.mask.raw() & (1 << lane_id)) != 0
    }

    /// Check if all lanes in the original mask voted true
    #[inline(always)]
    pub fn all_voted(&self, original_mask: WarpMask) -> bool {
        self.mask.raw() == original_mask.raw()
    }

    /// Check if no lanes voted true
    #[inline(always)]
    pub const fn none_voted(&self) -> bool {
        self.mask.raw() == 0
    }

    /// Iterate over lanes that voted true
    pub fn true_lanes(&self) -> impl Iterator<Item = u32> {
        let mask = self.mask.raw();
        (0..32).filter(move |&lane| (mask & (1 << lane)) != 0)
    }
}

// ============================================================================
// Vote Operations - Type-safe warp voting with semantic clarity
// ============================================================================

/// A vote operation builder for type-safe voting
pub struct Vote {
    mask: WarpMask,
}

impl Vote {
    /// Create a new vote operation for the given mask
    #[inline(always)]
    pub const fn new(mask: WarpMask) -> Self {
        Self { mask }
    }

    /// Create a vote operation for all threads
    #[inline(always)]
    pub const fn all_threads() -> Self {
        Self::new(WarpMask::all())
    }

    /// Vote whether all threads satisfy the predicate
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn all<T>(&self, predicate: Predicate<T>) -> AllVoteResult {
        let mut out: u32;

        asm!(
            "{{",
            ".reg .pred %p<3>;",
            "setp.eq.u32 %p1, {}, 1;",
            "vote.sync.all.pred %p2, %p1, {};",
            "selp.u32 {}, 0, 1, %p2;",
            "}}",
            in(reg32) predicate.value() as u32,
            in(reg32) self.mask.raw(),
            out(reg32) out
        );

        AllVoteResult::new(out != 0)
    }

    /// Vote whether any thread satisfies the predicate
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn any<T>(&self, predicate: Predicate<T>) -> AnyVoteResult {
        let mut out: u32;

        asm!(
            "{{",
            ".reg .pred %p<3>;",
            "setp.eq.u32 %p1, {}, 1;",
            "vote.sync.any.pred %p2, %p1, {};",
            "selp.u32 {}, 0, 1, %p2;",
            "}}",
            in(reg32) predicate.value() as u32,
            in(reg32) self.mask.raw(),
            out(reg32) out
        );

        AnyVoteResult::new(out != 0)
    }

    /// Get a ballot (bitmask) of which threads satisfy the predicate
    #[gpu_only]
    #[inline(always)]
    pub unsafe fn ballot<T>(&self, predicate: Predicate<T>) -> BallotResult {
        let mut out: u32;

        asm!(
            "{{",
            ".reg .pred %p1;",
            "setp.eq.u32 %p1, {}, 1;",
            "vote.sync.ballot.b32 {}, %p1, {};",
            "}}",
            in(reg32) predicate.value() as u32,
            out(reg32) out,
            in(reg32) self.mask.raw(),
        );

        BallotResult::new(WarpMask::new(out))
    }
}

// ============================================================================
// Equality Voting - Check if all threads have the same value
// ============================================================================

/// Result of an equality vote with match mask
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqualityResult {
    all_equal: bool,
    match_mask: Option<WarpMask>,
}

impl EqualityResult {
    #[inline(always)]
    pub const fn new(all_equal: bool) -> Self {
        Self {
            all_equal,
            match_mask: None,
        }
    }

    #[inline(always)]
    pub const fn with_mask(all_equal: bool, mask: WarpMask) -> Self {
        Self {
            all_equal,
            match_mask: Some(mask),
        }
    }

    /// Returns true if all threads have the same value
    #[inline(always)]
    pub const fn all_equal(&self) -> bool {
        self.all_equal
    }

    /// Returns true if threads have different values
    #[inline(always)]
    pub const fn has_divergence(&self) -> bool {
        !self.all_equal
    }

    /// Returns the mask of threads that have matching values
    /// This is only available when using the match intrinsics
    #[inline(always)]
    pub const fn match_mask(&self) -> Option<WarpMask> {
        self.match_mask
    }
}

/// Trait for types that support equality voting
pub trait VoteEquality: Copy {
    /// Check if all threads have the same value
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult;
}

// C-compatible struct to match LLVM IR's {i32, i8} return type for match intrinsics
#[doc(hidden)]
#[repr(C)]
pub struct WarpMatchResult {
    matched_mask: u32,
    all_matched: u8,
}

// Use the match.all intrinsic for equality checking
impl VoteEquality for i32 {
    #[gpu_only]
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
        extern "C" {
            // Returns mask of matching threads and whether all matched
            fn __nvvm_warp_match_all_32(mask: u32, value: u32) -> WarpMatchResult;
        }

        let result = __nvvm_warp_match_all_32(mask.raw(), value as u32);
        EqualityResult::with_mask(result.all_matched != 0, WarpMask::new(result.matched_mask))
    }
}

impl VoteEquality for u32 {
    #[gpu_only]
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
        extern "C" {
            fn __nvvm_warp_match_all_32(mask: u32, value: u32) -> WarpMatchResult;
        }

        let result = __nvvm_warp_match_all_32(mask.raw(), value);
        EqualityResult::with_mask(result.all_matched != 0, WarpMask::new(result.matched_mask))
    }
}

// For smaller types, cast to i32
macro_rules! impl_vote_equality_small {
    ($($ty:ty),* $(,)?) => {
        $(
            impl VoteEquality for $ty {
                #[gpu_only]
                unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
                    i32::vote_all_equal(mask, value as i32)
                }
            }
        )*
    };
}

impl_vote_equality_small! {
    i8, i16,
    u8, u16,
}

impl VoteEquality for i64 {
    #[gpu_only]
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
        extern "C" {
            fn __nvvm_warp_match_all_64(mask: u32, value: u64) -> WarpMatchResult;
        }

        let result = __nvvm_warp_match_all_64(mask.raw(), value as u64);
        EqualityResult::with_mask(result.all_matched != 0, WarpMask::new(result.matched_mask))
    }
}

impl VoteEquality for u64 {
    #[gpu_only]
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
        extern "C" {
            fn __nvvm_warp_match_all_64(mask: u32, value: u64) -> WarpMatchResult;
        }

        let result = __nvvm_warp_match_all_64(mask.raw(), value);
        EqualityResult::with_mask(result.all_matched != 0, WarpMask::new(result.matched_mask))
    }
}

impl VoteEquality for f32 {
    #[gpu_only]
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
        extern "C" {
            fn __nvvm_warp_match_all_32(mask: u32, value: u32) -> WarpMatchResult;
        }

        let result = __nvvm_warp_match_all_32(mask.raw(), value.to_bits());
        EqualityResult::with_mask(result.all_matched != 0, WarpMask::new(result.matched_mask))
    }
}

impl VoteEquality for f64 {
    #[gpu_only]
    unsafe fn vote_all_equal(mask: WarpMask, value: Self) -> EqualityResult {
        extern "C" {
            fn __nvvm_warp_match_all_64(mask: u32, value: u64) -> WarpMatchResult;
        }

        let result = __nvvm_warp_match_all_64(mask.raw(), value.to_bits());
        EqualityResult::with_mask(result.all_matched != 0, WarpMask::new(result.matched_mask))
    }
}

// ============================================================================
// Extension Traits - Making the API intuitive and composable
// ============================================================================

/// Extension trait for WarpMask to enable fluent voting API
pub trait WarpMaskVoteExt {
    /// Start a vote operation on this mask
    fn vote(&self) -> Vote;

    /// Check if all threads in this mask satisfy a condition
    unsafe fn all_satisfy(&self, condition: bool) -> AllVoteResult;

    /// Check if any thread in this mask satisfies a condition
    unsafe fn any_satisfy(&self, condition: bool) -> AnyVoteResult;

    /// Get a ballot of threads satisfying a condition
    unsafe fn ballot_for(&self, condition: bool) -> BallotResult;
}

impl WarpMaskVoteExt for WarpMask {
    #[inline(always)]
    fn vote(&self) -> Vote {
        Vote::new(*self)
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn all_satisfy(&self, condition: bool) -> AllVoteResult {
        self.vote().all(Predicate::<()>::new(condition))
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn any_satisfy(&self, condition: bool) -> AnyVoteResult {
        self.vote().any(Predicate::<()>::new(condition))
    }

    #[gpu_only]
    #[inline(always)]
    unsafe fn ballot_for(&self, condition: bool) -> BallotResult {
        self.vote().ballot(Predicate::<()>::new(condition))
    }
}
