//! Warp-level primitives for CUDA.
//!
//! Warps in CUDA are groups of 32 threads that are dispatched together inside of
//! thread blocks and execute in SIMT fashion.
//!
//! This module is organized into submodules for different warp operations:
//! - `sync`: Warp synchronization primitives
//! - `vote`: Warp voting and ballot operations
//! - `shuffle`: Warp shuffle operations for data exchange
//! - `reduce`: Warp reduction operations
//! - `matrix`: Warp matrix (tensor core) operations

pub mod matrix;
pub mod reduce;
pub mod shuffle;
pub mod sync;
pub mod vote;

// Re-export commonly used items at the module level
pub use matrix::{Layout, MatrixElement, TensorCore, TensorCoreShape};
pub use reduce::{BitwiseReduceValue, ReduceValue, Reduction};
pub use shuffle::{InvalidLane, Shuffle, ShuffleValue, ShuffleWidth};
pub use sync::{active_mask, lane_id, sync, sync_mask, WarpMask};
pub use vote::{AllVoteResult, AnyVoteResult, BallotResult, Predicate, Vote};

// Constants
pub const WARP_SIZE: u32 = 32;
pub const FULL_MASK: u32 = 0xFFFFFFFF;
