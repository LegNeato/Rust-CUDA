//! Stride implementations for f32 type

use super::{StrideValidator, ValidStride};

// Include the auto-generated implementations
include!(concat!(env!("OUT_DIR"), "/stride/f32_impls.rs"));
