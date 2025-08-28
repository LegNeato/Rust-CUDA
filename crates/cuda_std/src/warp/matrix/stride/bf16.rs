//! Stride implementations for bf16 type

use super::{StrideValidator, ValidStride};
use crate::bf16;

// Include the auto-generated implementations
include!(concat!(env!("OUT_DIR"), "/stride/bf16_impls.rs"));