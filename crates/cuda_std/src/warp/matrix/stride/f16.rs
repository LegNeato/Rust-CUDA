//! Stride implementations for f16 type

use super::{StrideValidator, ValidStride};
use crate::f16;

// Include the auto-generated implementations
include!(concat!(env!("OUT_DIR"), "/stride/f16_impls.rs"));