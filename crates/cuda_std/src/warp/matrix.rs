//! Extremely type-safe and ergonomic warp matrix (tensor core) operations.
//!
//! This module makes tensor core operations foolproof through the type system.
//! Invalid operations won't compile, and valid operations are intuitive.

use crate::gpu_only;
use core::marker::PhantomData;
use half::{bf16, f16};

// ============================================================================
// Shape Types with Compile-Time Validation
// ============================================================================

/// Type-level dimensions for matrix operations
pub mod dims {
    /// Complete shape specification
    pub struct Shape<const M: usize, const N: usize, const K: usize>;
}

/// Valid tensor core shape combinations
pub trait TensorCoreShape: sealed::Sealed {
    const M: usize;
    const N: usize;
    const K: usize;
}

// Only these exact combinations are valid for tensor cores
impl TensorCoreShape for dims::Shape<16, 16, 16> {
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 16;
}

impl TensorCoreShape for dims::Shape<32, 8, 16> {
    const M: usize = 32;
    const N: usize = 8;
    const K: usize = 16;
}

impl TensorCoreShape for dims::Shape<8, 32, 16> {
    const M: usize = 8;
    const N: usize = 32;
    const K: usize = 16;
}

impl TensorCoreShape for dims::Shape<16, 16, 8> {
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 8;
}

impl TensorCoreShape for dims::Shape<8, 8, 32> {
    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 32;
}

impl TensorCoreShape for dims::Shape<8, 8, 128> {
    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 128;
}

impl TensorCoreShape for dims::Shape<8, 8, 4> {
    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 4;
}

// ============================================================================
// Layout Types
// ============================================================================

/// Type-level layout specification
pub mod layout {
    /// Row-major layout
    pub struct Row;

    /// Column-major layout
    pub struct Col;
}

/// Trait for valid layouts
pub trait Layout: sealed::Sealed {
    const IS_ROW_MAJOR: bool;
}

impl Layout for layout::Row {
    const IS_ROW_MAJOR: bool = true;
}

impl Layout for layout::Col {
    const IS_ROW_MAJOR: bool = false;
}

// ============================================================================
// Element Types with Compatibility Rules
// ============================================================================

/// Trait for types that can be matrix elements
pub trait MatrixElement: Copy + sealed::Sealed {
    /// The accumulator type this element requires
    type Accumulator: AccumulatorElement;

    /// Storage type in fragment
    type Storage: Copy;

    /// Number of elements per thread
    const ELEMENTS_PER_THREAD: usize;
}

/// Trait for types that can be accumulator elements
pub trait AccumulatorElement: Copy + sealed::Sealed {
    type Storage: Copy;
    const ELEMENTS_PER_THREAD: usize;
}

// f16 matrices accumulate to f32 or f16
impl MatrixElement for f16 {
    type Accumulator = f32; // Default to f32 for better precision
    type Storage = f16;
    const ELEMENTS_PER_THREAD: usize = 16;
}

// bf16 matrices accumulate to f32
impl MatrixElement for bf16 {
    type Accumulator = f32;
    type Storage = bf16;
    const ELEMENTS_PER_THREAD: usize = 16;
}

// i8 matrices accumulate to i32
impl MatrixElement for i8 {
    type Accumulator = i32;
    type Storage = i32; // Packed
    const ELEMENTS_PER_THREAD: usize = 4;
}

// u8 matrices accumulate to i32
impl MatrixElement for u8 {
    type Accumulator = i32;
    type Storage = i32; // Packed
    const ELEMENTS_PER_THREAD: usize = 4;
}

// f32 matrices (TF32) accumulate to f32
impl MatrixElement for f32 {
    type Accumulator = f32;
    type Storage = f32;
    const ELEMENTS_PER_THREAD: usize = 8;
}

// i32 matrices accumulate to i32 (for integer tensor cores)
impl MatrixElement for i32 {
    type Accumulator = i32;
    type Storage = i32; // Packed
    const ELEMENTS_PER_THREAD: usize = 4;
}

impl AccumulatorElement for f32 {
    type Storage = f32;
    const ELEMENTS_PER_THREAD: usize = 8;
}

impl AccumulatorElement for f16 {
    type Storage = f16;
    const ELEMENTS_PER_THREAD: usize = 8;
}

impl AccumulatorElement for i32 {
    type Storage = i32;
    const ELEMENTS_PER_THREAD: usize = 8;
}

impl AccumulatorElement for f64 {
    type Storage = f64;
    const ELEMENTS_PER_THREAD: usize = 8;
}

// ============================================================================
// Matrix Fragments with Role-Specific Types
// ============================================================================

/// Matrix A fragment (left operand)
#[repr(C)]
pub struct MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    data: [T::Storage; 32], // Max size
    _phantom: PhantomData<(T, Shape, L)>,
}

/// Matrix B fragment (right operand)
#[repr(C)]
pub struct MatrixB<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    data: [T::Storage; 32], // Max size
    _phantom: PhantomData<(T, Shape, L)>,
}

/// Accumulator matrix fragment
#[repr(C)]
pub struct Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    data: [T::Storage; 32], // Max size
    _phantom: PhantomData<(T, Shape)>,
}

// ============================================================================
// Safe Constructors and Operations
// ============================================================================

impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    /// Create a new matrix A fragment
    #[inline]
    pub fn new() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
            _phantom: PhantomData,
        }
    }

    /// Load from memory with compile-time stride validation
    #[gpu_only]
    pub unsafe fn load<const STRIDE: usize>(&mut self, _ptr: *const T)
    where
        StrideValidator<T, STRIDE>: ValidStride,
    {
        // LLVM intrinsic call would go here
        // The StrideValidator ensures STRIDE is valid at compile time
    }
}

impl<T, Shape, L> Default for MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Shape, L> MatrixB<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    /// Create a new matrix B fragment
    #[inline]
    pub fn new() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
            _phantom: PhantomData,
        }
    }

    /// Load from memory with compile-time stride validation
    #[gpu_only]
    pub unsafe fn load<const STRIDE: usize>(&mut self, _ptr: *const T)
    where
        StrideValidator<T, STRIDE>: ValidStride,
    {
        // LLVM intrinsic call would go here
    }
}

impl<T, Shape, L> Default for MatrixB<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Shape> Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    /// Create a new accumulator fragment
    #[inline]
    pub fn new() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
            _phantom: PhantomData,
        }
    }

    /// Fill with a constant value
    #[gpu_only]
    pub fn fill(&mut self, value: T) {
        for i in 0..T::ELEMENTS_PER_THREAD {
            self.data[i] = unsafe { core::mem::transmute_copy(&value) };
        }
    }

    /// Store to memory with layout specification
    #[gpu_only]
    pub unsafe fn store<L, const STRIDE: usize>(&self, _ptr: *mut T)
    where
        L: Layout,
        StrideValidator<T, STRIDE>: ValidStride,
    {
        // LLVM intrinsic call would go here
    }
}

impl<T, Shape> Default for Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type-Safe Matrix Multiply-Accumulate
// ============================================================================

/// Trait for valid MMA operations - only implemented for valid combinations
pub trait Mma<A, B, C>: sealed::Sealed
where
    A: MatrixElement,
    B: MatrixElement,
    C: AccumulatorElement,
{
    type Output: AccumulatorElement;

    /// Perform the matrix multiply-accumulate operation
    fn mma<S, LA, LB>(
        a: &MatrixA<A, S, LA>,
        b: &MatrixB<B, S, LB>,
        c: &Accumulator<C, S>,
    ) -> Accumulator<Self::Output, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout;
}

/// Implementation for f16 × f16 + f32 → f32
impl Mma<f16, f16, f32> for f32 {
    type Output = f32;

    #[gpu_only]
    fn mma<S, LA, LB>(
        _a: &MatrixA<f16, S, LA>,
        _b: &MatrixB<f16, S, LB>,
        c: &Accumulator<f32, S>,
    ) -> Accumulator<f32, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        // LLVM intrinsic call would go here
        // For now, return a copy of c
        Accumulator {
            data: c.data,
            _phantom: PhantomData,
        }
    }
}

/// Implementation for f16 × f16 + f16 → f16
impl Mma<f16, f16, f16> for f16 {
    type Output = f16;

    #[gpu_only]
    fn mma<S, LA, LB>(
        _a: &MatrixA<f16, S, LA>,
        _b: &MatrixB<f16, S, LB>,
        c: &Accumulator<f16, S>,
    ) -> Accumulator<f16, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        Accumulator {
            data: c.data,
            _phantom: PhantomData,
        }
    }
}

/// Implementation for i8 × i8 + i32 → i32
impl Mma<i8, i8, i32> for i32 {
    type Output = i32;

    #[gpu_only]
    fn mma<S, LA, LB>(
        _a: &MatrixA<i8, S, LA>,
        _b: &MatrixB<i8, S, LB>,
        c: &Accumulator<i32, S>,
    ) -> Accumulator<i32, S>
    where
        S: TensorCoreShape,
        LA: Layout,
        LB: Layout,
    {
        Accumulator {
            data: c.data,
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// Stride Validation
// ============================================================================

/// Compile-time stride validation
pub struct StrideValidator<T, const STRIDE: usize>(PhantomData<T>);

/// Trait for valid strides
pub trait ValidStride: sealed::Sealed {}

// f16 requires stride to be multiple of 8
impl ValidStride for StrideValidator<f16, 8> {}
impl ValidStride for StrideValidator<f16, 16> {}
impl ValidStride for StrideValidator<f16, 24> {}
impl ValidStride for StrideValidator<f16, 32> {}
impl ValidStride for StrideValidator<f16, 40> {}
impl ValidStride for StrideValidator<f16, 48> {}
impl ValidStride for StrideValidator<f16, 56> {}
impl ValidStride for StrideValidator<f16, 64> {}

// f32 requires stride to be multiple of 4
impl ValidStride for StrideValidator<f32, 4> {}
impl ValidStride for StrideValidator<f32, 8> {}
impl ValidStride for StrideValidator<f32, 12> {}
impl ValidStride for StrideValidator<f32, 16> {}
impl ValidStride for StrideValidator<f32, 20> {}
impl ValidStride for StrideValidator<f32, 24> {}
impl ValidStride for StrideValidator<f32, 28> {}
impl ValidStride for StrideValidator<f32, 32> {}

// ============================================================================
// Ergonomic Builder API
// ============================================================================

/// Entry point for tensor core operations with type-driven API
pub struct TensorCore<T: MatrixElement, Shape: TensorCoreShape> {
    _phantom: PhantomData<(T, Shape)>,
}

impl<T: MatrixElement, Shape: TensorCoreShape> TensorCore<T, Shape> {
    /// Create a new tensor core operation for the given element type and shape
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Create matrix A with specified layout
    pub fn matrix_a<L: Layout>(&self) -> MatrixA<T, Shape, L> {
        MatrixA::new()
    }

    /// Create matrix B with specified layout
    pub fn matrix_b<L: Layout>(&self) -> MatrixB<T, Shape, L> {
        MatrixB::new()
    }
}

// Specialized implementations for different element types
impl<Shape: TensorCoreShape> TensorCore<f16, Shape> {
    /// Create an f32 accumulator (default precision for f16 inputs)
    pub fn accumulator(&self) -> Accumulator<f32, Shape> {
        Accumulator::new()
    }

    /// Create an f16 accumulator (lower precision, potentially faster)
    pub fn accumulator_f16(&self) -> Accumulator<f16, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<bf16, Shape> {
    /// Create an f32 accumulator for bf16 operations
    pub fn accumulator(&self) -> Accumulator<f32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<f32, Shape> {
    /// Create an f32 accumulator for TF32 operations
    pub fn accumulator(&self) -> Accumulator<f32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<i8, Shape> {
    /// Create an i32 accumulator (required for i8 operations)
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<u8, Shape> {
    /// Create an i32 accumulator (required for u8 operations)
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

impl<Shape: TensorCoreShape> TensorCore<i32, Shape> {
    /// Create an i32 accumulator for integer tensor cores
    pub fn accumulator(&self) -> Accumulator<i32, Shape> {
        Accumulator::new()
    }
}

// Builder structs are no longer needed with the type-driven API
// The TensorCore type itself now provides all the necessary methods

// ============================================================================
// Extension trait for ergonomic MMA operations
// ============================================================================

/// Extension trait for accumulator to provide ergonomic MMA syntax
pub trait MmaExt<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    /// Perform MMA: self = a × b + self
    fn mma_inplace<E, LA, LB>(&mut self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>)
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>;

    /// Perform MMA: result = a × b + self
    fn mma<E, LA, LB>(self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>) -> Self
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>;
}

impl<T, Shape> MmaExt<T, Shape> for Accumulator<T, Shape>
where
    T: AccumulatorElement,
    Shape: TensorCoreShape,
{
    fn mma_inplace<E, LA, LB>(&mut self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>)
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>,
    {
        *self = T::mma(a, b, self);
    }

    fn mma<E, LA, LB>(self, a: &MatrixA<E, Shape, LA>, b: &MatrixB<E, Shape, LB>) -> Self
    where
        E: MatrixElement<Accumulator = T>,
        LA: Layout,
        LB: Layout,
        T: Mma<E, E, T, Output = T>,
    {
        T::mma(a, b, &self)
    }
}

// ============================================================================
// Convenient type aliases
// ============================================================================

// Type aliases removed - TensorCore now requires element type parameter
// Users should use TensorCore::<T, dims::Shape<M, N, K>>::new() directly

// ============================================================================
// Private sealing
// ============================================================================

mod sealed {
    use super::*;

    pub trait Sealed {}

    // Seal shapes
    impl Sealed for dims::Shape<16, 16, 16> {}
    impl Sealed for dims::Shape<32, 8, 16> {}
    impl Sealed for dims::Shape<8, 32, 16> {}
    impl Sealed for dims::Shape<16, 16, 8> {}
    impl Sealed for dims::Shape<8, 8, 32> {}
    impl Sealed for dims::Shape<8, 8, 128> {}
    impl Sealed for dims::Shape<8, 8, 4> {}

    // Seal layouts
    impl Sealed for layout::Row {}
    impl Sealed for layout::Col {}

    // Seal elements
    impl Sealed for f16 {}
    impl Sealed for bf16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i32 {}

    // Seal stride validators
    impl<T, const S: usize> Sealed for StrideValidator<T, S> {}
}
