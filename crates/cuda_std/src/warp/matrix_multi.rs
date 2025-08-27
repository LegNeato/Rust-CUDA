//! Zero-cost compile-time abstraction for multiple matrix fragments per warp
//!
//! This module provides idiomatic Rust patterns for handling multiple matrix fragments
//! in a single warp, enabling efficient register blocking patterns like 2x8 tiles.

use super::matrix::layout::{Layout, Row};
use super::matrix::{
    Accumulator, AccumulatorElement, MatrixA, MatrixB, MatrixElement, MmaWithShapeAndLayout,
    TensorCoreShape,
};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

// ============================================================================
// Core Abstraction: Fragment Arrays with Const Generics
// ============================================================================

/// A compile-time array of matrix fragments
///
/// This is the core abstraction that provides zero-cost multiple fragments.
/// Uses const generics for compile-time size guarantees.
#[repr(transparent)]
pub struct FragmentArray<T, const N: usize> {
    fragments: [T; N],
}

impl<T, const N: usize> FragmentArray<T, N> {
    /// Create a new fragment array from an array
    #[inline(always)]
    pub const fn new(fragments: [T; N]) -> Self {
        Self { fragments }
    }

    /// Get an immutable iterator over fragments
    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.fragments.iter()
    }

    /// Get a mutable iterator over fragments
    #[inline(always)]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.fragments.iter_mut()
    }

    /// Map each fragment to a new type
    #[inline(always)]
    pub fn map<U, F>(self, f: F) -> FragmentArray<U, N>
    where
        F: FnMut(T) -> U,
    {
        FragmentArray {
            fragments: self.fragments.map(f),
        }
    }

    /// Apply a function to each fragment with its neighbor
    #[inline(always)]
    pub fn zip_with<U, V, F>(&self, other: &FragmentArray<U, N>, mut f: F) -> FragmentArray<V, N>
    where
        F: FnMut(&T, &U) -> V,
        V: Default + Copy,
    {
        let mut result = [V::default(); N];
        for i in 0..N {
            result[i] = f(&self.fragments[i], &other.fragments[i]);
        }
        FragmentArray::new(result)
    }
}

// Implement Index for ergonomic access
impl<T, const N: usize> Index<usize> for FragmentArray<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.fragments[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for FragmentArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.fragments[index]
    }
}

// Implement Default when T is Default + Copy
impl<T: Default + Copy, const N: usize> Default for FragmentArray<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            fragments: [T::default(); N],
        }
    }
}

// ============================================================================
// 2D Grid of Fragments using Const Generics
// ============================================================================

/// A 2D grid of matrix fragments for advanced tiling patterns
#[repr(transparent)]
pub struct FragmentGrid<T, const ROWS: usize, const COLS: usize> {
    grid: [[T; COLS]; ROWS],
}

impl<T, const ROWS: usize, const COLS: usize> FragmentGrid<T, ROWS, COLS> {
    /// Create a new fragment grid
    #[inline(always)]
    pub const fn new(grid: [[T; COLS]; ROWS]) -> Self {
        Self { grid }
    }

    /// Get a reference to a specific tile
    #[inline(always)]
    pub fn tile(&self, row: usize, col: usize) -> &T {
        &self.grid[row][col]
    }

    /// Get a mutable reference to a specific tile
    #[inline(always)]
    pub fn tile_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.grid[row][col]
    }

    /// Get a row of fragments
    #[inline(always)]
    pub fn row(&self, row: usize) -> &[T; COLS] {
        &self.grid[row]
    }

    /// Get a mutable row of fragments
    #[inline(always)]
    pub fn row_mut(&mut self, row: usize) -> &mut [T; COLS] {
        &mut self.grid[row]
    }

    /// Iterate over all fragments in row-major order
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.grid.iter().flat_map(|row| row.iter())
    }

    /// Map each fragment to a new type
    #[inline(always)]
    pub fn map<U, F>(&self, mut f: F) -> FragmentGrid<U, ROWS, COLS>
    where
        F: FnMut(&T) -> U,
        U: Copy + Default,
    {
        let mut result = [[U::default(); COLS]; ROWS];
        for r in 0..ROWS {
            for c in 0..COLS {
                result[r][c] = f(&self.grid[r][c]);
            }
        }
        FragmentGrid::new(result)
    }
}

impl<T: Default + Copy, const ROWS: usize, const COLS: usize> Default
    for FragmentGrid<T, ROWS, COLS>
{
    #[inline(always)]
    fn default() -> Self {
        Self {
            grid: [[T::default(); COLS]; ROWS],
        }
    }
}

// Implement 2D indexing
impl<T, const ROWS: usize, const COLS: usize> Index<(usize, usize)>
    for FragmentGrid<T, ROWS, COLS>
{
    type Output = T;

    #[inline(always)]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.grid[row][col]
    }
}

impl<T, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)>
    for FragmentGrid<T, ROWS, COLS>
{
    #[inline(always)]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.grid[row][col]
    }
}

// ============================================================================
// Builder Pattern for Ergonomic Fragment Creation
// ============================================================================

/// Builder for creating fragment arrays with type inference
pub struct FragmentBuilder<T, Shape, L = Row> {
    _phantom: PhantomData<(T, Shape, L)>,
}

impl<T, Shape, L> FragmentBuilder<T, Shape, L>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
{
    /// Create a 1D array of fragments
    #[inline(always)]
    pub fn array<const N: usize>() -> FragmentArray<MatrixA<T, Shape, L>, N>
    where
        MatrixA<T, Shape, L>: Default + Copy,
    {
        Default::default()
    }

    /// Create a 2D grid of fragments
    #[inline(always)]
    pub fn grid<const R: usize, const C: usize>() -> FragmentGrid<MatrixA<T, Shape, L>, R, C>
    where
        MatrixA<T, Shape, L>: Default + Copy,
    {
        Default::default()
    }
}

// ============================================================================
// Trait for Batch Operations
// ============================================================================

/// Trait for types that support batch loading from memory
pub trait BatchLoad<const STRIDE: usize>: Sized {
    /// Load fragments from memory with strided access
    unsafe fn load_batch(&mut self, ptr: *const u8, offset: usize);
}

/// Trait for types that support batch storing to memory
pub trait BatchStore<const STRIDE: usize>: Sized {
    /// Store fragments to memory with strided access
    unsafe fn store_batch(&self, ptr: *mut u8, offset: usize);
}

// Implement batch operations for FragmentArray
impl<T, Shape, L, const N: usize, const STRIDE: usize> BatchLoad<STRIDE>
    for FragmentArray<MatrixA<T, Shape, L>, N>
where
    T: MatrixElement,
    Shape: TensorCoreShape,
    L: Layout,
    super::matrix::StrideValidator<T, STRIDE>: super::matrix::ValidStride,
    T: super::matrix::ops::LoadMatrixA<Shape, L>,
{
    #[inline(always)]
    unsafe fn load_batch(&mut self, ptr: *const u8, offset: usize) {
        for (i, fragment) in self.fragments.iter_mut().enumerate() {
            fragment.load::<STRIDE>(ptr.add(i * offset) as *const T);
        }
    }
}

// Note: Store is not implemented for FragmentArray<Accumulator> as it would require
// the Layout type parameter at the call site. Users should iterate and store individually.

// ============================================================================
// MMA Operations for Fragment Collections
// ============================================================================

/// Extension trait for batch MMA operations
pub trait BatchMma<A, B>: Sized {
    /// Output type of the MMA operation
    type Output;

    /// Perform MMA on batches of fragments
    fn mma_batch(self, a: &A, b: &B) -> Self::Output;
}

// Implement batch MMA for arrays
impl<E, T, Shape, LA, LB, const N: usize>
    BatchMma<FragmentArray<MatrixA<E, Shape, LA>, N>, FragmentArray<MatrixB<E, Shape, LB>, N>>
    for FragmentArray<Accumulator<T, Shape>, N>
where
    E: MatrixElement,
    T: AccumulatorElement,
    Shape: TensorCoreShape,
    LA: Layout,
    LB: Layout,
    T: MmaWithShapeAndLayout<E, E, T, Shape, LA, LB, Output = T>,
{
    type Output = FragmentArray<Accumulator<T, Shape>, N>;

    #[inline(always)]
    fn mma_batch(
        self,
        a: &FragmentArray<MatrixA<E, Shape, LA>, N>,
        b: &FragmentArray<MatrixB<E, Shape, LB>, N>,
    ) -> Self::Output {
        // Use MaybeUninit for safe uninitialized memory
        unsafe {
            let mut result =
                core::mem::MaybeUninit::<FragmentArray<Accumulator<T, Shape>, N>>::uninit();
            let result_ptr = result.as_mut_ptr();
            let self_ptr = &self.fragments as *const [Accumulator<T, Shape>; N];
            let a_ptr = &a.fragments as *const [MatrixA<E, Shape, LA>; N];
            let b_ptr = &b.fragments as *const [MatrixB<E, Shape, LB>; N];

            for i in 0..N {
                let acc = core::ptr::read(&(*self_ptr)[i]);
                let mat_a = &(*a_ptr)[i];
                let mat_b = &(*b_ptr)[i];
                core::ptr::write(&mut (*result_ptr).fragments[i], acc.mma(mat_a, mat_b));
            }

            core::mem::forget(self); // We've moved the values out manually
            result.assume_init()
        }
    }
}

// ============================================================================
// Zero-Cost Pattern Matching with Const Generics
// ============================================================================

/// Helper type for compile-time tiling patterns
pub struct TilePattern<const M_TILES: usize, const N_TILES: usize, const K_TILES: usize>;

impl<const M: usize, const N: usize, const K: usize> TilePattern<M, N, K> {
    /// Create fragment arrays for this tiling pattern
    #[inline(always)]
    pub fn create_fragments<E, T, Shape>() -> (
        FragmentGrid<MatrixA<E, Shape, Row>, M, K>,
        FragmentGrid<MatrixB<E, Shape, Row>, K, N>,
        FragmentGrid<Accumulator<T, Shape>, M, N>,
    )
    where
        E: MatrixElement,
        T: AccumulatorElement,
        Shape: TensorCoreShape,
        MatrixA<E, Shape, Row>: Default + Copy,
        MatrixB<E, Shape, Row>: Default + Copy,
        Accumulator<T, Shape>: Default + Copy,
    {
        (Default::default(), Default::default(), Default::default())
    }
}

// ============================================================================
// Convenience Type Aliases
// ============================================================================

/// Common tiling pattern for Flash Attention (2x8 tiles)
pub type FlashAttentionTiles<T> = FragmentGrid<T, 2, 8>;

/// Common tiling pattern for GEMM (4x4 tiles)
pub type GemmTiles<T> = FragmentGrid<T, 4, 4>;

/// Single row of fragments
pub type FragmentRow<T, const N: usize> = FragmentArray<T, N>;

/// Single column of fragments
pub type FragmentCol<T, const N: usize> = FragmentArray<T, N>;

// ============================================================================
// Example Usage
// ============================================================================

#[cfg(test)]
mod examples {
    use super::*;
    use crate::warp::matrix::dims;
    use half::f16;

    /// Example: Flash Attention with idiomatic Rust patterns
    fn flash_attention_example() {
        // Use const generics for compile-time guarantees
        const TILE_M: usize = 2;
        const TILE_N: usize = 8;

        type Shape = dims::Shape<16, 8, 16>;

        // Create tiled fragments with type inference
        let mut q_tiles: FragmentGrid<MatrixA<f16, Shape, Row>, TILE_M, TILE_N> =
            Default::default();
        let mut k_tiles: FragmentGrid<MatrixB<f16, Shape, Row>, TILE_M, TILE_N> =
            Default::default();
        let mut acc_tiles: FragmentGrid<Accumulator<f32, Shape>, TILE_M, TILE_N> =
            Default::default();

        // Load with iterator pattern
        for (i, tile) in q_tiles.iter_mut().enumerate() {
            // tile.load(ptr.offset(i * stride), stride);
        }

        // Use indexing for specific tiles
        let tile_0_0 = &q_tiles[(0, 0)];

        // Map operations are zero-cost
        let scaled_tiles = acc_tiles.map(|tile| {
            // Scale each tile
            tile
        });
    }

    /// Example: Using the builder pattern
    fn builder_example() {
        type Shape = dims::Shape<16, 16, 16>;

        // Type inference makes this clean
        let a_tiles = FragmentBuilder::<f16, Shape>::array::<4>();
        let b_tiles = FragmentBuilder::<f16, Shape>::grid::<2, 2>();
    }

    /// Example: Pattern matching for optimization
    fn pattern_matching_example() {
        // Compiler can optimize based on const values
        const PATTERN: TilePattern<2, 4, 2> = TilePattern;

        type Shape = dims::Shape<16, 16, 16>;
        let (a, b, c) = PATTERN.create_fragments::<f16, f32, Shape>();
    }
}
