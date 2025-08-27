// Test multi-fragment support with FragmentArray and FragmentGrid
// build-pass

#![no_std]

use cuda_std::prelude::*;
use cuda_std::warp::matrix::{dims, layout, Accumulator, MatrixA, MatrixB};
use cuda_std::warp::matrix_multi::{BatchLoad, BatchMma, FragmentArray, FragmentGrid};
use cuda_std::{bf16, f16};

#[kernel]
pub unsafe fn test_fragment_array() {
    type Shape = dims::Shape<16, 8, 16>;

    // Create an array of 4 matrix fragments
    let mut a_fragments: FragmentArray<MatrixA<f16, Shape, layout::Row>, 4> = Default::default();
    let mut b_fragments: FragmentArray<MatrixB<f16, Shape, layout::Row>, 4> = Default::default();
    let mut acc_fragments: FragmentArray<Accumulator<f32, Shape>, 4> = Default::default();

    // Load fragments with batch operation
    let ptr = 0 as *const f16;
    for i in 0..4 {
        a_fragments[i].load::<16>(ptr.add(i * 256));
        b_fragments[i].load::<16>(ptr.add(i * 256));
    }

    // Perform batch MMA
    acc_fragments = acc_fragments.mma_batch(&a_fragments, &b_fragments);

    // Store results
    let out_ptr = 0 as *mut f32;
    for i in 0..4 {
        acc_fragments[i].store::<layout::Row, 16>(out_ptr.add(i * 128));
    }
}

#[kernel]
pub unsafe fn test_fragment_grid() {
    type Shape = dims::Shape<16, 16, 16>;

    // Create a 2x2 grid of fragments for tiling
    let mut q_grid: FragmentGrid<MatrixA<f16, Shape, layout::Row>, 2, 2> = Default::default();
    let mut k_grid: FragmentGrid<MatrixB<f16, Shape, layout::Row>, 2, 2> = Default::default();
    let mut v_grid: FragmentGrid<MatrixB<f16, Shape, layout::Row>, 2, 2> = Default::default();
    let mut acc_grid: FragmentGrid<Accumulator<f32, Shape>, 2, 2> = Default::default();

    let ptr = 0 as *const f16;

    // Load tiles into grid
    for r in 0..2 {
        for c in 0..2 {
            q_grid[(r, c)].load::<16>(ptr.add((r * 2 + c) * 512));
            k_grid[(r, c)].load::<16>(ptr.add((r * 2 + c) * 512));
        }
    }

    // Perform MMA on each tile
    for r in 0..2 {
        for c in 0..2 {
            acc_grid[(r, c)] = acc_grid[(r, c)].mma(&q_grid[(r, c)], &k_grid[(r, c)]);
        }
    }

    // Map operation on grid
    let _scaled = acc_grid.map(|tile| {
        // In real code, would scale the tile
        *tile
    });
}

#[kernel]
pub unsafe fn test_flash_attention_pattern() {
    type Shape = dims::Shape<16, 8, 16>;

    // Flash Attention uses 2x8 tiling
    type QTiles = FragmentGrid<MatrixA<bf16, Shape, layout::Row>, 2, 8>;
    type KTiles = FragmentGrid<MatrixB<bf16, Shape, layout::Row>, 2, 8>;
    type VTiles = FragmentGrid<MatrixB<bf16, Shape, layout::Row>, 2, 8>;
    type AccTiles = FragmentGrid<Accumulator<f32, Shape>, 2, 8>;

    let mut q_tiles: QTiles = Default::default();
    let mut k_tiles: KTiles = Default::default();
    let mut v_tiles: VTiles = Default::default();
    let mut s_tiles: AccTiles = Default::default();

    let q_ptr = 0 as *const bf16;
    let k_ptr = 0 as *const bf16;

    // Load Q and K tiles
    for i in 0..2 {
        for j in 0..8 {
            q_tiles[(i, j)].load::<16>(q_ptr.add((i * 8 + j) * 256));
            k_tiles[(i, j)].load::<16>(k_ptr.add((i * 8 + j) * 256));
        }
    }

    // Compute attention scores S = Q @ K^T
    for i in 0..2 {
        for j in 0..8 {
            s_tiles[(i, j)] = s_tiles[(i, j)].mma(&q_tiles[(i, j)], &k_tiles[(i, j)]);
        }
    }
}
