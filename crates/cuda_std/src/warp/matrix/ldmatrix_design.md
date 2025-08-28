# LdMatrix Integration Design

## Overview
`ldmatrix` provides a way to load matrix data from shared memory into registers formatted for MMA operations. This enables memory loading for MMA-only shapes like Shape<16, 8, 16>.

## Trait Hierarchy

```rust
// Base trait - all shapes support MMA compute
pub trait MmaShape: TensorCoreShape {
    // MMA compute operations
}

// Shapes that support loading from shared memory
pub trait LdMatrixShape: MmaShape {
    const LDMATRIX_SHAPE: LdMatrixShapeType;
    const LDMATRIX_NUM: LdMatrixNum;
}

// Shapes that support WMMA load/store from global memory  
pub trait WmmaShape: MmaShape {
    // WMMA global memory operations
}

// Some shapes implement both!
impl LdMatrixShape for Shape<16, 16, 16> { ... }
impl WmmaShape for Shape<16, 16, 16> { ... }

// Shape<16, 8, 16> only supports ldmatrix (not WMMA)
impl LdMatrixShape for Shape<16, 8, 16> { 
    const LDMATRIX_SHAPE: LdMatrixShapeType = LdMatrixShapeType::M8N8;
    const LDMATRIX_NUM: LdMatrixNum = LdMatrixNum::X2; // Need 2 matrices for 16x8
}
```

## API Methods

```rust
impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: MmaShape + FragmentSize<T>,
    L: Layout,
{
    /// Available for all MMA shapes - register initialization
    pub fn from_array(values: [T; Shape::A_REGISTERS]) -> Self { ... }
    pub fn splat(value: T) -> Self { ... }
}

impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: LdMatrixShape + FragmentSize<T>,
    L: Layout,
{
    /// Load from shared memory using ldmatrix
    pub unsafe fn load_shared(&mut self, ptr: *const T, stride: usize) {
        // Use ldmatrix intrinsic
    }
}

impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: WmmaShape + FragmentSize<T>,
    L: Layout,
{
    /// Load from global memory using WMMA
    pub unsafe fn load<const STRIDE: usize>(&mut self, ptr: *const T) {
        // Use WMMA load intrinsic
    }
}
```

## Usage Examples

```rust
// Shape<16, 8, 16> - MMA + ldmatrix (no WMMA)
type Shape = dims::Shape<16, 8, 16>;

// Can use register API
let a = MatrixA::<bf16, Shape, Row>::from_array([bf16::ZERO; 8]);

// Can use shared memory load
extern "C" __shared__ static mut SMEM: [bf16; 1024];
let mut b = MatrixA::<bf16, Shape, Row>::new();
b.load_shared(SMEM.as_ptr(), 16);  // OK - has LdMatrixShape

// Cannot use global memory load
// b.load::<16>(global_ptr);  // ERROR: Shape doesn't implement WmmaShape

// Shape<16, 16, 16> - has all three APIs
type WmmaShape = dims::Shape<16, 16, 16>;

let mut c = MatrixA::<bf16, WmmaShape, Row>::new();
c.load_shared(SMEM.as_ptr(), 16);  // OK - has LdMatrixShape  
c.load::<16>(global_ptr);          // OK - has WmmaShape
```

## Implementation Strategy

1. Define ldmatrix intrinsics for each shape/type combination
2. Implement LdMatrixShape trait for applicable shapes
3. Add load_shared methods conditionally based on LdMatrixShape
4. Update documentation to explain memory hierarchy

## Benefits

1. **Type-safe**: Compile-time enforcement of what each shape supports
2. **Ergonomic**: Same API style, just different method names for different memory
3. **Zero-cost**: All resolved at compile time
4. **Clear semantics**: Method names indicate memory source
5. **Flexible**: Shapes can support multiple loading strategies