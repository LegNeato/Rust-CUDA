# MMA-Only Shape API Design

## Problem
Shape<16, 8, 16> supports MMA compute but not WMMA load/store. We need an ergonomic way to:
1. Get data into fragments for MMA-only shapes
2. Keep the API consistent with WMMA shapes where possible
3. Make it compile-time safe with zero runtime cost

## Proposed Solution

### 1. Register-based construction (works for ALL shapes)
```rust
impl<T, Shape, L> MatrixA<T, Shape, L> 
where 
    T: MatrixElement,
    Shape: MmaShape,  // Note: only requires MmaShape, not WmmaShape
    L: Layout,
{
    /// Create from register values directly
    /// This works for both WMMA and MMA-only shapes
    pub fn from_registers(values: &[T]) -> Self {
        // Implementation details
    }
    
    /// Fill with a single value (broadcast)
    pub fn splat(value: T) -> Self {
        // Implementation details
    }
}
```

### 2. Conditional load method (only available for WMMA shapes)
```rust
impl<T, Shape, L> MatrixA<T, Shape, L> 
where 
    Shape: WmmaShape,  // Only for WMMA-capable shapes
{
    pub unsafe fn load<const STRIDE: usize>(&mut self, ptr: *const T) { ... }
}
```

### 3. Builder pattern for complex initialization
```rust
impl<T, Shape, L> MatrixA<T, Shape, L>
where
    Shape: MmaShape,
{
    pub fn builder() -> MatrixBuilder<T, Shape, L> { ... }
}

pub struct MatrixBuilder<T, Shape, L> { ... }

impl<T, Shape, L> MatrixBuilder<T, Shape, L> {
    pub fn set_lane(self, lane: usize, value: T) -> Self { ... }
    pub fn set_row(self, row: usize, values: &[T]) -> Self { ... }
    pub fn build(self) -> MatrixA<T, Shape, L> { ... }
}
```

### 4. Associated types for fragment size
```rust
pub trait MmaShape: TensorCoreShape {
    type FragmentA<T: MatrixElement>: AsRef<[T::Storage]>;
    type FragmentB<T: MatrixElement>: AsRef<[T::Storage]>;
    type FragmentC<T: AccumulatorElement>: AsRef<[T::Storage]>;
}
```

## Usage Examples

### For WMMA shapes (unchanged):
```rust
// Traditional WMMA load still works
let mut a_frag = tc.matrix_a();
a_frag.load::<16>(data_ptr);
```

### For MMA-only shapes:
```rust
// Shape<16, 8, 16> - MMA only
type Shape = dims::Shape<16, 8, 16>;
let tc = TensorCore::<bf16, Shape>::new();

// Option 1: From registers
let values = [bf16::from_f32(1.0); 8];
let a_frag = MatrixA::from_registers(&values);

// Option 2: Splat
let b_frag = MatrixB::splat(bf16::from_f32(2.0));

// Option 3: Builder
let c_frag = Accumulator::builder()
    .set_lane(0, 1.0)
    .set_lane(1, 2.0)
    .build();

// MMA operations work the same way
let result = c_frag.mma(&a_frag, &b_frag);
```

## Benefits
1. **Ergonomic**: Similar API for both WMMA and MMA shapes
2. **Type-safe**: Compile-time errors if you try to load on MMA-only shapes
3. **Zero-cost**: All resolved at compile time
4. **Intuitive**: Methods clearly indicate data source (registers vs memory)
5. **Flexible**: Multiple ways to construct fragments based on needs