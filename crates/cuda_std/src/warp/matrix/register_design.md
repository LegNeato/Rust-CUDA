# Register-Based Type-Safe API Design

## Compile-Time Register Abstraction

```rust
/// Type-level register pack for fragments
/// This is purely compile-time, zero runtime cost
pub struct RegisterPack<T, const N: usize> {
    values: [T; N],
}

impl<T, const N: usize> RegisterPack<T, N> {
    pub const fn new(values: [T; N]) -> Self {
        Self { values }
    }
}

/// Trait that defines the register requirements for each shape/type combination
pub trait RegisterLayout<T: MatrixElement> {
    type RegistersA;
    type RegistersB; 
    type RegistersC<Acc: AccumulatorElement>;
    
    const A_SIZE: usize;
    const B_SIZE: usize;
    const C_SIZE: usize;
}

// Example implementation for Shape<16, 8, 16> with bf16
impl RegisterLayout<bf16> for dims::Shape<16, 8, 16> {
    type RegistersA = RegisterPack<bf16, 8>;
    type RegistersB = RegisterPack<bf16, 8>;
    type RegistersC<Acc: AccumulatorElement> = RegisterPack<Acc, 4>;
    
    const A_SIZE: usize = 8;
    const B_SIZE: usize = 8;
    const C_SIZE: usize = 4;
}

// Then the API becomes type-safe:
impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: MmaShape + RegisterLayout<T>,
    L: Layout,
{
    /// Create from a register pack - compile-time safe size
    pub fn from_register_pack(regs: Shape::RegistersA) -> Self {
        // Implementation
    }
}
```

## Even Better: Const Generic Arrays

```rust
/// Associated constants for fragment sizes
pub trait FragmentSize<T: MatrixElement> {
    const A_REGISTERS: usize;
    const B_REGISTERS: usize;
    const C_REGISTERS: usize;
}

impl FragmentSize<bf16> for dims::Shape<16, 8, 16> {
    const A_REGISTERS: usize = 8;
    const B_REGISTERS: usize = 8;  
    const C_REGISTERS: usize = 4;
}

impl<T, Shape, L> MatrixA<T, Shape, L>
where
    T: MatrixElement,
    Shape: MmaShape + FragmentSize<T>,
    L: Layout,
{
    /// Type-safe register initialization
    pub fn from_array(values: [T; Shape::A_REGISTERS]) -> Self 
    where
        [T; Shape::A_REGISTERS]:, // const generic bound
    {
        let mut fragment = Self::new();
        // Direct copy, size guaranteed at compile time
        fragment.data[..Shape::A_REGISTERS].copy_from_slice(&values);
        fragment
    }
}
```

## Benefits

1. **Compile-time size checking**: Can't pass wrong number of registers
2. **Zero runtime cost**: All resolved at compile time
3. **Self-documenting**: Types show exact register requirements
4. **No runtime bounds checking**: Array sizes known at compile time
5. **Optimizable**: Compiler can inline and optimize better

## Usage

```rust
// For MMA-only shape
type Shape = dims::Shape<16, 8, 16>;

// Compile error if wrong size!
let a_frag = MatrixA::<bf16, Shape, Row>::from_array([bf16::ZERO; 8]); // OK
// let bad = MatrixA::<bf16, Shape, Row>::from_array([bf16::ZERO; 16]); // ERROR: wrong size

// For WMMA shape - can use either API
type WmmaShape = dims::Shape<16, 16, 16>;
let b_frag = MatrixB::<bf16, WmmaShape, Row>::from_array([bf16::ONE; 16]); // Register API
let mut c_frag = MatrixB::<bf16, WmmaShape, Row>::new();
c_frag.load::<16>(ptr); // Memory API also available
```