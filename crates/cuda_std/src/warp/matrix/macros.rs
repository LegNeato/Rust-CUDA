//! Macros to reduce repetition in matrix module

/// Macro for implementing MMA operations with common pattern
macro_rules! impl_mma {
    // Standard MMA implementation pattern
    ($elem:ty, $acc:ty, $shape:ty, $la:ty, $lb:ty, $intrinsic:ident, 
     $a_size:literal, $b_size:literal, $c_size:literal) => {
        impl MmaWithShapeAndLayout<$elem, $elem, $acc, $shape, $la, $lb> for $acc {
            type Output = $acc;

            #[gpu_only]
            fn mma(
                a: &MatrixA<$elem, $shape, $la>,
                b: &MatrixB<$elem, $shape, $lb>,
                c: &Accumulator<$acc, $shape>,
            ) -> Accumulator<$acc, $shape> {
                let mut result = Accumulator::new();

                // Extract A matrix values
                let a_vals = unsafe {
                    core::mem::transmute::<[<$elem as MatrixElement>::Storage; $a_size], [i16; $a_size]>(
                        *(&a.data[..$a_size] as *const [<$elem as MatrixElement>::Storage]
                            as *const [<$elem as MatrixElement>::Storage; $a_size]),
                    )
                };
                
                // Extract B matrix values
                let b_vals = unsafe {
                    core::mem::transmute::<[<$elem as MatrixElement>::Storage; $b_size], [i16; $b_size]>(
                        *(&b.data[..$b_size] as *const [<$elem as MatrixElement>::Storage]
                            as *const [<$elem as MatrixElement>::Storage; $b_size]),
                    )
                };
                
                // Extract C accumulator values
                let c_vals = unsafe {
                    impl_mma!(@extract_c $acc, c, $c_size)
                };

                // Call the intrinsic with unpacked values
                let result_vals = unsafe {
                    impl_mma!(@call_intrinsic $intrinsic, a_vals, b_vals, c_vals, 
                              $a_size, $b_size, $c_size)
                };

                // Store results
                impl_mma!(@store_result result, result_vals, $acc, $c_size);
                
                result
            }
        }
    };
    
    // Helper: Extract C values for f32 accumulator
    (@extract_c f32, $c:expr, $size:literal) => {
        core::mem::transmute::<[<f32 as AccumulatorElement>::Storage; $size], [f32; $size]>(
            *(&$c.data[..$size] as *const [<f32 as AccumulatorElement>::Storage]
                as *const [<f32 as AccumulatorElement>::Storage; $size]),
        )
    };
    
    // Helper: Extract C values for f16 accumulator  
    (@extract_c f16, $c:expr, $size:literal) => {
        core::mem::transmute::<[<f16 as AccumulatorElement>::Storage; $size], [i16; $size]>(
            *(&$c.data[..$size] as *const [<f16 as AccumulatorElement>::Storage]
                as *const [<f16 as AccumulatorElement>::Storage; $size]),
        )
    };
    
    // Helper: Extract C values for i32 accumulator
    (@extract_c i32, $c:expr, $size:literal) => {
        core::mem::transmute::<[<i32 as AccumulatorElement>::Storage; $size], [i32; $size]>(
            *(&$c.data[..$size] as *const [<i32 as AccumulatorElement>::Storage]
                as *const [<i32 as AccumulatorElement>::Storage; $size]),
        )
    };
    
    // Helper: Call intrinsic for 16x16x16 (16 A, 16 B, 8 C)
    (@call_intrinsic $intrinsic:ident, $a:expr, $b:expr, $c:expr, 16, 16, 8) => {
        $intrinsic(
            $a[0], $a[1], $a[2], $a[3], $a[4], $a[5], $a[6], $a[7],
            $a[8], $a[9], $a[10], $a[11], $a[12], $a[13], $a[14], $a[15],
            $b[0], $b[1], $b[2], $b[3], $b[4], $b[5], $b[6], $b[7],
            $b[8], $b[9], $b[10], $b[11], $b[12], $b[13], $b[14], $b[15],
            $c[0], $c[1], $c[2], $c[3], $c[4], $c[5], $c[6], $c[7],
        )
    };
    
    // Helper: Call intrinsic for 16x8x16 (8 A, 8 B, 4 C)
    (@call_intrinsic $intrinsic:ident, $a:expr, $b:expr, $c:expr, 8, 8, 4) => {
        $intrinsic(
            $a[0], $a[1], $a[2], $a[3], $a[4], $a[5], $a[6], $a[7],
            $b[0], $b[1], $b[2], $b[3], $b[4], $b[5], $b[6], $b[7],
            $c[0], $c[1], $c[2], $c[3],
        )
    };
    
    // Helper: Call intrinsic for 32x8x16 (16 A, 8 B, 8 C)
    (@call_intrinsic $intrinsic:ident, $a:expr, $b:expr, $c:expr, 16, 8, 8) => {
        $intrinsic(
            $a[0], $a[1], $a[2], $a[3], $a[4], $a[5], $a[6], $a[7],
            $a[8], $a[9], $a[10], $a[11], $a[12], $a[13], $a[14], $a[15],
            $b[0], $b[1], $b[2], $b[3], $b[4], $b[5], $b[6], $b[7],
            $c[0], $c[1], $c[2], $c[3], $c[4], $c[5], $c[6], $c[7],
        )
    };
    
    // Helper: Call intrinsic for 8x32x16 (8 A, 16 B, 8 C)
    (@call_intrinsic $intrinsic:ident, $a:expr, $b:expr, $c:expr, 8, 16, 8) => {
        $intrinsic(
            $a[0], $a[1], $a[2], $a[3], $a[4], $a[5], $a[6], $a[7],
            $b[0], $b[1], $b[2], $b[3], $b[4], $b[5], $b[6], $b[7],
            $b[8], $b[9], $b[10], $b[11], $b[12], $b[13], $b[14], $b[15],
            $c[0], $c[1], $c[2], $c[3], $c[4], $c[5], $c[6], $c[7],
        )
    };
    
    // Helper: Call intrinsic for 8x8x4 (2 A, 2 B, 2 C) - f64
    (@call_intrinsic $intrinsic:ident, $a:expr, $b:expr, $c:expr, 2, 2, 2) => {
        $intrinsic(
            $a[0], $a[1],
            $b[0], $b[1],
            $c[0], $c[1],
        )
    };
    
    // Helper: Store result for f32
    (@store_result $result:expr, $vals:expr, f32, $size:literal) => {
        $result.data[..$size].copy_from_slice(&$vals);
    };
    
    // Helper: Store result for f16
    (@store_result $result:expr, $vals:expr, f16, $size:literal) => {
        $result.data[..$size].copy_from_slice(&unsafe {
            core::mem::transmute::<[i16; $size], [<f16 as AccumulatorElement>::Storage; $size]>($vals)
        });
    };
    
    // Helper: Store result for i32
    (@store_result $result:expr, $vals:expr, i32, $size:literal) => {
        $result.data[..$size].copy_from_slice(&$vals);
    };
}

/// Macro for implementing all 4 layout combinations at once
macro_rules! impl_mma_all_layouts {
    ($elem:ty, $acc:ty, $shape:ty, $base_name:ident, $shape_suffix:ident, $a_size:literal, $b_size:literal, $c_size:literal) => {
        paste::paste! {
            // Row-Row
            impl_mma!($elem, $acc, $shape, layout::Row, layout::Row,
                     [<$base_name _row_row_ $shape_suffix>], $a_size, $b_size, $c_size);
            
            // Row-Col
            impl_mma!($elem, $acc, $shape, layout::Row, layout::Col,
                     [<$base_name _row_col_ $shape_suffix>], $a_size, $b_size, $c_size);
            
            // Col-Row
            impl_mma!($elem, $acc, $shape, layout::Col, layout::Row,
                     [<$base_name _col_row_ $shape_suffix>], $a_size, $b_size, $c_size);
            
            // Col-Col
            impl_mma!($elem, $acc, $shape, layout::Col, layout::Col,
                     [<$base_name _col_col_ $shape_suffix>], $a_size, $b_size, $c_size);
        }
    };
}