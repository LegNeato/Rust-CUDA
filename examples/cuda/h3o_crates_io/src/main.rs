use cust::memory::DeviceBuffer;
use cust::prelude::*;
use h3o::{CellIndex, LatLng, Resolution};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;

    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut test_failed = false;

    println!("H3O Geographic to Cell Conversion");
    {
        let kernel = module.get_function("lat_lng_to_cell")?;

        let lats = vec![37.7749, 40.7128, 51.5074, 35.6762];
        let lngs = vec![-122.4194, -74.0060, -0.1278, 139.6503];
        let resolution = 9u8;

        let lats_gpu = DeviceBuffer::from_slice(&lats)?;
        let lngs_gpu = DeviceBuffer::from_slice(&lngs)?;

        let mut output = vec![0u64; lats.len()];
        let output_gpu = DeviceBuffer::from_slice(&output)?;

        let grid_size = ((lats.len() + 255) / 256) as u32;
        let block_size = 256;

        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    lats_gpu.as_device_ptr(),
                    lats_gpu.len(),
                    lngs_gpu.as_device_ptr(),
                    lngs_gpu.len(),
                    resolution,
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output)?;

        println!("   GPU Results:");
        for (i, &cell_u64) in output.iter().enumerate() {
            println!("      [{}, {}] -> 0x{:x}", lats[i], lngs[i], cell_u64);
        }

        println!("   CPU Verification:");
        for i in 0..lats.len() {
            let coord = LatLng::new(lats[i], lngs[i]).expect("valid coordinates");
            let resolution = Resolution::try_from(resolution).expect("valid resolution");
            let cpu_cell = coord.to_cell(resolution);
            let cpu_u64 = u64::from(cpu_cell);

            println!("      [{}, {}] -> 0x{:x}", lats[i], lngs[i], cpu_u64);

            if output[i] != cpu_u64 {
                println!("      ERROR: Results differ for index {}", i);
                test_failed = true;
            }
        }

        if !test_failed {
            println!("   Results match!");
        }
    }

    println!("\nH3O Cell to Parent Conversion");
    {
        let kernel = module.get_function("cell_to_parent")?;

        let coord = LatLng::new(37.7749, -122.4194).expect("valid coordinates");
        let res9 = Resolution::try_from(9).expect("valid resolution");
        let cell = coord.to_cell(res9);
        let cell_u64 = u64::from(cell);

        let cells = vec![cell_u64; 4];
        let parent_resolution = 7u8;

        let cells_gpu = DeviceBuffer::from_slice(&cells)?;
        let mut output = vec![0u64; cells.len()];
        let output_gpu = DeviceBuffer::from_slice(&output)?;

        let grid_size = ((cells.len() + 255) / 256) as u32;
        let block_size = 256;

        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    cells_gpu.as_device_ptr(),
                    cells_gpu.len(),
                    parent_resolution,
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output)?;

        println!("   GPU Results:");
        for &parent in &output {
            println!("      Cell 0x{:x} -> Parent 0x{:x}", cell_u64, parent);
        }

        let res7 = Resolution::try_from(parent_resolution).expect("valid resolution");
        let cpu_parent = cell.parent(res7).expect("valid parent");
        let cpu_parent_u64 = u64::from(cpu_parent);

        println!(
            "   CPU Result: Cell 0x{:x} -> Parent 0x{:x}",
            cell_u64, cpu_parent_u64
        );

        for &gpu_parent in &output {
            if gpu_parent != cpu_parent_u64 {
                println!("   ERROR: Results differ");
                test_failed = true;
            }
        }

        if !test_failed {
            println!("   Results match!");
        }
    }

    println!("\nH3O Children Count Calculation");
    {
        let kernel = module.get_function("cell_to_children_count")?;

        let coord = LatLng::new(37.7749, -122.4194).expect("valid coordinates");
        let res5 = Resolution::try_from(5).expect("valid resolution");
        let cell = coord.to_cell(res5);
        let cell_u64 = u64::from(cell);

        let cells = vec![cell_u64; 3];
        let child_resolution = 7u8;

        let cells_gpu = DeviceBuffer::from_slice(&cells)?;
        let mut output = vec![0u64; cells.len()];
        let output_gpu = DeviceBuffer::from_slice(&output)?;

        let grid_size = ((cells.len() + 255) / 256) as u32;
        let block_size = 256;

        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    cells_gpu.as_device_ptr(),
                    cells_gpu.len(),
                    child_resolution,
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output)?;

        println!("   GPU Results:");
        for &count in &output {
            println!(
                "      Cell 0x{:x} has {} children at resolution {}",
                cell_u64, count, child_resolution
            );
        }

        let res7 = Resolution::try_from(child_resolution).expect("valid resolution");
        let cpu_count = cell.children_count(res7).expect("valid count");

        println!(
            "   CPU Result: Cell 0x{:x} has {} children at resolution {}",
            cell_u64, cpu_count, child_resolution
        );

        for &gpu_count in &output {
            if gpu_count != cpu_count {
                println!("   ERROR: Results differ");
                test_failed = true;
            }
        }

        if !test_failed {
            println!("   Results match!");
        }
    }

    println!("\nH3O Area Calculation");
    {
        let kernel = module.get_function("cell_area_calculation")?;

        let coords = vec![
            LatLng::new(37.7749, -122.4194).expect("valid coordinates"),
            LatLng::new(40.7128, -74.0060).expect("valid coordinates"),
        ];

        let res9 = Resolution::try_from(9).expect("valid resolution");
        let cells: Vec<u64> = coords
            .iter()
            .map(|coord| u64::from(coord.to_cell(res9)))
            .collect();

        let cells_gpu = DeviceBuffer::from_slice(&cells)?;
        let mut output_m2 = vec![0.0f64; cells.len()];
        let mut output_km2 = vec![0.0f64; cells.len()];
        let output_m2_gpu = DeviceBuffer::from_slice(&output_m2)?;
        let output_km2_gpu = DeviceBuffer::from_slice(&output_km2)?;

        let grid_size = ((cells.len() + 255) / 256) as u32;
        let block_size = 256;

        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    cells_gpu.as_device_ptr(),
                    cells_gpu.len(),
                    output_m2_gpu.as_device_ptr(),
                    output_km2_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_m2_gpu.copy_to(&mut output_m2)?;
        output_km2_gpu.copy_to(&mut output_km2)?;

        println!("   GPU Results:");
        for (i, &cell_u64) in cells.iter().enumerate() {
            println!(
                "      Cell 0x{:x}: {:.2} m², {:.6} km²",
                cell_u64, output_m2[i], output_km2[i]
            );
        }

        println!("   CPU Verification:");
        for (i, &cell_u64) in cells.iter().enumerate() {
            let cell = CellIndex::try_from(cell_u64).expect("valid cell");
            let cpu_m2 = cell.area_m2();
            let cpu_km2 = cell.area_km2();

            println!(
                "      Cell 0x{:x}: {:.2} m², {:.6} km²",
                cell_u64, cpu_m2, cpu_km2
            );

            let m2_diff = (output_m2[i] - cpu_m2).abs();
            let km2_diff = (output_km2[i] - cpu_km2).abs();

            if m2_diff > 0.01 || km2_diff > 0.000001 {
                println!("      ERROR: Results differ significantly");
                test_failed = true;
            }
        }

        if !test_failed {
            println!("   Results match!");
        }
    }

    if test_failed {
        Err("One or more tests failed".into())
    } else {
        println!("\nAll H3O tests passed successfully!");
        Ok(())
    }
}
