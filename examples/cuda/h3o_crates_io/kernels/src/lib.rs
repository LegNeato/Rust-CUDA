#![no_std]
#![feature(abi_ptx)]

use cuda_std::prelude::*;
use h3o::{CellIndex, LatLng, Resolution};

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn lat_lng_to_cell(
    lats: &[f64],
    lngs: &[f64],
    resolution: u8,
    output: *mut u64,
) {
    let idx = thread::index_1d() as usize;
    
    if idx < lats.len() && idx < lngs.len() {
        let lat = lats[idx];
        let lng = lngs[idx];
        
        let coord = LatLng::new(lat, lng).expect("valid coordinates");
        let resolution = Resolution::try_from(resolution).expect("valid resolution");
        let cell = coord.to_cell(resolution);
        
        unsafe {
            let output_slice = core::slice::from_raw_parts_mut(output, lats.len());
            output_slice[idx] = u64::from(cell);
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn cell_to_parent(
    cells: &[u64],
    parent_resolution: u8,
    output: *mut u64,
) {
    let idx = thread::index_1d() as usize;
    
    if idx < cells.len() {
        let cell = CellIndex::try_from(cells[idx]).expect("valid cell");
        let resolution = Resolution::try_from(parent_resolution).expect("valid resolution");
        let parent = cell.parent(resolution).expect("valid parent");
        
        unsafe {
            let output_slice = core::slice::from_raw_parts_mut(output, cells.len());
            output_slice[idx] = u64::from(parent);
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn cell_to_children_count(
    cells: &[u64],
    child_resolution: u8,
    output: *mut u64,
) {
    let idx = thread::index_1d() as usize;
    
    if idx < cells.len() {
        let cell = CellIndex::try_from(cells[idx]).expect("valid cell");
        let resolution = Resolution::try_from(child_resolution).expect("valid resolution");
        let children_count = cell.children_count(resolution);
        
        unsafe {
            let output_slice = core::slice::from_raw_parts_mut(output, cells.len());
            output_slice[idx] = children_count;
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn cell_area_calculation(
    cells: &[u64],
    output_m2: *mut f64,
    output_km2: *mut f64,
) {
    let idx = thread::index_1d() as usize;
    
    if idx < cells.len() {
        let cell = CellIndex::try_from(cells[idx]).expect("valid cell");
        let area_m2 = cell.area_m2();
        let area_km2 = cell.area_km2();
        
        unsafe {
            let output_m2_slice = core::slice::from_raw_parts_mut(output_m2, cells.len());
            let output_km2_slice = core::slice::from_raw_parts_mut(output_km2, cells.len());
            output_m2_slice[idx] = area_m2;
            output_km2_slice[idx] = area_km2;
        }
    }
}