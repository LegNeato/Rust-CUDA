use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../..")
        .copy_to("kernels/src/lib.rs")
        .build()
        .unwrap();
}