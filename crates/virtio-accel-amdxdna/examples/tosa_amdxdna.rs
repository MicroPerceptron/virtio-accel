//! Scaffold example: the native TOSA-to-XDNA execution path is not implemented yet.
//!
//! It arrives with the HRX FFI, native `Accelerator`, and compiler-helper tickets. For now this
//! example reports the placeholder state and exits successfully, keeping the example lane green.

fn main() {
    match virtio_accel_amdxdna::AmdXdnaAccelerator::new() {
        Ok(_) => unreachable!("the scaffold placeholder never initializes a backend"),
        Err(error) => {
            eprintln!("virtio-accel-amdxdna is scaffolded but not yet executing: {error}");
        }
    }
}
