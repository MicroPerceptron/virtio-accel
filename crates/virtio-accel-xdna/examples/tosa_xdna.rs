//! Availability example: report whether the native backend can initialize.
//!
//! Without a detected HRX runtime this prints the placeholder state. With one, it initializes the
//! device/stream and reports the enumerated NPU. This example only reports availability; see the
//! crate docs and `tests/hardware.rs` for end-to-end TOSA program loading and dispatch.

fn main() {
    match virtio_accel_xdna::XdnaAccelerator::new() {
        Ok(_backend) => {
            eprintln!("virtio-accel-xdna initialized the HRX device and stream");
        }
        Err(error) => {
            eprintln!("virtio-accel-xdna backend unavailable: {error}");
        }
    }
}
