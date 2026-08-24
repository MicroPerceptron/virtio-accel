//! Scaffold example: report backend availability.
//!
//! Without a detected HRX runtime this prints the placeholder state. With one, it initializes the
//! device/stream and reports the enumerated NPU. The TOSA execution path (program loading and
//! dispatch) lands in a later ticket.

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
