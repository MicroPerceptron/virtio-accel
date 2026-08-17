//! Backend-local entry point for the Qualcomm Hexagon TOSA path.

use virtio_accel_hexagon::{HexagonAccelerator, TESTED_QAIRT_RELEASE};

fn main() {
    match HexagonAccelerator::new() {
        Ok(_) => {
            println!(
                "QAIRT {TESTED_QAIRT_RELEASE} was detected, but native TOSA execution is not enabled in this revision"
            );
        }
        Err(error) => {
            println!(
                "Qualcomm Hexagon backend unavailable ({error}); install the complete QAIRT/QNN C SDK to enable native development"
            );
        }
    }
}
