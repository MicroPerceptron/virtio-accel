//! Execute the shared FP16 identity corpus through QNN on Qualcomm Hexagon HTP.

#[cfg(va_hexagon)]
mod support;

#[cfg(va_hexagon)]
fn main() {
    use virtio_accel_conformance::numerics::IDENTITY_EDGES_FP16;

    support::run_fp16_model(
        "TOSA FP16 identity",
        IDENTITY_EDGES_FP16.artifact,
        &[IDENTITY_EDGES_FP16.inputs[0].bits],
        IDENTITY_EDGES_FP16.outputs[0].bits,
    );
}

#[cfg(not(va_hexagon))]
fn main() {
    println!(
        "Qualcomm Hexagon HTP unavailable: build on Windows ARM64 with a complete QAIRT/QNN SDK"
    );
}
