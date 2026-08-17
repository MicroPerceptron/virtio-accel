//! Run a tiny FP16 linear classifier on Qualcomm Hexagon HTP.
//!
//! The graph computes two sets of class logits from three input features using a learned 3x2
//! weight matrix. The weights are direct-bound inputs because embedded model constants are outside
//! the initial Hexagon tier.

#[cfg(va_hexagon)]
mod support;

#[cfg(va_hexagon)]
fn main() {
    use virtio_accel_conformance::numerics::MOCK_LINEAR_CLASSIFIER_FP16;

    support::run_fp16_model(
        "mock FP16 linear classifier",
        MOCK_LINEAR_CLASSIFIER_FP16.artifact,
        &[
            MOCK_LINEAR_CLASSIFIER_FP16.inputs[0].bits,
            MOCK_LINEAR_CLASSIFIER_FP16.inputs[1].bits,
        ],
        MOCK_LINEAR_CLASSIFIER_FP16.outputs[0].bits,
    );
}

#[cfg(not(va_hexagon))]
fn main() {
    println!(
        "Qualcomm Hexagon HTP unavailable: build on Windows ARM64 with a complete QAIRT/QNN SDK"
    );
}
