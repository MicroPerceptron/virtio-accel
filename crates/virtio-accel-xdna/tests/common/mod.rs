use std::time::{Duration, Instant};

use virtio_accel_core::{Accelerator, BackendError, EventState};
use virtio_accel_tosa::DType;
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
use virtio_accel_xdna::{XDNA_TOSA_FP8_TARGET, XDNA_TOSA_INTEGER_TARGET, XDNA_TOSA_TARGET};

pub fn bf16_identity_tosa(elements: usize) -> Vec<u8> {
    let shape = vec![1, 1, elements as i32];
    let mut graph = OwnedGraph::new("main");
    graph.push_tensor(OwnedTensor::new("x", shape.clone(), DType::BF16));
    graph.push_tensor(OwnedTensor::new("y", shape, DType::BF16));
    graph.push_operator(OwnedOperator::new(
        OperatorKind::Identity,
        vec!["x".into()],
        vec!["y".into()],
    ));
    graph.push_input("x");
    graph.push_output("y");
    graph.build(XDNA_TOSA_TARGET).expect("build bf16 identity")
}

#[allow(dead_code)] // The hardware benchmark uses this; conformance.rs compiles this module too.
pub fn fp8e4m3_to_bf16_tosa(elements: usize) -> Vec<u8> {
    let elements = i32::try_from(elements).expect("FP8 benchmark shape fits i32");
    let shape = vec![elements];
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", shape.clone(), DType::FP8E4M3))
        .push_tensor(OwnedTensor::new("y", shape, DType::BF16))
        .push_operator(OwnedOperator::new(
            OperatorKind::Cast,
            vec!["x".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    graph
        .build(XDNA_TOSA_FP8_TARGET)
        .expect("build fp8e4m3 to bf16 cast")
}

#[allow(dead_code)] // This shared module is compiled separately into conformance.rs, which uses only IDENTITY.
pub fn bf16_matmul_tosa(m: i32, k: i32, n: i32) -> Vec<u8> {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("lhs", vec![1, m, k], DType::BF16))
        .push_tensor(OwnedTensor::new("rhs", vec![1, k, n], DType::BF16))
        .push_tensor(OwnedTensor::constant(
            "lhs_zp",
            vec![1],
            DType::BF16,
            vec![0u8; 2],
        ))
        .push_tensor(OwnedTensor::constant(
            "rhs_zp",
            vec![1],
            DType::BF16,
            vec![0u8; 2],
        ))
        .push_tensor(OwnedTensor::new("output", vec![1, m, n], DType::FP32))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["lhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["rhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["lhs".into(), "rhs".into(), "lhs_zp".into(), "rhs_zp".into()],
            vec!["output".into()],
        ))
        .push_input("lhs")
        .push_input("rhs")
        .push_output("output");
    graph.build(XDNA_TOSA_TARGET).expect("build bf16 matmul")
}

#[allow(dead_code)] // The hardware benchmark uses this; conformance.rs compiles this module too.
pub fn int8_matmul_tosa(
    m: i32,
    k: i32,
    n: i32,
    left_zero_point: i8,
    right_zero_point: i8,
) -> Vec<u8> {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("lhs", vec![1, m, k], DType::INT8))
        .push_tensor(OwnedTensor::new("rhs", vec![1, k, n], DType::INT8))
        .push_tensor(OwnedTensor::constant(
            "lhs_zp",
            vec![1],
            DType::INT8,
            vec![left_zero_point as u8],
        ))
        .push_tensor(OwnedTensor::constant(
            "rhs_zp",
            vec![1],
            DType::INT8,
            vec![right_zero_point as u8],
        ))
        .push_tensor(OwnedTensor::new("output", vec![1, m, n], DType::INT32))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["lhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["rhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["lhs".into(), "rhs".into(), "lhs_zp".into(), "rhs_zp".into()],
            vec!["output".into()],
        ))
        .push_input("lhs")
        .push_input("rhs")
        .push_output("output");
    graph
        .build(XDNA_TOSA_INTEGER_TARGET)
        .expect("build int8 matmul")
}

pub fn poll_to_terminal<A: Accelerator>(
    backend: &A,
    event: &A::Event,
    timeout: Duration,
) -> Result<EventState, BackendError> {
    let deadline = Instant::now() + timeout;
    loop {
        match backend.poll_event(event)? {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => return Err(BackendError::DeadlineExpired),
            terminal => return Ok(terminal),
        }
    }
}

/// A fused FP8 MATMUL graph: two FP8E4M3 block inputs, each promoted by its own explicit CAST,
/// multiplied to FP32. The promotion stays in the graph; only its placement is the backend's
/// choice.
#[allow(dead_code)] // Used by the hardware suite; this module is compiled into conformance.rs too.
pub fn fp8e4m3_matmul_tosa(m: i32, k: i32, n: i32) -> Vec<u8> {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("lhs_fp8", vec![1, m, k], DType::FP8E4M3))
        .push_tensor(OwnedTensor::new("rhs_fp8", vec![1, k, n], DType::FP8E4M3))
        .push_tensor(OwnedTensor::new("lhs_bf16", vec![1, m, k], DType::BF16))
        .push_tensor(OwnedTensor::new("rhs_bf16", vec![1, k, n], DType::BF16))
        .push_tensor(OwnedTensor::constant(
            "lhs_zp",
            vec![1],
            DType::BF16,
            vec![0u8; 2],
        ))
        .push_tensor(OwnedTensor::constant(
            "rhs_zp",
            vec![1],
            DType::BF16,
            vec![0u8; 2],
        ))
        .push_tensor(OwnedTensor::new("output", vec![1, m, n], DType::FP32))
        .push_operator(OwnedOperator::new(
            OperatorKind::Cast,
            vec!["lhs_fp8".into()],
            vec!["lhs_bf16".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Cast,
            vec!["rhs_fp8".into()],
            vec!["rhs_bf16".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["lhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["rhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec![
                "lhs_bf16".into(),
                "rhs_bf16".into(),
                "lhs_zp".into(),
                "rhs_zp".into(),
            ],
            vec!["output".into()],
        ))
        .push_input("lhs_fp8")
        .push_input("rhs_fp8")
        .push_output("output");
    graph
        .build(XDNA_TOSA_FP8_TARGET)
        .expect("build fused fp8 matmul")
}
