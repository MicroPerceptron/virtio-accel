//! Native validation/serialization boundary for the experimental Python frontend.
#![deny(unsafe_code)]

use serde::Deserialize;
use serde_json::Value;
use virtio_accel_tosa::{
    DType, ExtensionSet, Level, NanPropagationMode, ProfileSet, Target, Version,
};
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedShape, OwnedTensor};

const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
const MAX_NODES: usize = 4096;
const MAX_TENSOR_BYTES: usize = 256 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    nodes: Vec<Node>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    op: String,
    shape: Vec<i32>,
    inputs: Vec<usize>,
    attrs: serde_json::Map<String, Value>,
}

fn name(index: usize) -> String {
    format!("v{index}")
}

fn constant(node: &Node, bytes: usize) -> Result<Vec<u8>, String> {
    let hex = node
        .attrs
        .get("data")
        .and_then(Value::as_str)
        .ok_or("missing constant bytes")?;
    if hex.len() != bytes * 2 {
        return Err("constant byte count does not match shape".into());
    }
    fn digit(c: u8) -> Result<u8, String> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            _ => Err("constant must be lowercase hexadecimal".into()),
        }
    }
    hex.as_bytes()
        .chunks_exact(2)
        .map(|p| Ok((digit(p[0])? << 4) | digit(p[1])?))
        .collect()
}

/// Validate a bounded diagnostic graph and serialize it through the existing TOSA builder.
/// This does not imply admission by any particular physical backend.
pub fn compile_json(json: &str) -> Result<Vec<u8>, String> {
    if json.len() > MAX_JSON_BYTES {
        return Err("frontend graph exceeds 16 MiB JSON limit".into());
    }
    let doc: Document = serde_json::from_str(json).map_err(|e| e.to_string())?;
    if doc.version != 1 || doc.nodes.is_empty() || doc.nodes.len() > MAX_NODES {
        return Err("unsupported graph version or node count".into());
    }
    if doc.outputs.is_empty() || doc.outputs.len() > MAX_NODES || doc.inputs.len() > MAX_NODES {
        return Err("invalid graph boundary count".into());
    }
    let input_ids: Vec<_> = doc
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| (n.op == "input").then_some(i))
        .collect();
    if input_ids != doc.inputs {
        return Err("input list must name every input node in order".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    if doc
        .outputs
        .iter()
        .any(|i| *i >= doc.nodes.len() || !seen.insert(*i))
    {
        return Err("invalid or duplicate output index".into());
    }
    let mut graph = OwnedGraph::new("main");
    for (index, node) in doc.nodes.iter().enumerate() {
        let result = lower_node(&mut graph, index, node);
        result.map_err(|e| format!("node {index} ({}): {e}", node.op))?;
    }
    for index in doc.inputs {
        graph.push_input(name(index));
    }
    for index in doc.outputs {
        graph.push_output(name(index));
    }
    let target = Target::new(
        Version::TOSA_1_0,
        ProfileSet::FLOATING_POINT,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    graph
        .build(target)
        .map_err(|e| format!("TOSA validation failed: {e:?}"))
}

fn lower_node(graph: &mut OwnedGraph<'static>, index: usize, node: &Node) -> Result<(), String> {
    if node.shape.len() > 6 || node.shape.iter().any(|n| *n <= 0) {
        return Err("expected positive dimensions and rank <= 6".into());
    }
    let bytes = node
        .shape
        .iter()
        .try_fold(4usize, |size, n| size.checked_mul(*n as usize))
        .filter(|n| *n <= MAX_TENSOR_BYTES)
        .ok_or("tensor exceeds size limit")?;
    if node.inputs.iter().any(|i| *i >= index) {
        return Err("input references must be topologically ordered".into());
    }
    let expected_attrs: &[&str] = match node.op.as_str() {
        "const" => &["data"],
        "transpose" => &["axes"],
        "reduce_sum" => &["axis"],
        _ => &[],
    };
    if node.attrs.len() != expected_attrs.len()
        || expected_attrs.iter().any(|k| !node.attrs.contains_key(*k))
    {
        return Err("unexpected or missing attributes".into());
    }
    let arity = match node.op.as_str() {
        "input" | "const" => 0,
        "add" | "sub" | "mul" | "maximum" | "minimum" | "matmul" => 2,
        _ => 1,
    };
    if node.inputs.len() != arity {
        return Err("incorrect operand count".into());
    }
    let output = name(index);
    let tensor = if node.op == "const" {
        OwnedTensor::constant(
            &output,
            node.shape.clone(),
            DType::FP32,
            constant(node, bytes)?,
        )
    } else {
        OwnedTensor::new(&output, node.shape.clone(), DType::FP32)
    };
    graph.push_tensor(tensor);
    let mut inputs: Vec<_> = node.inputs.iter().copied().map(name).collect();
    // Released TOSA 1.0 carries zero points and MUL's shift as operands even
    // for floating-point graphs. Keep those details out of the Python API.
    if matches!(node.op.as_str(), "matmul" | "negate" | "mul") {
        let auxiliary = format!("aux{index}");
        let (dtype, data) = if node.op == "mul" {
            (DType::INT8, vec![0_u8])
        } else {
            (DType::FP32, vec![0_u8; 4])
        };
        graph.push_tensor(OwnedTensor::constant(&auxiliary, vec![1], dtype, data));
        graph.push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![auxiliary.clone()],
        ));
        inputs.push(auxiliary.clone());
        if node.op != "mul" {
            inputs.push(auxiliary);
        }
    }
    let kind = match node.op.as_str() {
        "input" => return Ok(()),
        "const" => OperatorKind::Const,
        "identity" => OperatorKind::Identity,
        "add" => OperatorKind::Add,
        "sub" => OperatorKind::Sub,
        "mul" => OperatorKind::Mul,
        "maximum" => OperatorKind::Maximum {
            nan_mode: NanPropagationMode::PROPAGATE,
        },
        "minimum" => OperatorKind::Minimum {
            nan_mode: NanPropagationMode::PROPAGATE,
        },
        "negate" => OperatorKind::Negate,
        "abs" => OperatorKind::Abs,
        "exp" => OperatorKind::Exp,
        "log" => OperatorKind::Log,
        "tanh" => OperatorKind::Tanh,
        "sin" => OperatorKind::Sin,
        "cos" => OperatorKind::Cos,
        "matmul" => OperatorKind::MatMul,
        "reshape" => {
            let shape = format!("s{index}");
            graph.push_shape(OwnedShape::new(
                &shape,
                node.shape.iter().map(|n| i64::from(*n)).collect(),
            ));
            graph.push_operator(OwnedOperator::new(
                OperatorKind::ConstShape,
                vec![],
                vec![shape.clone()],
            ));
            inputs.push(shape);
            OperatorKind::Reshape
        }
        "transpose" => {
            let axes: Vec<i32> =
                serde_json::from_value(node.attrs["axes"].clone()).map_err(|e| e.to_string())?;
            if axes.len() != node.shape.len() {
                return Err("transpose rank mismatch".into());
            }
            let mut perms = [0; 6];
            perms[..axes.len()].copy_from_slice(&axes);
            OperatorKind::Transpose {
                perms,
                rank: axes.len() as u8,
            }
        }
        "reduce_sum" => {
            let axis = node.attrs["axis"]
                .as_i64()
                .and_then(|x| i32::try_from(x).ok())
                .ok_or("invalid axis")?;
            OperatorKind::ReduceSum { axis }
        }
        _ => return Err("unsupported frontend operator".into()),
    };
    graph.push_operator(OwnedOperator::new(kind, inputs, vec![output]));
    Ok(())
}

#[cfg(feature = "python")]
mod python {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyfunction]
    fn build_tosa<'py>(py: Python<'py>, graph_json: String) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| super::compile_json(&graph_json))
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pymodule]
    fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(build_tosa, module)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn graph() -> Value {
        json!({"version":1,"nodes":[
            {"op":"input","shape":[2,3],"inputs":[],"attrs":{}},
            {"op":"const","shape":[1,1],"inputs":[],"attrs":{"data":"0000803f"}},
            {"op":"add","shape":[2,3],"inputs":[0,1],"attrs":{}}
        ],"inputs":[0],"outputs":[2]})
    }

    #[test]
    fn round_trip_and_deterministic() {
        let json = graph().to_string();
        let bytes = compile_json(&json).unwrap();
        virtio_accel_tosa::parse(&bytes).unwrap();
        assert_eq!(bytes, compile_json(&json).unwrap());
    }

    #[test]
    fn projection_fixture_includes_required_tosa_auxiliary_operands() {
        let bytes = compile_json(include_str!("../tests/data/projection.json")).unwrap();
        virtio_accel_tosa::parse(&bytes).unwrap();
    }

    #[test]
    fn rejects_wrong_shape_through_tosa_validator() {
        let mut g = graph();
        g["nodes"][2]["shape"] = json!([2, 4]);
        assert!(
            compile_json(&g.to_string())
                .unwrap_err()
                .contains("TOSA validation")
        );
    }

    #[test]
    fn rejects_malformed_boundaries_and_operands() {
        for (field, value) in [
            ("inputs", json!([1])),
            ("outputs", json!([8])),
            ("outputs", json!([2, 2])),
        ] {
            let mut g = graph();
            g[field] = value;
            assert!(compile_json(&g.to_string()).is_err());
        }
        let mut g = graph();
        g["nodes"][2]["inputs"] = json!([2, 0]);
        assert!(compile_json(&g.to_string()).is_err());
        g = graph();
        g["nodes"][1]["attrs"]["data"] = json!("00");
        assert!(compile_json(&g.to_string()).is_err());
        g = graph();
        g["nodes"][2]["attrs"]["out"] = json!(0);
        assert!(compile_json(&g.to_string()).is_err());
    }
}
