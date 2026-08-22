//! Dependency-free Core ML ML Program encoding for the exact TOSA integer tier.
//!
//! Core ML's older `NeuralNetwork` model family cannot expose INT8 multi-array boundaries. The
//! ML Program format added that boundary on macOS 26. This module deliberately implements only
//! operations whose integer semantics have been executed against the shared Rust oracle:
//! same-type identity and INT8 MATMUL with explicit zero-point subtraction and INT32 accumulation.
//! No path converts integer tensors through floating point.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::lower::{
    LoweredFeature, LoweredFeatureRole, LoweredModel, LoweringError, encode_feature, static_shape,
};
use virtio_accel_tosa::{
    AnalyzedValueKind, CapabilityDescriptor, DType, DTypeCapability, ExtensionSet,
    GraphCapabilities, Level, Op, OperatorCapability, ProfileSet, RuntimeConditionSupport, Target,
    TosaAnalysis, ValueId, ValueRoles, Version, parse,
};

/// TOSA integer-profile target lowered to Core ML ML Program on macOS 26 or newer.
pub const COREML_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

const INTEGER_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::INT8, ValueRoles::ALL),
    DTypeCapability::new(
        DType::INT32,
        ValueRoles::OUTPUT
            .union(ValueRoles::CONSTANT)
            .union(ValueRoles::INTERMEDIATE),
    ),
];

const INTEGER_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::new(Op::MATMUL),
];

/// Conservative integer-profile boundary for the macOS 26+ ML Program tier.
pub const COREML_TOSA_INTEGER_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: COREML_TOSA_INTEGER_TARGET,
    dtypes: INTEGER_DTYPES,
    operators: INTEGER_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

const COREML_SPECIFICATION_VERSION: u64 = 10;
const MLPROGRAM_VERSION: u64 = 1;
const OPSET: &str = "CoreML9";

// MILSpec.DataType values. These are distinct from MLMultiArrayDataType values in the model
// description encoded by `encode_feature`.
const MIL_BOOL: u64 = 1;
const MIL_STRING: u64 = 2;
const MIL_INT8: u64 = 21;
const MIL_INT32: u64 = 23;

pub(crate) fn lower_integer_tosa(
    bytes: &[u8],
    target: Target,
) -> Result<LoweredModel, LoweringError> {
    if target != COREML_TOSA_INTEGER_TARGET {
        return Err(LoweringError::UnsupportedGraph);
    }
    let model = parse(bytes).map_err(LoweringError::Parse)?;
    let analysis = model.analyze_for(target).map_err(LoweringError::Analysis)?;
    if analysis.regions().len() != 1
        || analysis.blocks().len() != 1
        || !analysis.conditions().is_empty()
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let block = analysis.blocks()[0].id();
    let inputs = analysis.block_inputs(block);
    let outputs = analysis.block_outputs(block);
    if inputs.is_empty() || outputs.is_empty() || inputs.iter().any(|id| outputs.contains(id)) {
        return Err(LoweringError::UnsupportedGraph);
    }

    let mut description = Vec::new();
    let mut features = Vec::new();
    features
        .try_reserve_exact(
            inputs
                .len()
                .checked_add(outputs.len())
                .ok_or(LoweringError::ResourceLimit)?,
        )
        .map_err(|_| LoweringError::ResourceLimit)?;
    for (index, value) in inputs.iter().copied().enumerate() {
        let tensor = tensor(&analysis, value)?;
        if tensor.dtype() != DType::INT8 {
            return Err(LoweringError::UnsupportedType(tensor.dtype()));
        }
        let name = format!("input_{index}");
        encode_feature(&mut description, 1, &name, tensor)?;
        features.push(LoweredFeature {
            slot: u32::try_from(index).map_err(|_| LoweringError::ResourceLimit)?,
            role: LoweredFeatureRole::Input,
            name,
        });
    }
    for (index, value) in outputs.iter().copied().enumerate() {
        let tensor = tensor(&analysis, value)?;
        if !matches!(tensor.dtype(), DType::INT8 | DType::INT32) {
            return Err(LoweringError::UnsupportedType(tensor.dtype()));
        }
        let name = format!("output_{index}");
        encode_feature(&mut description, 10, &name, tensor)?;
        features.push(LoweredFeature {
            slot: u32::try_from(inputs.len() + index).map_err(|_| LoweringError::ResourceLimit)?,
            role: LoweredFeatureRole::Output,
            name,
        });
    }

    let executable = analysis
        .execution_order(block)
        .iter()
        .copied()
        .filter(|operator| {
            !matches!(
                analysis.operator(*operator).op(),
                Op::CONST | Op::CONST_SHAPE
            )
        })
        .collect::<Vec<_>>();
    if executable.len() != 1 {
        return Err(LoweringError::UnsupportedGraph);
    }
    let operator = executable[0];
    let operations = match analysis.operator(operator).op() {
        Op::IDENTITY => encode_identity(&analysis, operator, inputs, outputs)?,
        Op::MATMUL => encode_matmul(&analysis, operator, inputs, outputs)?,
        op => return Err(LoweringError::UnsupportedOperator(op)),
    };

    let mut block_body = Vec::new();
    for index in 0..outputs.len() {
        field_string(&mut block_body, 2, &format!("output_{index}"));
    }
    for operation in operations {
        field_message(&mut block_body, 3, &operation);
    }

    let mut function = Vec::new();
    for (index, value) in inputs.iter().copied().enumerate() {
        let shape = static_shape(tensor(&analysis, value)?)?;
        field_message(
            &mut function,
            1,
            &named_value_type(&format!("input_{index}"), MIL_INT8, &shape),
        );
    }
    field_string(&mut function, 2, OPSET);
    field_message(&mut function, 3, &map_entry(OPSET, &block_body));

    let mut program = Vec::new();
    field_varint(&mut program, 1, MLPROGRAM_VERSION);
    field_message(&mut program, 2, &map_entry("main", &function));

    let mut encoded = Vec::new();
    field_varint(&mut encoded, 1, COREML_SPECIFICATION_VERSION);
    field_message(&mut encoded, 2, &description);
    field_message(&mut encoded, 502, &program);
    Ok(LoweredModel {
        bytes: encoded,
        features,
    })
}

fn encode_identity(
    analysis: &TosaAnalysis<'_>,
    operator: virtio_accel_tosa::OperatorId,
    block_inputs: &[ValueId],
    block_outputs: &[ValueId],
) -> Result<Vec<Vec<u8>>, LoweringError> {
    let inputs = analysis.operator_inputs(operator);
    let outputs = analysis.operator_outputs(operator);
    if block_inputs.len() != 1
        || block_outputs.len() != 1
        || inputs != block_inputs
        || outputs != block_outputs
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let input = tensor(analysis, inputs[0])?;
    let output = tensor(analysis, outputs[0])?;
    let input_shape = static_shape(input)?;
    let output_shape = static_shape(output)?;
    if input.dtype() != DType::INT8 || output.dtype() != DType::INT8 || input_shape != output_shape
    {
        return Err(LoweringError::UnsupportedGraph);
    }

    // Core ML rejects an INT8 `identity` MIL op. A widening cast, exact add-zero, and narrowing
    // cast is accepted and preserves every signed byte value exactly.
    Ok(vec![
        const_string("identity_wide_dtype", "int32"),
        unary_operation(
            "cast",
            &[("x", "input_0"), ("dtype", "identity_wide_dtype")],
            "identity_wide",
            MIL_INT32,
            &input_shape,
        ),
        const_int32("identity_zero", 0),
        unary_operation(
            "add",
            &[("x", "identity_wide"), ("y", "identity_zero")],
            "identity_exact",
            MIL_INT32,
            &input_shape,
        ),
        const_string("identity_narrow_dtype", "int8"),
        unary_operation(
            "cast",
            &[("x", "identity_exact"), ("dtype", "identity_narrow_dtype")],
            "output_0",
            MIL_INT8,
            &output_shape,
        ),
    ])
}

fn encode_matmul(
    analysis: &TosaAnalysis<'_>,
    operator: virtio_accel_tosa::OperatorId,
    block_inputs: &[ValueId],
    block_outputs: &[ValueId],
) -> Result<Vec<Vec<u8>>, LoweringError> {
    let inputs = analysis.operator_inputs(operator);
    let outputs = analysis.operator_outputs(operator);
    if block_inputs.len() != 2
        || block_outputs.len() != 1
        || inputs.len() != 4
        || outputs.len() != 1
        || inputs[..2] != *block_inputs
        || outputs != block_outputs
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let lhs = tensor(analysis, inputs[0])?;
    let rhs = tensor(analysis, inputs[1])?;
    let output = tensor(analysis, outputs[0])?;
    let lhs_shape = static_shape(lhs)?;
    let rhs_shape = static_shape(rhs)?;
    let output_shape = static_shape(output)?;
    if lhs.dtype() != DType::INT8
        || rhs.dtype() != DType::INT8
        || output.dtype() != DType::INT32
        || lhs_shape.len() != 3
        || rhs_shape.len() != 3
        || output_shape.len() != 3
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let zero_point = |value: ValueId| {
        let bytes = analysis
            .serialized_constant(value)
            .ok_or(LoweringError::UnsupportedGraph)?;
        if tensor(analysis, value)?.dtype() != DType::INT8 || bytes.len() != 1 {
            return Err(LoweringError::UnsupportedGraph);
        }
        Ok(i32::from(bytes[0] as i8))
    };
    let lhs_zero_point = zero_point(inputs[2])?;
    let rhs_zero_point = zero_point(inputs[3])?;

    let mut operations = Vec::new();
    operations
        .try_reserve_exact(11)
        .map_err(|_| LoweringError::ResourceLimit)?;
    operations.push(const_string("lhs_wide_dtype", "int32"));
    operations.push(unary_operation(
        "cast",
        &[("x", "input_0"), ("dtype", "lhs_wide_dtype")],
        "lhs_wide",
        MIL_INT32,
        &lhs_shape,
    ));
    operations.push(const_string("rhs_wide_dtype", "int32"));
    operations.push(unary_operation(
        "cast",
        &[("x", "input_1"), ("dtype", "rhs_wide_dtype")],
        "rhs_wide",
        MIL_INT32,
        &rhs_shape,
    ));
    operations.push(const_int32("lhs_zero_point", lhs_zero_point));
    operations.push(unary_operation(
        "sub",
        &[("x", "lhs_wide"), ("y", "lhs_zero_point")],
        "lhs_centered",
        MIL_INT32,
        &lhs_shape,
    ));
    operations.push(const_int32("rhs_zero_point", rhs_zero_point));
    operations.push(unary_operation(
        "sub",
        &[("x", "rhs_wide"), ("y", "rhs_zero_point")],
        "rhs_centered",
        MIL_INT32,
        &rhs_shape,
    ));
    operations.push(const_bool("matmul_transpose_x", false));
    operations.push(const_bool("matmul_transpose_y", false));
    operations.push(unary_operation(
        "matmul",
        &[
            ("x", "lhs_centered"),
            ("y", "rhs_centered"),
            ("transpose_x", "matmul_transpose_x"),
            ("transpose_y", "matmul_transpose_y"),
        ],
        "output_0",
        MIL_INT32,
        &output_shape,
    ));
    Ok(operations)
}

fn tensor<'a>(
    analysis: &'a TosaAnalysis<'a>,
    value: ValueId,
) -> Result<virtio_accel_tosa::Tensor<'a>, LoweringError> {
    match analysis.value(value).kind() {
        AnalyzedValueKind::Tensor(tensor) => Ok(tensor),
        AnalyzedValueKind::Shape(_) => Err(LoweringError::UnsupportedGraph),
    }
}

fn unary_operation(
    kind: &str,
    inputs: &[(&str, &str)],
    output: &str,
    dtype: u64,
    shape: &[i32],
) -> Vec<u8> {
    let mut operation = Vec::new();
    field_string(&mut operation, 1, kind);
    for (key, name) in inputs {
        field_message(&mut operation, 2, &argument_entry(key, name));
    }
    field_message(&mut operation, 3, &named_value_type(output, dtype, shape));
    field_message(
        &mut operation,
        5,
        &attribute_entry("name", &string_value(output)),
    );
    operation
}

fn const_string(name: &str, value: &str) -> Vec<u8> {
    const_operation(name, MIL_STRING, &string_value(value))
}

fn const_int32(name: &str, value: i32) -> Vec<u8> {
    const_operation(name, MIL_INT32, &int32_value(value))
}

fn const_bool(name: &str, value: bool) -> Vec<u8> {
    const_operation(name, MIL_BOOL, &bool_value(value))
}

fn const_operation(name: &str, dtype: u64, value: &[u8]) -> Vec<u8> {
    let mut operation = Vec::new();
    field_string(&mut operation, 1, "const");
    field_message(&mut operation, 3, &named_value_type(name, dtype, &[]));
    field_message(&mut operation, 5, &attribute_entry("val", value));
    field_message(
        &mut operation,
        5,
        &attribute_entry("name", &string_value(name)),
    );
    operation
}

fn argument_entry(key: &str, name: &str) -> Vec<u8> {
    let mut binding = Vec::new();
    field_string(&mut binding, 1, name);
    let mut argument = Vec::new();
    field_message(&mut argument, 1, &binding);
    map_entry(key, &argument)
}

fn attribute_entry(key: &str, value: &[u8]) -> Vec<u8> {
    map_entry(key, value)
}

fn map_entry(key: &str, value: &[u8]) -> Vec<u8> {
    let mut entry = Vec::new();
    field_string(&mut entry, 1, key);
    field_message(&mut entry, 2, value);
    entry
}

fn named_value_type(name: &str, dtype: u64, shape: &[i32]) -> Vec<u8> {
    let mut named = Vec::new();
    field_string(&mut named, 1, name);
    field_message(&mut named, 2, &value_type(dtype, shape));
    named
}

fn value_type(dtype: u64, shape: &[i32]) -> Vec<u8> {
    let mut tensor = Vec::new();
    field_varint(&mut tensor, 1, dtype);
    if !shape.is_empty() {
        field_varint(&mut tensor, 2, shape.len() as u64);
        for dimension in shape {
            let mut constant = Vec::new();
            field_varint(&mut constant, 1, *dimension as u64);
            let mut dimension_message = Vec::new();
            field_message(&mut dimension_message, 1, &constant);
            field_message(&mut tensor, 3, &dimension_message);
        }
    }
    let mut value_type = Vec::new();
    field_message(&mut value_type, 1, &tensor);
    value_type
}

fn string_value(value: &str) -> Vec<u8> {
    let mut repeated = Vec::new();
    field_string(&mut repeated, 1, value);
    immediate_tensor_value(MIL_STRING, 4, &repeated)
}

fn int32_value(value: i32) -> Vec<u8> {
    let mut packed = Vec::new();
    varint(&mut packed, value as i64 as u64);
    let mut repeated = Vec::new();
    field_bytes(&mut repeated, 1, &packed);
    immediate_tensor_value(MIL_INT32, 2, &repeated)
}

fn bool_value(value: bool) -> Vec<u8> {
    let mut packed = Vec::new();
    varint(&mut packed, value as u64);
    let mut repeated = Vec::new();
    field_bytes(&mut repeated, 1, &packed);
    immediate_tensor_value(MIL_BOOL, 3, &repeated)
}

fn immediate_tensor_value(dtype: u64, tensor_field: u32, repeated: &[u8]) -> Vec<u8> {
    let mut tensor_value = Vec::new();
    field_message(&mut tensor_value, tensor_field, repeated);
    let mut immediate = Vec::new();
    field_message(&mut immediate, 1, &tensor_value);
    let mut value = Vec::new();
    field_message(&mut value, 2, &value_type(dtype, &[]));
    field_message(&mut value, 3, &immediate);
    value
}

fn field_varint(target: &mut Vec<u8>, field: u32, value: u64) {
    varint(target, u64::from(field) << 3);
    varint(target, value);
}

fn field_string(target: &mut Vec<u8>, field: u32, value: &str) {
    field_bytes(target, field, value.as_bytes());
}

fn field_message(target: &mut Vec<u8>, field: u32, message: &[u8]) {
    field_bytes(target, field, message);
}

fn field_bytes(target: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    varint(target, (u64::from(field) << 3) | 2);
    varint(target, bytes.len() as u64);
    target.extend_from_slice(bytes);
}

fn varint(target: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        target.push((value as u8) | 0x80);
        value >>= 7;
    }
    target.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_conformance::numerics::{IDENTITY_INT8, MATMUL_INT8};

    #[test]
    fn lowers_shared_int8_identity_to_ml_program() {
        let lowered =
            lower_integer_tosa(IDENTITY_INT8.artifact, COREML_TOSA_INTEGER_TARGET).unwrap();
        assert_eq!(lowered.features.len(), 2);
        assert!(lowered.bytes.windows(7).any(|bytes| bytes == b"CoreML9"));
        assert!(lowered.bytes.windows(4).any(|bytes| bytes == b"cast"));
        assert!(lowered.bytes.windows(3).any(|bytes| bytes == b"add"));
    }

    #[test]
    fn lowers_shared_int8_matmul_with_explicit_zero_points() {
        let lowered = lower_integer_tosa(MATMUL_INT8.artifact, COREML_TOSA_INTEGER_TARGET).unwrap();
        assert_eq!(lowered.features.len(), 3);
        assert!(lowered.bytes.windows(6).any(|bytes| bytes == b"matmul"));
        assert!(lowered.bytes.windows(3).any(|bytes| bytes == b"sub"));
        // Negative lhs zero-point is encoded as a sign-extended protobuf int32 varint.
        assert!(
            lowered
                .bytes
                .windows(10)
                .any(|bytes| bytes == [0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01])
        );
    }

    #[test]
    fn rejects_float_artifacts_at_the_integer_target() {
        let error = lower_integer_tosa(
            virtio_accel_conformance::numerics::MATMUL_FP32.artifact,
            COREML_TOSA_INTEGER_TARGET,
        )
        .unwrap_err();
        assert!(matches!(error, LoweringError::Analysis(_)));
    }
}
