use alloc::vec::Vec;

extern crate std;

use crate::generated::tosa as wire;
use crate::{
    DType, Error, ExtensionSet, Level, Limits, Op, ProfileSet, Resource, SemanticError,
    SemanticErrorKind, Target, Version, parse, parse_with_limits,
};

#[derive(Clone, Copy)]
struct Fixture {
    version: (i32, i32, i32, bool),
    op: u32,
    dtype: u32,
    output_reference: &'static str,
    duplicate_tensor: bool,
    repeat_operator: bool,
    variable_output: bool,
    shape_value: Option<(u32, &'static [u8])>,
    tensor_shape: &'static [i32],
}

impl Default for Fixture {
    fn default() -> Self {
        Self {
            version: (1, 0, 0, false),
            op: wire::Op::IDENTITY.0,
            dtype: wire::DType::INT8.0,
            output_reference: "output",
            duplicate_tensor: false,
            repeat_operator: false,
            variable_output: false,
            shape_value: None,
            tensor_shape: &[1],
        }
    }
}

fn model_bytes(fixture: Fixture) -> Vec<u8> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let region_name = builder.create_string("main");
    let block_name = builder.create_string("entry");
    let input_name = builder.create_string("input");
    let output_name = builder.create_string("output");
    let output_reference = builder.create_string(fixture.output_reference);
    let tensor_shape = builder.create_vector(fixture.tensor_shape);
    let shape_value = fixture.shape_value.map(|(rank, data)| {
        let name = builder.create_string("shape");
        let data = builder.create_vector(data);
        wire::TosaShape::create(
            &mut builder,
            &wire::TosaShapeArgs {
                name: Some(name),
                rank,
                data: Some(data),
            },
        )
    });
    let shapes = shape_value.map(|shape| builder.create_vector(&[shape]));

    let input = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(input_name),
            shape: Some(tensor_shape),
            type_: wire::DType(fixture.dtype),
            ..Default::default()
        },
    );
    let output = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(if fixture.duplicate_tensor {
                input_name
            } else {
                output_name
            }),
            shape: Some(tensor_shape),
            type_: wire::DType(fixture.dtype),
            variable: fixture.variable_output,
            ..Default::default()
        },
    );
    let attribute = wire::IdentityAttribute::create(&mut builder, &Default::default());
    let inputs = builder.create_vector(&[input_name]);
    let outputs = builder.create_vector(&[output_reference]);
    let operator = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op(fixture.op),
            attribute_type: wire::Attribute::IdentityAttribute,
            attribute: Some(attribute.as_union_value()),
            inputs: Some(inputs),
            outputs: Some(outputs),
            location: None,
        },
    );
    let tensors = builder.create_vector(&[input, output]);
    let operators = if fixture.repeat_operator {
        builder.create_vector(&[operator, operator])
    } else {
        builder.create_vector(&[operator])
    };
    let block_inputs = builder.create_vector(&[input_name]);
    let block_outputs = builder.create_vector(&[output_name]);
    let block = wire::TosaBasicBlock::create(
        &mut builder,
        &wire::TosaBasicBlockArgs {
            name: Some(block_name),
            operators: Some(operators),
            tensors: Some(tensors),
            inputs: Some(block_inputs),
            outputs: Some(block_outputs),
            shapes,
        },
    );
    let blocks = builder.create_vector(&[block]);
    let region = wire::TosaRegion::create(
        &mut builder,
        &wire::TosaRegionArgs {
            name: Some(region_name),
            blocks: Some(blocks),
        },
    );
    let regions = builder.create_vector(&[region]);
    let (major, minor, patch, draft) = fixture.version;
    let version = wire::Version::create(
        &mut builder,
        &wire::VersionArgs {
            _major: major,
            _minor: minor,
            _patch: patch,
            _draft: draft,
        },
    );
    let graph = wire::TosaGraph::create(
        &mut builder,
        &wire::TosaGraphArgs {
            version: Some(version),
            regions: Some(regions),
            software_version: None,
        },
    );
    wire::finish_tosa_graph_buffer(&mut builder, graph);
    builder.finished_data().to_vec()
}

fn conv2d_bytes(output_height: i32, connect_ctc: bool) -> Vec<u8> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let region_name = builder.create_string("main");
    let block_name = builder.create_string("entry");
    let input_name = builder.create_string("input");
    let weight_name = builder.create_string("weight");
    let bias_name = builder.create_string("bias");
    let input_zp_name = builder.create_string("input_zp");
    let weight_zp_name = builder.create_string("weight_zp");
    let output_name = builder.create_string("output");

    let input_shape = builder.create_vector(&[1_i32, 4, 4, 1]);
    let input = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(input_name),
            shape: Some(input_shape),
            type_: wire::DType::INT8,
            ..Default::default()
        },
    );
    let weight_shape = builder.create_vector(&[1_i32, 3, 3, 1]);
    let weight = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(weight_name),
            shape: Some(weight_shape),
            type_: wire::DType::INT8,
            ..Default::default()
        },
    );
    let bias_shape = builder.create_vector(&[1_i32]);
    let bias = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(bias_name),
            shape: Some(bias_shape),
            type_: wire::DType::INT32,
            ..Default::default()
        },
    );
    let input_zp_shape = builder.create_vector(&[1_i32]);
    let input_zp_data = builder.create_vector(&[0_u8]);
    let input_zp = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(input_zp_name),
            shape: Some(input_zp_shape),
            type_: wire::DType::INT8,
            data: Some(input_zp_data),
            ..Default::default()
        },
    );
    let weight_zp_shape = builder.create_vector(&[1_i32]);
    let weight_zp_data = builder.create_vector(&[0_u8]);
    let weight_zp = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(weight_zp_name),
            shape: Some(weight_zp_shape),
            type_: wire::DType::INT8,
            data: Some(weight_zp_data),
            ..Default::default()
        },
    );
    let output_shape = builder.create_vector(&[1_i32, output_height, 2, 1]);
    let output = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(output_name),
            shape: Some(output_shape),
            type_: wire::DType::INT32,
            ..Default::default()
        },
    );

    let pad = builder.create_vector(&[0_i32; 4]);
    let stride = builder.create_vector(&[1_i32; 2]);
    let dilation = builder.create_vector(&[1_i32; 2]);
    let attribute = wire::Conv2dAttribute::create(
        &mut builder,
        &wire::Conv2dAttributeArgs {
            pad: Some(pad),
            stride: Some(stride),
            dilation: Some(dilation),
            local_bound: false,
            acc_type: wire::DType::INT32,
        },
    );
    let operator_inputs = builder.create_vector(&[
        input_name,
        weight_name,
        bias_name,
        input_zp_name,
        weight_zp_name,
    ]);
    let operator_outputs = builder.create_vector(&[output_name]);
    let conv = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONV2D,
            attribute_type: wire::Attribute::Conv2dAttribute,
            attribute: Some(attribute.as_union_value()),
            inputs: Some(operator_inputs),
            outputs: Some(operator_outputs),
            location: None,
        },
    );
    let const_attribute = wire::ConstAttribute::create(&mut builder, &Default::default());
    let input_zp_outputs = builder.create_vector(&[input_zp_name]);
    let input_zp_const = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONST,
            attribute_type: wire::Attribute::ConstAttribute,
            attribute: Some(const_attribute.as_union_value()),
            inputs: None,
            outputs: Some(input_zp_outputs),
            location: None,
        },
    );
    let weight_zp_outputs = builder.create_vector(&[weight_zp_name]);
    let weight_zp_const = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONST,
            attribute_type: wire::Attribute::ConstAttribute,
            attribute: Some(const_attribute.as_union_value()),
            inputs: None,
            outputs: Some(weight_zp_outputs),
            location: None,
        },
    );
    let tensors = builder.create_vector(&[input, weight, bias, input_zp, weight_zp, output]);
    let operators = if connect_ctc {
        builder.create_vector(&[input_zp_const, weight_zp_const, conv])
    } else {
        builder.create_vector(&[conv])
    };
    let block_inputs = if connect_ctc {
        builder.create_vector(&[input_name, weight_name, bias_name])
    } else {
        builder.create_vector(&[
            input_name,
            weight_name,
            bias_name,
            input_zp_name,
            weight_zp_name,
        ])
    };
    let block_outputs = builder.create_vector(&[output_name]);
    let block = wire::TosaBasicBlock::create(
        &mut builder,
        &wire::TosaBasicBlockArgs {
            name: Some(block_name),
            operators: Some(operators),
            tensors: Some(tensors),
            inputs: Some(block_inputs),
            outputs: Some(block_outputs),
            shapes: None,
        },
    );
    let blocks = builder.create_vector(&[block]);
    let region = wire::TosaRegion::create(
        &mut builder,
        &wire::TosaRegionArgs {
            name: Some(region_name),
            blocks: Some(blocks),
        },
    );
    let regions = builder.create_vector(&[region]);
    let version = wire::Version::create(
        &mut builder,
        &wire::VersionArgs {
            _major: 1,
            _minor: 0,
            _patch: 0,
            _draft: false,
        },
    );
    let graph = wire::TosaGraph::create(
        &mut builder,
        &wire::TosaGraphArgs {
            version: Some(version),
            regions: Some(regions),
            software_version: None,
        },
    );
    wire::finish_tosa_graph_buffer(&mut builder, graph);
    builder.finished_data().to_vec()
}

fn matmul_float_bytes(dtype: wire::DType) -> Vec<u8> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let region_name = builder.create_string("main");
    let block_name = builder.create_string("entry");
    let lhs_name = builder.create_string("lhs");
    let rhs_name = builder.create_string("rhs");
    let lhs_zp_name = builder.create_string("lhs_zp");
    let rhs_zp_name = builder.create_string("rhs_zp");
    let output_name = builder.create_string("output");

    let lhs_shape = builder.create_vector(&[1_i32, 2, 3]);
    let lhs = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(lhs_name),
            shape: Some(lhs_shape),
            type_: dtype,
            ..Default::default()
        },
    );
    let rhs_shape = builder.create_vector(&[1_i32, 3, 2]);
    let rhs = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(rhs_name),
            shape: Some(rhs_shape),
            type_: dtype,
            ..Default::default()
        },
    );
    let zero_shape = builder.create_vector(&[1_i32]);
    let zero_bytes = if dtype == wire::DType::FP16 {
        0_u16.to_le_bytes().to_vec()
    } else {
        0_f32.to_le_bytes().to_vec()
    };
    let zero_data = builder.create_vector(&zero_bytes);
    let lhs_zp = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(lhs_zp_name),
            shape: Some(zero_shape),
            type_: dtype,
            data: Some(zero_data),
            ..Default::default()
        },
    );
    let rhs_zp = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(rhs_zp_name),
            shape: Some(zero_shape),
            type_: dtype,
            data: Some(zero_data),
            ..Default::default()
        },
    );
    let output_shape = builder.create_vector(&[1_i32, 2, 2]);
    let output = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(output_name),
            shape: Some(output_shape),
            type_: dtype,
            ..Default::default()
        },
    );

    let const_attribute = wire::ConstAttribute::create(&mut builder, &Default::default());
    let lhs_zp_outputs = builder.create_vector(&[lhs_zp_name]);
    let lhs_zp_const = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONST,
            attribute_type: wire::Attribute::ConstAttribute,
            attribute: Some(const_attribute.as_union_value()),
            inputs: None,
            outputs: Some(lhs_zp_outputs),
            location: None,
        },
    );
    let rhs_zp_outputs = builder.create_vector(&[rhs_zp_name]);
    let rhs_zp_const = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONST,
            attribute_type: wire::Attribute::ConstAttribute,
            attribute: Some(const_attribute.as_union_value()),
            inputs: None,
            outputs: Some(rhs_zp_outputs),
            location: None,
        },
    );
    let matmul_attribute = wire::MatMulAttribute::create(&mut builder, &Default::default());
    let matmul_inputs = builder.create_vector(&[lhs_name, rhs_name, lhs_zp_name, rhs_zp_name]);
    let matmul_outputs = builder.create_vector(&[output_name]);
    let matmul = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::MATMUL,
            attribute_type: wire::Attribute::MatMulAttribute,
            attribute: Some(matmul_attribute.as_union_value()),
            inputs: Some(matmul_inputs),
            outputs: Some(matmul_outputs),
            location: None,
        },
    );

    let tensors = builder.create_vector(&[lhs, rhs, lhs_zp, rhs_zp, output]);
    let operators = builder.create_vector(&[lhs_zp_const, rhs_zp_const, matmul]);
    let block_inputs = builder.create_vector(&[lhs_name, rhs_name]);
    let block_outputs = builder.create_vector(&[output_name]);
    let block = wire::TosaBasicBlock::create(
        &mut builder,
        &wire::TosaBasicBlockArgs {
            name: Some(block_name),
            operators: Some(operators),
            tensors: Some(tensors),
            inputs: Some(block_inputs),
            outputs: Some(block_outputs),
            shapes: None,
        },
    );
    let blocks = builder.create_vector(&[block]);
    let region = wire::TosaRegion::create(
        &mut builder,
        &wire::TosaRegionArgs {
            name: Some(region_name),
            blocks: Some(blocks),
        },
    );
    let regions = builder.create_vector(&[region]);
    let version = wire::Version::create(
        &mut builder,
        &wire::VersionArgs {
            _major: 1,
            _minor: 0,
            _patch: 0,
            _draft: false,
        },
    );
    let graph = wire::TosaGraph::create(
        &mut builder,
        &wire::TosaGraphArgs {
            version: Some(version),
            regions: Some(regions),
            software_version: None,
        },
    );
    wire::finish_tosa_graph_buffer(&mut builder, graph);
    builder.finished_data().to_vec()
}

fn matmul_int8_bytes() -> Vec<u8> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let region_name = builder.create_string("main");
    let block_name = builder.create_string("entry");
    let lhs_name = builder.create_string("lhs");
    let rhs_name = builder.create_string("rhs");
    let lhs_zp_name = builder.create_string("lhs_zp");
    let rhs_zp_name = builder.create_string("rhs_zp");
    let output_name = builder.create_string("output");

    let lhs_shape = builder.create_vector(&[1_i32, 2, 3]);
    let lhs = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(lhs_name),
            shape: Some(lhs_shape),
            type_: wire::DType::INT8,
            ..Default::default()
        },
    );
    let rhs_shape = builder.create_vector(&[1_i32, 3, 2]);
    let rhs = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(rhs_name),
            shape: Some(rhs_shape),
            type_: wire::DType::INT8,
            ..Default::default()
        },
    );
    let zero_shape = builder.create_vector(&[1_i32]);
    let lhs_zp_data = builder.create_vector(&[(-2_i8) as u8]);
    let lhs_zp = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(lhs_zp_name),
            shape: Some(zero_shape),
            type_: wire::DType::INT8,
            data: Some(lhs_zp_data),
            ..Default::default()
        },
    );
    let rhs_zp_data = builder.create_vector(&[3_u8]);
    let rhs_zp = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(rhs_zp_name),
            shape: Some(zero_shape),
            type_: wire::DType::INT8,
            data: Some(rhs_zp_data),
            ..Default::default()
        },
    );
    let output_shape = builder.create_vector(&[1_i32, 2, 2]);
    let output = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(output_name),
            shape: Some(output_shape),
            type_: wire::DType::INT32,
            ..Default::default()
        },
    );

    let const_attribute = wire::ConstAttribute::create(&mut builder, &Default::default());
    let lhs_zp_outputs = builder.create_vector(&[lhs_zp_name]);
    let lhs_zp_const = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONST,
            attribute_type: wire::Attribute::ConstAttribute,
            attribute: Some(const_attribute.as_union_value()),
            inputs: None,
            outputs: Some(lhs_zp_outputs),
            location: None,
        },
    );
    let rhs_zp_outputs = builder.create_vector(&[rhs_zp_name]);
    let rhs_zp_const = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::CONST,
            attribute_type: wire::Attribute::ConstAttribute,
            attribute: Some(const_attribute.as_union_value()),
            inputs: None,
            outputs: Some(rhs_zp_outputs),
            location: None,
        },
    );
    let matmul_attribute = wire::MatMulAttribute::create(&mut builder, &Default::default());
    let matmul_inputs = builder.create_vector(&[lhs_name, rhs_name, lhs_zp_name, rhs_zp_name]);
    let matmul_outputs = builder.create_vector(&[output_name]);
    let matmul = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::MATMUL,
            attribute_type: wire::Attribute::MatMulAttribute,
            attribute: Some(matmul_attribute.as_union_value()),
            inputs: Some(matmul_inputs),
            outputs: Some(matmul_outputs),
            location: None,
        },
    );

    let tensors = builder.create_vector(&[lhs, rhs, lhs_zp, rhs_zp, output]);
    let operators = builder.create_vector(&[lhs_zp_const, rhs_zp_const, matmul]);
    let block_inputs = builder.create_vector(&[lhs_name, rhs_name]);
    let block_outputs = builder.create_vector(&[output_name]);
    let block = wire::TosaBasicBlock::create(
        &mut builder,
        &wire::TosaBasicBlockArgs {
            name: Some(block_name),
            operators: Some(operators),
            tensors: Some(tensors),
            inputs: Some(block_inputs),
            outputs: Some(block_outputs),
            shapes: None,
        },
    );
    let blocks = builder.create_vector(&[block]);
    let region = wire::TosaRegion::create(
        &mut builder,
        &wire::TosaRegionArgs {
            name: Some(region_name),
            blocks: Some(blocks),
        },
    );
    let regions = builder.create_vector(&[region]);
    let version = wire::Version::create(
        &mut builder,
        &wire::VersionArgs {
            _major: 1,
            _minor: 0,
            _patch: 0,
            _draft: false,
        },
    );
    let graph = wire::TosaGraph::create(
        &mut builder,
        &wire::TosaGraphArgs {
            version: Some(version),
            regions: Some(regions),
            software_version: None,
        },
    );
    wire::finish_tosa_graph_buffer(&mut builder, graph);
    builder.finished_data().to_vec()
}

fn max_pool2d_float_bytes(dtype: wire::DType) -> Vec<u8> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let region_name = builder.create_string("main");
    let block_name = builder.create_string("entry");
    let input_name = builder.create_string("input");
    let output_name = builder.create_string("output");
    let input_shape = builder.create_vector(&[1_i32, 4, 4, 2]);
    let output_shape = builder.create_vector(&[1_i32, 2, 2, 2]);
    let input = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(input_name),
            shape: Some(input_shape),
            type_: dtype,
            ..Default::default()
        },
    );
    let output = wire::TosaTensor::create(
        &mut builder,
        &wire::TosaTensorArgs {
            name: Some(output_name),
            shape: Some(output_shape),
            type_: dtype,
            ..Default::default()
        },
    );
    let kernel = builder.create_vector(&[2_i32, 2]);
    let stride = builder.create_vector(&[2_i32, 2]);
    let pad = builder.create_vector(&[0_i32; 4]);
    let attribute = wire::MaxPool2dAttribute::create(
        &mut builder,
        &wire::MaxPool2dAttributeArgs {
            kernel: Some(kernel),
            stride: Some(stride),
            pad: Some(pad),
            nan_mode: wire::NanPropagationMode::PROPAGATE,
        },
    );
    let operator_inputs = builder.create_vector(&[input_name]);
    let operator_outputs = builder.create_vector(&[output_name]);
    let operator = wire::TosaOperator::create(
        &mut builder,
        &wire::TosaOperatorArgs {
            op: wire::Op::MAX_POOL2D,
            attribute_type: wire::Attribute::MaxPool2dAttribute,
            attribute: Some(attribute.as_union_value()),
            inputs: Some(operator_inputs),
            outputs: Some(operator_outputs),
            location: None,
        },
    );
    let tensors = builder.create_vector(&[input, output]);
    let operators = builder.create_vector(&[operator]);
    let block_inputs = builder.create_vector(&[input_name]);
    let block_outputs = builder.create_vector(&[output_name]);
    let block = wire::TosaBasicBlock::create(
        &mut builder,
        &wire::TosaBasicBlockArgs {
            name: Some(block_name),
            operators: Some(operators),
            tensors: Some(tensors),
            inputs: Some(block_inputs),
            outputs: Some(block_outputs),
            shapes: None,
        },
    );
    let blocks = builder.create_vector(&[block]);
    let region = wire::TosaRegion::create(
        &mut builder,
        &wire::TosaRegionArgs {
            name: Some(region_name),
            blocks: Some(blocks),
        },
    );
    let regions = builder.create_vector(&[region]);
    let version = wire::Version::create(
        &mut builder,
        &wire::VersionArgs {
            _major: 1,
            _minor: 0,
            _patch: 0,
            _draft: false,
        },
    );
    let graph = wire::TosaGraph::create(
        &mut builder,
        &wire::TosaGraphArgs {
            version: Some(version),
            regions: Some(regions),
            software_version: None,
        },
    );
    wire::finish_tosa_graph_buffer(&mut builder, graph);
    builder.finished_data().to_vec()
}

#[test]
fn matmul_fixture_is_semantically_valid() {
    for dtype in [wire::DType::FP16, wire::DType::FP32] {
        let bytes = matmul_float_bytes(dtype);
        parse(&bytes)
            .unwrap()
            .validate_for(Target::new(
                Version::TOSA_1_0,
                ProfileSet::FLOATING_POINT,
                Level::Level8K,
                ExtensionSet::NONE,
            ))
            .unwrap();
    }
}

#[test]
fn int8_matmul_fixture_is_semantically_valid() {
    parse(&matmul_int8_bytes())
        .unwrap()
        .validate_for(Target::new(
            Version::TOSA_1_0,
            ProfileSet::INTEGER,
            Level::Level8K,
            ExtensionSet::NONE,
        ))
        .unwrap();
}

#[test]
fn max_pool2d_fixture_is_semantically_valid() {
    for dtype in [wire::DType::FP16, wire::DType::FP32] {
        let bytes = max_pool2d_float_bytes(dtype);
        parse(&bytes)
            .unwrap()
            .validate_for(Target::new(
                Version::TOSA_1_0,
                ProfileSet::FLOATING_POINT,
                Level::Level8K,
                ExtensionSet::NONE,
            ))
            .unwrap();
    }
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_matmul_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(destination, matmul_float_bytes(wire::DType::FP32)).unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_matmul_fp16_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(destination, matmul_float_bytes(wire::DType::FP16)).unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_matmul_int8_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(destination, matmul_int8_bytes()).unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_max_pool2d_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(destination, max_pool2d_float_bytes(wire::DType::FP32)).unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_max_pool2d_fp16_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(destination, max_pool2d_float_bytes(wire::DType::FP16)).unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_identity_edges_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(
        destination,
        model_bytes(Fixture {
            dtype: wire::DType::FP32.0,
            tensor_shape: &[8],
            ..Default::default()
        }),
    )
    .unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_identity_edges_fp16_fixture() {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(
        destination,
        model_bytes(Fixture {
            dtype: wire::DType::FP16.0,
            tensor_shape: &[8],
            ..Default::default()
        }),
    )
    .unwrap();
}

fn regenerate_identity_fixture(dtype: wire::DType) {
    let destination = std::env::var_os("VIRTIO_ACCEL_TOSA_FIXTURE_OUT")
        .expect("set VIRTIO_ACCEL_TOSA_FIXTURE_OUT to the exact output path");
    std::fs::write(
        destination,
        model_bytes(Fixture {
            dtype: dtype.0,
            tensor_shape: &[8],
            ..Default::default()
        }),
    )
    .unwrap();
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_identity_int8_fixture() {
    regenerate_identity_fixture(wire::DType::INT8);
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_identity_int4_fixture() {
    regenerate_identity_fixture(wire::DType::INT4);
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_identity_fp8e4m3_fixture() {
    regenerate_identity_fixture(wire::DType::FP8E4M3);
}

#[test]
#[ignore = "writes a requested checked-in test fixture"]
fn regenerate_identity_fp8e5m2_fixture() {
    regenerate_identity_fixture(wire::DType::FP8E5M2);
}

#[test]
fn valid_graph_is_zero_copy_and_iterable() {
    let bytes = model_bytes(Fixture::default());
    let model = parse(&bytes).unwrap();
    assert_eq!(model.as_bytes().as_ptr(), bytes.as_ptr());
    assert_eq!(model.version(), Version::TOSA_1_0);
    assert_eq!(model.stats().regions, 1);
    assert_eq!(model.stats().blocks, 1);
    assert_eq!(model.stats().tensors, 2);
    assert_eq!(model.stats().operators, 1);

    let region = model.regions().next().unwrap();
    assert_eq!(region.name(), "main");
    let block = region.blocks().next().unwrap();
    assert_eq!(block.name(), "entry");
    assert_eq!(
        block
            .tensors()
            .map(|tensor| tensor.name())
            .collect::<Vec<_>>(),
        ["input", "output"]
    );
    let operator = block.operators().next().unwrap();
    assert_eq!(operator.op(), Op::IDENTITY);
    assert_eq!(operator.inputs().collect::<Vec<_>>(), ["input"]);
    assert_eq!(operator.outputs().collect::<Vec<_>>(), ["output"]);
}

#[test]
fn rejects_wrong_identifier_and_truncation() {
    assert_eq!(parse(b"TOSA").unwrap_err(), Error::MissingIdentifier);

    let mut bytes = model_bytes(Fixture::default());
    bytes[4] = b'X';
    assert_eq!(parse(&bytes).unwrap_err(), Error::MissingIdentifier);

    let bytes = model_bytes(Fixture::default());
    assert_eq!(parse(&bytes[..16]).unwrap_err(), Error::InvalidFlatbuffer);
}

#[test]
fn rejects_nonstable_version_schema_entries_and_bad_symbols() {
    let bytes = model_bytes(Fixture {
        version: (1, 1, 0, true),
        ..Fixture::default()
    });
    assert!(matches!(
        parse(&bytes),
        Err(Error::UnsupportedVersion { .. })
    ));

    let bytes = model_bytes(Fixture {
        op: wire::Op::DIM.0,
        ..Fixture::default()
    });
    assert_eq!(
        parse(&bytes).unwrap_err(),
        Error::UnsupportedOperator(wire::Op::DIM.0)
    );

    let bytes = model_bytes(Fixture {
        dtype: wire::DType::FP6E2M3.0,
        ..Fixture::default()
    });
    assert_eq!(
        parse(&bytes).unwrap_err(),
        Error::UnsupportedDataType(wire::DType::FP6E2M3.0)
    );

    let bytes = model_bytes(Fixture {
        output_reference: "missing",
        ..Fixture::default()
    });
    assert_eq!(parse(&bytes).unwrap_err(), Error::UnknownSymbol);

    let bytes = model_bytes(Fixture {
        duplicate_tensor: true,
        ..Fixture::default()
    });
    assert_eq!(
        parse(&bytes).unwrap_err(),
        Error::DuplicateName(crate::NameKind::Tensor)
    );
}

#[test]
fn applies_caller_limits_before_returning_views() {
    let bytes = model_bytes(Fixture::default());
    let limits = Limits {
        max_operators: 0,
        ..Limits::default()
    };
    assert_eq!(
        parse_with_limits(&bytes, limits).unwrap_err(),
        Error::LimitExceeded {
            resource: Resource::Operators,
            limit: 0,
        }
    );
}

#[test]
fn single_assignment_allows_variable_tensor_writes_only() {
    let bytes = model_bytes(Fixture {
        repeat_operator: true,
        ..Fixture::default()
    });
    assert_eq!(parse(&bytes).unwrap_err(), Error::MultipleProducers);

    let bytes = model_bytes(Fixture {
        repeat_operator: true,
        variable_output: true,
        ..Fixture::default()
    });
    assert_eq!(parse(&bytes).unwrap().stats().operators, 2);
}

#[test]
fn validates_and_decodes_shape_data() {
    static ENCODED: [u8; 16] = [
        42, 0, 0, 0, 0, 0, 0, 0, 249, 255, 255, 255, 255, 255, 255, 255,
    ];
    let bytes = model_bytes(Fixture {
        shape_value: Some((2, &ENCODED)),
        ..Fixture::default()
    });
    let shape = parse(&bytes)
        .unwrap()
        .regions()
        .next()
        .unwrap()
        .blocks()
        .next()
        .unwrap()
        .shapes()
        .next()
        .unwrap();
    assert_eq!(shape.values().unwrap().collect::<Vec<_>>(), [42, -7]);

    let bytes = model_bytes(Fixture {
        shape_value: Some((1, &[0; 7])),
        ..Fixture::default()
    });
    assert_eq!(parse(&bytes).unwrap_err(), Error::InvalidShapeData);

    let bytes = model_bytes(Fixture {
        shape_value: Some((0, &[])),
        ..Fixture::default()
    });
    let shape = parse(&bytes)
        .unwrap()
        .regions()
        .next()
        .unwrap()
        .blocks()
        .next()
        .unwrap()
        .shapes()
        .next()
        .unwrap();
    assert_eq!(shape.values().unwrap().len(), 0);
}

#[test]
fn target_identity_and_artifact_envelope_round_trip() {
    let target = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER.union(ProfileSet::FLOATING_POINT),
        Level::Level8K,
        ExtensionSet::INT4.union(ExtensionSet::FFT),
    );
    assert_eq!(Target::from_identity(target.to_identity()), Ok(target));

    let bytes = model_bytes(Fixture::default());
    let model = parse(&bytes).unwrap();
    let artifact = model.artifact_ref(target, bytes.len() as u64).unwrap();
    assert_eq!(artifact.format, crate::ARTIFACT_FORMAT);
    assert_eq!(artifact.target, target.to_identity());
    assert_eq!(artifact.payload.as_contiguous(), Some(bytes.as_slice()));
}

#[test]
fn semantic_pass_accepts_valid_graph_and_enforces_extensions() {
    let integer = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    let bytes = model_bytes(Fixture::default());
    parse(&bytes).unwrap().validate_for(integer).unwrap();

    let int4 = model_bytes(Fixture {
        dtype: wire::DType::INT4.0,
        ..Fixture::default()
    });
    assert!(matches!(
        parse(&int4).unwrap().validate_for(integer),
        Err(SemanticError::Graph {
            operator: Some(0),
            kind: SemanticErrorKind::UnsupportedTypeProfile(Op::IDENTITY),
            ..
        })
    ));
    let with_int4 = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::INT4,
    );
    parse(&int4).unwrap().validate_for(with_int4).unwrap();
}

#[test]
fn compact_analysis_precomputes_ids_liveness_hints_and_dynamic_obligations() {
    let integer = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    let bytes = model_bytes(Fixture::default());
    let model = parse(&bytes).unwrap();
    let analysis = model.analyze_for(integer).unwrap();
    assert_eq!(analysis.regions().len(), 1);
    assert_eq!(analysis.blocks().len(), 1);
    assert_eq!(analysis.values().len(), 2);
    assert_eq!(analysis.operators().len(), 1);
    let block = analysis.blocks()[0].id();
    let operator = analysis.execution_order(block)[0];
    assert_eq!(analysis.operator(operator).op(), Op::IDENTITY);
    assert!(
        analysis
            .operator(operator)
            .hints()
            .contains(crate::OptimizationHints::ALIAS_INPUT_ZERO)
    );
    let input = analysis.operator_inputs(operator)[0];
    let output = analysis.operator_outputs(operator)[0];
    assert_eq!(analysis.value(input).name(), "input");
    assert_eq!(analysis.value(input).producer(), None);
    assert_eq!(analysis.value(input).first_use(), Some(0));
    assert_eq!(analysis.value(output).producer(), Some(operator));
    assert_eq!(analysis.value(output).last_use(), Some(1));
    assert!(analysis.conditions().is_empty());

    let dynamic = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::DYNAMIC,
    );
    let bytes = conv2d_bytes(2, false);
    let model = parse(&bytes).unwrap();
    let analysis = model.analyze_for(dynamic).unwrap();
    assert_eq!(
        analysis
            .conditions()
            .iter()
            .filter(|condition| condition.error_detection_required())
            .count(),
        2
    );
    assert!(analysis.conditions().iter().all(|condition| matches!(
        condition,
        crate::RuntimeCondition::DynamicCompileTimeInput { .. }
    )));
    let input_zp = analysis
        .values()
        .iter()
        .find(|value| value.name() == "input_zp")
        .unwrap()
        .id();
    let weight_zp = analysis
        .values()
        .iter()
        .find(|value| value.name() == "weight_zp")
        .unwrap()
        .id();
    let mut runtime = [
        crate::RuntimeValue {
            value: input_zp,
            bytes: &[0],
        },
        crate::RuntimeValue {
            value: weight_zp,
            bytes: &[0],
        },
    ];
    runtime.sort_unstable_by_key(|value| value.value);
    assert_eq!(
        crate::validate_runtime_values(&analysis, &runtime),
        Ok(crate::RuntimeValidation {
            unpredictable: false
        })
    );
    assert!(matches!(
        crate::validate_runtime_values(&analysis, &runtime[..1]),
        Err(crate::RuntimeError {
            kind: crate::RuntimeErrorKind::MissingValue,
            ..
        })
    ));
}

#[test]
fn semantic_pass_checks_convolution_numerics_and_geometry() {
    let target = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    let valid = conv2d_bytes(2, true);
    let model = parse(&valid).unwrap();
    model.validate_for(target).unwrap();
    assert!(matches!(
        model
            .regions()
            .next()
            .unwrap()
            .blocks()
            .next()
            .unwrap()
            .operators()
            .nth(2)
            .unwrap()
            .attributes(),
        crate::OpAttributes::Conv2d {
            acc_type: DType::INT32,
            ..
        }
    ));

    let invalid = conv2d_bytes(3, true);
    assert!(matches!(
        parse(&invalid).unwrap().validate_for(target),
        Err(SemanticError::Graph {
            operator: Some(2),
            kind: SemanticErrorKind::InvalidShape(Op::CONV2D),
            ..
        })
    ));

    let disconnected_ctc = conv2d_bytes(2, false);
    assert!(matches!(
        parse(&disconnected_ctc).unwrap().validate_for(target),
        Err(SemanticError::Graph {
            operator: Some(0),
            kind: SemanticErrorKind::ConstantRequired {
                op: Op::CONV2D,
                input: 3,
            },
            ..
        })
    ));
}
