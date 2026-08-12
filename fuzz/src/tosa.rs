use virtio_accel_tosa::{
    AnalyzedValueKind, ExtensionSet, Level, ProfileSet, RuntimeCondition, RuntimeValue, Target,
    Version, parse, validate_runtime_values,
};

pub fn fuzz_tosa_parse(data: &[u8]) {
    let Ok(model) = parse(data) else {
        return;
    };

    let mut regions = 0;
    let mut blocks = 0;
    let mut tensors = 0;
    let mut shapes = 0;
    let mut operators = 0;
    let mut edges = 0;
    let mut constant_bytes = 0;

    for region in model.regions() {
        regions += 1;
        assert!(!region.name().is_empty());
        for block in region.blocks() {
            blocks += 1;
            assert!(!block.name().is_empty());
            edges += block.inputs().count() + block.outputs().count();
            for tensor in block.tensors() {
                tensors += 1;
                assert!(!tensor.name().is_empty());
                assert!(tensor.dtype().is_tosa_1_0());
                if let Some(rank) = tensor.rank() {
                    assert_eq!(tensor.dimensions().count(), rank);
                }
                constant_bytes += tensor.data().len();
                if let Some((_, size)) = tensor.external_data_range() {
                    constant_bytes += usize::try_from(size).unwrap();
                }
            }
            for shape in block.shapes() {
                shapes += 1;
                assert!(!shape.name().is_empty());
                constant_bytes += shape.data().len();
                if let Some(values) = shape.values() {
                    assert_eq!(values.len(), usize::try_from(shape.rank()).unwrap());
                    for _ in values {}
                }
            }
            for operator in block.operators() {
                operators += 1;
                assert!(operator.op().is_tosa_1_0());
                assert_eq!(
                    operator.op().get(),
                    u32::from(operator.attribute_kind().get())
                );
                edges += operator.inputs().count() + operator.outputs().count();
                let _ = operator.attributes();
                let _ = operator.location();
            }
        }
    }

    let stats = model.stats();
    assert_eq!(regions, stats.regions);
    assert_eq!(blocks, stats.blocks);
    assert_eq!(tensors, stats.tensors);
    assert_eq!(shapes, stats.shapes);
    assert_eq!(operators, stats.operators);
    assert_eq!(edges, stats.edges);
    assert_eq!(constant_bytes, stats.constant_bytes);
    assert_eq!(model.as_bytes(), data);

    let complete = Target::new(
        Version::TOSA_1_0,
        ProfileSet::ALL,
        Level::Unbounded,
        ExtensionSet::ALL,
    );
    let minimal_integer = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    for target in [complete, minimal_integer] {
        if let Ok(analysis) = model.analyze_for(target) {
            assert_eq!(analysis.regions().len(), stats.regions);
            assert_eq!(analysis.blocks().len(), stats.blocks);
            assert_eq!(analysis.values().len(), stats.tensors + stats.shapes);
            assert_eq!(analysis.operators().len(), stats.operators);
            for block in analysis.blocks() {
                for &operator in analysis.execution_order(block.id()) {
                    let _ = analysis.operator_inputs(operator);
                    let _ = analysis.operator_outputs(operator);
                    let _ = analysis.operator_conditions(operator);
                }
            }
            let mut runtime = Vec::new();
            for condition in analysis.conditions() {
                let RuntimeCondition::DynamicCompileTimeInput { value, .. } = *condition else {
                    continue;
                };
                if runtime
                    .last()
                    .is_some_and(|prior: &RuntimeValue<'_>| prior.value == value)
                {
                    continue;
                }
                let bytes = match analysis.value(value).kind() {
                    AnalyzedValueKind::Tensor(tensor) if !tensor.data().is_empty() => tensor.data(),
                    AnalyzedValueKind::Tensor(tensor) => tensor
                        .external_data_range()
                        .and_then(|(offset, size)| {
                            let start = usize::try_from(offset).ok()?;
                            let size = usize::try_from(size).ok()?;
                            data.get(start..start.checked_add(size)?)
                        })
                        .unwrap_or_default(),
                    AnalyzedValueKind::Shape(shape) => shape.data(),
                };
                runtime.push(RuntimeValue { value, bytes });
            }
            runtime.sort_unstable_by_key(|value| value.value);
            runtime.dedup_by_key(|value| value.value);
            let _ = validate_runtime_values(&analysis, &runtime);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fuzz_tosa_parse;

    #[test]
    fn hostile_and_valid_inputs_do_not_panic() {
        fuzz_tosa_parse(&[]);
        fuzz_tosa_parse(b"TOSA");
        fuzz_tosa_parse(include_bytes!(
            "../../crates/virtio-accel-tosa/tests/data/select-v1.0.0.tosa"
        ));
    }
}
