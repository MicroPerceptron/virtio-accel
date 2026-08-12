use virtio_accel_tosa::{
    DType, ExtensionSet, Level, Op, ProfileSet, SemanticError, SemanticErrorKind, Target, Version,
    parse,
};

const SELECT: &[u8] = include_bytes!("data/select-v1.0.0.tosa");

#[test]
fn parses_flatc_encoded_upstream_stable_select_graph() {
    let model = parse(SELECT).unwrap();
    assert_eq!(model.version(), Version::TOSA_1_0);
    assert_eq!(model.stats().regions, 1);
    assert_eq!(model.stats().blocks, 1);
    assert_eq!(model.stats().tensors, 4);
    assert_eq!(model.stats().operators, 1);

    let block = model.regions().next().unwrap().blocks().next().unwrap();
    let dtypes = block
        .tensors()
        .map(|tensor| tensor.dtype())
        .collect::<Vec<_>>();
    assert_eq!(dtypes, [DType::BOOL, DType::INT8, DType::INT8, DType::INT8]);
    assert_eq!(block.operators().next().unwrap().op(), Op::SELECT);

    let integer = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    model.validate_for(integer).unwrap();

    let floating = Target::new(
        Version::TOSA_1_0,
        ProfileSet::FLOATING_POINT,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    assert!(matches!(
        model.validate_for(floating),
        Err(SemanticError::Graph {
            operator: Some(0),
            kind: SemanticErrorKind::UnsupportedTypeProfile(Op::SELECT),
            ..
        })
    ));
}
