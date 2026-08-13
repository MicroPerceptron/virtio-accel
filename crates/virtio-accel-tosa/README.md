# virtio-accel-tosa

`virtio-accel-tosa` is the device-neutral TOSA artifact layer for `virtio-accel`. It verifies an
untrusted `.tosa` FlatBuffer under finite caller-selected limits, then exposes borrowed graph views
without copying tensor data or allocating an owned graph.

The crate is `no_std + alloc`, works on the workspace's bare-metal and Wasm targets, and has no
native library, LLVM, Python, `flatc`, filesystem, or operating-system dependency. Its only runtime
dependencies are `virtio-accel-core` and the official FlatBuffers Rust runtime with default
features disabled.

```rust
use virtio_accel_tosa::{
    ExtensionSet, Level, ProfileSet, Target, Version, parse,
};

# fn load(bytes: &[u8]) -> Result<(), virtio_accel_tosa::Error> {
let model = parse(bytes)?;
for region in model.regions() {
    for block in region.blocks() {
        for operator in block.operators() {
            println!("{}: {:?}", block.name(), operator.op());
        }
    }
}

let target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::INT4,
);
let analysis = model.analyze_for(target).unwrap();
for block in analysis.blocks() {
    for operator in analysis.execution_order(block.id()) {
        let operator = analysis.operator(*operator);
        println!("lower {:?} with {:?}", operator.op(), operator.hints());
    }
}
// This is the caller-authorized bound on storage retained by the eventual provider; it need not
// equal the payload length.
let artifact = model.artifact_ref(target, 16 * 1024 * 1024).unwrap();
assert_eq!(artifact.format, virtio_accel_tosa::ARTIFACT_FORMAT);
# Ok(())
# }
```

## Validation boundary

The built-in pass checks:

- the `TOSA` file identifier and complete FlatBuffers structure;
- stable, non-draft TOSA graph version 1.0.0, corresponding to TOSA specification 1.0.1;
- exclusion of the draft 1.1 operators and data types appended to the pinned tools schema;
- finite model, verifier, table, graph-object, edge, name, rank, and constant-data limits;
- required and unique region, block, tensor, and shape names;
- tensor/shape references, single-assignment operator outputs except declared variables, ranks,
  dimensions, and external constant-data ranges; and
- known operator/data-type values plus the operator/attribute union correspondence.

After parsing, `Model::validate_for` provides the production semantic pass for all 75 stable TOSA
1.0 operators. It validates exact or variadic arity, tensor-versus-shape operands, rank and level
limits, every supported data-type/profile/extension row, typed attributes, compile-time-constant
requirements, zero points and scaling parameters, static output geometry, control-flow region
signatures, cycles, and nesting limits. `Operator::attributes` exposes every stable attribute field
through safe borrowed Rust values for backend lowering.

`Model::analyze_for` combines that validation with a compact lowering overlay. Dense value/operator
IDs, operand spans, topological order, use counts, liveness intervals, serialized/foldable constant
state, conservative layout/constant/dead-code hints, and runtime conditions are computed once while
the original names, attributes, shapes, and constant bytes remain borrowed. It is an indexed plan,
not a second owned compute graph.

`validate_runtime_values` handles the remaining host-readable `EXT-DYNAMIC` CTC boundary without
scanning ordinary tensor inputs. It validates exact encodings and mandatory dynamic `ERROR_IF`
conditions; per-element `REQUIRE` conditions remain explicitly classified as advisory because TOSA
defines their failure as unpredictable. `SpecializationKeyBuilder` and `SpecializationCache` provide
bounded, exact-key shape/CTC specialization without host dependencies. A backend still decides
which `CUSTOM` domains/operators and implementation payloads it supports.

The crate also exposes stable-Rust helpers for the TOSA low-precision wire formats:
`low_precision_storage_bytes`, low-nibble-first `pack_int4`/`unpack_int4`, and exact
`fp8e4m3_to_f32`/`fp8e5m2_to_f32` reference conversion. They are serialization and numerical
utilities, not a claim that every backend natively executes those dtypes; providers must publish
and enforce their own capability boundary.

## Schema provenance

`schema/tosa.fbs` is copied byte-for-byte from TOSA Tools `v2026.05.0`; its SHA-256 is
`a1d6383bdecddf9cc6a00c33e5a2fac7e8479dc560a9da2f68ba41da35e143f8`. The private Rust bindings
were generated once with `flatc 25.2.10 --rust` and are checked in, so consumers do not need a code
generator. The upstream schema is copyright Arm Limited and licensed under Apache-2.0. See
[`SAFETY.md`](SAFETY.md) for the generated-code audit boundary.

The crate has no dependency on `tosa-rs`; that prototype is reference material only. Third-party
utility crates remain welcome where they add focused validation, transforms, builders, or compiler
integration without weakening this crate's portable parsing boundary.

This project is not affiliated with or endorsed by Arm. TOSA is an Arm trademark.

## License

Project-authored code is licensed under either Apache-2.0 or MIT, at your option. The pinned
upstream TOSA schema retains its Apache-2.0 notice.
