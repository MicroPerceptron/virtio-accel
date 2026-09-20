//! Checksum verification for the committed binary fixtures under `tests/data/`.
//!
//! Committed binaries cannot be reviewed line by line, so each one is pinned by its BLAKE3
//! hash in `tests/data/BLAKE3SUMS` (regenerate with `b3sum *.xdnp *.xbfp > BLAKE3SUMS` after
//! any deliberate rebuild). This test runs on every host — it is deliberately NOT gated on
//! `va_xdna` — so CI rejects a fixture that drifts from its recorded hash, and a hash change
//! always appears in review next to the binary it covers.

const SUMS: &str = include_str!("data/BLAKE3SUMS");

const FIXTURES: &[(&str, &[u8])] = &[
    (
        "passthrough-dmas-npu2.xdnp",
        include_bytes!("data/passthrough-dmas-npu2.xdnp"),
    ),
    (
        "xbfp-mxint8-matmul-8x512x8-v1.xbfp",
        include_bytes!("data/xbfp-mxint8-matmul-8x512x8-v1.xbfp"),
    ),
];

#[test]
fn every_committed_binary_fixture_matches_its_recorded_blake3_hash() {
    let mut recorded = std::collections::BTreeMap::new();
    for line in SUMS.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (hash, name) = line
            .split_once("  ")
            .expect("BLAKE3SUMS line: <hash>  <name>");
        recorded.insert(name.trim(), hash);
    }
    assert_eq!(
        recorded.len(),
        FIXTURES.len(),
        "BLAKE3SUMS and the FIXTURES table must cover the same files"
    );
    for (name, bytes) in FIXTURES {
        let expected = recorded
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing from BLAKE3SUMS"));
        let actual = blake3::hash(bytes).to_hex();
        assert_eq!(
            &actual.as_str(),
            expected,
            "{name}: committed bytes do not match the recorded BLAKE3 hash"
        );
    }
}
