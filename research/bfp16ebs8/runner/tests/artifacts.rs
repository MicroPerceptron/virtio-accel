//! BLAKE3 verification of the committed probe artifacts (review requirement: committed
//! binaries are pinned by checksum). Regenerate `../artifacts/BLAKE3SUMS` with
//! `b3sum * > BLAKE3SUMS` in that directory after any deliberate probe rebuild.

use std::path::Path;

#[test]
fn every_committed_probe_artifact_matches_its_recorded_blake3_hash() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../artifacts");
    let sums = std::fs::read_to_string(dir.join("BLAKE3SUMS")).expect("read BLAKE3SUMS");
    let mut checked = 0usize;
    for line in sums.lines() {
        let line = line.trim();
        if line.is_empty() || line.ends_with("BLAKE3SUMS") {
            continue;
        }
        let (hash, name) = line
            .split_once("  ")
            .expect("BLAKE3SUMS line: <hash>  <name>");
        let bytes =
            std::fs::read(dir.join(name.trim())).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            blake3::hash(&bytes).to_hex().as_str(),
            hash,
            "{name}: committed bytes do not match the recorded BLAKE3 hash"
        );
        checked += 1;
    }
    let files = std::fs::read_dir(&dir)
        .expect("list artifacts")
        .filter(|entry| {
            entry
                .as_ref()
                .is_ok_and(|e| e.file_name() != "BLAKE3SUMS" && e.path().is_file())
        })
        .count();
    assert_eq!(
        checked, files,
        "every committed artifact must be listed in BLAKE3SUMS"
    );
}
