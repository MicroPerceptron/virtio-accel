# Fuzzing

This workspace keeps coverage-guided fuzzing separate from the portable crates. The shared harness
code is regular Rust and can be tested without libFuzzer; the `fuzzing` feature only enables the
`cargo fuzz` binaries.

## Targets

| Target | Contract |
|---|---|
| `protocol_decode` | Feeds arbitrary bounded byte frames through contiguous and segmented decode paths and checks accepted frames against the clean-room codec. |
| `descriptor_end_to_end` | Builds raw or canonical split-virtqueue descriptor chains, runs valid chains through the command processor, and verifies used-length, truncation, and response-tail behavior. |
| `stateful_commands` | Generates bounded create/use/destroy/reset command sequences and checks resource counts, retained bytes, stale IDs, and backend health after every action. |
| `guest_client` | Drives the reference guest client against an arbitrary, non-conforming device and checks in-flight bounds, epoch staleness, and caller-chain ownership after every action. |
| `tosa_parse` | Mutates a stable upstream TOSA graph, checks every safe graph/attribute view against collected statistics, and runs complete plus minimal Level 8K semantic targets. |

All targets cap input length before allocation. A resource-limit rejection, malformed descriptor,
or protocol error is not a crash unless it violates a processor, queue, or codec invariant.

## Guest client input layout

`guest_client` reads a little-endian `u16` response-byte pool length, that many pool bytes, and then
a stream of eight-byte actions. Seeds fill the pool with reviewed response frames from
`conformance/v1.0/vectors.json` so a mutated header or used length is applied to otherwise
canonical bytes. The device side is driven by the harness itself rather than the split-queue model,
so it can publish responses no conforming transport would produce.

## Local Commands

```sh
cargo test --manifest-path fuzz/Cargo.toml --lib --no-default-features
python3 ci/seed-fuzz-corpus.py
cargo fuzz list
cargo fuzz run protocol_decode fuzz/corpus/protocol_decode -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run descriptor_end_to_end fuzz/corpus/descriptor_end_to_end -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run stateful_commands fuzz/corpus/stateful_commands -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run guest_client fuzz/corpus/guest_client -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run tosa_parse fuzz/corpus/tosa_parse -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
```

Minimize a failing input with `cargo fuzz tmin <target> <artifact>`, then commit the minimized file
under `fuzz/regressions/<target>/`. CI replays both generated seeds and committed regressions.
