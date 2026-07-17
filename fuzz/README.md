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

All targets cap input length before allocation. A resource-limit rejection, malformed descriptor,
or protocol error is not a crash unless it violates a processor, queue, or codec invariant.

## Local Commands

```sh
cargo test --manifest-path fuzz/Cargo.toml --lib --no-default-features
python3 ci/seed-fuzz-corpus.py
cargo fuzz list
cargo fuzz run protocol_decode fuzz/corpus/protocol_decode -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run descriptor_end_to_end fuzz/corpus/descriptor_end_to_end -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run stateful_commands fuzz/corpus/stateful_commands -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
```

Minimize a failing input with `cargo fuzz tmin <target> <artifact>`, then commit the minimized file
under `fuzz/regressions/<target>/`. CI replays both generated seeds and committed regressions.
