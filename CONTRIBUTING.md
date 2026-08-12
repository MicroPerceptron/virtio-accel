# Contributing to virtio-accel

Thanks for looking. This project is a portable, pre-standardization protocol and Rust foundation for
a transport-neutral accelerator device. Contributions are welcome, including the kind that argue the
design is wrong — a well-reasoned objection to a frozen decision is worth more to us than a
workaround built on top of it.

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before participating.

## Where to start

| You want to | Go here |
|---|---|
| Report a suspected vulnerability | **Do not open an issue.** Follow [SECURITY.md](SECURITY.md) |
| Report a bug | [Open a bug report](https://github.com/MicroPerceptron/virtio-accel/issues/new?template=bug_report.yml) |
| Propose a protocol or wire change | [Open a protocol change proposal](https://github.com/MicroPerceptron/virtio-accel/issues/new?template=protocol_change.yml) — read the classification rules below first |
| Ask how to port a backend | [Discussions → Q&A](https://github.com/MicroPerceptron/virtio-accel/discussions) |
| Float a design idea | [Discussions → Ideas](https://github.com/MicroPerceptron/virtio-accel/discussions) |
| Implement a backend | [Backend implementer guide](docs/backend-implementer-guide.md) |

If you are not sure whether something is a bug or a design disagreement, open a discussion. It is
easier to promote a discussion to an issue than to unwind an argument held in a bug tracker.

## Before you open a pull request

Requires Rust 1.85 or newer (edition 2024). Run the same gates CI runs:

```sh
cargo fmt --all -- --check
python3 ci/check-release-policy.py
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo run --example backend_conformance
cargo run --example reference_execution
python3 ci/publish-dry-run.py
```

Portable-target checks need the bare-metal standard libraries:

```sh
rustup target add aarch64-unknown-none riscv64gc-unknown-none-elf wasm32-unknown-unknown
```

CI additionally enforces MSRV, the portable target matrix, the Cargo feature powerset, dependency
and license policy, and bounded fuzz smoke runs. `publish-dry-run.py` is the gate that matters most
before a release: `cargo package`'s own verify step only builds the library target, which is how
four cross-package `include_str!` sites once shipped assertions that could not compile from a
published tarball.

## Protocol changes are classified before code is merged

This is the one rule that is not negotiable, because it is what makes the protocol reviewable by
independent implementers. Protocol 1.0 is frozen by the
[freeze audit](conformance/v1.0/freeze-audit.md).

Every proposed wire change must first be classified under
[docs/wire-abi.md §9](docs/wire-abi.md):

1. **Erratum** — changes no accepted or emitted bytes. May clarify the 1.0 documents and tests.
2. **Compatible extension** — uses a previously reserved number with explicit feature or new-opcode
   negotiation, preserves every 1.0 frame, and gets a new minor-version conformance directory.
3. **Breaking change** — any changed assigned number, payload length, field meaning, required
   response, or ownership interpretation. Requires a new protocol *major* version and its own
   conformance directory.

A change in category 2 or 3 must update the normative documents, the Rust constants and layout
assertions, the machine layout manifest, the canonical vectors, and the compatibility tests *in the
same reviewed change*. Version directories are never regenerated opportunistically from current Rust
types — `layout.json`, `vectors.json`, `scenarios.json`, and `requirements.json` are authoritative
inputs, not build outputs.

Cargo versioning is a separate axis from protocol versioning; see
[docs/release-policy.md](docs/release-policy.md).

## What will not be merged

These are scope boundaries, not judgments about the code:

- **Platform adapters in portable crates.** Linux ioctls, macOS frameworks, Windows APIs, VMM or
  kernel glue, and vendor SDKs belong in downstream crates that depend on these layers.
- **A default feature that selects host behavior** in a portable crate. Features must be additive:
  disabling them may remove convenience, never change protocol interpretation.
- **Unaudited `unsafe` code.** Project-authored portable code forbids or denies unsafe code. The
  existing Core ML FFI and private generated TOSA bindings are confined by their `SAFETY.md`
  audits. A new exception requires a discussion and the release-policy evidence before a patch.
- **A claimed Virtio device ID.** Its absence is deliberate and documented.
- **Advertising a deferred feature** — multi-queue, event queues, external memory import/export,
  timeline fences, secure contexts, or packed virtqueues — without the negotiation, ownership,
  synchronization, and conformance rules that a future protocol version would have to assign.

## Pull requests

Keep them focused; a small PR that does one thing gets reviewed faster than a broad one. Explain
**why** in the description — the *what* is visible in the diff.

The PR template carries a short checklist derived from the
[release review checklist](docs/release-policy.md). The questions that catch the most problems:

- Does this alter accepted or emitted bytes, payload lengths, ownership, reset, error, timeout, or
  feature-negotiation behavior?
- Does it change a public Rust item that backend, guest, device, or transport authors depend on?
- Does it move platform behavior into a portable crate via a dependency, feature, or target?

Commit messages: short, imperative, and explain the reasoning when it is not obvious.

## Licensing

By contributing, you agree that your contributions are licensed under the same terms as the project:
[MIT](LICENSE-MIT) **or** [Apache-2.0](LICENSE-APACHE) at the user's option. Inbound licensing
matches outbound. There is no separate CLA.

If you are contributing on behalf of an employer, please make sure you have the right to do so
before opening the PR.
