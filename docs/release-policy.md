# Release and evolution policy

This policy applies after the protocol 1.0 freeze audit. It keeps Cargo package evolution, wire
compatibility, feature selection, unsafe code, dependency selection, and target support aligned.

## Version dimensions

There are two separate version axes:

- Cargo crate versions describe the Rust API and package graph.
- Protocol versions describe driver/device wire compatibility and conformance artifacts.

The two axes advance independently. A Cargo patch, minor, or major change does not by itself change
the wire protocol, and a wire protocol change must follow the protocol classification below even when
the Rust crate version is still pre-1.0. A public protocol 1.0 release needs a matching release note
and a frozen `conformance/v1.0` directory; it does not require a `1.0.0` Cargo version.

### Cargo version posture

The workspace publishes at `0.1.x` while carrying the frozen protocol 1.0 baseline. This is
deliberate rather than an unreconciled gap:

- The Cargo version tracks the **Rust API and package graph**, which is young and expected to change
  as backend, guest, device, and transport adapter authors build against it.
- The protocol version tracks **wire compatibility**, which is frozen by the
  [v1.0 freeze audit](../conformance/v1.0/freeze-audit.md) and governed by the classification table
  below.

A pre-1.0 Cargo version is therefore the accurate signal on both axes, and is not a statement about
protocol stability. Consumers who need the stable artifact should depend on protocol 1.0 and its
conformance directory, not on a Cargo version number.

Moving the workspace to `1.0.0` is a separate, later decision. It requires the public Rust API to
have real downstream users and an explicit semver promise recorded in a release note. Until then,
breaking Rust API changes ship as `0.x` minor bumps under the classification table below.

## Change classification examples

| Change | Cargo classification | Protocol classification | Required evidence |
|---|---|---|---|
| Fix rustdoc, examples, comments, non-normative rationale, or tests without changing accepted/emitted bytes | Patch | Erratum or no protocol change | CI plus updated docs when relevant |
| Add a new helper type or trait method with a default implementation that preserves existing behavior | Minor while pre-1.0/public policy permits it; otherwise semver-compatible minor | No protocol change | API review and downstream compile coverage |
| Remove, rename, or change the meaning of a public Rust item used by backend, guest, device, or transport authors | Cargo major | No protocol change unless wire behavior also changes | Migration note and affected-crate review |
| Raise MSRV or remove a supported portable target | Cargo minor only if release notes document it and no public semver promise forbids it; otherwise Cargo major | No protocol change | Target/MSRV rationale and CI matrix update |
| Add a platform adapter crate that depends on portable crates but is not a default dependency | Additive Cargo minor | No protocol change | Dependency-policy and portability review |
| Add a default feature that selects Linux, macOS, Windows, VMM, kernel, vendor SDK, filesystem, socket, thread, or runtime behavior in a portable crate | Forbidden | Forbidden unless it is a negotiated protocol feature and isolated from portable defaults | Must be redesigned |
| Assign a reserved feature bit, opcode, status, flag, field, or capability with negotiated behavior and unchanged 1.0 frames | Cargo minor or major depending on Rust API impact | Protocol minor with a new conformance directory | Normative docs, feature negotiation tests, vectors, scenarios, and clean-room coverage |
| Append fields to an existing response a 1.0 driver can receive without a negotiated feature | Forbidden in protocol 1.x | Protocol major if required | New major-version directory |
| Change an assigned opcode value, structure size, field meaning, status success/failure interpretation, ownership rule, reset rule, or existing payload length | Cargo major if Rust API also changes | Protocol major | New normative documents and conformance directory |

## Wire evolution

Protocol 1.0 freezes the assigned values, exact payload lengths, ownership rules, and golden bytes in
`conformance/v1.0`. Unknown fields are not a baseline extension mechanism: protocol 1.0 receivers
validate exact payload lengths and reject trailing bytes unless a negotiated feature explicitly
selects a different layout. Unknown opcodes remain unsupported without side effects, unknown request
or object flags are rejected before backend invocation, unknown response statuses are opaque
failures, and unknown event states require recovery rather than being guessed terminal.

New behavior should prefer one of these forms, in order:

1. A new opcode with exact request and response payloads.
2. A previously reserved feature bit that gates all changed behavior.
3. A previously reserved value whose semantics are specified in full before it is advertised.
4. A new protocol major version when compatibility cannot be preserved.

Reserved values are invalid until assigned by a later policy. A constant that records a reserved
number is not permission for a device to advertise it or a driver to accept it.

## Cargo feature policy

Cargo features are additive. Disabling a default feature may remove convenience code but must not
select a different protocol interpretation. Enabling a feature must not make a portable crate depend
on an operating system, VMM, kernel, guest-memory library, vendor SDK, filesystem, socket, thread,
global runtime, or platform synchronization primitive.

Platform integrations must live in adapter crates that depend inward on the portable layers. They
must not become default dependencies of `virtio-accel-core`, `virtio-accel-proto`,
`virtio-accel-transport`, `virtio-accel-device`, `virtio-accel-guest`,
`virtio-accel-split-queue`, or the facade crate.

## MSRV and supported targets

The minimum supported Rust version is the workspace `rust-version`. A change to MSRV requires a
release-note entry, a CI matrix update, and an explanation of why the old compiler cannot preserve
the current API or implementation invariants.

The supported portable target set is the one documented in `docs/portability.md` and enforced by
CI. Removing a target or moving a crate to a less-portable runtime tier requires a release-note
entry and an explicit portability review. Adding a platform adapter cannot reduce the portability
tier of an existing crate.

## Unsafe-code policy

Project-authored code in portable crates, reference crates, and fuzz harness support code forbids or
denies unsafe code at the crate root. This is an intentional v1 invariant, not incidental linting.
There are two reviewed, confined exceptions. The host-native `virtio-accel-coreml` adapter's Rust
FFI and aligned-allocation code is documented in `crates/virtio-accel-coreml/SAFETY.md`; non-macOS
builds still forbid unsafe code. `virtio-accel-tosa` denies unsafe code globally but locally permits
its private, checked-in official FlatBuffers bindings after bounded verification; the boundary and
regeneration procedure are documented in `crates/virtio-accel-tosa/SAFETY.md`.

A future unsafe exception requires all of the following in one reviewed change:

- the crate-level `forbid(unsafe_code)` removal or replacement is explicit;
- every unsafe block has a local safety comment naming the invariant it relies on;
- the release review records why a safe abstraction, `zerocopy` validation, ownership token, or
  adapter boundary could not preserve the invariant;
- tests or conformance evidence exercise the unsafe boundary; and
- the public API does not transfer unsafe obligations to downstream users unless those obligations
  are documented on the item that requires them.

## Dependency and license policy

Workspace dependencies must be centralized in `[workspace.dependencies]` unless there is a narrow
crate-local reason to diverge. Normal dependencies for portable crates should use minimal features
and `default-features = false` when the dependency supports it.

Dependency review must check:

- cargo-deny advisories, yanked crates, duplicate versions, wildcard requirements, unknown sources,
  and licenses;
- whether a build dependency leaks `std` or `alloc` into a target graph;
- whether a proc macro or helper crate is build-host-only or runtime-visible;
- whether a dependency introduces platform defaults; and
- whether its license remains inside the workspace allowlist.

The workspace license is `MIT OR Apache-2.0`. Every published manifest inherits `license`,
`rust-version`, `repository`, `homepage`, `keywords`, and `categories` from `[workspace.package]`,
declares its own `description` and `readme`, and carries its own byte-identical copies of
`LICENSE-MIT` and `LICENSE-APACHE`. Cargo only packages files inside a package directory, so the
root license files do not reach the sub-crate tarballs; the copies exist for that reason and are
copies rather than symlinks because CI runs `windows-latest`.

`ci/check-release-policy.py` enforces all of this against an explicit twelve-crate allowlist. A new
package fails that check until it is added to the allowlist, which forces a decision about whether
it is public rather than letting it default either way. The check also validates the crates.io
keyword and category limits, which neither `cargo package` nor `cargo publish --dry-run` catches
before an upload is attempted.

The `fuzz/` harness is a separate workspace at version `0.0.0` and stays `publish = false`.

## Publication, yank, and rollback

Twelve packages are published to crates.io. Publication is ordered: a crate cannot be published before
every crate it depends on, and that includes development dependencies, because a published crate's
versioned dev-dependencies must resolve from the registry for `cargo test` to run on the packaged
source.

| # | Crate | Normal dependencies | Development dependencies |
|---|---|---|---|
| 1 | `virtio-accel-transport` | — | — |
| 2 | `virtio-accel-cleanroom` | — | — |
| 3 | `virtio-accel-proto` | — | `cleanroom` |
| 4 | `virtio-accel-core` | `transport` | — |
| 5 | `virtio-accel-tosa` | `core`, FlatBuffers | — |
| 6 | `virtio-accel-split-queue` | `transport` | — |
| 7 | `virtio-accel-guest` | `proto`, `transport` | `split-queue` |
| 8 | `virtio-accel-mock` | `core` | — |
| 9 | `virtio-accel-device` | `core`, `proto`, `transport` | `mock` |
| 10 | `virtio-accel-conformance` | `core` | `mock` |
| 11 | `virtio-accel-coreml` | `core`, `tosa` | `conformance` |
| 12 | `virtio-accel` | the six runtime crates | `conformance`, `mock`, `cleanroom` |

This order is executable, not just documentary: `ci/publish-dry-run.py` walks it against an isolated
local registry, adding each crate only after it has been built, tested, and documented from its own
extracted tarball. A crate can therefore only ever resolve its predecessors, so a wrong order fails
with an unresolvable dependency instead of passing quietly. The same script is a required CI job.

`cargo package`'s own verify step is not sufficient and must not be treated as sufficient: it builds
only the library target. That is how four cross-package `include_str!` sites reached outside their
package directories unnoticed, leaving assertions that could never have compiled from a published
tarball. Any check on packaged output must run the tests inside the packaged source.

### When a mid-order publish fails

crates.io publication is not transactional across crates. If crate N fails after 1..N-1 succeeded,
those earlier versions are live and permanent.

1. Stop. Do not publish the remaining crates, and do not attempt to reuse the version number.
2. Diagnose against the local registry, not against crates.io. Reproduce with
   `ci/publish-dry-run.py`.
3. Fix forward. Bump the patch version of the crate that failed and of any crate that must depend on
   the fixed version, then re-run the ordered publication from the first crate whose version
   changed. Earlier crates that published correctly are left alone.
4. Yank only if a published version is actively harmful — see below. A version that is merely
   stranded, because its dependents were never published, is not harmful; it is unreachable.

### Yank versus patch

A crates.io version is immutable. It cannot be edited, replaced, or deleted, and its contents remain
downloadable even after a yank. Publishing is therefore a one-way action, and a mistaken publish is
corrected by publishing again, never by trying to undo.

Yanking only stops *new* resolution: existing `Cargo.lock` files continue to resolve a yanked
version, so a yank is not a security control and never a substitute for an advisory.

Publish a patch, and do not yank, when:

- the defect is a bug, a missing file, or wrong metadata that a newer version supersedes;
- the version is stranded but harmless; or
- downstream users are better served by upgrading than by a broken resolution.

Yank, in addition to publishing a patch, when:

- the version is a security risk to anyone who resolves it — coordinate with
  [SECURITY.md](../SECURITY.md) and publish an advisory, since the yank alone protects nobody;
- it claims a protocol conformance it does not have, so a driver or device could interoperate
  incorrectly on the wire; or
- it was published in error and has no valid use, such as a wrong version number or a crate
  published out of order with an unsatisfiable dependency.

Never un-yank to "restore" a version that was yanked for a wire-compatibility or security reason.
Publish a new version instead.

### Rollback

There is no rollback. The recovery path for every publication mistake is a new version, in the same
documented order, with a release-note entry recording what happened and why. If a protocol-affecting
defect ships, the classification table above governs whether the fix is an erratum, a protocol minor
extension, or a new protocol major version with its own conformance directory — a security fix is
not exempt from that classification.

## Release review checklist

Every release or compatibility-affecting PR should answer:

- Does this change alter accepted or emitted protocol bytes, exact payload lengths, ownership,
  reset, error, timeout, or feature-negotiation behavior?
- If yes, is it a protocol erratum, minor extension, or major-version change under this policy?
- Are `layout.json`, `vectors.json`, `scenarios.json`, `requirements.json`, and performance budgets
  still authoritative inputs rather than regenerated by accident?
- Did any public Rust API change affect backend implementers, guest/device authors, or transport
  adapters?
- Did any default dependency, Cargo feature, or target move platform behavior into a portable crate?
- Did any crate add or permit unsafe code, and is the audit trail complete?
- Did dependency, license, advisory, and MSRV checks pass?
- Are deferred optional features still unadvertised and documented as out of scope?
