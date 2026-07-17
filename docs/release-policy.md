# Release and evolution policy

This policy applies after the protocol 1.0 freeze audit. It keeps Cargo package evolution, wire
compatibility, feature selection, unsafe code, dependency selection, and target support aligned.

## Version dimensions

There are two separate version axes:

- Cargo crate versions describe the Rust API and package graph.
- Protocol versions describe driver/device wire compatibility and conformance artifacts.

The root workspace version is allowed to remain lower than `1.0.0` while the repository is private
or unpublished, but a public protocol 1.0 release needs a matching release note and a frozen
`conformance/v1.0` directory. A Cargo patch, minor, or major change does not by itself change the
wire protocol. A wire protocol change must follow the protocol classification below even when the
Rust crate version is still pre-1.0.

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

All current crates and fuzz harness support code forbid unsafe code at the crate root. This is an
intentional v1 invariant, not incidental linting.

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

The workspace license is `MIT OR Apache-2.0`. Current package manifests inherit the workspace
license and rust-version, include descriptions, and set `publish = false`. Publishing a crate later
requires an explicit metadata review before changing that field.

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
