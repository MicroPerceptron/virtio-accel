<div id="landing-hero" class="landing-hero">

<svg class="landing-logo" width="160" height="107" viewBox="0 0 240 160" aria-hidden="true">
  <g transform="translate(120 80) rotate(-18)">
    <g stroke="var(--fg)" stroke-width="1.5" opacity="0.3">
      <line x1="38" y1="0" x2="98" y2="0"/>
      <line x1="32.9" y1="7" x2="84.9" y2="17.5"/>
      <line x1="19" y1="12.1" x2="49" y2="30.3"/>
      <line x1="0" y1="14" x2="0" y2="35"/>
      <line x1="-19" y1="12.1" x2="-49" y2="30.3"/>
      <line x1="-32.9" y1="7" x2="-84.9" y2="17.5"/>
      <line x1="-38" y1="0" x2="-98" y2="0"/>
      <line x1="-32.9" y1="-7" x2="-84.9" y2="-17.5"/>
      <line x1="-19" y1="-12.1" x2="-49" y2="-30.3"/>
      <line x1="0" y1="-14" x2="0" y2="-35"/>
      <line x1="19" y1="-12.1" x2="49" y2="-30.3"/>
      <line x1="32.9" y1="-7" x2="84.9" y2="-17.5"/>
    </g>
    <ellipse cx="0" cy="0" rx="100" ry="36" fill="none" stroke="var(--links)" stroke-width="7"/>
    <ellipse cx="0" cy="0" rx="80" ry="29" fill="none" stroke="var(--sidebar-active)" stroke-width="6" opacity="0.85"/>
    <ellipse cx="0" cy="0" rx="60" ry="22" fill="none" stroke="var(--links)" stroke-width="5" opacity="0.7"/>
    <ellipse cx="0" cy="0" rx="38" ry="14" fill="none" stroke="var(--fg)" stroke-width="3" opacity="0.4"/>
    <circle cx="0" cy="0" r="6" fill="var(--fg)"/>
  </g>
</svg>

<p class="landing-kicker">Open source &middot; Rust</p>

# virtio-accel

<p class="landing-lede">An experimental, native-Rust foundation for a transport-neutral virtual
accelerator device. First target: NPU execution — with a core model that leaves room for GPUs,
DSPs, and other program-driven accelerators.</p>

<p class="landing-status"><span class="landing-status-dot"></span> Active &middot; protocol 1.0 frozen &middot; workspace at 0.3.x</p>

</div>

<div class="landing-sections">

<div class="landing-section">

## What it is

virtio-accel concentrates on the portable majority of the system: protocol wire structures,
transport ports, device-owned state, a guest client, a mock backend, and a transport-free
conformance suite — without Linux ioctls, macOS frameworks, Windows APIs, guest physical
addresses, vendor command formats, or a claimed Virtio device ID.

Platform adapters are meant to be written against these layers rather than inside them. What
ships here is the part that should not have to be rewritten per platform.

</div>

<div class="landing-section">

## The workspace

Ten crates publish together in dependency order, each pinned to a portability tier that CI
enforces on bare-metal `aarch64`, `riscv64`, and `wasm32` targets, so a crate cannot quietly
acquire a host dependency. Every crate is `#![forbid(unsafe_code)]`.

<dl class="landing-crates">
<div><dt>virtio-accel</dt><dd>Facade re-exporting the portable layers.</dd></div>
<div><dt>virtio-accel-proto</dt><dd>Pointer-free, little-endian protocol 1.0 wire structures.</dd></div>
<div><dt>virtio-accel-transport</dt><dd>Descriptor-chain, queue, reset, and notification ports.</dd></div>
<div><dt>virtio-accel-core</dt><dd>Backend lifecycle, memory, program, queue, and event contracts.</dd></div>
<div><dt>virtio-accel-conformance</dt><dd>Transport-free semantic suite and numerical corpus.</dd></div>
</dl>

[Full crate reference →](api.md)

</div>

<div class="landing-section">

## Protocol 1.0

A frozen, versioned wire contract for exposing an accelerator to a guest: contexts, buffers,
programs, execution queues, submissions, and events.

Unknown opcodes, statuses, and event states stay raw integers until validated, so decoding
untrusted bytes never constructs an invalid Rust enum. Failure still returns an event: a
successful submit yields one, and an indeterminate failure must yield one too, because the
operation's resources are still owned by the device.

[Read the specification →](docs/specification.md)

</div>

<div class="landing-section">

## Supported backends

macOS Core ML / ANE, Intel OpenVINO, and Qualcomm Hexagon adapters lower device-neutral TOSA to
native runtimes with direct buffer bindings, alongside an in-memory mock backend for portable
conformance testing.

[Backend implementer guide →](docs/backend-implementer-guide.md)

</div>

<div class="landing-section">

## Status

The workspace publishes together at 0.3.x. The Cargo version tracks the Rust API, which is
still young and expected to change as adapter authors build against it; Protocol 1.0 itself is
frozen separately by the v1.0 freeze audit — a pre-1.0 crate version is not a statement about
protocol stability.

</div>

</div>

<div class="landing-note">

This project is pre-standardization and experimental. Protocol 1.0 is frozen as
a versioned review input for independent implementation — stable enough to build
against and to disagree with in writing, not an approved Virtio specification.

</div>

<div class="landing-links">

<p class="landing-kicker">Links</p>

- [GitHub](https://github.com/MicroPerceptron/virtio-accel)
- [crates.io](https://crates.io/crates/virtio-accel)
- [Getting started](getting-started.md)
- [API reference](api.md)

</div>
