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

# virtio-accel

An experimental virtual-accelerator protocol plus production-oriented Rust
implementations.

</div>

<div class="landing-grid">

<div class="landing-card">

## Protocol 1.0

A frozen, versioned wire contract for exposing an accelerator to a guest:
contexts, buffers, programs, execution queues, submissions, and events.

[Read the specification](docs/specification.md)

</div>

<div class="landing-card">

## Portable Rust

Executable `no_std` guest, device, transport, queue, and TOSA layers with a
transport-free conformance suite and an independent clean-room codec.

[Browse the API](api.md)

</div>

<div class="landing-card">

## Real backends

macOS Core ML / ANE, Intel OpenVINO, and Qualcomm Hexagon adapters that lower
device-neutral TOSA to native runtimes with direct buffer bindings.

[Getting started](getting-started.md)

</div>

</div>

<div class="landing-note">

This project is pre-standardization and experimental. Protocol 1.0 is frozen as
a versioned review input for independent implementation — stable enough to build
against and to disagree with in writing, not an approved Virtio specification.

</div>
