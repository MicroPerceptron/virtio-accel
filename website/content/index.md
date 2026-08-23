<div id="landing-hero" class="landing-hero">

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
