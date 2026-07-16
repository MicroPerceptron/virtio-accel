# Protocol 1.0 conformance artifacts

This directory contains implementation-independent inputs for the frozen portable wire contract.

- [`layout.json`](layout.json) records protocol constants and every structure size, byte alignment,
  and field offset.
- [`vectors.json`](vectors.json) contains canonical hexadecimal bytes for the device configuration,
  all 15 request opcodes, every success and command-specific response shape, all event states, and
  reviewed malformed/unknown boundary cases.

The files are deliberately plain JSON with hexadecimal byte strings so implementations do not need
Rust tooling to consume them.

Ordinary tests parse these checked-in files as inputs. They do not regenerate them. An intentional
protocol revision must update the normative specification, Rust layout assertions, manifest, and
vectors in one reviewed change. Incompatible changes require a new versioned directory.
