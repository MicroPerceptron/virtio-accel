# Unsafe-code audit

This crate will become a host-native exception to the portable workspace's `forbid(unsafe_code)`
rule, on the same terms as `virtio-accel-openvino`: unsafe Rust confined to the `va_xdna` build
configuration — `src/ffi.rs` (HRX C ABI declarations only) and `src/native.rs` (the calls) — with
the crate root relaxed to `cfg_attr(not(va_xdna), forbid(unsafe_code))` and this file registered
as the audited exception in `ci/check-release-policy.py`.

**Scaffold state.** The native modules do not exist yet. This build compiles **no `unsafe` at
all** — only the portable admission surface (`src/lower.rs`) and a placeholder — so the crate root
still forbids unsafe outright. The full audit is authored with the FFI and hardware tickets. Its
planned structure, transcribed from §4 of the crate-layout design record (issue #83), is:

1. **Boundary statement** — safe admission (`lower.rs`) and safe compiler subprocess
   (`compiler.rs`); unsafe confined to `ffi.rs` declarations and `native.rs` call sites.
2. **ABI and runtime lifetime** — refcounted HRX handles with one Rust owner each; construction
   stores each handle before the next fallible step and releases the initialized prefix on error;
   every non-NULL `hrx_status_t` is consumed exactly once (`hrx_status_to_string` →
   `hrx_status_free_message` → `hrx_status_ignore` for both statuses); one process-lifetime HRX
   context through a single audited shared owner, never shut down.
3. **Buffers and mappings** — one persistent mapping per buffer, valid until buffer release;
   mapped access bounded by the allocation; `hrx_buffer_flush_range` after every host write and
   `hrx_buffer_invalidate_range` before every host read; zero-size allocations rejected before FFI;
   all `hrx_amdxdna_executable_create` input storage borrowed only for the call.
4. **Concurrency and the dispatch worker** — all dispatch/synchronize serialized on one worker;
   device loss is two-tier (kernel TDR yields a terminal `Failed` with buffers safely releasable;
   a longer worker watchdog covers a true wedge with a never-terminal event whose guards leak until
   process exit). No cancellation is advertised.
5. **Teardown ordering** — events terminal → executables released → buffers released → stream →
   device.
6. **Audited unsafe operations** — every `unsafe` block carries a local `SAFETY:` comment; the
   native test inventory is listed here.
