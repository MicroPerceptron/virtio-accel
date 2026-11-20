# virtio-accel vhost-user host

Experimental **Linux host** adapter for protocol 1.0. One daemon process owns one virtual device
and one selected accelerator. QEMU presents modern Virtio MMIO to its guest and forwards its
queues to this daemon using vhost-user. The host contains no guest-OS driver.

See [the binding and validation guide](../../docs/vhost-user.md) for builds, QEMU patches,
backend selection, resource budgets, and current limitations.

This is an unpublished, separate Cargo workspace. Linux and rust-vmm dependencies do not enter
any portable crate. Project-authored Rust in this adapter forbids unsafe code. The QEMU patches
retain the source licenses of the upstream QEMU files they modify; the Rust adapter
uses the repository's MIT OR Apache-2.0 license.
