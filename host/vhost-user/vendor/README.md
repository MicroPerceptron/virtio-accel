# Vendored dependencies of the vhost-user host

## `vhost` — rust-vmm `vhost` 0.16.0, with the backend half of the IOTLB exchange

`vhost/` is the `vhost` 0.16.0 crate exactly as published on crates.io
(upstream <https://github.com/rust-vmm/vhost>, commit
`fd08435326a4dd7144efd42304a2a2beb43c0a78`, directory `vhost/`), plus one
change, and `../Cargo.toml` substitutes it for the registry crate.

**Why it is here.** A device behind an IOMMU learns its translations from the
frontend one range at a time (vhost-user `IOTLB_MSG`). No released `vhost`,
through 0.18.1, has the backend side of that: it rejects the frontend's
message as unknown and cannot send a miss. The change adds

- `VhostUserIotlb`, the wire structure;
- `iotlb_msg` on the backend request handler traits, and its dispatch;
- `Backend::send_iotlb_miss`, `recv_iotlb_miss_reply` and `reply_stream`. The
  send and the wait are separate calls because the frontend answers a miss in
  two steps on two sockets — the translation on the main channel, which it
  waits to have acknowledged, and only then the reply — so a backend that
  blocks on the reply never acknowledges the update.

The first commit that adds this directory is the published crate unmodified;
the change is the commit after it, so `git diff` between the two is the whole
fork. It belongs upstream, and this directory goes away when a release has it.

**Licence.** `Apache-2.0 OR BSD-3-Clause`; used here under Apache-2.0.
`LICENSE` and `LICENSE-BSD-3-Clause` are upstream's files at that commit —
the published crate ships neither. Five source files name a
`LICENSE-BSD-Google` file in their headers; upstream's repository has no file
of that name at that commit.
