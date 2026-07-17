use virtio_accel_conformance::{BindingFixture, ProgramFixture, TargetDescription};
use virtio_accel_core::{AccessMode, MemoryDomain};
use virtio_accel_mock::reference;

pub fn target() -> TargetDescription {
    target_in(MemoryDomain::Shared)
}

pub fn target_in(domain: MemoryDomain) -> TargetDescription {
    let initial = vec![0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];
    let expected = initial.iter().map(|byte| byte ^ 0x5a).collect::<Vec<_>>();
    TargetDescription::new(
        ProgramFixture::new(
            reference::ARTIFACT_FORMAT,
            reference::TARGET_IDENTITY,
            reference::ReferenceArtifact::xor(7, 0x5a).as_bytes(),
            reference::RESIDENT_BYTES,
        )
        .unwrap(),
        BindingFixture::new(7, AccessMode::ReadWrite, domain, 16, initial, expected).unwrap(),
    )
}
