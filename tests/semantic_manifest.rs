use virtio_accel::core::Capabilities;

#[test]
fn semantic_capabilities_match_the_versioned_manifest() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../conformance/v1.0/layout.json")).unwrap();
    let assigned = &manifest["capabilities"]["assigned"];
    let reserved = &manifest["capabilities"]["reserved"];

    for (name, value) in [
        ("HOST_VISIBLE_MEMORY", Capabilities::HOST_VISIBLE_MEMORY),
        ("DEVICE_LOCAL_MEMORY", Capabilities::DEVICE_LOCAL_MEMORY),
        ("EVENT_CANCELLATION", Capabilities::EVENT_CANCELLATION),
        ("SHARED_MEMORY", Capabilities::SHARED_MEMORY),
    ] {
        assert_eq!(assigned[name], format!("0x{:016x}", value.bits()));
    }

    for (name, value) in [
        ("EXTERNAL_MEMORY", Capabilities::EXTERNAL_MEMORY),
        ("SECURE_CONTEXTS", Capabilities::SECURE_CONTEXTS),
    ] {
        assert_eq!(reserved[name], format!("0x{:016x}", value.bits()));
    }
}
