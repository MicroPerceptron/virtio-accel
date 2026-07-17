use std::cell::Cell;
use std::collections::BTreeSet;
use std::mem::size_of;

use serde_json::Value;
use virtio_accel::core::{
    AcceleratorClass, BackendError, ByteSource, Capabilities, DeviceIdentity, DeviceInfo,
    DeviceLimits,
};
use virtio_accel::device::{DecodedRequestBody, DecoderLimits, FrameDecodeError, FrameDecoder};
use virtio_accel::proto::{
    BASELINE_COMMAND_QUEUES, HARD_MAX_BINDINGS, HARD_MAX_CHAIN_DESCRIPTORS, HARD_MAX_REQUEST_BYTES,
    HARD_MAX_RESPONSE_BYTES, KnownOpcode, Le16, Le32, LoadProgramRequest, PROTOCOL_MAJOR,
    PROTOCOL_MINOR, RequestHeader, StatusCode, SubmitRequest, WireConfig,
};

#[derive(Debug)]
struct CountingSource {
    bytes: Vec<u8>,
    read_bytes: Cell<u64>,
}

impl CountingSource {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            read_bytes: Cell::new(0),
        }
    }

    fn read_bytes(&self) -> u64 {
        self.read_bytes.get()
    }
}

impl ByteSource for CountingSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(target.len())
            .filter(|end| *end <= self.bytes.len())
            .ok_or(BackendError::OutOfBounds)?;
        self.read_bytes
            .set(self.read_bytes.get() + u64::try_from(target.len()).unwrap());
        target.copy_from_slice(&self.bytes[start..end]);
        Ok(())
    }
}

fn decoder() -> FrameDecoder {
    let config = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(HARD_MAX_CHAIN_DESCRIPTORS),
        max_request_bytes: Le32::new(HARD_MAX_REQUEST_BYTES),
        max_response_bytes: Le32::new(HARD_MAX_RESPONSE_BYTES),
    };
    let info = DeviceInfo {
        identity: DeviceIdentity {
            uuid: [0; 16],
            class: AcceleratorClass::NPU,
            vendor_id: 0,
            device_id: 0,
        },
        capabilities: Capabilities::HOST_VISIBLE_MEMORY
            | Capabilities::DEVICE_LOCAL_MEMORY
            | Capabilities::SHARED_MEMORY,
        limits: DeviceLimits {
            max_contexts: 64,
            max_buffers_per_context: 1024,
            max_programs_per_context: 256,
            max_queues_per_context: 16,
            max_events_per_context: 4096,
            max_bindings_per_submission: HARD_MAX_BINDINGS,
            max_buffer_bytes: 1 << 30,
            max_artifact_bytes: 1 << 30,
        },
    };
    FrameDecoder::new(DecoderLimits::new(&config, info).unwrap())
}

fn scenario_request(scenario_name: &str, request_id: u64) -> Vec<u8> {
    let corpus: Value =
        serde_json::from_str(include_str!("../conformance/v1.0/scenarios.json")).unwrap();
    let scenario = corpus["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .find(|scenario| scenario["name"] == scenario_name)
        .unwrap();
    let request = scenario["trace"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["request_id"].as_u64() == Some(request_id))
        .unwrap()["request"]
        .as_str()
        .unwrap();
    decode_hex(request)
}

fn decode_hex(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).unwrap() as u8;
            let low = (pair[1] as char).to_digit(16).unwrap() as u8;
            (high << 4) | low
        })
        .collect()
}

#[test]
fn performance_budget_manifest_covers_every_v1_hot_path() {
    let budgets: Value =
        serde_json::from_str(include_str!("../conformance/v1.0/performance-budgets.json")).unwrap();
    assert_eq!(budgets["schema"], "virtio-accel-performance-budgets-1");
    let operations = budgets["operations"].as_array().unwrap();
    let ids = operations
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        ids,
        BTreeSet::from([
            "wire.config_decode",
            "wire.request_decode_non_submit",
            "wire.submit_decode",
            "transport.segmented_region_access",
            "state.object_lookup",
            "device.command_dispatch",
            "device.submission_admission",
            "device.polling",
            "device.reset",
            "provider.explicit_transfer",
            "provider.copy_path_diagnostics",
        ])
    );
    for entry in operations {
        assert_eq!(entry["allocates_from_unvalidated_guest_count"], false);
        assert!(entry["complexity"].as_str().unwrap().starts_with("O("));
        assert!(entry["allocation_profile"].is_string());
        assert!(entry["copy_profile"].is_string());
        assert!(entry["permitted_copy_boundary"].is_string());
        assert!(entry["thresholds"].is_object());
    }
}

#[test]
fn baseline_metadata_tracks_the_checked_in_budget_manifest() {
    let baseline: Value = serde_json::from_str(include_str!(
        "../conformance/v1.0/performance-baseline.json"
    ))
    .unwrap();
    assert_eq!(baseline["schema"], "virtio-accel-performance-baseline-1");
    assert_eq!(baseline["budgets"], "performance-budgets.json");
    assert_eq!(baseline["protocol"], "1.0");
    assert!(baseline["toolchain"]["msrv"].is_string());
    assert_eq!(baseline["host"]["hardware_acceleration"], "none");
    assert!(
        baseline["deterministic_results"]
            .as_array()
            .unwrap()
            .iter()
            .all(
                |result| result["value"].as_u64().unwrap() <= result["threshold"].as_u64().unwrap()
            )
    );
}

#[test]
fn non_submit_decode_does_not_read_or_stage_bulk_artifact_tail() {
    let request = scenario_request("complete_lifecycle", 6);
    let source = CountingSource::new(request);
    let decoded = decoder().decode(&source, 24).unwrap();
    let DecodedRequestBody::LoadProgram { payload, .. } = decoded.body() else {
        panic!("scenario request 6 is not LOAD_PROGRAM");
    };
    assert_eq!(decoded.body().opcode(), KnownOpcode::LoadProgram);
    assert_eq!(payload.len(), 24);
    assert_eq!(
        source.read_bytes(),
        u64::try_from(size_of::<RequestHeader>() + size_of::<LoadProgramRequest>()).unwrap()
    );
}

#[test]
fn submit_rejects_unvalidated_binding_count_before_reading_binding_tail() {
    let mut request = scenario_request("complete_lifecycle", 8);
    let binding_count_offset = size_of::<RequestHeader>() + 16;
    request[binding_count_offset..binding_count_offset + 4]
        .copy_from_slice(&(HARD_MAX_BINDINGS + 1).to_le_bytes());
    let source = CountingSource::new(request);

    assert!(matches!(
        decoder().decode(&source, 24),
        Err(FrameDecodeError::Protocol {
            status: StatusCode::RESOURCE_LIMIT,
            ..
        })
    ));
    assert_eq!(
        source.read_bytes(),
        u64::try_from(size_of::<RequestHeader>() + size_of::<SubmitRequest>()).unwrap()
    );
}
