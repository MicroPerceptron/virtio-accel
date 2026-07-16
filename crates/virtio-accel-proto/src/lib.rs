//! Pointer-free, little-endian wire structures for portable virtio-accel protocol 1.0.
//!
//! The byte contract is frozen for independent implementation. It intentionally does not assign or
//! claim a Virtio device ID or standardize provider artifact contents.

#![no_std]
#![forbid(unsafe_code)]

use bitflags::bitflags;
use zerocopy::byteorder::{LE, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub type Le16 = U16<LE>;
pub type Le32 = U32<LE>;
pub type Le64 = U64<LE>;

/// Frozen portable protocol major version.
pub const PROTOCOL_MAJOR: u16 = 1;
/// Frozen portable protocol minor version.
pub const PROTOCOL_MINOR: u16 = 0;
/// Index of the baseline command virtqueue.
///
/// This transport queue carries protocol requests and responses. It is distinct from an
/// accelerator execution queue created with [`KnownOpcode::CreateQueue`].
pub const COMMAND_QUEUE: u16 = 0;
/// Number of command virtqueues in the baseline device model.
pub const BASELINE_COMMAND_QUEUES: u16 = 1;
/// Protocol-wide upper bound for flattened descriptors in one command chain.
pub const HARD_MAX_CHAIN_DESCRIPTORS: u16 = 256;
/// Protocol-wide upper bound for one complete request frame, including its header.
pub const HARD_MAX_REQUEST_BYTES: u32 = 16 * 1024 * 1024;
/// Protocol-wide upper bound for one complete response frame, including its header.
pub const HARD_MAX_RESPONSE_BYTES: u32 = 16 * 1024 * 1024;
/// Protocol-wide upper bound for bindings carried by one submission.
pub const HARD_MAX_BINDINGS: u32 = 4_096;
/// Smallest request-frame limit that can carry every baseline opcode.
pub const MIN_MAX_REQUEST_BYTES: u32 = 97;
/// Smallest response-frame limit that can carry every baseline response.
pub const MIN_MAX_RESPONSE_BYTES: u32 = 92;

/// Mask of buffer-usage bits assigned by protocol 1.0.
pub const KNOWN_BUFFER_USAGE_BITS: u32 = 0x1f;
/// Request flags accepted by protocol 1.0.
pub const KNOWN_REQUEST_FLAG_BITS: u16 = 0;
/// Context flags accepted by protocol 1.0.
pub const KNOWN_CONTEXT_FLAG_BITS: u32 = 0;
/// Program-load flags accepted by protocol 1.0.
pub const KNOWN_PROGRAM_FLAG_BITS: u32 = 0;
/// Accelerator execution-queue flags accepted by protocol 1.0.
pub const KNOWN_QUEUE_FLAG_BITS: u32 = 0;
/// Submission flags accepted by protocol 1.0.
pub const KNOWN_SUBMIT_FLAG_BITS: u32 = 0;
/// Former draft assignment retained as a reserved-zero request bit.
pub const RESERVED_REQUEST_FLAG_NO_WAIT: u16 = 1 << 0;

bitflags! {
    /// Device-specific feature bits. Virtio transport feature bits are deliberately separate.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FeatureBits: u64 {
        const MULTI_QUEUE = 1 << 0;
        const EVENT_QUEUE = 1 << 1;
        const EXTERNAL_MEMORY = 1 << 2;
        const TIMELINE_FENCES = 1 << 3;
        const SECURE_CONTEXTS = 1 << 4;
    }
}

/// Feature set required by every implementation of portable protocol 1.0.
///
/// The baseline deliberately requires no device-specific feature bits. The commands and
/// object lifecycle described by the specification remain available; feature bits are reserved
/// for behavior that changes transport framing or synchronization.
pub const BASELINE_FEATURES: FeatureBits = FeatureBits::empty();

/// Reserved feature bits that a protocol 1.0 implementation must leave unadvertised.
///
/// Defining their numeric positions preserves the reviewed namespace without assigning protocol
/// semantics. Advertising any of these bits is a protocol error.
pub const RESERVED_FEATURES: FeatureBits = FeatureBits::from_bits_retain(
    FeatureBits::MULTI_QUEUE.bits()
        | FeatureBits::EVENT_QUEUE.bits()
        | FeatureBits::EXTERNAL_MEMORY.bits()
        | FeatureBits::TIMELINE_FENCES.bits()
        | FeatureBits::SECURE_CONTEXTS.bits(),
);

bitflags! {
    /// Per-request flags.
    ///
    /// Protocol 1.0 defines no request flags. Receivers must reject every nonzero raw flag value.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct RequestFlags: u16 {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum KnownOpcode {
    GetDeviceInfo = 0x0001,
    CreateContext = 0x0100,
    DestroyContext = 0x0101,
    AllocateBuffer = 0x0200,
    FreeBuffer = 0x0201,
    WriteBuffer = 0x0202,
    ReadBuffer = 0x0203,
    LoadProgram = 0x0300,
    UnloadProgram = 0x0301,
    CreateQueue = 0x0400,
    DestroyQueue = 0x0401,
    Submit = 0x0500,
    PollEvent = 0x0501,
    CancelEvent = 0x0502,
    DestroyEvent = 0x0503,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownOpcode(pub u16);

impl TryFrom<u16> for KnownOpcode {
    type Error = UnknownOpcode;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0x0001 => Ok(Self::GetDeviceInfo),
            0x0100 => Ok(Self::CreateContext),
            0x0101 => Ok(Self::DestroyContext),
            0x0200 => Ok(Self::AllocateBuffer),
            0x0201 => Ok(Self::FreeBuffer),
            0x0202 => Ok(Self::WriteBuffer),
            0x0203 => Ok(Self::ReadBuffer),
            0x0300 => Ok(Self::LoadProgram),
            0x0301 => Ok(Self::UnloadProgram),
            0x0400 => Ok(Self::CreateQueue),
            0x0401 => Ok(Self::DestroyQueue),
            0x0500 => Ok(Self::Submit),
            0x0501 => Ok(Self::PollEvent),
            0x0502 => Ok(Self::CancelEvent),
            0x0503 => Ok(Self::DestroyEvent),
            _ => Err(UnknownOpcode(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct StatusCode(pub u16);

impl StatusCode {
    pub const OK: Self = Self(0);
    pub const UNSUPPORTED: Self = Self(1);
    pub const INCOMPATIBLE: Self = Self(2);
    pub const INVALID_ARGUMENT: Self = Self(3);
    pub const OUT_OF_BOUNDS: Self = Self(4);
    pub const BUSY: Self = Self(5);
    pub const OUT_OF_MEMORY: Self = Self(6);
    pub const RESOURCE_LIMIT: Self = Self(7);
    pub const DEADLINE_EXPIRED: Self = Self(8);
    pub const DEVICE_LOST: Self = Self(9);
    pub const PERMISSION_DENIED: Self = Self(10);
    pub const STALE_OBJECT: Self = Self(11);
    pub const INTERNAL_ERROR: Self = Self(0xffff);

    /// Returns whether this value has assigned protocol 1.0 semantics.
    pub const fn is_known(self) -> bool {
        matches!(self.0, 0..=11 | 0xffff)
    }

    pub const fn is_success(self) -> bool {
        self.0 == Self::OK.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum KnownEventState {
    Pending = 0,
    Complete = 1,
    Failed = 2,
    Cancelled = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownEventState(pub u16);

impl TryFrom<u16> for KnownEventState {
    type Error = UnknownEventState;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Pending),
            1 => Ok(Self::Complete),
            2 => Ok(Self::Failed),
            3 => Ok(Self::Cancelled),
            _ => Err(UnknownEventState(value)),
        }
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct WireConfig {
    pub protocol_major: Le16,
    pub protocol_minor: Le16,
    pub command_queue_count: Le16,
    pub max_chain_descriptors: Le16,
    pub max_request_bytes: Le32,
    pub max_response_bytes: Le32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigError {
    Version,
    CommandQueueCount,
    ChainDescriptorLimit,
    RequestByteLimit,
    ResponseByteLimit,
}

impl WireConfig {
    /// Validates that this configuration can provide the protocol 1.0 baseline.
    ///
    /// A higher minor version is accepted and used with 1.0 behavior until separately negotiated
    /// extensions are understood.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.protocol_major.get() != PROTOCOL_MAJOR {
            return Err(ConfigError::Version);
        }
        if self.command_queue_count.get() != BASELINE_COMMAND_QUEUES {
            return Err(ConfigError::CommandQueueCount);
        }
        if !(2..=HARD_MAX_CHAIN_DESCRIPTORS).contains(&self.max_chain_descriptors.get()) {
            return Err(ConfigError::ChainDescriptorLimit);
        }
        if !(MIN_MAX_REQUEST_BYTES..=HARD_MAX_REQUEST_BYTES).contains(&self.max_request_bytes.get())
        {
            return Err(ConfigError::RequestByteLimit);
        }
        if !(MIN_MAX_RESPONSE_BYTES..=HARD_MAX_RESPONSE_BYTES)
            .contains(&self.max_response_bytes.get())
        {
            return Err(ConfigError::ResponseByteLimit);
        }
        Ok(())
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct RequestHeader {
    pub opcode: Le16,
    pub flags: Le16,
    pub payload_bytes: Le32,
    pub request_id: Le64,
}

impl RequestHeader {
    pub fn new(
        opcode: KnownOpcode,
        flags: RequestFlags,
        payload_bytes: u32,
        request_id: u64,
    ) -> Self {
        Self {
            opcode: Le16::new(opcode as u16),
            flags: Le16::new(flags.bits()),
            payload_bytes: Le32::new(payload_bytes),
            request_id: Le64::new(request_id),
        }
    }

    pub fn known_opcode(&self) -> Result<KnownOpcode, UnknownOpcode> {
        self.opcode.get().try_into()
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ResponseHeader {
    pub status: Le16,
    pub flags: Le16,
    pub payload_bytes: Le32,
    pub request_id: Le64,
}

impl ResponseHeader {
    pub fn new(status: StatusCode, payload_bytes: u32, request_id: u64) -> Self {
        Self {
            status: Le16::new(status.0),
            flags: Le16::new(0),
            payload_bytes: Le32::new(payload_bytes),
            request_id: Le64::new(request_id),
        }
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct WireDeviceInfo {
    pub uuid: [u8; 16],
    pub class: Le16,
    pub reserved: Le16,
    pub vendor_id: Le32,
    pub device_id: Le32,
    pub capabilities: Le64,
    pub max_contexts: Le32,
    pub max_buffers_per_context: Le32,
    pub max_programs_per_context: Le32,
    pub max_queues_per_context: Le32,
    pub max_events_per_context: Le32,
    pub max_bindings_per_submission: Le32,
    pub max_buffer_bytes: Le64,
    pub max_artifact_bytes: Le64,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct CreateContextRequest {
    pub flags: Le32,
    pub reserved: Le32,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ObjectPayload {
    pub object_id: Le64,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct AllocateBufferRequest {
    pub context_id: Le64,
    pub bytes: Le64,
    pub alignment: Le64,
    pub memory_domain: u8,
    pub reserved0: [u8; 7],
    pub usage: Le32,
    pub reserved1: Le32,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct TransferBufferRequest {
    pub buffer_id: Le64,
    pub offset: Le64,
    pub bytes: Le64,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct LoadProgramRequest {
    pub context_id: Le64,
    pub format: Le32,
    pub flags: Le32,
    pub target: [Le32; 12],
    pub payload_bytes: Le64,
    pub resident_bytes: Le64,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct CreateQueueRequest {
    pub context_id: Le64,
    pub flags: Le32,
    pub reserved: Le32,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct SubmitRequest {
    pub queue_id: Le64,
    pub program_id: Le64,
    pub binding_count: Le32,
    pub flags: Le32,
    /// Relative timeout from device admission. Zero means infinite.
    pub timeout_ns: Le64,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct WireBinding {
    pub buffer_id: Le64,
    pub offset: Le64,
    pub bytes: Le64,
    pub slot: Le32,
    pub access: u8,
    pub reserved: [u8; 3],
}

/// Event identifier returned for an accepted or indeterminate submission.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct SubmitResponse {
    pub event_id: Le64,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
#[repr(C)]
pub struct WireEventState {
    pub state: Le16,
    pub error: Le16,
    pub reserved: Le32,
}

impl WireEventState {
    pub fn known_state(&self) -> Result<KnownEventState, UnknownEventState> {
        self.state.get().try_into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Size,
    CountOverflow,
}

pub fn read_exact<T: FromBytes>(bytes: &[u8]) -> Result<T, DecodeError> {
    T::read_from_bytes(bytes).map_err(|_| DecodeError::Size)
}

pub fn checked_array_bytes<T>(count: u32) -> Result<usize, DecodeError> {
    core::mem::size_of::<T>()
        .checked_mul(count as usize)
        .ok_or(DecodeError::CountOverflow)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use core::mem::{align_of, offset_of, size_of};
    use serde_json::Value;
    use std::collections::BTreeSet;
    use zerocopy::IntoBytes;

    const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

    fn corpus() -> Value {
        serde_json::from_str(include_str!("../../../conformance/v1.0/vectors.json")).unwrap()
    }

    fn layout_manifest() -> Value {
        serde_json::from_str(include_str!("../../../conformance/v1.0/layout.json")).unwrap()
    }

    fn decode_hex(hex: &str) -> std::vec::Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap())
            .collect()
    }

    fn vector(group: &str, name: &str) -> std::vec::Vec<u8> {
        let corpus = corpus();
        let entry = corpus[group]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"] == name)
            .unwrap_or_else(|| panic!("missing vector {name}"));
        decode_hex(entry["hex"].as_str().unwrap())
    }

    fn assert_layout<T>(name: &str, fields: &[(&str, usize)]) {
        let manifest = layout_manifest();
        let expected = &manifest["structures"][name];
        assert_eq!(size_of::<T>() as u64, expected["size"].as_u64().unwrap());
        assert_eq!(align_of::<T>() as u64, expected["align"].as_u64().unwrap());
        for (field, offset) in fields {
            assert_eq!(
                *offset as u64,
                expected["fields"][field].as_u64().unwrap(),
                "{name}.{field}"
            );
        }
    }

    #[test]
    fn headers_are_fixed_little_endian_frames() {
        let header = RequestHeader::new(KnownOpcode::Submit, RequestFlags::empty(), 32, 7);
        assert_eq!(size_of::<RequestHeader>(), 16);
        assert_eq!(header.as_bytes()[..2], 0x0500_u16.to_le_bytes());
        let decoded = read_exact::<RequestHeader>(header.as_bytes()).unwrap();
        assert_eq!(decoded.known_opcode(), Ok(KnownOpcode::Submit));
        assert_eq!(decoded.request_id.get(), 7);
    }

    #[test]
    fn wire_layouts_match_the_frozen_manifest() {
        assert_layout::<WireConfig>(
            "WireConfig",
            &[
                ("protocol_major", offset_of!(WireConfig, protocol_major)),
                ("protocol_minor", offset_of!(WireConfig, protocol_minor)),
                (
                    "command_queue_count",
                    offset_of!(WireConfig, command_queue_count),
                ),
                (
                    "max_chain_descriptors",
                    offset_of!(WireConfig, max_chain_descriptors),
                ),
                (
                    "max_request_bytes",
                    offset_of!(WireConfig, max_request_bytes),
                ),
                (
                    "max_response_bytes",
                    offset_of!(WireConfig, max_response_bytes),
                ),
            ],
        );
        assert_layout::<RequestHeader>(
            "RequestHeader",
            &[
                ("opcode", offset_of!(RequestHeader, opcode)),
                ("flags", offset_of!(RequestHeader, flags)),
                ("payload_bytes", offset_of!(RequestHeader, payload_bytes)),
                ("request_id", offset_of!(RequestHeader, request_id)),
            ],
        );
        assert_layout::<ResponseHeader>(
            "ResponseHeader",
            &[
                ("status", offset_of!(ResponseHeader, status)),
                ("flags", offset_of!(ResponseHeader, flags)),
                ("payload_bytes", offset_of!(ResponseHeader, payload_bytes)),
                ("request_id", offset_of!(ResponseHeader, request_id)),
            ],
        );
        assert_layout::<WireDeviceInfo>(
            "WireDeviceInfo",
            &[
                ("uuid", offset_of!(WireDeviceInfo, uuid)),
                ("class", offset_of!(WireDeviceInfo, class)),
                ("reserved", offset_of!(WireDeviceInfo, reserved)),
                ("vendor_id", offset_of!(WireDeviceInfo, vendor_id)),
                ("device_id", offset_of!(WireDeviceInfo, device_id)),
                ("capabilities", offset_of!(WireDeviceInfo, capabilities)),
                ("max_contexts", offset_of!(WireDeviceInfo, max_contexts)),
                (
                    "max_buffers_per_context",
                    offset_of!(WireDeviceInfo, max_buffers_per_context),
                ),
                (
                    "max_programs_per_context",
                    offset_of!(WireDeviceInfo, max_programs_per_context),
                ),
                (
                    "max_queues_per_context",
                    offset_of!(WireDeviceInfo, max_queues_per_context),
                ),
                (
                    "max_events_per_context",
                    offset_of!(WireDeviceInfo, max_events_per_context),
                ),
                (
                    "max_bindings_per_submission",
                    offset_of!(WireDeviceInfo, max_bindings_per_submission),
                ),
                (
                    "max_buffer_bytes",
                    offset_of!(WireDeviceInfo, max_buffer_bytes),
                ),
                (
                    "max_artifact_bytes",
                    offset_of!(WireDeviceInfo, max_artifact_bytes),
                ),
            ],
        );
        assert_layout::<CreateContextRequest>(
            "CreateContextRequest",
            &[
                ("flags", offset_of!(CreateContextRequest, flags)),
                ("reserved", offset_of!(CreateContextRequest, reserved)),
            ],
        );
        assert_layout::<ObjectPayload>(
            "ObjectPayload",
            &[("object_id", offset_of!(ObjectPayload, object_id))],
        );
        assert_layout::<AllocateBufferRequest>(
            "AllocateBufferRequest",
            &[
                ("context_id", offset_of!(AllocateBufferRequest, context_id)),
                ("bytes", offset_of!(AllocateBufferRequest, bytes)),
                ("alignment", offset_of!(AllocateBufferRequest, alignment)),
                (
                    "memory_domain",
                    offset_of!(AllocateBufferRequest, memory_domain),
                ),
                ("reserved0", offset_of!(AllocateBufferRequest, reserved0)),
                ("usage", offset_of!(AllocateBufferRequest, usage)),
                ("reserved1", offset_of!(AllocateBufferRequest, reserved1)),
            ],
        );
        assert_layout::<TransferBufferRequest>(
            "TransferBufferRequest",
            &[
                ("buffer_id", offset_of!(TransferBufferRequest, buffer_id)),
                ("offset", offset_of!(TransferBufferRequest, offset)),
                ("bytes", offset_of!(TransferBufferRequest, bytes)),
            ],
        );
        assert_layout::<LoadProgramRequest>(
            "LoadProgramRequest",
            &[
                ("context_id", offset_of!(LoadProgramRequest, context_id)),
                ("format", offset_of!(LoadProgramRequest, format)),
                ("flags", offset_of!(LoadProgramRequest, flags)),
                ("target", offset_of!(LoadProgramRequest, target)),
                (
                    "payload_bytes",
                    offset_of!(LoadProgramRequest, payload_bytes),
                ),
                (
                    "resident_bytes",
                    offset_of!(LoadProgramRequest, resident_bytes),
                ),
            ],
        );
        assert_layout::<CreateQueueRequest>(
            "CreateQueueRequest",
            &[
                ("context_id", offset_of!(CreateQueueRequest, context_id)),
                ("flags", offset_of!(CreateQueueRequest, flags)),
                ("reserved", offset_of!(CreateQueueRequest, reserved)),
            ],
        );
        assert_layout::<SubmitRequest>(
            "SubmitRequest",
            &[
                ("queue_id", offset_of!(SubmitRequest, queue_id)),
                ("program_id", offset_of!(SubmitRequest, program_id)),
                ("binding_count", offset_of!(SubmitRequest, binding_count)),
                ("flags", offset_of!(SubmitRequest, flags)),
                ("timeout_ns", offset_of!(SubmitRequest, timeout_ns)),
            ],
        );
        assert_layout::<WireBinding>(
            "WireBinding",
            &[
                ("buffer_id", offset_of!(WireBinding, buffer_id)),
                ("offset", offset_of!(WireBinding, offset)),
                ("bytes", offset_of!(WireBinding, bytes)),
                ("slot", offset_of!(WireBinding, slot)),
                ("access", offset_of!(WireBinding, access)),
                ("reserved", offset_of!(WireBinding, reserved)),
            ],
        );
        assert_layout::<SubmitResponse>(
            "SubmitResponse",
            &[("event_id", offset_of!(SubmitResponse, event_id))],
        );
        assert_layout::<WireEventState>(
            "WireEventState",
            &[
                ("state", offset_of!(WireEventState, state)),
                ("error", offset_of!(WireEventState, error)),
                ("reserved", offset_of!(WireEventState, reserved)),
            ],
        );
    }

    #[test]
    fn manifest_constants_match_the_frozen_rust_namespace() {
        let manifest = layout_manifest();
        assert_eq!(manifest["protocol"]["major"], PROTOCOL_MAJOR);
        assert_eq!(manifest["protocol"]["minor"], PROTOCOL_MINOR);
        assert_eq!(manifest["queue"]["command_index"], COMMAND_QUEUE);
        assert_eq!(manifest["queue"]["baseline_count"], BASELINE_COMMAND_QUEUES);
        assert_eq!(
            manifest["queue"]["hard_max_chain_descriptors"],
            HARD_MAX_CHAIN_DESCRIPTORS
        );
        assert_eq!(
            manifest["queue"]["min_max_request_bytes"],
            MIN_MAX_REQUEST_BYTES
        );
        assert_eq!(
            manifest["queue"]["min_max_response_bytes"],
            MIN_MAX_RESPONSE_BYTES
        );
        assert_eq!(
            manifest["queue"]["hard_max_request_bytes"],
            HARD_MAX_REQUEST_BYTES
        );
        assert_eq!(
            manifest["queue"]["hard_max_response_bytes"],
            HARD_MAX_RESPONSE_BYTES
        );
        assert_eq!(manifest["queue"]["hard_max_bindings"], HARD_MAX_BINDINGS);
        assert_eq!(
            MIN_MAX_REQUEST_BYTES as usize,
            size_of::<RequestHeader>() + size_of::<LoadProgramRequest>() + 1
        );
        assert_eq!(
            MIN_MAX_RESPONSE_BYTES as usize,
            size_of::<ResponseHeader>() + size_of::<WireDeviceInfo>()
        );

        let opcodes = &manifest["opcodes"];
        for (name, value) in [
            ("GET_DEVICE_INFO", KnownOpcode::GetDeviceInfo as u16),
            ("CREATE_CONTEXT", KnownOpcode::CreateContext as u16),
            ("DESTROY_CONTEXT", KnownOpcode::DestroyContext as u16),
            ("ALLOCATE_BUFFER", KnownOpcode::AllocateBuffer as u16),
            ("FREE_BUFFER", KnownOpcode::FreeBuffer as u16),
            ("WRITE_BUFFER", KnownOpcode::WriteBuffer as u16),
            ("READ_BUFFER", KnownOpcode::ReadBuffer as u16),
            ("LOAD_PROGRAM", KnownOpcode::LoadProgram as u16),
            ("UNLOAD_PROGRAM", KnownOpcode::UnloadProgram as u16),
            ("CREATE_QUEUE", KnownOpcode::CreateQueue as u16),
            ("DESTROY_QUEUE", KnownOpcode::DestroyQueue as u16),
            ("SUBMIT", KnownOpcode::Submit as u16),
            ("POLL_EVENT", KnownOpcode::PollEvent as u16),
            ("CANCEL_EVENT", KnownOpcode::CancelEvent as u16),
            ("DESTROY_EVENT", KnownOpcode::DestroyEvent as u16),
        ] {
            assert_eq!(opcodes[name], std::format!("0x{value:04x}"));
        }

        let statuses = &manifest["statuses"];
        for (name, value) in [
            ("OK", StatusCode::OK.0),
            ("UNSUPPORTED", StatusCode::UNSUPPORTED.0),
            ("INCOMPATIBLE", StatusCode::INCOMPATIBLE.0),
            ("INVALID_ARGUMENT", StatusCode::INVALID_ARGUMENT.0),
            ("OUT_OF_BOUNDS", StatusCode::OUT_OF_BOUNDS.0),
            ("BUSY", StatusCode::BUSY.0),
            ("OUT_OF_MEMORY", StatusCode::OUT_OF_MEMORY.0),
            ("RESOURCE_LIMIT", StatusCode::RESOURCE_LIMIT.0),
            ("DEADLINE_EXPIRED", StatusCode::DEADLINE_EXPIRED.0),
            ("DEVICE_LOST", StatusCode::DEVICE_LOST.0),
            ("PERMISSION_DENIED", StatusCode::PERMISSION_DENIED.0),
            ("STALE_OBJECT", StatusCode::STALE_OBJECT.0),
            ("INTERNAL_ERROR", StatusCode::INTERNAL_ERROR.0),
        ] {
            assert_eq!(statuses[name], value);
        }

        let states = &manifest["event_states"];
        for (name, value) in [
            ("PENDING", KnownEventState::Pending as u16),
            ("COMPLETE", KnownEventState::Complete as u16),
            ("FAILED", KnownEventState::Failed as u16),
            ("CANCELLED", KnownEventState::Cancelled as u16),
        ] {
            assert_eq!(states[name], value);
        }

        let features = &manifest["features"];
        assert_eq!(features["baseline"], "0x0000000000000000");
        for (name, value) in [
            ("MULTI_QUEUE", FeatureBits::MULTI_QUEUE.bits()),
            ("EVENT_QUEUE", FeatureBits::EVENT_QUEUE.bits()),
            ("EXTERNAL_MEMORY", FeatureBits::EXTERNAL_MEMORY.bits()),
            ("TIMELINE_FENCES", FeatureBits::TIMELINE_FENCES.bits()),
            ("SECURE_CONTEXTS", FeatureBits::SECURE_CONTEXTS.bits()),
        ] {
            assert_eq!(features["reserved"][name], std::format!("0x{value:016x}"));
        }
    }

    #[test]
    fn canonical_frames_have_exact_header_lengths_and_known_namespaces() {
        let corpus = corpus();
        let frames = corpus["frames"].as_array().unwrap();
        let mut request_count = 0;
        let mut response_count = 0;
        let mut request_opcodes = BTreeSet::new();
        let mut response_names = BTreeSet::new();

        for frame in frames {
            let bytes = decode_hex(frame["hex"].as_str().unwrap());
            match frame["kind"].as_str().unwrap() {
                "config" => {
                    let config = read_exact::<WireConfig>(&bytes).unwrap();
                    config.validate().unwrap();
                }
                "request" => {
                    request_count += 1;
                    let header =
                        read_exact::<RequestHeader>(&bytes[..size_of::<RequestHeader>()]).unwrap();
                    assert!(header.known_opcode().is_ok(), "{}", frame["name"]);
                    let opcode_name = frame["opcode"].as_str().unwrap();
                    assert_eq!(
                        layout_manifest()["opcodes"][opcode_name],
                        std::format!("0x{:04x}", header.opcode.get())
                    );
                    assert!(request_opcodes.insert(header.opcode.get()));
                    assert_eq!(header.flags.get(), KNOWN_REQUEST_FLAG_BITS);
                    assert_ne!(header.request_id.get(), 0);
                    assert_eq!(
                        bytes.len(),
                        size_of::<RequestHeader>() + header.payload_bytes.get() as usize,
                        "{}",
                        frame["name"]
                    );
                }
                "response" => {
                    response_count += 1;
                    assert!(response_names.insert(frame["name"].as_str().unwrap()));
                    let header =
                        read_exact::<ResponseHeader>(&bytes[..size_of::<ResponseHeader>()])
                            .unwrap();
                    assert!(
                        StatusCode(header.status.get()).is_known(),
                        "{}",
                        frame["name"]
                    );
                    let status_name = frame["status"].as_str().unwrap();
                    assert_eq!(
                        layout_manifest()["statuses"][status_name],
                        header.status.get()
                    );
                    assert_eq!(header.flags.get(), 0);
                    assert_eq!(header.request_id.get(), REQUEST_ID);
                    assert_eq!(
                        bytes.len(),
                        size_of::<ResponseHeader>() + header.payload_bytes.get() as usize,
                        "{}",
                        frame["name"]
                    );
                }
                other => panic!("unknown frame kind {other}"),
            }
        }

        assert_eq!(request_count, 15);
        assert_eq!(request_opcodes.len(), 15);
        assert_eq!(response_count, 20);
        for name in [
            "response_get_device_info",
            "response_create_context",
            "response_destroy_context",
            "response_allocate_buffer",
            "response_free_buffer",
            "response_write_buffer",
            "response_read_buffer",
            "response_load_program",
            "response_unload_program",
            "response_create_queue",
            "response_destroy_queue",
            "response_submit_accepted",
            "response_submit_indeterminate",
            "response_poll_event_pending",
            "response_poll_event_complete",
            "response_poll_event_failed",
            "response_poll_event_cancelled",
            "response_cancel_event",
            "response_destroy_event",
            "response_unknown_opcode",
        ] {
            assert!(response_names.contains(name), "missing {name}");
        }
    }

    #[test]
    fn canonical_struct_encodings_match_reviewed_vectors() {
        let minimum_config = WireConfig {
            protocol_major: Le16::new(PROTOCOL_MAJOR),
            protocol_minor: Le16::new(PROTOCOL_MINOR),
            command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
            max_chain_descriptors: Le16::new(2),
            max_request_bytes: Le32::new(MIN_MAX_REQUEST_BYTES),
            max_response_bytes: Le32::new(MIN_MAX_RESPONSE_BYTES),
        };
        assert_eq!(
            minimum_config.as_bytes(),
            vector("frames", "config_minimum")
        );

        let get_info = RequestHeader::new(
            KnownOpcode::GetDeviceInfo,
            RequestFlags::empty(),
            0,
            REQUEST_ID,
        );
        assert_eq!(
            get_info.as_bytes(),
            vector("frames", "request_get_device_info")
        );

        let create_context = CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        };
        assert_eq!(
            create_context.as_bytes(),
            &vector("frames", "request_create_context")[16..]
        );

        let context = ObjectPayload {
            object_id: Le64::new(0x1112_1314_1516_1718),
        };
        assert_eq!(
            context.as_bytes(),
            &vector("frames", "request_destroy_context")[16..]
        );

        let allocate = AllocateBufferRequest {
            context_id: context.object_id,
            bytes: Le64::new(4_096),
            alignment: Le64::new(64),
            memory_domain: 3,
            reserved0: [0; 7],
            usage: Le32::new(0x0c),
            reserved1: Le32::new(0),
        };
        assert_eq!(
            allocate.as_bytes(),
            &vector("frames", "request_allocate_buffer")[16..]
        );

        let transfer = TransferBufferRequest {
            buffer_id: Le64::new(0x2122_2324_2526_2728),
            offset: Le64::new(4),
            bytes: Le64::new(4),
        };
        assert_eq!(
            transfer.as_bytes(),
            &vector("frames", "request_read_buffer")[16..]
        );

        let load_program = LoadProgramRequest {
            context_id: context.object_id,
            format: Le32::new(1),
            flags: Le32::new(0),
            target: core::array::from_fn(|index| Le32::new(index as u32)),
            payload_bytes: Le64::new(4),
            resident_bytes: Le64::new(4_096),
        };
        assert_eq!(
            load_program.as_bytes(),
            &vector("frames", "request_load_program")[16..96]
        );

        let create_queue = CreateQueueRequest {
            context_id: context.object_id,
            flags: Le32::new(0),
            reserved: Le32::new(0),
        };
        assert_eq!(
            create_queue.as_bytes(),
            &vector("frames", "request_create_queue")[16..]
        );

        let submit = SubmitRequest {
            queue_id: Le64::new(0x4142_4344_4546_4748),
            program_id: Le64::new(0x3132_3334_3536_3738),
            binding_count: Le32::new(1),
            flags: Le32::new(0),
            timeout_ns: Le64::new(1_000_000),
        };
        assert_eq!(
            submit.as_bytes(),
            &vector("frames", "request_submit")[16..48]
        );

        let binding = WireBinding {
            buffer_id: transfer.buffer_id,
            offset: Le64::new(0),
            bytes: Le64::new(4_096),
            slot: Le32::new(7),
            access: 3,
            reserved: [0; 3],
        };
        assert_eq!(
            binding.as_bytes(),
            &vector("frames", "request_submit")[48..]
        );

        let device_info = WireDeviceInfo {
            uuid: *b"virtio-accelmock",
            class: Le16::new(1),
            reserved: Le16::new(0),
            vendor_id: Le32::new(0x1234_5678),
            device_id: Le32::new(0x9abc_def0),
            capabilities: Le64::new(7),
            max_contexts: Le32::new(64),
            max_buffers_per_context: Le32::new(1_024),
            max_programs_per_context: Le32::new(256),
            max_queues_per_context: Le32::new(16),
            max_events_per_context: Le32::new(4_096),
            max_bindings_per_submission: Le32::new(256),
            max_buffer_bytes: Le64::new(1 << 30),
            max_artifact_bytes: Le64::new(16 * 1024 * 1024),
        };
        assert_eq!(
            device_info.as_bytes(),
            &vector("frames", "response_get_device_info")[16..]
        );

        let submit_response = SubmitResponse {
            event_id: Le64::new(0x5152_5354_5556_5758),
        };
        assert_eq!(
            submit_response.as_bytes(),
            &vector("frames", "response_submit_accepted")[16..]
        );

        let failed_event = WireEventState {
            state: Le16::new(KnownEventState::Failed as u16),
            error: Le16::new(StatusCode::DEVICE_LOST.0),
            reserved: Le32::new(0),
        };
        assert_eq!(
            failed_event.as_bytes(),
            &vector("frames", "response_poll_event_failed")[16..]
        );
    }

    #[test]
    fn edge_vectors_preserve_unknown_values_without_invalid_enums() {
        let truncated = vector("edge_cases", "request_header_truncated");
        assert_eq!(
            read_exact::<RequestHeader>(&truncated),
            Err(DecodeError::Size)
        );

        let unknown_opcode = vector("edge_cases", "request_unknown_opcode");
        let decoded = read_exact::<RequestHeader>(&unknown_opcode).unwrap();
        assert_eq!(decoded.known_opcode(), Err(UnknownOpcode(0xdead)));

        let unknown_status = vector("edge_cases", "response_unknown_status");
        let response = read_exact::<ResponseHeader>(&unknown_status).unwrap();
        let status = StatusCode(response.status.get());
        assert_eq!(status, StatusCode(0x1234));
        assert!(!status.is_known());
        assert!(!status.is_success());

        let unknown_event = vector("edge_cases", "response_unknown_event_state");
        let event = read_exact::<WireEventState>(&unknown_event[16..]).unwrap();
        assert_eq!(event.known_state(), Err(UnknownEventState(0xffff)));
    }

    #[test]
    fn edge_vectors_cover_limits_reserved_bits_and_exact_lengths() {
        let invalid_config = vector("edge_cases", "config_descriptor_limit_above_hard_max");
        assert_eq!(
            read_exact::<WireConfig>(&invalid_config)
                .unwrap()
                .validate(),
            Err(ConfigError::ChainDescriptorLimit)
        );

        let future_minor = vector("edge_cases", "config_future_minor_compatible");
        let config = read_exact::<WireConfig>(&future_minor).unwrap();
        assert!(config.protocol_minor.get() > PROTOCOL_MINOR);
        assert_eq!(config.validate(), Ok(()));

        let unknown_major = vector("edge_cases", "config_unknown_major");
        assert_eq!(
            read_exact::<WireConfig>(&unknown_major).unwrap().validate(),
            Err(ConfigError::Version)
        );

        let reserved_flag = vector("edge_cases", "request_reserved_flag");
        let header = read_exact::<RequestHeader>(&reserved_flag).unwrap();
        assert_eq!(header.flags.get(), RESERVED_REQUEST_FLAG_NO_WAIT);
        assert!(RequestFlags::from_bits(header.flags.get()).is_none());

        let trailing = vector("edge_cases", "request_trailing_byte");
        let header = read_exact::<RequestHeader>(&trailing[..16]).unwrap();
        assert_ne!(
            trailing.len(),
            size_of::<RequestHeader>() + header.payload_bytes.get() as usize
        );

        let excessive_bindings = vector("edge_cases", "request_binding_count_overflow");
        let submit = read_exact::<SubmitRequest>(&excessive_bindings[16..]).unwrap();
        assert!(submit.binding_count.get() > HARD_MAX_BINDINGS);
    }

    #[test]
    fn reserved_features_are_not_baseline_requirements() {
        assert!(BASELINE_FEATURES.is_empty());
        assert!(!RESERVED_FEATURES.is_empty());
        assert!(!BASELINE_FEATURES.intersects(RESERVED_FEATURES));
    }
}
