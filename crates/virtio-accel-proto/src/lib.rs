//! Pointer-free, little-endian wire structures for the experimental virtio-accel protocol.
//!
//! This is a draft ABI. It intentionally does not assign or claim a virtio device ID.

#![no_std]
#![forbid(unsafe_code)]

use bitflags::bitflags;
use zerocopy::byteorder::{LE, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub type Le16 = U16<LE>;
pub type Le32 = U32<LE>;
pub type Le64 = U64<LE>;

pub const PROTOCOL_MAJOR: u16 = 0;
pub const PROTOCOL_MINOR: u16 = 1;
pub const COMMAND_QUEUE: u16 = 0;

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

bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct RequestFlags: u16 {
        const NO_WAIT = 1 << 0;
    }
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
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct WireConfig {
    pub protocol_major: Le16,
    pub protocol_minor: Le16,
    pub max_command_queues: Le16,
    pub reserved: Le16,
    pub max_request_bytes: Le32,
    pub max_response_bytes: Le32,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
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

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
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

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct WireDeviceInfo {
    pub uuid: [u8; 16],
    pub class: Le16,
    pub reserved: Le16,
    pub vendor_id: Le32,
    pub device_id: Le32,
    pub capabilities: Le64,
    pub max_contexts: Le32,
    pub max_queues_per_context: Le32,
    pub max_buffers_per_context: Le32,
    pub max_bindings_per_submission: Le32,
    pub max_buffer_bytes: Le64,
    pub max_artifact_bytes: Le64,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct CreateContextRequest {
    pub flags: Le32,
    pub reserved: Le32,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct ObjectPayload {
    pub object_id: Le64,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
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

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct TransferBufferRequest {
    pub buffer_id: Le64,
    pub offset: Le64,
    pub bytes: Le64,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct LoadProgramRequest {
    pub context_id: Le64,
    pub format: Le32,
    pub flags: Le32,
    pub target: [Le32; 12],
    pub payload_bytes: Le64,
    pub resident_bytes: Le64,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct CreateQueueRequest {
    pub context_id: Le64,
    pub flags: Le32,
    pub reserved: Le32,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct SubmitRequest {
    pub queue_id: Le64,
    pub program_id: Le64,
    pub binding_count: Le32,
    pub flags: Le32,
    /// Relative timeout from device admission. Zero means infinite.
    pub timeout_ns: Le64,
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct WireBinding {
    pub buffer_id: Le64,
    pub offset: Le64,
    pub bytes: Le64,
    pub slot: Le32,
    pub access: u8,
    pub reserved: [u8; 3],
}

#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
#[repr(C)]
pub struct WireEventState {
    pub state: Le16,
    pub error: Le16,
    pub reserved: Le32,
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
    use super::*;
    use zerocopy::IntoBytes;

    #[test]
    fn headers_are_fixed_little_endian_frames() {
        let header = RequestHeader::new(KnownOpcode::Submit, RequestFlags::NO_WAIT, 32, 7);
        assert_eq!(core::mem::size_of::<RequestHeader>(), 16);
        assert_eq!(header.as_bytes()[..2], 0x0500_u16.to_le_bytes());
        let decoded = read_exact::<RequestHeader>(header.as_bytes()).unwrap();
        assert_eq!(decoded.known_opcode(), Ok(KnownOpcode::Submit));
        assert_eq!(decoded.request_id.get(), 7);
    }

    #[test]
    fn wire_layouts_contain_no_native_pointers_or_padding() {
        assert_eq!(core::mem::size_of::<WireConfig>(), 16);
        assert_eq!(core::mem::size_of::<ResponseHeader>(), 16);
        assert_eq!(core::mem::size_of::<WireDeviceInfo>(), 68);
        assert_eq!(core::mem::size_of::<AllocateBufferRequest>(), 40);
        assert_eq!(core::mem::size_of::<LoadProgramRequest>(), 80);
        assert_eq!(core::mem::size_of::<SubmitRequest>(), 32);
        assert_eq!(core::mem::size_of::<WireBinding>(), 32);
    }

    #[test]
    fn unknown_opcodes_remain_rejectable_without_invalid_enums() {
        let mut bytes = RequestHeader::new(KnownOpcode::GetDeviceInfo, RequestFlags::empty(), 0, 1)
            .as_bytes()
            .to_vec();
        bytes[..2].copy_from_slice(&0xdead_u16.to_le_bytes());
        let decoded = read_exact::<RequestHeader>(&bytes).unwrap();
        assert_eq!(decoded.known_opcode(), Err(UnknownOpcode(0xdead)));
    }
}
