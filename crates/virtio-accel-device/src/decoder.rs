//! Incremental validation for one untrusted command frame.
//!
//! Fixed wire values use an 80-byte stack scratch area. Transfer and artifact tails remain
//! borrowed views over the original [`ByteSource`]. `SUBMIT` is the only allocating decode path:
//! it retains one [`DecodedBinding`] per advertised binding and temporarily uses two `u32` arrays
//! per binding for four-pass radix duplicate detection. All three allocations are bounded by the
//! configured binding limit and use fallible reservation. The decoder performs a constant number
//! of fixed-prefix reads plus one 32-byte read and four fixed radix passes per binding, so its own
//! work is linear in the validated payload size.

use alloc::vec::Vec;
use core::mem::size_of;

use virtio_accel_core::{
    AccessMode, ArtifactFormat, BufferDesc, BufferRange, BufferUsage, ByteSource, Capabilities,
    ContextDesc, DeviceInfo, MemoryDomain, QueueDesc, TargetIdentity, Timeout,
};
use virtio_accel_proto::{
    AllocateBufferRequest, ConfigError, CreateContextRequest, CreateQueueRequest,
    HARD_MAX_BINDINGS, KnownOpcode, LoadProgramRequest, ObjectPayload, RequestHeader, StatusCode,
    SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding, WireConfig, WireDeviceInfo,
    WireEventState, read_exact,
};
use zerocopy::FromBytes;

use crate::{ObjectId, ReadableRegion};

const REQUEST_HEADER_BYTES: u64 = size_of::<RequestHeader>() as u64;
const RESPONSE_HEADER_BYTES: u64 = 16;
const MAX_FIXED_PREFIX_BYTES: usize = size_of::<LoadProgramRequest>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderLimitsError {
    Config(ConfigError),
    BindingLimit,
    BufferLimit,
    ArtifactLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderLimits {
    max_chain_descriptors: u16,
    max_request_bytes: u32,
    max_response_bytes: u32,
    max_bindings: u32,
    max_buffer_bytes: u64,
    max_artifact_bytes: u64,
    capabilities: Capabilities,
}

impl DecoderLimits {
    pub fn new(config: &WireConfig, info: DeviceInfo) -> Result<Self, DecoderLimitsError> {
        config.validate().map_err(DecoderLimitsError::Config)?;
        if !(1..=HARD_MAX_BINDINGS).contains(&info.limits.max_bindings_per_submission) {
            return Err(DecoderLimitsError::BindingLimit);
        }
        if info.limits.max_buffer_bytes == 0 {
            return Err(DecoderLimitsError::BufferLimit);
        }
        if info.limits.max_artifact_bytes == 0 {
            return Err(DecoderLimitsError::ArtifactLimit);
        }
        Ok(Self {
            max_chain_descriptors: config.max_chain_descriptors.get(),
            max_request_bytes: config.max_request_bytes.get(),
            max_response_bytes: config.max_response_bytes.get(),
            max_bindings: info.limits.max_bindings_per_submission,
            max_buffer_bytes: info.limits.max_buffer_bytes,
            max_artifact_bytes: info.limits.max_artifact_bytes,
            capabilities: info.capabilities,
        })
    }

    pub const fn max_chain_descriptors(self) -> u16 {
        self.max_chain_descriptors
    }

    pub const fn max_request_bytes(self) -> u32 {
        self.max_request_bytes
    }

    pub const fn max_response_bytes(self) -> u32 {
        self.max_response_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnrecoverableDecodeError {
    RequestHeader,
    RequestAccess,
    ResponseHeader,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDecodeError {
    Unrecoverable(UnrecoverableDecodeError),
    Protocol {
        request_id: u64,
        status: StatusCode,
    },
    InsufficientResponse {
        request_id: u64,
        required: u64,
        available: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedBinding {
    pub buffer_id: ObjectId,
    pub range: BufferRange,
    pub slot: u32,
    pub access: AccessMode,
}

#[derive(Debug)]
pub enum DecodedRequestBody<'a> {
    GetDeviceInfo,
    CreateContext(ContextDesc),
    DestroyContext {
        context_id: ObjectId,
    },
    AllocateBuffer {
        context_id: ObjectId,
        desc: BufferDesc,
    },
    FreeBuffer {
        buffer_id: ObjectId,
    },
    WriteBuffer {
        buffer_id: ObjectId,
        range: BufferRange,
        data: ReadableRegion<'a>,
    },
    ReadBuffer {
        buffer_id: ObjectId,
        range: BufferRange,
    },
    LoadProgram {
        context_id: ObjectId,
        format: ArtifactFormat,
        target: TargetIdentity,
        payload: ReadableRegion<'a>,
        resident_bytes: u64,
    },
    UnloadProgram {
        program_id: ObjectId,
    },
    CreateQueue {
        context_id: ObjectId,
        desc: QueueDesc,
    },
    DestroyQueue {
        queue_id: ObjectId,
    },
    Submit {
        queue_id: ObjectId,
        program_id: ObjectId,
        bindings: Vec<DecodedBinding>,
        timeout: Timeout,
    },
    PollEvent {
        event_id: ObjectId,
    },
    CancelEvent {
        event_id: ObjectId,
    },
    DestroyEvent {
        event_id: ObjectId,
    },
}

impl DecodedRequestBody<'_> {
    pub const fn opcode(&self) -> KnownOpcode {
        match self {
            Self::GetDeviceInfo => KnownOpcode::GetDeviceInfo,
            Self::CreateContext(_) => KnownOpcode::CreateContext,
            Self::DestroyContext { .. } => KnownOpcode::DestroyContext,
            Self::AllocateBuffer { .. } => KnownOpcode::AllocateBuffer,
            Self::FreeBuffer { .. } => KnownOpcode::FreeBuffer,
            Self::WriteBuffer { .. } => KnownOpcode::WriteBuffer,
            Self::ReadBuffer { .. } => KnownOpcode::ReadBuffer,
            Self::LoadProgram { .. } => KnownOpcode::LoadProgram,
            Self::UnloadProgram { .. } => KnownOpcode::UnloadProgram,
            Self::CreateQueue { .. } => KnownOpcode::CreateQueue,
            Self::DestroyQueue { .. } => KnownOpcode::DestroyQueue,
            Self::Submit { .. } => KnownOpcode::Submit,
            Self::PollEvent { .. } => KnownOpcode::PollEvent,
            Self::CancelEvent { .. } => KnownOpcode::CancelEvent,
            Self::DestroyEvent { .. } => KnownOpcode::DestroyEvent,
        }
    }
}

#[derive(Debug)]
pub struct DecodedRequest<'a> {
    request_id: u64,
    required_response_bytes: u32,
    body: DecodedRequestBody<'a>,
}

impl<'a> DecodedRequest<'a> {
    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    pub const fn required_response_bytes(&self) -> u32 {
        self.required_response_bytes
    }

    pub const fn body(&self) -> &DecodedRequestBody<'a> {
        &self.body
    }

    pub fn into_body(self) -> DecodedRequestBody<'a> {
        self.body
    }
}

/// Stateless decoder configured with protocol and backend-advertised limits.
///
/// A successful return means the complete frame, every fixed field, every binding, and the maximum
/// success-response capacity have been validated. No backend operation is reachable through this
/// type.
#[derive(Clone, Copy, Debug)]
pub struct FrameDecoder {
    limits: DecoderLimits,
}

impl FrameDecoder {
    pub const fn new(limits: DecoderLimits) -> Self {
        Self { limits }
    }

    pub const fn limits(&self) -> DecoderLimits {
        self.limits
    }

    pub fn decode<'a>(
        &self,
        request: &'a dyn ByteSource,
        response_capacity: u64,
    ) -> Result<DecodedRequest<'a>, FrameDecodeError> {
        if response_capacity < RESPONSE_HEADER_BYTES {
            return Err(FrameDecodeError::Unrecoverable(
                UnrecoverableDecodeError::ResponseHeader,
            ));
        }
        if request.len() < REQUEST_HEADER_BYTES {
            return Err(FrameDecodeError::Unrecoverable(
                UnrecoverableDecodeError::RequestHeader,
            ));
        }

        let header = read_wire(
            request,
            0,
            size_of::<RequestHeader>(),
            read_exact::<RequestHeader>,
        )
        .map_err(|_| FrameDecodeError::Unrecoverable(UnrecoverableDecodeError::RequestAccess))?;
        let request_id = header.request_id.get();
        let expected_frame_bytes = REQUEST_HEADER_BYTES + u64::from(header.payload_bytes.get());

        if request.len() > u64::from(self.limits.max_request_bytes) {
            return Err(protocol(request_id, StatusCode::RESOURCE_LIMIT));
        }
        if request.len() != expected_frame_bytes {
            return Err(protocol(request_id, StatusCode::INVALID_ARGUMENT));
        }
        if request_id == 0 {
            return Err(protocol(request_id, StatusCode::INVALID_ARGUMENT));
        }
        if header.flags.get() != 0 {
            return Err(protocol(request_id, StatusCode::UNSUPPORTED));
        }
        let opcode = header
            .known_opcode()
            .map_err(|_| protocol(request_id, StatusCode::UNSUPPORTED))?;

        let (body, required_response_bytes) = self
            .decode_body(request, header.payload_bytes.get(), opcode)
            .map_err(|error| match error {
                BodyDecodeError::Protocol(status) => protocol(request_id, status),
                BodyDecodeError::Access => {
                    FrameDecodeError::Unrecoverable(UnrecoverableDecodeError::RequestAccess)
                }
            })?;

        if required_response_bytes > u64::from(self.limits.max_response_bytes) {
            return Err(protocol(request_id, StatusCode::RESOURCE_LIMIT));
        }
        if response_capacity < required_response_bytes {
            return Err(FrameDecodeError::InsufficientResponse {
                request_id,
                required: required_response_bytes,
                available: response_capacity,
            });
        }

        Ok(DecodedRequest {
            request_id,
            required_response_bytes: required_response_bytes as u32,
            body,
        })
    }

    fn decode_body<'a>(
        &self,
        request: &'a dyn ByteSource,
        payload_bytes: u32,
        opcode: KnownOpcode,
    ) -> Result<(DecodedRequestBody<'a>, u64), BodyDecodeError> {
        match opcode {
            KnownOpcode::GetDeviceInfo => {
                expect_payload_bytes(payload_bytes, 0)?;
                Ok((
                    DecodedRequestBody::GetDeviceInfo,
                    RESPONSE_HEADER_BYTES + size_of::<WireDeviceInfo>() as u64,
                ))
            }
            KnownOpcode::CreateContext => {
                expect_payload_bytes(payload_bytes, size_of::<CreateContextRequest>())?;
                let value = read_payload::<CreateContextRequest>(request)?;
                if value.flags.get() != 0 {
                    return Err(BodyDecodeError::Protocol(StatusCode::UNSUPPORTED));
                }
                if value.reserved.get() != 0 {
                    return Err(invalid());
                }
                Ok((
                    DecodedRequestBody::CreateContext(ContextDesc::default()),
                    RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
                ))
            }
            KnownOpcode::DestroyContext => Ok((
                DecodedRequestBody::DestroyContext {
                    context_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES,
            )),
            KnownOpcode::AllocateBuffer => {
                expect_payload_bytes(payload_bytes, size_of::<AllocateBufferRequest>())?;
                let value = read_payload::<AllocateBufferRequest>(request)?;
                let context_id = object_id(value.context_id.get())?;
                if value.reserved0 != [0; 7] || value.reserved1.get() != 0 {
                    return Err(invalid());
                }

                let domain = MemoryDomain::try_from(value.memory_domain).map_err(|_| invalid())?;
                let usage = BufferUsage::from_bits(value.usage.get())
                    .ok_or(BodyDecodeError::Protocol(StatusCode::UNSUPPORTED))?;
                if usage.is_empty() {
                    return Err(invalid());
                }
                let desc = BufferDesc::new(value.bytes.get(), value.alignment.get(), domain, usage)
                    .map_err(|_| invalid())?;
                if desc.bytes() > self.limits.max_buffer_bytes {
                    return Err(BodyDecodeError::Protocol(StatusCode::RESOURCE_LIMIT));
                }
                if !self.limits.capabilities.supports_memory_domain(domain) {
                    return Err(BodyDecodeError::Protocol(StatusCode::UNSUPPORTED));
                }

                Ok((
                    DecodedRequestBody::AllocateBuffer { context_id, desc },
                    RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
                ))
            }
            KnownOpcode::FreeBuffer => Ok((
                DecodedRequestBody::FreeBuffer {
                    buffer_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES,
            )),
            KnownOpcode::WriteBuffer => {
                let (buffer_id, range) = decode_transfer(request, payload_bytes, true)?;
                let data = ReadableRegion::new(
                    request,
                    REQUEST_HEADER_BYTES + size_of::<TransferBufferRequest>() as u64,
                    range.bytes(),
                )
                .map_err(|_| BodyDecodeError::Access)?;
                Ok((
                    DecodedRequestBody::WriteBuffer {
                        buffer_id,
                        range,
                        data,
                    },
                    RESPONSE_HEADER_BYTES,
                ))
            }
            KnownOpcode::ReadBuffer => {
                let (buffer_id, range) = decode_transfer(request, payload_bytes, false)?;
                let required_response_bytes = RESPONSE_HEADER_BYTES
                    .checked_add(range.bytes())
                    .ok_or_else(resource_limit)?;
                Ok((
                    DecodedRequestBody::ReadBuffer { buffer_id, range },
                    required_response_bytes,
                ))
            }
            KnownOpcode::LoadProgram => {
                if u64::from(payload_bytes) < size_of::<LoadProgramRequest>() as u64 {
                    return Err(invalid());
                }
                let value = read_payload::<LoadProgramRequest>(request)?;
                if value.flags.get() != 0 {
                    return Err(BodyDecodeError::Protocol(StatusCode::UNSUPPORTED));
                }
                let context_id = object_id(value.context_id.get())?;
                let format = ArtifactFormat::new(value.format.get()).ok_or_else(invalid)?;
                let payload_len = value.payload_bytes.get();
                if payload_len == 0 || value.resident_bytes.get() == 0 {
                    return Err(invalid());
                }
                let expected = (size_of::<LoadProgramRequest>() as u64)
                    .checked_add(payload_len)
                    .ok_or_else(resource_limit)?;
                if expected != u64::from(payload_bytes) {
                    return Err(invalid());
                }
                if payload_len > self.limits.max_artifact_bytes {
                    return Err(BodyDecodeError::Protocol(StatusCode::RESOURCE_LIMIT));
                }
                let payload = ReadableRegion::new(
                    request,
                    REQUEST_HEADER_BYTES + size_of::<LoadProgramRequest>() as u64,
                    payload_len,
                )
                .map_err(|_| BodyDecodeError::Access)?;
                let target =
                    TargetIdentity(core::array::from_fn(|index| value.target[index].get()));
                Ok((
                    DecodedRequestBody::LoadProgram {
                        context_id,
                        format,
                        target,
                        payload,
                        resident_bytes: value.resident_bytes.get(),
                    },
                    RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
                ))
            }
            KnownOpcode::UnloadProgram => Ok((
                DecodedRequestBody::UnloadProgram {
                    program_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES,
            )),
            KnownOpcode::CreateQueue => {
                expect_payload_bytes(payload_bytes, size_of::<CreateQueueRequest>())?;
                let value = read_payload::<CreateQueueRequest>(request)?;
                if value.flags.get() != 0 {
                    return Err(BodyDecodeError::Protocol(StatusCode::UNSUPPORTED));
                }
                if value.reserved.get() != 0 {
                    return Err(invalid());
                }
                Ok((
                    DecodedRequestBody::CreateQueue {
                        context_id: object_id(value.context_id.get())?,
                        desc: QueueDesc::default(),
                    },
                    RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
                ))
            }
            KnownOpcode::DestroyQueue => Ok((
                DecodedRequestBody::DestroyQueue {
                    queue_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES,
            )),
            KnownOpcode::Submit => {
                let (queue_id, program_id, bindings, timeout) =
                    self.decode_submit(request, payload_bytes)?;
                Ok((
                    DecodedRequestBody::Submit {
                        queue_id,
                        program_id,
                        bindings,
                        timeout,
                    },
                    RESPONSE_HEADER_BYTES + size_of::<SubmitResponse>() as u64,
                ))
            }
            KnownOpcode::PollEvent => Ok((
                DecodedRequestBody::PollEvent {
                    event_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES + size_of::<WireEventState>() as u64,
            )),
            KnownOpcode::CancelEvent => Ok((
                DecodedRequestBody::CancelEvent {
                    event_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES,
            )),
            KnownOpcode::DestroyEvent => Ok((
                DecodedRequestBody::DestroyEvent {
                    event_id: decode_object(request, payload_bytes)?,
                },
                RESPONSE_HEADER_BYTES,
            )),
        }
    }

    fn decode_submit(
        &self,
        request: &dyn ByteSource,
        payload_bytes: u32,
    ) -> Result<(ObjectId, ObjectId, Vec<DecodedBinding>, Timeout), BodyDecodeError> {
        if u64::from(payload_bytes) < size_of::<SubmitRequest>() as u64 {
            return Err(invalid());
        }
        let value = read_payload::<SubmitRequest>(request)?;
        if value.flags.get() != 0 {
            return Err(BodyDecodeError::Protocol(StatusCode::UNSUPPORTED));
        }

        let binding_count = value.binding_count.get();
        if binding_count == 0 {
            return Err(invalid());
        }
        if binding_count > self.limits.max_bindings || binding_count > HARD_MAX_BINDINGS {
            return Err(BodyDecodeError::Protocol(StatusCode::RESOURCE_LIMIT));
        }
        let binding_bytes = u64::from(binding_count)
            .checked_mul(size_of::<WireBinding>() as u64)
            .ok_or_else(resource_limit)?;
        let expected = (size_of::<SubmitRequest>() as u64)
            .checked_add(binding_bytes)
            .ok_or_else(resource_limit)?;
        if expected != u64::from(payload_bytes) {
            return Err(invalid());
        }

        let count = binding_count as usize;
        let mut bindings = Vec::new();
        bindings
            .try_reserve_exact(count)
            .map_err(|_| BodyDecodeError::Protocol(StatusCode::OUT_OF_MEMORY))?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .map_err(|_| BodyDecodeError::Protocol(StatusCode::OUT_OF_MEMORY))?;

        let binding_base = REQUEST_HEADER_BYTES + size_of::<SubmitRequest>() as u64;
        for index in 0..count {
            let offset = binding_base + (index * size_of::<WireBinding>()) as u64;
            let binding = read_wire(
                request,
                offset,
                size_of::<WireBinding>(),
                read_exact::<WireBinding>,
            )?;
            if binding.reserved != [0; 3] {
                return Err(invalid());
            }
            let bytes = binding.bytes.get();
            if bytes == 0 || binding.offset.get().checked_add(bytes).is_none() {
                return Err(invalid());
            }
            let range = BufferRange::new(binding.offset.get(), bytes).map_err(|_| invalid())?;
            let access = AccessMode::try_from(binding.access).map_err(|_| invalid())?;
            slots.push(binding.slot.get());
            bindings.push(DecodedBinding {
                buffer_id: object_id(binding.buffer_id.get())?,
                range,
                slot: binding.slot.get(),
                access,
            });
        }

        if has_duplicate_slots(&mut slots)? {
            return Err(invalid());
        }

        Ok((
            object_id(value.queue_id.get())?,
            object_id(value.program_id.get())?,
            bindings,
            Timeout::from_wire_ns(value.timeout_ns.get()),
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BodyDecodeError {
    Protocol(StatusCode),
    Access,
}

fn protocol(request_id: u64, status: StatusCode) -> FrameDecodeError {
    FrameDecodeError::Protocol { request_id, status }
}

fn invalid() -> BodyDecodeError {
    BodyDecodeError::Protocol(StatusCode::INVALID_ARGUMENT)
}

fn resource_limit() -> BodyDecodeError {
    BodyDecodeError::Protocol(StatusCode::RESOURCE_LIMIT)
}

fn expect_payload_bytes(payload_bytes: u32, expected: usize) -> Result<(), BodyDecodeError> {
    if u64::from(payload_bytes) != expected as u64 {
        return Err(invalid());
    }
    Ok(())
}

fn read_payload<T>(request: &dyn ByteSource) -> Result<T, BodyDecodeError>
where
    T: FromBytes,
{
    read_wire(
        request,
        REQUEST_HEADER_BYTES,
        size_of::<T>(),
        read_exact::<T>,
    )
}

fn read_wire<T>(
    source: &dyn ByteSource,
    offset: u64,
    bytes: usize,
    decode: impl FnOnce(&[u8]) -> Result<T, virtio_accel_proto::DecodeError>,
) -> Result<T, BodyDecodeError> {
    if bytes > MAX_FIXED_PREFIX_BYTES {
        return Err(BodyDecodeError::Protocol(StatusCode::INTERNAL_ERROR));
    }
    let mut scratch = [0_u8; MAX_FIXED_PREFIX_BYTES];
    source
        .read_at(offset, &mut scratch[..bytes])
        .map_err(|_| BodyDecodeError::Access)?;
    decode(&scratch[..bytes]).map_err(|_| invalid())
}

fn decode_object(
    request: &dyn ByteSource,
    payload_bytes: u32,
) -> Result<ObjectId, BodyDecodeError> {
    expect_payload_bytes(payload_bytes, size_of::<ObjectPayload>())?;
    object_id(read_payload::<ObjectPayload>(request)?.object_id.get())
}

fn object_id(raw: u64) -> Result<ObjectId, BodyDecodeError> {
    ObjectId::from_raw(raw).ok_or_else(invalid)
}

fn decode_transfer(
    request: &dyn ByteSource,
    payload_bytes: u32,
    has_data: bool,
) -> Result<(ObjectId, BufferRange), BodyDecodeError> {
    if u64::from(payload_bytes) < size_of::<TransferBufferRequest>() as u64 {
        return Err(invalid());
    }
    let value = read_payload::<TransferBufferRequest>(request)?;
    let bytes = value.bytes.get();
    if bytes == 0 || value.offset.get().checked_add(bytes).is_none() {
        return Err(invalid());
    }
    let expected = if has_data {
        (size_of::<TransferBufferRequest>() as u64)
            .checked_add(bytes)
            .ok_or_else(resource_limit)?
    } else {
        size_of::<TransferBufferRequest>() as u64
    };
    if expected != u64::from(payload_bytes) {
        return Err(invalid());
    }
    Ok((
        object_id(value.buffer_id.get())?,
        BufferRange::new(value.offset.get(), bytes).map_err(|_| invalid())?,
    ))
}

fn has_duplicate_slots(slots: &mut [u32]) -> Result<bool, BodyDecodeError> {
    if slots.len() < 2 {
        return Ok(false);
    }

    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(slots.len())
        .map_err(|_| BodyDecodeError::Protocol(StatusCode::OUT_OF_MEMORY))?;
    scratch.resize(slots.len(), 0);

    radix_pass(slots, &mut scratch, 0);
    radix_pass(&scratch, slots, 8);
    radix_pass(slots, &mut scratch, 16);
    radix_pass(&scratch, slots, 24);
    Ok(slots.windows(2).any(|window| window[0] == window[1]))
}

fn radix_pass(input: &[u32], output: &mut [u32], shift: u32) {
    let mut counts = [0_usize; 256];
    for value in input {
        counts[((value >> shift) & 0xff) as usize] += 1;
    }

    let mut next = 0;
    for count in &mut counts {
        let start = next;
        next += *count;
        *count = start;
    }

    for value in input {
        let bucket = ((value >> shift) & 0xff) as usize;
        output[counts[bucket]] = *value;
        counts[bucket] += 1;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::SegmentedSource;
    use serde_json::Value;
    use std::vec::Vec;
    use virtio_accel_core::{AcceleratorClass, Capabilities, DeviceIdentity, DeviceLimits};
    use virtio_accel_proto::{
        BASELINE_COMMAND_QUEUES, HARD_MAX_CHAIN_DESCRIPTORS, HARD_MAX_REQUEST_BYTES,
        HARD_MAX_RESPONSE_BYTES, Le16, Le32, PROTOCOL_MAJOR, PROTOCOL_MINOR,
    };

    const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

    fn corpus() -> Value {
        serde_json::from_str(include_str!("../../../conformance/v1.0/vectors.json")).unwrap()
    }

    fn vector(section: &str, name: &str) -> Vec<u8> {
        let corpus = corpus();
        let entry = corpus[section]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"] == name)
            .unwrap();
        decode_hex(entry["hex"].as_str().unwrap())
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
                class: AcceleratorClass::OTHER,
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
                max_artifact_bytes: 1 << 28,
            },
        };
        FrameDecoder::new(DecoderLimits::new(&config, info).unwrap())
    }

    fn opcode(name: &str) -> KnownOpcode {
        match name {
            "GET_DEVICE_INFO" => KnownOpcode::GetDeviceInfo,
            "CREATE_CONTEXT" => KnownOpcode::CreateContext,
            "DESTROY_CONTEXT" => KnownOpcode::DestroyContext,
            "ALLOCATE_BUFFER" => KnownOpcode::AllocateBuffer,
            "FREE_BUFFER" => KnownOpcode::FreeBuffer,
            "WRITE_BUFFER" => KnownOpcode::WriteBuffer,
            "READ_BUFFER" => KnownOpcode::ReadBuffer,
            "LOAD_PROGRAM" => KnownOpcode::LoadProgram,
            "UNLOAD_PROGRAM" => KnownOpcode::UnloadProgram,
            "CREATE_QUEUE" => KnownOpcode::CreateQueue,
            "DESTROY_QUEUE" => KnownOpcode::DestroyQueue,
            "SUBMIT" => KnownOpcode::Submit,
            "POLL_EVENT" => KnownOpcode::PollEvent,
            "CANCEL_EVENT" => KnownOpcode::CancelEvent,
            "DESTROY_EVENT" => KnownOpcode::DestroyEvent,
            other => panic!("unknown opcode name {other}"),
        }
    }

    #[test]
    fn every_canonical_request_decodes_across_every_byte_split() {
        let decoder = decoder();
        let corpus = corpus();
        for frame in corpus["frames"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|frame| frame["kind"] == "request")
        {
            let bytes = decode_hex(frame["hex"].as_str().unwrap());
            let expected_opcode = opcode(frame["opcode"].as_str().unwrap());

            let contiguous_segments = [bytes.as_slice()];
            let contiguous = SegmentedSource::new(&contiguous_segments).unwrap();
            let decoded = decoder
                .decode(&contiguous, u64::from(HARD_MAX_RESPONSE_BYTES))
                .unwrap();
            assert_eq!(decoded.request_id(), REQUEST_ID);
            assert_eq!(decoded.body().opcode(), expected_opcode);

            for split in 1..bytes.len() {
                let segments = [&bytes[..split], &bytes[split..]];
                let segmented = SegmentedSource::new(&segments).unwrap();
                let decoded = decoder
                    .decode(&segmented, u64::from(HARD_MAX_RESPONSE_BYTES))
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} failed at split {split}: {error:?}",
                            frame["name"].as_str().unwrap()
                        )
                    });
                assert_eq!(decoded.request_id(), REQUEST_ID);
                assert_eq!(decoded.body().opcode(), expected_opcode);
            }
        }
    }

    #[test]
    fn transfer_and_artifact_tails_remain_borrowed_segmented_sources() {
        let decoder = decoder();

        let write = vector("frames", "request_write_buffer");
        let write_segments = [&write[..41], &write[41..43], &write[43..]];
        let write_source = SegmentedSource::new(&write_segments).unwrap();
        let decoded = decoder
            .decode(&write_source, u64::from(HARD_MAX_RESPONSE_BYTES))
            .unwrap();
        let DecodedRequestBody::WriteBuffer { data, range, .. } = decoded.body() else {
            panic!("write request decoded as wrong body");
        };
        assert_eq!(range.bytes(), 4);
        assert!(data.as_contiguous().is_none());
        let mut write_payload = [0_u8; 4];
        data.read_at(0, &mut write_payload).unwrap();
        assert_eq!(write_payload, [0xde, 0xad, 0xbe, 0xef]);

        let load = vector("frames", "request_load_program");
        let load_segments = [&load[..97], &load[97..99], &load[99..]];
        let load_source = SegmentedSource::new(&load_segments).unwrap();
        let decoded = decoder
            .decode(&load_source, u64::from(HARD_MAX_RESPONSE_BYTES))
            .unwrap();
        let DecodedRequestBody::LoadProgram {
            payload,
            resident_bytes,
            ..
        } = decoded.body()
        else {
            panic!("load request decoded as wrong body");
        };
        assert_eq!(*resident_bytes, 4096);
        let mut artifact = [0_u8; 4];
        payload.read_at(0, &mut artifact).unwrap();
        assert_eq!(artifact, [0xa1, 0xb2, 0xc3, 0xd4]);
    }

    #[test]
    fn malformed_conformance_vectors_map_to_defined_failures() {
        let decoder = decoder();
        let cases = [
            (
                "request_unknown_opcode",
                FrameDecodeError::Protocol {
                    request_id: REQUEST_ID,
                    status: StatusCode::UNSUPPORTED,
                },
            ),
            (
                "request_reserved_flag",
                FrameDecodeError::Protocol {
                    request_id: REQUEST_ID,
                    status: StatusCode::UNSUPPORTED,
                },
            ),
            (
                "request_trailing_byte",
                FrameDecodeError::Protocol {
                    request_id: REQUEST_ID,
                    status: StatusCode::INVALID_ARGUMENT,
                },
            ),
            (
                "request_reserved_context_flag",
                FrameDecodeError::Protocol {
                    request_id: REQUEST_ID,
                    status: StatusCode::UNSUPPORTED,
                },
            ),
            (
                "request_unknown_memory_domain",
                FrameDecodeError::Protocol {
                    request_id: REQUEST_ID,
                    status: StatusCode::INVALID_ARGUMENT,
                },
            ),
            (
                "request_binding_count_overflow",
                FrameDecodeError::Protocol {
                    request_id: REQUEST_ID,
                    status: StatusCode::RESOURCE_LIMIT,
                },
            ),
        ];

        for (name, expected) in cases {
            let bytes = vector("edge_cases", name);
            let segments = [bytes.as_slice()];
            let source = SegmentedSource::new(&segments).unwrap();
            match decoder.decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)) {
                Err(actual) => assert_eq!(actual, expected, "{name}"),
                Ok(_) => panic!("{name} unexpectedly decoded"),
            }
        }

        let truncated = vector("edge_cases", "request_header_truncated");
        let segments = [truncated.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        assert!(matches!(
            decoder.decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)),
            Err(FrameDecodeError::Unrecoverable(
                UnrecoverableDecodeError::RequestHeader
            ))
        ));
    }

    #[test]
    fn duplicate_submission_slots_are_rejected_before_dispatch() {
        let mut request = vector("frames", "request_submit");
        let binding = request[48..80].to_vec();
        request[4..8].copy_from_slice(&96_u32.to_le_bytes());
        request[32..36].copy_from_slice(&2_u32.to_le_bytes());
        request.extend_from_slice(&binding);

        let segments = [request.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        assert!(matches!(
            decoder().decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)),
            Err(FrameDecodeError::Protocol {
                status: StatusCode::INVALID_ARGUMENT,
                ..
            })
        ));
    }

    #[test]
    fn transfer_range_overflow_is_a_protocol_error() {
        let mut request = vector("frames", "request_read_buffer");
        request[24..32].copy_from_slice(&u64::MAX.to_le_bytes());

        let segments = [request.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        assert!(matches!(
            decoder().decode(&source, u64::from(HARD_MAX_RESPONSE_BYTES)),
            Err(FrameDecodeError::Protocol {
                status: StatusCode::INVALID_ARGUMENT,
                ..
            })
        ));
    }

    #[test]
    fn short_success_capacity_writes_nothing() {
        let request = vector("frames", "request_read_buffer");
        let segments = [request.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        match decoder().decode(&source, 19) {
            Err(actual) => assert_eq!(
                actual,
                FrameDecodeError::InsufficientResponse {
                    request_id: REQUEST_ID,
                    required: 20,
                    available: 19,
                }
            ),
            Ok(_) => panic!("short response capacity unexpectedly decoded"),
        }
    }

    #[test]
    fn malformed_request_needs_only_error_header_capacity() {
        let request = vector("edge_cases", "request_reserved_context_flag");
        let segments = [request.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        assert!(matches!(
            decoder().decode(&source, 16),
            Err(FrameDecodeError::Protocol {
                status: StatusCode::UNSUPPORTED,
                ..
            })
        ));
    }

    #[test]
    fn unsupported_memory_domain_capability_is_rejected_before_backend_use() {
        let config = WireConfig {
            protocol_major: Le16::new(PROTOCOL_MAJOR),
            protocol_minor: Le16::new(PROTOCOL_MINOR),
            command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
            max_chain_descriptors: Le16::new(8),
            max_request_bytes: Le32::new(1024),
            max_response_bytes: Le32::new(1024),
        };
        let info = DeviceInfo {
            identity: DeviceIdentity {
                uuid: [0; 16],
                class: AcceleratorClass::OTHER,
                vendor_id: 0,
                device_id: 0,
            },
            capabilities: Capabilities::HOST_VISIBLE_MEMORY,
            limits: DeviceLimits {
                max_contexts: 1,
                max_buffers_per_context: 1,
                max_programs_per_context: 1,
                max_queues_per_context: 1,
                max_events_per_context: 1,
                max_bindings_per_submission: 1,
                max_buffer_bytes: 1 << 20,
                max_artifact_bytes: 1 << 20,
            },
        };
        let decoder = FrameDecoder::new(DecoderLimits::new(&config, info).unwrap());
        let request = vector("frames", "request_allocate_buffer");
        let segments = [request.as_slice()];
        let source = SegmentedSource::new(&segments).unwrap();
        assert!(matches!(
            decoder.decode(&source, 1024),
            Err(FrameDecodeError::Protocol {
                status: StatusCode::UNSUPPORTED,
                ..
            })
        ));
    }
}
