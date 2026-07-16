//! Independent, dependency-free codec for the virtio-accel protocol 1.0 candidate.
//!
//! This crate intentionally does not depend on `virtio-accel-proto`, `zerocopy`, or any shared
//! wire type. It implements the normative byte contract with explicit little-endian reads and
//! writes so conformance tests can exercise interoperability through bytes alone.

#![no_std]
#![forbid(unsafe_code)]

use core::convert::TryFrom;

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 0;
pub const REQUEST_HEADER_BYTES: usize = 16;
pub const RESPONSE_HEADER_BYTES: usize = 16;
pub const HARD_MAX_CHAIN_DESCRIPTORS: u16 = 256;
pub const HARD_MAX_REQUEST_BYTES: u32 = 16 * 1024 * 1024;
pub const HARD_MAX_RESPONSE_BYTES: u32 = 16 * 1024 * 1024;
pub const HARD_MAX_BINDINGS: u32 = 4_096;
pub const MIN_MAX_REQUEST_BYTES: u32 = 97;
pub const MIN_MAX_RESPONSE_BYTES: u32 = 92;
pub const KNOWN_BUFFER_USAGE_BITS: u32 = 0x1f;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Size,
    Version,
    CommandQueueCount,
    ChainDescriptorLimit,
    RequestByteLimit,
    ResponseByteLimit,
    Unsupported,
    InvalidArgument,
    ResourceLimit,
    RecoveryRequired,
    OutputTooSmall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub protocol_major: u16,
    pub protocol_minor: u16,
    pub command_queue_count: u16,
    pub max_chain_descriptors: u16,
    pub max_request_bytes: u32,
    pub max_response_bytes: u32,
}

impl Config {
    pub const ENCODED_BYTES: usize = 16;

    pub fn validate(self, queue_size: u16) -> Result<(), Error> {
        if self.protocol_major != PROTOCOL_MAJOR {
            return Err(Error::Version);
        }
        if self.command_queue_count != 1 {
            return Err(Error::CommandQueueCount);
        }
        if !(2..=HARD_MAX_CHAIN_DESCRIPTORS).contains(&self.max_chain_descriptors)
            || self.max_chain_descriptors > queue_size
        {
            return Err(Error::ChainDescriptorLimit);
        }
        if !(MIN_MAX_REQUEST_BYTES..=HARD_MAX_REQUEST_BYTES).contains(&self.max_request_bytes) {
            return Err(Error::RequestByteLimit);
        }
        if !(MIN_MAX_RESPONSE_BYTES..=HARD_MAX_RESPONSE_BYTES).contains(&self.max_response_bytes) {
            return Err(Error::ResponseByteLimit);
        }
        Ok(())
    }

    pub fn encode(self, queue_size: u16, output: &mut [u8]) -> Result<usize, Error> {
        self.validate(queue_size)?;
        require_output(output, Self::ENCODED_BYTES)?;
        write_u16(output, 0, self.protocol_major)?;
        write_u16(output, 2, self.protocol_minor)?;
        write_u16(output, 4, self.command_queue_count)?;
        write_u16(output, 6, self.max_chain_descriptors)?;
        write_u32(output, 8, self.max_request_bytes)?;
        write_u32(output, 12, self.max_response_bytes)?;
        Ok(Self::ENCODED_BYTES)
    }
}

pub fn decode_config(bytes: &[u8], queue_size: u16) -> Result<Config, Error> {
    if bytes.len() != Config::ENCODED_BYTES {
        return Err(Error::Size);
    }
    let config = Config {
        protocol_major: read_u16(bytes, 0)?,
        protocol_minor: read_u16(bytes, 2)?,
        command_queue_count: read_u16(bytes, 4)?,
        max_chain_descriptors: read_u16(bytes, 6)?,
        max_request_bytes: read_u32(bytes, 8)?,
        max_response_bytes: read_u32(bytes, 12)?,
    };
    config.validate(queue_size)?;
    Ok(config)
}

pub fn validate_features(features: u64) -> Result<(), Error> {
    if features == 0 {
        Ok(())
    } else {
        Err(Error::Unsupported)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Opcode {
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

impl TryFrom<u16> for Opcode {
    type Error = Error;

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
            _ => Err(Error::Unsupported),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocateBuffer {
    pub context_id: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub memory_domain: u8,
    pub usage: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferBuffer {
    pub buffer_id: u64,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadProgram<'a> {
    pub context_id: u64,
    pub format: u32,
    pub target: [u32; 12],
    pub resident_bytes: u64,
    pub artifact: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub buffer_id: u64,
    pub offset: u64,
    pub bytes: u64,
    pub slot: u32,
    pub access: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bindings<'a> {
    Encoded { bytes: &'a [u8], count: u32 },
    Values(&'a [Binding]),
}

impl<'a> Bindings<'a> {
    pub fn count(self) -> Result<u32, Error> {
        match self {
            Self::Encoded { count, .. } => Ok(count),
            Self::Values(values) => u32::try_from(values.len()).map_err(|_| Error::ResourceLimit),
        }
    }

    pub fn get(self, index: u32) -> Result<Binding, Error> {
        match self {
            Self::Encoded { bytes, count } => {
                if index >= count {
                    return Err(Error::InvalidArgument);
                }
                let offset = usize::try_from(index)
                    .map_err(|_| Error::ResourceLimit)?
                    .checked_mul(32)
                    .ok_or(Error::ResourceLimit)?;
                decode_binding(bytes, offset)
            }
            Self::Values(values) => values
                .get(usize::try_from(index).map_err(|_| Error::ResourceLimit)?)
                .copied()
                .ok_or(Error::InvalidArgument),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Submit<'a> {
    pub queue_id: u64,
    pub program_id: u64,
    pub timeout_ns: u64,
    pub bindings: Bindings<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestBody<'a> {
    GetDeviceInfo,
    CreateContext,
    DestroyContext {
        context_id: u64,
    },
    AllocateBuffer(AllocateBuffer),
    FreeBuffer {
        buffer_id: u64,
    },
    WriteBuffer {
        transfer: TransferBuffer,
        data: &'a [u8],
    },
    ReadBuffer(TransferBuffer),
    LoadProgram(LoadProgram<'a>),
    UnloadProgram {
        program_id: u64,
    },
    CreateQueue {
        context_id: u64,
    },
    DestroyQueue {
        queue_id: u64,
    },
    Submit(Submit<'a>),
    PollEvent {
        event_id: u64,
    },
    CancelEvent {
        event_id: u64,
    },
    DestroyEvent {
        event_id: u64,
    },
}

impl RequestBody<'_> {
    pub const fn opcode(&self) -> Opcode {
        match self {
            Self::GetDeviceInfo => Opcode::GetDeviceInfo,
            Self::CreateContext => Opcode::CreateContext,
            Self::DestroyContext { .. } => Opcode::DestroyContext,
            Self::AllocateBuffer(_) => Opcode::AllocateBuffer,
            Self::FreeBuffer { .. } => Opcode::FreeBuffer,
            Self::WriteBuffer { .. } => Opcode::WriteBuffer,
            Self::ReadBuffer(_) => Opcode::ReadBuffer,
            Self::LoadProgram(_) => Opcode::LoadProgram,
            Self::UnloadProgram { .. } => Opcode::UnloadProgram,
            Self::CreateQueue { .. } => Opcode::CreateQueue,
            Self::DestroyQueue { .. } => Opcode::DestroyQueue,
            Self::Submit(_) => Opcode::Submit,
            Self::PollEvent { .. } => Opcode::PollEvent,
            Self::CancelEvent { .. } => Opcode::CancelEvent,
            Self::DestroyEvent { .. } => Opcode::DestroyEvent,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request<'a> {
    pub request_id: u64,
    pub body: RequestBody<'a>,
}

impl Request<'_> {
    pub fn encoded_len(&self) -> Result<usize, Error> {
        let payload = request_payload_len(&self.body)?;
        let total = REQUEST_HEADER_BYTES
            .checked_add(payload)
            .ok_or(Error::ResourceLimit)?;
        if total > HARD_MAX_REQUEST_BYTES as usize {
            return Err(Error::ResourceLimit);
        }
        Ok(total)
    }

    pub fn encode(&self, output: &mut [u8]) -> Result<usize, Error> {
        if self.request_id == 0 {
            return Err(Error::InvalidArgument);
        }
        let payload_len = request_payload_len(&self.body)?;
        let total = REQUEST_HEADER_BYTES
            .checked_add(payload_len)
            .ok_or(Error::ResourceLimit)?;
        let payload_u32 = u32::try_from(payload_len).map_err(|_| Error::ResourceLimit)?;
        if total > HARD_MAX_REQUEST_BYTES as usize {
            return Err(Error::ResourceLimit);
        }
        require_output(output, total)?;
        write_u16(output, 0, self.body.opcode() as u16)?;
        write_u16(output, 2, 0)?;
        write_u32(output, 4, payload_u32)?;
        write_u64(output, 8, self.request_id)?;
        encode_request_body(&self.body, &mut output[REQUEST_HEADER_BYTES..total])?;
        Ok(total)
    }
}

pub fn decode_request(bytes: &[u8]) -> Result<Request<'_>, Error> {
    if bytes.len() < REQUEST_HEADER_BYTES {
        return Err(Error::Size);
    }
    let opcode = Opcode::try_from(read_u16(bytes, 0)?)?;
    if read_u16(bytes, 2)? != 0 {
        return Err(Error::Unsupported);
    }
    let payload_len = usize::try_from(read_u32(bytes, 4)?).map_err(|_| Error::ResourceLimit)?;
    let total = REQUEST_HEADER_BYTES
        .checked_add(payload_len)
        .ok_or(Error::ResourceLimit)?;
    if total != bytes.len() {
        return Err(Error::InvalidArgument);
    }
    if total > HARD_MAX_REQUEST_BYTES as usize {
        return Err(Error::ResourceLimit);
    }
    let request_id = read_u64(bytes, 8)?;
    if request_id == 0 {
        return Err(Error::InvalidArgument);
    }
    let payload = &bytes[REQUEST_HEADER_BYTES..];
    let body = decode_request_body(opcode, payload)?;
    if request_payload_len(&body)? != payload_len {
        return Err(Error::InvalidArgument);
    }
    Ok(Request { request_id, body })
}

fn decode_request_body(opcode: Opcode, payload: &[u8]) -> Result<RequestBody<'_>, Error> {
    match opcode {
        Opcode::GetDeviceInfo => {
            require_exact(payload, 0)?;
            Ok(RequestBody::GetDeviceInfo)
        }
        Opcode::CreateContext => {
            require_exact(payload, 8)?;
            if read_u32(payload, 0)? != 0 {
                return Err(Error::Unsupported);
            }
            if read_u32(payload, 4)? != 0 {
                return Err(Error::InvalidArgument);
            }
            Ok(RequestBody::CreateContext)
        }
        Opcode::DestroyContext => Ok(RequestBody::DestroyContext {
            context_id: decode_object(payload)?,
        }),
        Opcode::AllocateBuffer => {
            require_exact(payload, 40)?;
            if payload[25..32].iter().any(|byte| *byte != 0) || read_u32(payload, 36)? != 0 {
                return Err(Error::InvalidArgument);
            }
            Ok(RequestBody::AllocateBuffer(AllocateBuffer {
                context_id: read_u64(payload, 0)?,
                bytes: read_u64(payload, 8)?,
                alignment: read_u64(payload, 16)?,
                memory_domain: payload[24],
                usage: read_u32(payload, 32)?,
            }))
        }
        Opcode::FreeBuffer => Ok(RequestBody::FreeBuffer {
            buffer_id: decode_object(payload)?,
        }),
        Opcode::WriteBuffer => {
            if payload.len() < 24 {
                return Err(Error::InvalidArgument);
            }
            Ok(RequestBody::WriteBuffer {
                transfer: decode_transfer(payload)?,
                data: &payload[24..],
            })
        }
        Opcode::ReadBuffer => {
            require_exact(payload, 24)?;
            Ok(RequestBody::ReadBuffer(decode_transfer(payload)?))
        }
        Opcode::LoadProgram => {
            if payload.len() < 80 {
                return Err(Error::InvalidArgument);
            }
            if read_u32(payload, 12)? != 0 {
                return Err(Error::Unsupported);
            }
            let mut target = [0_u32; 12];
            for (index, word) in target.iter_mut().enumerate() {
                *word = read_u32(payload, 16 + index * 4)?;
            }
            let artifact = &payload[80..];
            let declared_bytes =
                usize::try_from(read_u64(payload, 64)?).map_err(|_| Error::ResourceLimit)?;
            if declared_bytes != artifact.len() {
                return Err(Error::InvalidArgument);
            }
            Ok(RequestBody::LoadProgram(LoadProgram {
                context_id: read_u64(payload, 0)?,
                format: read_u32(payload, 8)?,
                target,
                resident_bytes: read_u64(payload, 72)?,
                artifact,
            }))
        }
        Opcode::UnloadProgram => Ok(RequestBody::UnloadProgram {
            program_id: decode_object(payload)?,
        }),
        Opcode::CreateQueue => {
            require_exact(payload, 16)?;
            if read_u32(payload, 8)? != 0 {
                return Err(Error::Unsupported);
            }
            if read_u32(payload, 12)? != 0 {
                return Err(Error::InvalidArgument);
            }
            Ok(RequestBody::CreateQueue {
                context_id: read_u64(payload, 0)?,
            })
        }
        Opcode::DestroyQueue => Ok(RequestBody::DestroyQueue {
            queue_id: decode_object(payload)?,
        }),
        Opcode::Submit => {
            if payload.len() < 32 {
                return Err(Error::InvalidArgument);
            }
            let count = read_u32(payload, 16)?;
            if count == 0 {
                return Err(Error::InvalidArgument);
            }
            if count > HARD_MAX_BINDINGS {
                return Err(Error::ResourceLimit);
            }
            if read_u32(payload, 20)? != 0 {
                return Err(Error::Unsupported);
            }
            Ok(RequestBody::Submit(Submit {
                queue_id: read_u64(payload, 0)?,
                program_id: read_u64(payload, 8)?,
                timeout_ns: read_u64(payload, 24)?,
                bindings: Bindings::Encoded {
                    bytes: &payload[32..],
                    count,
                },
            }))
        }
        Opcode::PollEvent => Ok(RequestBody::PollEvent {
            event_id: decode_object(payload)?,
        }),
        Opcode::CancelEvent => Ok(RequestBody::CancelEvent {
            event_id: decode_object(payload)?,
        }),
        Opcode::DestroyEvent => Ok(RequestBody::DestroyEvent {
            event_id: decode_object(payload)?,
        }),
    }
}

fn request_payload_len(body: &RequestBody<'_>) -> Result<usize, Error> {
    match body {
        RequestBody::GetDeviceInfo => Ok(0),
        RequestBody::CreateContext => Ok(8),
        RequestBody::DestroyContext { context_id } => object_len(*context_id),
        RequestBody::AllocateBuffer(request) => {
            validate_allocate(*request)?;
            Ok(40)
        }
        RequestBody::FreeBuffer { buffer_id } => object_len(*buffer_id),
        RequestBody::WriteBuffer { transfer, data } => {
            validate_transfer(*transfer)?;
            let bytes = usize::try_from(transfer.bytes).map_err(|_| Error::ResourceLimit)?;
            if bytes != data.len() {
                return Err(Error::InvalidArgument);
            }
            24_usize.checked_add(bytes).ok_or(Error::ResourceLimit)
        }
        RequestBody::ReadBuffer(transfer) => {
            validate_transfer(*transfer)?;
            Ok(24)
        }
        RequestBody::LoadProgram(request) => {
            if request.context_id == 0
                || request.format == 0
                || request.artifact.is_empty()
                || request.resident_bytes == 0
            {
                return Err(Error::InvalidArgument);
            }
            80_usize
                .checked_add(request.artifact.len())
                .ok_or(Error::ResourceLimit)
        }
        RequestBody::UnloadProgram { program_id } => object_len(*program_id),
        RequestBody::CreateQueue { context_id } => {
            object_len(*context_id)?;
            Ok(16)
        }
        RequestBody::DestroyQueue { queue_id } => object_len(*queue_id),
        RequestBody::Submit(request) => {
            if request.queue_id == 0 || request.program_id == 0 {
                return Err(Error::InvalidArgument);
            }
            validate_bindings(request.bindings)
        }
        RequestBody::PollEvent { event_id }
        | RequestBody::CancelEvent { event_id }
        | RequestBody::DestroyEvent { event_id } => object_len(*event_id),
    }
}

fn encode_request_body(body: &RequestBody<'_>, output: &mut [u8]) -> Result<(), Error> {
    output.fill(0);
    match body {
        RequestBody::GetDeviceInfo | RequestBody::CreateContext => {}
        RequestBody::DestroyContext { context_id } => write_u64(output, 0, *context_id)?,
        RequestBody::AllocateBuffer(request) => {
            write_u64(output, 0, request.context_id)?;
            write_u64(output, 8, request.bytes)?;
            write_u64(output, 16, request.alignment)?;
            output[24] = request.memory_domain;
            write_u32(output, 32, request.usage)?;
        }
        RequestBody::FreeBuffer { buffer_id } => write_u64(output, 0, *buffer_id)?,
        RequestBody::WriteBuffer { transfer, data } => {
            encode_transfer(*transfer, output)?;
            write_slice(output, 24, data)?;
        }
        RequestBody::ReadBuffer(transfer) => encode_transfer(*transfer, output)?,
        RequestBody::LoadProgram(request) => {
            write_u64(output, 0, request.context_id)?;
            write_u32(output, 8, request.format)?;
            for (index, word) in request.target.iter().enumerate() {
                write_u32(output, 16 + index * 4, *word)?;
            }
            write_u64(
                output,
                64,
                u64::try_from(request.artifact.len()).map_err(|_| Error::ResourceLimit)?,
            )?;
            write_u64(output, 72, request.resident_bytes)?;
            write_slice(output, 80, request.artifact)?;
        }
        RequestBody::UnloadProgram { program_id } => write_u64(output, 0, *program_id)?,
        RequestBody::CreateQueue { context_id } => write_u64(output, 0, *context_id)?,
        RequestBody::DestroyQueue { queue_id } => write_u64(output, 0, *queue_id)?,
        RequestBody::Submit(request) => {
            write_u64(output, 0, request.queue_id)?;
            write_u64(output, 8, request.program_id)?;
            let count = request.bindings.count()?;
            write_u32(output, 16, count)?;
            write_u64(output, 24, request.timeout_ns)?;
            for index in 0..count {
                encode_binding(
                    request.bindings.get(index)?,
                    output,
                    32 + index as usize * 32,
                )?;
            }
        }
        RequestBody::PollEvent { event_id }
        | RequestBody::CancelEvent { event_id }
        | RequestBody::DestroyEvent { event_id } => write_u64(output, 0, *event_id)?,
    }
    Ok(())
}

fn validate_allocate(request: AllocateBuffer) -> Result<(), Error> {
    if request.context_id == 0
        || request.bytes == 0
        || request.alignment == 0
        || !request.alignment.is_power_of_two()
        || !(1..=3).contains(&request.memory_domain)
        || request.usage == 0
    {
        return Err(Error::InvalidArgument);
    }
    if request.usage & !KNOWN_BUFFER_USAGE_BITS != 0 {
        return Err(Error::Unsupported);
    }
    Ok(())
}

fn decode_transfer(bytes: &[u8]) -> Result<TransferBuffer, Error> {
    Ok(TransferBuffer {
        buffer_id: read_u64(bytes, 0)?,
        offset: read_u64(bytes, 8)?,
        bytes: read_u64(bytes, 16)?,
    })
}

fn validate_transfer(transfer: TransferBuffer) -> Result<(), Error> {
    if transfer.buffer_id == 0 || transfer.bytes == 0 {
        return Err(Error::InvalidArgument);
    }
    transfer
        .offset
        .checked_add(transfer.bytes)
        .ok_or(Error::InvalidArgument)?;
    Ok(())
}

fn encode_transfer(transfer: TransferBuffer, output: &mut [u8]) -> Result<(), Error> {
    write_u64(output, 0, transfer.buffer_id)?;
    write_u64(output, 8, transfer.offset)?;
    write_u64(output, 16, transfer.bytes)
}

fn validate_bindings(bindings: Bindings<'_>) -> Result<usize, Error> {
    let count = bindings.count()?;
    if count == 0 {
        return Err(Error::InvalidArgument);
    }
    if count > HARD_MAX_BINDINGS {
        return Err(Error::ResourceLimit);
    }
    if let Bindings::Encoded { bytes, count } = bindings {
        let expected = usize::try_from(count)
            .map_err(|_| Error::ResourceLimit)?
            .checked_mul(32)
            .ok_or(Error::ResourceLimit)?;
        if bytes.len() != expected {
            return Err(Error::InvalidArgument);
        }
    }
    for index in 0..count {
        let binding = bindings.get(index)?;
        validate_binding(binding)?;
        for previous in 0..index {
            if bindings.get(previous)?.slot == binding.slot {
                return Err(Error::InvalidArgument);
            }
        }
    }
    32_usize
        .checked_add(
            usize::try_from(count)
                .map_err(|_| Error::ResourceLimit)?
                .checked_mul(32)
                .ok_or(Error::ResourceLimit)?,
        )
        .ok_or(Error::ResourceLimit)
}

fn decode_binding(bytes: &[u8], offset: usize) -> Result<Binding, Error> {
    let end = offset.checked_add(32).ok_or(Error::ResourceLimit)?;
    let binding = bytes.get(offset..end).ok_or(Error::InvalidArgument)?;
    if binding[29..32].iter().any(|byte| *byte != 0) {
        return Err(Error::InvalidArgument);
    }
    Ok(Binding {
        buffer_id: read_u64(binding, 0)?,
        offset: read_u64(binding, 8)?,
        bytes: read_u64(binding, 16)?,
        slot: read_u32(binding, 24)?,
        access: binding[28],
    })
}

fn validate_binding(binding: Binding) -> Result<(), Error> {
    if binding.buffer_id == 0 || binding.bytes == 0 || !(1..=3).contains(&binding.access) {
        return Err(Error::InvalidArgument);
    }
    binding
        .offset
        .checked_add(binding.bytes)
        .ok_or(Error::InvalidArgument)?;
    Ok(())
}

fn encode_binding(binding: Binding, output: &mut [u8], offset: usize) -> Result<(), Error> {
    write_u64(output, offset, binding.buffer_id)?;
    write_u64(output, offset + 8, binding.offset)?;
    write_u64(output, offset + 16, binding.bytes)?;
    write_u32(output, offset + 24, binding.slot)?;
    *output.get_mut(offset + 28).ok_or(Error::OutputTooSmall)? = binding.access;
    Ok(())
}

fn decode_object(payload: &[u8]) -> Result<u64, Error> {
    require_exact(payload, 8)?;
    let object = read_u64(payload, 0)?;
    object_len(object)?;
    Ok(object)
}

fn object_len(object: u64) -> Result<usize, Error> {
    if object == 0 {
        Err(Error::InvalidArgument)
    } else {
        Ok(8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub uuid: [u8; 16],
    pub class: u16,
    pub vendor_id: u32,
    pub device_id: u32,
    pub capabilities: u64,
    pub max_contexts: u32,
    pub max_buffers_per_context: u32,
    pub max_programs_per_context: u32,
    pub max_queues_per_context: u32,
    pub max_events_per_context: u32,
    pub max_bindings_per_submission: u32,
    pub max_buffer_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl DeviceInfo {
    pub const ENCODED_BYTES: usize = 76;

    fn validate(self) -> Result<(), Error> {
        if self.max_contexts == 0
            || self.max_buffers_per_context == 0
            || self.max_programs_per_context == 0
            || self.max_queues_per_context == 0
            || self.max_events_per_context == 0
            || self.max_bindings_per_submission == 0
            || self.max_buffer_bytes == 0
            || self.max_artifact_bytes == 0
        {
            return Err(Error::InvalidArgument);
        }
        if self.max_bindings_per_submission > HARD_MAX_BINDINGS {
            return Err(Error::ResourceLimit);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum EventKind {
    Pending = 0,
    Complete = 1,
    Failed = 2,
    Cancelled = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventState {
    pub kind: EventKind,
    pub error: u16,
}

impl EventState {
    fn validate(self) -> Result<(), Error> {
        match self.kind {
            EventKind::Failed if self.error != 0 => Ok(()),
            EventKind::Pending | EventKind::Complete | EventKind::Cancelled if self.error == 0 => {
                Ok(())
            }
            _ => Err(Error::InvalidArgument),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseBody<'a> {
    Empty,
    DeviceInfo(DeviceInfo),
    Object(u64),
    Data(&'a [u8]),
    EventId(u64),
    EventState(EventState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Response<'a> {
    pub status: u16,
    pub request_id: u64,
    pub body: ResponseBody<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseContext {
    pub opcode: Option<Opcode>,
    pub request_id: u64,
    pub read_bytes: Option<usize>,
}

impl Response<'_> {
    pub fn encoded_len(&self, context: ResponseContext) -> Result<usize, Error> {
        let payload = response_payload_len(self, context)?;
        let total = RESPONSE_HEADER_BYTES
            .checked_add(payload)
            .ok_or(Error::ResourceLimit)?;
        if total > HARD_MAX_RESPONSE_BYTES as usize {
            return Err(Error::ResourceLimit);
        }
        Ok(total)
    }

    pub fn encode(&self, context: ResponseContext, output: &mut [u8]) -> Result<usize, Error> {
        let payload_len = response_payload_len(self, context)?;
        let total = RESPONSE_HEADER_BYTES
            .checked_add(payload_len)
            .ok_or(Error::ResourceLimit)?;
        if total > HARD_MAX_RESPONSE_BYTES as usize {
            return Err(Error::ResourceLimit);
        }
        require_output(output, total)?;
        write_u16(output, 0, self.status)?;
        write_u16(output, 2, 0)?;
        write_u32(
            output,
            4,
            u32::try_from(payload_len).map_err(|_| Error::ResourceLimit)?,
        )?;
        write_u64(output, 8, self.request_id)?;
        encode_response_body(self.body, &mut output[RESPONSE_HEADER_BYTES..total])?;
        Ok(total)
    }
}

pub fn decode_response(bytes: &[u8], context: ResponseContext) -> Result<Response<'_>, Error> {
    if bytes.len() < RESPONSE_HEADER_BYTES {
        return Err(Error::Size);
    }
    let status = read_u16(bytes, 0)?;
    if read_u16(bytes, 2)? != 0 {
        return Err(Error::InvalidArgument);
    }
    let payload_len = usize::try_from(read_u32(bytes, 4)?).map_err(|_| Error::ResourceLimit)?;
    let total = RESPONSE_HEADER_BYTES
        .checked_add(payload_len)
        .ok_or(Error::ResourceLimit)?;
    if total != bytes.len() {
        return Err(Error::InvalidArgument);
    }
    if total > HARD_MAX_RESPONSE_BYTES as usize {
        return Err(Error::ResourceLimit);
    }
    let request_id = read_u64(bytes, 8)?;
    if request_id != context.request_id || request_id == 0 {
        return Err(Error::InvalidArgument);
    }
    let payload = &bytes[RESPONSE_HEADER_BYTES..];
    let body = decode_response_body(status, payload, context)?;
    let response = Response {
        status,
        request_id,
        body,
    };
    if response_payload_len(&response, context)? != payload_len {
        return Err(Error::InvalidArgument);
    }
    Ok(response)
}

fn decode_response_body<'a>(
    status: u16,
    payload: &'a [u8],
    context: ResponseContext,
) -> Result<ResponseBody<'a>, Error> {
    if status != 0 {
        if context.opcode == Some(Opcode::Submit) && payload.len() == 8 {
            return Ok(ResponseBody::EventId(decode_object(payload)?));
        }
        require_exact(payload, 0)?;
        return Ok(ResponseBody::Empty);
    }

    match context.opcode.ok_or(Error::InvalidArgument)? {
        Opcode::GetDeviceInfo => Ok(ResponseBody::DeviceInfo(decode_device_info(payload)?)),
        Opcode::CreateContext
        | Opcode::AllocateBuffer
        | Opcode::LoadProgram
        | Opcode::CreateQueue => Ok(ResponseBody::Object(decode_object(payload)?)),
        Opcode::DestroyContext
        | Opcode::FreeBuffer
        | Opcode::WriteBuffer
        | Opcode::UnloadProgram
        | Opcode::DestroyQueue
        | Opcode::CancelEvent
        | Opcode::DestroyEvent => {
            require_exact(payload, 0)?;
            Ok(ResponseBody::Empty)
        }
        Opcode::ReadBuffer => {
            let expected = context.read_bytes.ok_or(Error::InvalidArgument)?;
            require_exact(payload, expected)?;
            Ok(ResponseBody::Data(payload))
        }
        Opcode::Submit => Ok(ResponseBody::EventId(decode_object(payload)?)),
        Opcode::PollEvent => Ok(ResponseBody::EventState(decode_event_state(payload)?)),
    }
}

fn response_payload_len(response: &Response<'_>, context: ResponseContext) -> Result<usize, Error> {
    if response.request_id == 0 || response.request_id != context.request_id {
        return Err(Error::InvalidArgument);
    }
    if response.status != 0 {
        return match (context.opcode, response.body) {
            (Some(Opcode::Submit), ResponseBody::EventId(event_id)) => object_len(event_id),
            (_, ResponseBody::Empty) => Ok(0),
            _ => Err(Error::InvalidArgument),
        };
    }

    match (context.opcode, response.body) {
        (Some(Opcode::GetDeviceInfo), ResponseBody::DeviceInfo(info)) => {
            info.validate()?;
            Ok(DeviceInfo::ENCODED_BYTES)
        }
        (
            Some(
                Opcode::CreateContext
                | Opcode::AllocateBuffer
                | Opcode::LoadProgram
                | Opcode::CreateQueue,
            ),
            ResponseBody::Object(object),
        ) => object_len(object),
        (
            Some(
                Opcode::DestroyContext
                | Opcode::FreeBuffer
                | Opcode::WriteBuffer
                | Opcode::UnloadProgram
                | Opcode::DestroyQueue
                | Opcode::CancelEvent
                | Opcode::DestroyEvent,
            ),
            ResponseBody::Empty,
        ) => Ok(0),
        (Some(Opcode::ReadBuffer), ResponseBody::Data(data)) => {
            if Some(data.len()) != context.read_bytes {
                return Err(Error::InvalidArgument);
            }
            Ok(data.len())
        }
        (Some(Opcode::Submit), ResponseBody::EventId(event_id)) => object_len(event_id),
        (Some(Opcode::PollEvent), ResponseBody::EventState(state)) => {
            state.validate()?;
            Ok(8)
        }
        _ => Err(Error::InvalidArgument),
    }
}

fn decode_device_info(payload: &[u8]) -> Result<DeviceInfo, Error> {
    require_exact(payload, DeviceInfo::ENCODED_BYTES)?;
    if read_u16(payload, 18)? != 0 {
        return Err(Error::InvalidArgument);
    }
    let mut uuid = [0_u8; 16];
    uuid.copy_from_slice(&payload[..16]);
    let info = DeviceInfo {
        uuid,
        class: read_u16(payload, 16)?,
        vendor_id: read_u32(payload, 20)?,
        device_id: read_u32(payload, 24)?,
        capabilities: read_u64(payload, 28)?,
        max_contexts: read_u32(payload, 36)?,
        max_buffers_per_context: read_u32(payload, 40)?,
        max_programs_per_context: read_u32(payload, 44)?,
        max_queues_per_context: read_u32(payload, 48)?,
        max_events_per_context: read_u32(payload, 52)?,
        max_bindings_per_submission: read_u32(payload, 56)?,
        max_buffer_bytes: read_u64(payload, 60)?,
        max_artifact_bytes: read_u64(payload, 68)?,
    };
    info.validate()?;
    Ok(info)
}

fn decode_event_state(payload: &[u8]) -> Result<EventState, Error> {
    require_exact(payload, 8)?;
    if read_u32(payload, 4)? != 0 {
        return Err(Error::InvalidArgument);
    }
    let kind = match read_u16(payload, 0)? {
        0 => EventKind::Pending,
        1 => EventKind::Complete,
        2 => EventKind::Failed,
        3 => EventKind::Cancelled,
        _ => return Err(Error::RecoveryRequired),
    };
    let state = EventState {
        kind,
        error: read_u16(payload, 2)?,
    };
    state.validate()?;
    Ok(state)
}

fn encode_response_body(body: ResponseBody<'_>, output: &mut [u8]) -> Result<(), Error> {
    output.fill(0);
    match body {
        ResponseBody::Empty => {}
        ResponseBody::DeviceInfo(info) => {
            write_slice(output, 0, &info.uuid)?;
            write_u16(output, 16, info.class)?;
            write_u32(output, 20, info.vendor_id)?;
            write_u32(output, 24, info.device_id)?;
            write_u64(output, 28, info.capabilities)?;
            write_u32(output, 36, info.max_contexts)?;
            write_u32(output, 40, info.max_buffers_per_context)?;
            write_u32(output, 44, info.max_programs_per_context)?;
            write_u32(output, 48, info.max_queues_per_context)?;
            write_u32(output, 52, info.max_events_per_context)?;
            write_u32(output, 56, info.max_bindings_per_submission)?;
            write_u64(output, 60, info.max_buffer_bytes)?;
            write_u64(output, 68, info.max_artifact_bytes)?;
        }
        ResponseBody::Object(object) | ResponseBody::EventId(object) => {
            write_u64(output, 0, object)?;
        }
        ResponseBody::Data(data) => write_slice(output, 0, data)?,
        ResponseBody::EventState(state) => {
            write_u16(output, 0, state.kind as u16)?;
            write_u16(output, 2, state.error)?;
        }
    }
    Ok(())
}

fn require_exact(bytes: &[u8], expected: usize) -> Result<(), Error> {
    if bytes.len() == expected {
        Ok(())
    } else {
        Err(Error::InvalidArgument)
    }
}

fn require_output(output: &[u8], required: usize) -> Result<(), Error> {
    if output.len() >= required {
        Ok(())
    } else {
        Err(Error::OutputTooSmall)
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, Error> {
    let raw: [u8; 2] = bytes
        .get(offset..offset.checked_add(2).ok_or(Error::Size)?)
        .ok_or(Error::Size)?
        .try_into()
        .map_err(|_| Error::Size)?;
    Ok(u16::from_le_bytes(raw))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Error> {
    let raw: [u8; 4] = bytes
        .get(offset..offset.checked_add(4).ok_or(Error::Size)?)
        .ok_or(Error::Size)?
        .try_into()
        .map_err(|_| Error::Size)?;
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, Error> {
    let raw: [u8; 8] = bytes
        .get(offset..offset.checked_add(8).ok_or(Error::Size)?)
        .ok_or(Error::Size)?
        .try_into()
        .map_err(|_| Error::Size)?;
    Ok(u64::from_le_bytes(raw))
}

fn write_u16(output: &mut [u8], offset: usize, value: u16) -> Result<(), Error> {
    write_slice(output, offset, &value.to_le_bytes())
}

fn write_u32(output: &mut [u8], offset: usize, value: u32) -> Result<(), Error> {
    write_slice(output, offset, &value.to_le_bytes())
}

fn write_u64(output: &mut [u8], offset: usize, value: u64) -> Result<(), Error> {
    write_slice(output, offset, &value.to_le_bytes())
}

fn write_slice(output: &mut [u8], offset: usize, value: &[u8]) -> Result<(), Error> {
    let end = offset
        .checked_add(value.len())
        .ok_or(Error::OutputTooSmall)?;
    output
        .get_mut(offset..end)
        .ok_or(Error::OutputTooSmall)?
        .copy_from_slice(value);
    Ok(())
}
