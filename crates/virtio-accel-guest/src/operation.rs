use core::mem::size_of;
use core::num::NonZeroU64;

use virtio_accel_proto::{
    KnownEventState, KnownOpcode, ObjectPayload, StatusCode, SubmitResponse, WireDeviceInfo,
    WireEventState, read_exact,
};
use virtio_accel_transport::{DriverChainBuffer, QueueEpoch};
use zerocopy::FromBytes;

use crate::config::GuestConfig;
use crate::types::{
    Buffer, BufferDesc, Context, DeviceInfo, DeviceInfoError, Event, EventState, ExecutionQueue,
    FailureDisposition, Handle, Program, ProgramDesc, ReadBufferOutput, SubmissionOutcome,
};

const RESPONSE_HEADER_BYTES: u64 = 16;
const MAX_FIXED_PAYLOAD_BYTES: usize = size_of::<WireDeviceInfo>();

/// Malformed or inaccessible device response.
#[derive(Debug, PartialEq, Eq)]
pub enum ResponseError<E> {
    /// Used length cannot contain the required response header or exact payload.
    UsedLength {
        /// Published used length.
        used: u32,
    },
    /// The response exceeds the negotiated frame limit.
    ResponseLimit,
    /// The response header cannot be read.
    HeaderAccess(E),
    /// The response payload cannot be read.
    PayloadAccess(E),
    /// Response request ID does not match the chain's request.
    RequestId {
        /// Expected request ID.
        expected: u64,
        /// Device-provided request ID.
        actual: u64,
    },
    /// Protocol 1.0 response flags are nonzero.
    Flags(u16),
    /// Payload shape does not match the operation and status.
    PayloadLength {
        /// Required payload bytes.
        expected: u64,
        /// Device-provided payload bytes.
        actual: u32,
    },
    /// A fixed payload cannot be decoded from its exact bytes.
    PayloadEncoding,
    /// A successful object-creation response returned object ID zero.
    ObjectId,
    /// Device discovery returned invalid limits or reserved values.
    DeviceInfo(DeviceInfoError),
    /// Event state uses an unknown numeric value.
    EventState(u16),
    /// Event state and error status contradict each other.
    EventStatus {
        /// Raw event state.
        state: u16,
        /// Raw event error status.
        error: StatusCode,
    },
}

#[doc(hidden)]
pub enum OperationResult<T> {
    Success(T),
    DeviceError(StatusCode),
}

mod sealed {
    pub trait Sealed {}
}

/// Sealed response contract for one typed pending operation.
pub trait Operation: sealed::Sealed + Sized {
    /// Validated successful response value.
    type Output;

    /// Command opcode represented by this operation.
    fn opcode(&self) -> KnownOpcode;

    /// Whether device discovery must complete before this operation can be published.
    fn requires_discovery(&self) -> bool {
        true
    }

    #[doc(hidden)]
    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        epoch: QueueEpoch,
        config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>>;

    #[doc(hidden)]
    fn discovered_info(_output: &Self::Output) -> Option<DeviceInfo> {
        None
    }

    #[doc(hidden)]
    fn failure_disposition(status: StatusCode) -> FailureDisposition {
        if !status.is_known() {
            FailureDisposition::Unknown
        } else if status == StatusCode::STALE_OBJECT {
            FailureDisposition::Invalidated
        } else if status == StatusCode::DEVICE_LOST {
            FailureDisposition::Indeterminate
        } else {
            FailureDisposition::Retryable
        }
    }

    #[doc(hidden)]
    fn output_requires_reset(_output: &Self::Output) -> bool {
        false
    }
}

macro_rules! empty_operation {
    ($name:ident, $opcode:ident) => {
        #[doc = concat!("Pending `", stringify!($opcode), "` operation.")]
        #[derive(Debug)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl Operation for $name {
            type Output = ();

            fn opcode(&self) -> KnownOpcode {
                KnownOpcode::$opcode
            }

            fn decode<C: DriverChainBuffer>(
                &self,
                _chain: &C,
                status: StatusCode,
                payload_bytes: u32,
                _epoch: QueueEpoch,
                _config: GuestConfig,
            ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
                ordinary(status, payload_bytes, 0, || Ok(()))
            }
        }
    };
}

/// Pending device-discovery operation.
#[derive(Debug)]
pub struct GetDeviceInfo;

impl sealed::Sealed for GetDeviceInfo {}

impl Operation for GetDeviceInfo {
    type Output = DeviceInfo;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::GetDeviceInfo
    }

    fn requires_discovery(&self) -> bool {
        false
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(
            status,
            payload_bytes,
            size_of::<WireDeviceInfo>() as u64,
            || {
                let wire = read_payload::<WireDeviceInfo, C>(chain)?;
                DeviceInfo::from_wire(wire, config.wire().max_request_bytes.get())
                    .map_err(ResponseError::DeviceInfo)
            },
        )
    }

    fn discovered_info(output: &Self::Output) -> Option<DeviceInfo> {
        Some(*output)
    }
}

/// Pending context-creation operation.
#[derive(Debug)]
pub struct CreateContext;

impl sealed::Sealed for CreateContext {}

impl Operation for CreateContext {
    type Output = Context;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::CreateContext
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary_object(status, payload_bytes, chain, |id| Context {
            handle: Handle::new(id, epoch),
            context: None,
        })
    }
}

/// Pending context destruction; retained on failure for explicit retry.
#[derive(Debug)]
pub struct DestroyContext {
    pub(crate) context: Context,
}

impl DestroyContext {
    /// Inspect or recover the consumed context after failure.
    ///
    /// Retry it only when the completion disposition is `Retryable`.
    pub fn into_context(self) -> Context {
        self.context
    }
}

impl sealed::Sealed for DestroyContext {}

impl Operation for DestroyContext {
    type Output = ();

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::DestroyContext
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        _chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(status, payload_bytes, 0, || Ok(()))
    }
}

/// Pending buffer allocation.
#[derive(Debug)]
pub struct AllocateBuffer {
    pub(crate) context: NonZeroU64,
    pub(crate) desc: BufferDesc,
}

impl sealed::Sealed for AllocateBuffer {}

impl Operation for AllocateBuffer {
    type Output = Buffer;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::AllocateBuffer
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary_object(status, payload_bytes, chain, |id| Buffer {
            handle: Handle::new(id, epoch),
            context: self.context,
            desc: self.desc,
        })
    }
}

/// Pending buffer release; retained on failure for explicit retry.
#[derive(Debug)]
pub struct FreeBuffer {
    pub(crate) buffer: Buffer,
}

impl FreeBuffer {
    /// Inspect or recover the consumed buffer after failure.
    ///
    /// Retry it only when the completion disposition is `Retryable`.
    pub fn into_buffer(self) -> Buffer {
        self.buffer
    }
}

impl sealed::Sealed for FreeBuffer {}

impl Operation for FreeBuffer {
    type Output = ();

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::FreeBuffer
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        _chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(status, payload_bytes, 0, || Ok(()))
    }
}

empty_operation!(WriteBuffer, WriteBuffer);

/// Pending buffer-read operation.
#[derive(Debug)]
pub struct ReadBuffer {
    pub(crate) bytes: u64,
}

impl Operation for ReadBuffer {
    type Output = ReadBufferOutput;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::ReadBuffer
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        _chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(status, payload_bytes, self.bytes, || {
            Ok(ReadBufferOutput { bytes: self.bytes })
        })
    }
}

impl sealed::Sealed for ReadBuffer {}

/// Pending program load.
#[derive(Debug)]
pub struct LoadProgram {
    pub(crate) context: NonZeroU64,
    pub(crate) desc: ProgramDesc,
}

impl sealed::Sealed for LoadProgram {}

impl Operation for LoadProgram {
    type Output = Program;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::LoadProgram
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        let _ = self.desc;
        ordinary_object(status, payload_bytes, chain, |id| Program {
            handle: Handle::new(id, epoch),
            context: Some(self.context),
        })
    }
}

/// Pending program unload; retained on failure for explicit retry.
#[derive(Debug)]
pub struct UnloadProgram {
    pub(crate) program: Program,
}

impl UnloadProgram {
    /// Inspect or recover the consumed program after failure.
    ///
    /// Retry it only when the completion disposition is `Retryable`.
    pub fn into_program(self) -> Program {
        self.program
    }
}

impl sealed::Sealed for UnloadProgram {}

impl Operation for UnloadProgram {
    type Output = ();

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::UnloadProgram
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        _chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(status, payload_bytes, 0, || Ok(()))
    }
}

/// Pending accelerator execution-queue creation.
#[derive(Debug)]
pub struct CreateExecutionQueue {
    pub(crate) context: NonZeroU64,
}

impl sealed::Sealed for CreateExecutionQueue {}

impl Operation for CreateExecutionQueue {
    type Output = ExecutionQueue;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::CreateQueue
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary_object(status, payload_bytes, chain, |id| ExecutionQueue {
            handle: Handle::new(id, epoch),
            context: Some(self.context),
        })
    }
}

/// Pending execution-queue destruction; retained on failure for explicit retry.
#[derive(Debug)]
pub struct DestroyExecutionQueue {
    pub(crate) queue: ExecutionQueue,
}

impl DestroyExecutionQueue {
    /// Inspect or recover the consumed queue after failure.
    ///
    /// Retry it only when the completion disposition is `Retryable`.
    pub fn into_queue(self) -> ExecutionQueue {
        self.queue
    }
}

impl sealed::Sealed for DestroyExecutionQueue {}

impl Operation for DestroyExecutionQueue {
    type Output = ();

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::DestroyQueue
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        _chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(status, payload_bytes, 0, || Ok(()))
    }
}

/// Pending submission admission.
#[derive(Debug)]
pub struct Submit {
    pub(crate) context: NonZeroU64,
}

impl sealed::Sealed for Submit {}

impl Operation for Submit {
    type Output = SubmissionOutcome;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::Submit
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        if status.is_success() {
            expect_payload::<C::Error>(payload_bytes, size_of::<SubmitResponse>() as u64)?;
            let event = decode_event(chain, epoch, self.context)?;
            return Ok(OperationResult::Success(SubmissionOutcome::Accepted(event)));
        }
        if payload_bytes == 0 {
            return Ok(OperationResult::DeviceError(status));
        }
        if !status.is_known() {
            return Err(ResponseError::PayloadLength {
                expected: 0,
                actual: payload_bytes,
            });
        }
        expect_payload::<C::Error>(payload_bytes, size_of::<SubmitResponse>() as u64)?;
        let event = decode_event(chain, epoch, self.context)?;
        Ok(OperationResult::Success(SubmissionOutcome::Indeterminate {
            status,
            event,
        }))
    }

    fn output_requires_reset(output: &Self::Output) -> bool {
        matches!(
            output,
            SubmissionOutcome::Indeterminate {
                status: StatusCode::DEVICE_LOST,
                ..
            }
        )
    }
}

/// Pending nonblocking event poll.
#[derive(Debug)]
pub struct PollEvent;

impl sealed::Sealed for PollEvent {}

impl Operation for PollEvent {
    type Output = EventState;

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::PollEvent
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(
            status,
            payload_bytes,
            size_of::<WireEventState>() as u64,
            || decode_event_state(chain),
        )
    }

    fn output_requires_reset(output: &Self::Output) -> bool {
        matches!(output, EventState::Failed(StatusCode::DEVICE_LOST))
    }
}

empty_operation!(CancelEvent, CancelEvent);

/// Pending event destruction; retained on failure for explicit retry.
#[derive(Debug)]
pub struct DestroyEvent {
    pub(crate) event: Event,
}

impl DestroyEvent {
    /// Inspect or recover the consumed event after failure.
    ///
    /// Retry it only when the completion disposition is `Retryable`.
    pub fn into_event(self) -> Event {
        self.event
    }
}

impl sealed::Sealed for DestroyEvent {}

impl Operation for DestroyEvent {
    type Output = ();

    fn opcode(&self) -> KnownOpcode {
        KnownOpcode::DestroyEvent
    }

    fn decode<C: DriverChainBuffer>(
        &self,
        _chain: &C,
        status: StatusCode,
        payload_bytes: u32,
        _epoch: QueueEpoch,
        _config: GuestConfig,
    ) -> Result<OperationResult<Self::Output>, ResponseError<C::Error>> {
        ordinary(status, payload_bytes, 0, || Ok(()))
    }
}

fn ordinary<T, E>(
    status: StatusCode,
    payload_bytes: u32,
    success_bytes: u64,
    success: impl FnOnce() -> Result<T, ResponseError<E>>,
) -> Result<OperationResult<T>, ResponseError<E>> {
    if !status.is_success() {
        expect_payload::<E>(payload_bytes, 0)?;
        return Ok(OperationResult::DeviceError(status));
    }
    expect_payload::<E>(payload_bytes, success_bytes)?;
    success().map(OperationResult::Success)
}

fn ordinary_object<T, C: DriverChainBuffer>(
    status: StatusCode,
    payload_bytes: u32,
    chain: &C,
    construct: impl FnOnce(NonZeroU64) -> T,
) -> Result<OperationResult<T>, ResponseError<C::Error>> {
    ordinary(
        status,
        payload_bytes,
        size_of::<ObjectPayload>() as u64,
        || {
            let object = read_payload::<ObjectPayload, C>(chain)?;
            let id = NonZeroU64::new(object.object_id.get()).ok_or(ResponseError::ObjectId)?;
            Ok(construct(id))
        },
    )
}

fn expect_payload<E>(actual: u32, expected: u64) -> Result<(), ResponseError<E>> {
    if u64::from(actual) == expected {
        Ok(())
    } else {
        Err(ResponseError::PayloadLength { expected, actual })
    }
}

fn read_payload<T: FromBytes, C: DriverChainBuffer>(
    chain: &C,
) -> Result<T, ResponseError<C::Error>> {
    let bytes = size_of::<T>();
    if bytes > MAX_FIXED_PAYLOAD_BYTES {
        return Err(ResponseError::PayloadEncoding);
    }
    let mut scratch = [0_u8; MAX_FIXED_PAYLOAD_BYTES];
    chain
        .read_device_writable(RESPONSE_HEADER_BYTES, &mut scratch[..bytes])
        .map_err(ResponseError::PayloadAccess)?;
    read_exact(&scratch[..bytes]).map_err(|_| ResponseError::PayloadEncoding)
}

fn decode_event<C: DriverChainBuffer>(
    chain: &C,
    epoch: QueueEpoch,
    context: NonZeroU64,
) -> Result<Event, ResponseError<C::Error>> {
    let response = read_payload::<SubmitResponse, C>(chain)?;
    let id = NonZeroU64::new(response.event_id.get()).ok_or(ResponseError::ObjectId)?;
    Ok(Event {
        handle: Handle::new(id, epoch),
        context: Some(context),
    })
}

fn decode_event_state<C: DriverChainBuffer>(
    chain: &C,
) -> Result<EventState, ResponseError<C::Error>> {
    let value = read_payload::<WireEventState, C>(chain)?;
    if value.reserved.get() != 0 {
        return Err(ResponseError::PayloadEncoding);
    }
    let raw_state = value.state.get();
    let error = StatusCode(value.error.get());
    match value
        .known_state()
        .map_err(|_| ResponseError::EventState(raw_state))?
    {
        KnownEventState::Pending if error.is_success() => Ok(EventState::Pending),
        KnownEventState::Complete if error.is_success() => Ok(EventState::Complete),
        KnownEventState::Failed if !error.is_success() => Ok(EventState::Failed(error)),
        KnownEventState::Cancelled if error.is_success() => Ok(EventState::Cancelled),
        _ => Err(ResponseError::EventStatus {
            state: raw_state,
            error,
        }),
    }
}
