//! A whole protocol session with the driver and the device on opposite sides of shared memory.
//!
//! `portable_end_to_end` joins the reference guest to the reference device engine through the
//! in-memory queue model, where both halves are one value. Here they are two: the guest publishes
//! through `DriverRing`, the engine consumes through `DeviceRing`, and the only thing they have in
//! common is a buffer of bytes laid out as Virtio 1.2 §2.7 says. If the session below computes the
//! right answer, the ring carried every frame of it.

use std::cell::{Cell, RefCell};
use std::mem::{size_of, size_of_val};
use std::num::{NonZeroU32, NonZeroU64};
use std::rc::Rc;
use std::sync::atomic::AtomicU64;

use virtio_accel::core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef,
    BufferDesc as BackendBufferDesc, ByteSink, ByteSource, ContextDesc,
    DeviceInfo as BackendDeviceInfo, EventState as BackendEventState, QueueDesc, ReleaseFailure,
    SubmitFailure, Timeout, TransportByteSink, TransportByteSource,
};
use virtio_accel::device::{CommandOutcome, CommandProcessor, ObjectNamespace, ResourcePolicy};
use virtio_accel::guest::{
    AccessMode, Binding, BufferDesc, BufferRange, BufferUsage, Completion, EventState, GuestClient,
    GuestConfig, MemoryDomain, Operation, Pending, ProgramDesc, RequestPoll, SubmissionOutcome,
};
use virtio_accel::proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, Le16, Le32,
    LoadProgramRequest, ObjectPayload, PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestHeader,
    ResponseHeader, SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding, WireConfig,
    WireDeviceInfo, WireEventState,
};
use virtio_accel::transport::{
    DeviceChain, DeviceQueue, DriverChainBuffer, DriverQueue, NotificationRecheck, PublishError,
    PublishedChain, QueueControl, QueueEpoch, QueueError, QueuePort, QueueSize, QueueState,
    UsedChain, UsedLength,
};
use virtio_accel_mock::{
    MockAccelerator, MockBuffer, MockContext, MockEvent, MockProgram, MockQueue, reference,
};
use virtio_accel_shared_queue::{
    Access, DeviceRing, DriverChain, DriverRing, DriverRingError, MemoryFault, ReclaimedChains,
    Region, RingAddresses, SharedMemory,
};

/// Device addresses start well above zero, so an offset is never an address by accident.
const BASE: u64 = 0x4000_0000;
const BYTES: usize = 1 << 20;
const QUEUE: u16 = 32;
const LIMIT: u16 = 8;
const REQUEST: usize = size_of::<RequestHeader>();
const RESPONSE: usize = size_of::<ResponseHeader>();

/// The memory the two sides share, with no `unsafe`: a platform's implementation maps real
/// memory, this one indexes a vector.
#[derive(Clone, Debug)]
struct Memory(Rc<RefCell<Vec<u8>>>);

impl Memory {
    fn range(&self, at: u64, len: usize) -> Result<std::ops::Range<usize>, MemoryFault> {
        if self.admits(at, len as u64, Access::Read) {
            let from = (at - BASE) as usize;
            Ok(from..from + len)
        } else {
            Err(MemoryFault::Range)
        }
    }
}

impl SharedMemory for Memory {
    fn admits(&self, at: u64, len: u64, _access: Access) -> bool {
        at >= BASE
            && at
                .checked_add(len)
                .is_some_and(|end| end <= BASE + BYTES as u64)
    }

    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), MemoryFault> {
        let range = self.range(at, target.len())?;
        target.copy_from_slice(&self.0.borrow()[range]);
        Ok(())
    }

    fn write(&self, at: u64, source: &[u8]) -> Result<(), MemoryFault> {
        let range = self.range(at, source.len())?;
        self.0.borrow_mut()[range].copy_from_slice(source);
        Ok(())
    }

    fn load_u16(&self, at: u64) -> Result<u16, MemoryFault> {
        if at % 2 != 0 {
            return Err(MemoryFault::Alignment);
        }
        let mut word = [0; 2];
        self.read(at, &mut word)?;
        Ok(u16::from_le_bytes(word))
    }

    fn store_u16(&self, at: u64, value: u16) -> Result<(), MemoryFault> {
        if at % 2 != 0 {
            return Err(MemoryFault::Alignment);
        }
        self.write(at, &value.to_le_bytes())
    }
}

/// The mock accelerator with limits a device of this size can honour. A guest refuses a device
/// whose largest artifact would not fit one request frame, and the mock's own limit is a gibibyte.
struct Backend(MockAccelerator);

impl Accelerator for Backend {
    type Context = MockContext;
    type Buffer = MockBuffer;
    type Program = MockProgram;
    type Queue = MockQueue;
    type Event = MockEvent;

    fn device_info(&self) -> Result<BackendDeviceInfo, BackendError> {
        let mut info = self.0.device_info()?;
        info.limits.max_buffer_bytes = 4_096;
        info.limits.max_artifact_bytes = 4_000;
        Ok(info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.0.create_context(desc)
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        self.0.destroy_context(context)
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BackendBufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.0.allocate_buffer(context, desc)
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        self.0.write_buffer(buffer, offset, data)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        self.0.read_buffer(buffer, offset, data)
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.0.free_buffer(buffer)
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        self.0.load_program(context, artifact)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.0.unload_program(program)
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.0.create_queue(context, desc)
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        self.0.destroy_queue(queue)
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        self.0.submit(queue, program, bindings, timeout)
    }

    fn poll_event(&self, event: &Self::Event) -> Result<BackendEventState, BackendError> {
        self.0.poll_event(event)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        self.0.cancel_event(event)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        self.0.destroy_event(event)
    }
}

type Driver = DriverRing<Memory, 32, 8>;
type Device = DeviceRing<Memory, Rc<AtomicU64>, 32, 8>;
type Chain = DriverChain<Memory, 8>;

/// The driver's queue as the guest sees it, with a device on the other side of the memory that
/// runs whenever the driver looks for an answer. Nothing passes between the two halves here.
struct Link {
    driver: Driver,
    device: Device,
    engine: CommandProcessor<Backend>,
    served: Rc<Cell<u32>>,
}

impl Link {
    fn serve(&mut self) {
        while let Some(mut chain) = self
            .device
            .pop_available()
            .expect("a ring the driver laid out")
        {
            let used = {
                let (regions, request, response) =
                    chain.io().expect("a guest's chain").into_parts();
                let source = TransportByteSource::new(request);
                let mut sink = TransportByteSink::new(response);
                match self
                    .engine
                    .process(regions, &source, &mut sink)
                    .expect("a frame the engine can answer")
                {
                    CommandOutcome::Response { used, .. } => used,
                    CommandOutcome::Unusable(_) => panic!("the guest published an unusable frame"),
                }
            };
            self.device
                .complete(chain, UsedLength::new(used))
                .expect("room the chain declared");
            self.served.set(self.served.get() + 1);
        }
    }
}

impl QueuePort for Link {
    fn state(&self) -> QueueState {
        self.driver.state()
    }
}

impl DriverQueue for Link {
    type Chain = Chain;
    type Reclaimed = ReclaimedChains<Memory, 32, 8>;
    type Error = DriverRingError;

    fn publish(
        &mut self,
        chain: Chain,
    ) -> Result<PublishedChain, PublishError<Chain, Self::Error>> {
        self.driver.publish(chain)
    }

    fn pop_used(&mut self) -> Result<Option<UsedChain<Chain>>, QueueError<Self::Error>> {
        self.serve();
        self.driver.pop_used()
    }

    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        self.driver.disable_used_notifications()
    }

    fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        self.driver.enable_used_notifications()
    }

    fn reset(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
        DriverQueue::reset(&mut self.driver, next_epoch)
    }
}

/// Buffers for chains, each split across descriptors and none adjacent to another.
struct Buffers {
    memory: Memory,
    next: u64,
}

impl Buffers {
    fn region(&mut self, len: usize) -> Region {
        let address = self.next;
        self.next += len as u64 + 40;
        Region {
            address,
            len: len as u32,
        }
    }

    /// A chain for `request` bytes and up to `response`, each in two descriptors where it can be.
    fn chain(&mut self, request: usize, response: usize) -> Chain {
        let split = |total: usize, first: usize| {
            if total > first {
                vec![first, total - first]
            } else {
                vec![total]
            }
        };
        let readable: Vec<Region> = split(request, 7)
            .into_iter()
            .map(|len| self.region(len))
            .collect();
        let writable: Vec<Region> = split(response, 5)
            .into_iter()
            .map(|len| self.region(len))
            .collect();
        Chain::new(self.memory.clone(), &readable, &writable).expect("a command's shape")
    }
}

fn answer<O: Operation>(client: &mut GuestClient<Link>, pending: Pending<O>) -> (O::Output, Chain) {
    match client.poll(pending) {
        RequestPoll::Ready(Completion::Success { output, chain }) => (output, chain),
        RequestPoll::Ready(Completion::DeviceError { status, .. }) => {
            panic!("the device refused a well-formed command: {status:?}")
        }
        RequestPoll::Ready(Completion::InvalidResponse { error, .. }) => {
            panic!("the response did not survive the ring: {error:?}")
        }
        RequestPoll::Pending(_) => panic!("the device did not answer"),
        RequestPoll::QueueError { error, .. } => panic!("the used ring failed: {error:?}"),
        _ => panic!("the queue was reset under the session"),
    }
}

#[test]
fn a_session_runs_with_only_shared_memory_between_guest_and_device() {
    let memory = Memory(Rc::new(RefCell::new(vec![0; BYTES])));
    let size = QueueSize::new(QUEUE).unwrap();
    let (at, ring_bytes) = RingAddresses::contiguous(BASE, size).unwrap();

    // The driver lays the queue out and makes it ready; then the device is told where it is.
    let mut driver = Driver::new(memory.clone(), size, LIMIT).unwrap();
    driver.configure(size).unwrap();
    driver.set_addresses(at).unwrap();
    driver.set_ready(true).unwrap();
    let mut device = Device::new(memory.clone(), Rc::default(), size, LIMIT).unwrap();
    device.configure(size).unwrap();
    device.set_addresses(driver.addresses()).unwrap();
    device.set_ready(true).unwrap();

    let config = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(1),
        max_chain_descriptors: Le16::new(LIMIT),
        max_request_bytes: Le32::new(4_096),
        max_response_bytes: Le32::new(4_096),
    };
    let engine = CommandProcessor::new(
        Backend(MockAccelerator::completing_on_poll()),
        &config,
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(1 << 30, 1 << 30).unwrap(),
    )
    .unwrap();
    let served = Rc::new(Cell::new(0));
    let link = Link {
        driver,
        device,
        engine,
        served: served.clone(),
    };
    let mut client = GuestClient::new(link, GuestConfig::new(config, 0, 8).unwrap()).unwrap();
    let mut buffers = Buffers {
        memory: memory.clone(),
        next: BASE + ring_bytes.next_multiple_of(64) + 64,
    };

    let pending = client
        .get_device_info(buffers.chain(REQUEST, RESPONSE + size_of::<WireDeviceInfo>()))
        .unwrap();
    let (info, _) = answer(&mut client, pending);
    assert_eq!(info.uuid, *b"virtio-accelmock");

    let pending = client
        .create_context(buffers.chain(
            REQUEST + size_of::<CreateContextRequest>(),
            RESPONSE + size_of::<ObjectPayload>(),
        ))
        .unwrap();
    let (context, _) = answer(&mut client, pending);

    let usage = BufferUsage::TRANSFER_SOURCE
        | BufferUsage::TRANSFER_DESTINATION
        | BufferUsage::PROGRAM_INPUT
        | BufferUsage::PROGRAM_OUTPUT
        | BufferUsage::MUTABLE_STATE;
    let pending = client
        .allocate_buffer(
            buffers.chain(
                REQUEST + size_of::<AllocateBufferRequest>(),
                RESPONSE + size_of::<ObjectPayload>(),
            ),
            &context,
            BufferDesc::new(64, 16, MemoryDomain::Shared, usage).unwrap(),
        )
        .unwrap();
    let (buffer, _) = answer(&mut client, pending);

    let operands: [u8; 16] = core::array::from_fn(|index| index as u8 * 3 + 1);
    let pending = client
        .write_buffer(
            buffers.chain(
                REQUEST + size_of::<TransferBufferRequest>() + size_of_val(&operands),
                RESPONSE,
            ),
            &buffer,
            0,
            operands.as_slice(),
        )
        .unwrap();
    answer(&mut client, pending);

    // A program that XORs its one bound buffer with a constant.
    let artifact = reference::ReferenceArtifact::xor(0, 0x5a);
    let pending = client
        .load_program(
            buffers.chain(
                REQUEST + size_of::<LoadProgramRequest>() + reference::ARTIFACT_BYTES,
                RESPONSE + size_of::<ObjectPayload>(),
            ),
            &context,
            ProgramDesc::new(
                NonZeroU32::new(reference::ARTIFACT_FORMAT.get()).unwrap(),
                reference::TARGET_IDENTITY.0,
                NonZeroU64::new(reference::RESIDENT_BYTES).unwrap(),
            ),
            artifact.as_bytes().as_slice(),
        )
        .unwrap();
    let (program, _) = answer(&mut client, pending);

    let pending = client
        .create_execution_queue(
            buffers.chain(
                REQUEST + size_of::<CreateQueueRequest>(),
                RESPONSE + size_of::<ObjectPayload>(),
            ),
            &context,
        )
        .unwrap();
    let (queue, _) = answer(&mut client, pending);

    let whole = BufferRange::new(0, 16).unwrap();
    let binding = [Binding {
        buffer: &buffer,
        range: whole,
        slot: 0,
        access: AccessMode::ReadWrite,
    }];
    let pending = client
        .submit(
            buffers.chain(
                REQUEST + size_of::<SubmitRequest>() + size_of::<WireBinding>(),
                RESPONSE + size_of::<SubmitResponse>(),
            ),
            &queue,
            &program,
            &binding,
            0,
        )
        .unwrap();
    let (outcome, _) = answer(&mut client, pending);
    let SubmissionOutcome::Accepted(event) = outcome else {
        panic!("the mock accepted nothing")
    };
    let pending = client
        .poll_event(
            buffers.chain(
                REQUEST + size_of::<ObjectPayload>(),
                RESPONSE + size_of::<WireEventState>(),
            ),
            &event,
        )
        .unwrap();
    let (state, _) = answer(&mut client, pending);
    assert_eq!(state, EventState::Complete);
    let pending = client
        .destroy_event(
            buffers.chain(REQUEST + size_of::<ObjectPayload>(), RESPONSE),
            event,
        )
        .unwrap();
    answer(&mut client, pending);

    let pending = client
        .read_buffer(
            buffers.chain(REQUEST + size_of::<TransferBufferRequest>(), RESPONSE + 16),
            &buffer,
            whole,
        )
        .unwrap();
    let (read, chain) = answer(&mut client, pending);
    assert_eq!(read.bytes, 16);
    let mut results = [0u8; 16];
    chain
        .read_device_writable(RESPONSE as u64, &mut results)
        .unwrap();
    assert_eq!(
        results,
        operands.map(|byte| byte ^ 0x5a),
        "the device computed over the ring"
    );
    assert_eq!(
        served.get(),
        10,
        "one chain crossed the ring for each command, and no other"
    );
}
