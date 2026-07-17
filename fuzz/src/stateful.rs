use core::mem::size_of;

use virtio_accel_core::{BackendError, BufferUsage, TransportByteSink, TransportByteSource};
use virtio_accel_device::{
    ChainRegion, CommandOutcome, CommandProcessor, DeviceHealth, ObjectId, ObjectNamespace,
    ResetDisposition, ResourceCounts, ResourcePolicy,
};
use virtio_accel_mock::{MockAccelerator, reference};
use virtio_accel_proto::{
    AllocateBufferRequest, BASELINE_COMMAND_QUEUES, CreateContextRequest, CreateQueueRequest,
    KnownOpcode, Le16, Le32, Le64, LoadProgramRequest, ObjectPayload, PROTOCOL_MAJOR,
    PROTOCOL_MINOR, RequestFlags, RequestHeader, ResponseHeader, StatusCode, SubmitRequest,
    SubmitResponse, TransferBufferRequest, WireBinding, WireConfig, read_exact,
};
use zerocopy::IntoBytes;

use crate::Input;

const MAX_SEQUENCE_BYTES: usize = 1_024;
const ACTION_BYTES: usize = 8;
const RESPONSE_BYTES: usize = 128;
const MAX_BUFFER_BYTES: u64 = 64;
const BUFFER_POLICY_BYTES: u64 = 256;
const PROGRAM_POLICY_BYTES: u64 = 256;

#[derive(Clone, Copy, Debug)]
struct Child {
    id: ObjectId,
    context: ObjectId,
    retained_bytes: u64,
}

#[derive(Default)]
struct Registry {
    contexts: Vec<ObjectId>,
    buffers: Vec<Child>,
    programs: Vec<Child>,
    queues: Vec<Child>,
    events: Vec<ObjectId>,
    stale: Vec<u64>,
}

impl Registry {
    fn remember_live_as_stale(&mut self) {
        self.stale.extend(self.contexts.iter().map(|id| id.get()));
        self.stale
            .extend(self.buffers.iter().map(|child| child.id.get()));
        self.stale
            .extend(self.programs.iter().map(|child| child.id.get()));
        self.stale
            .extend(self.queues.iter().map(|child| child.id.get()));
        self.stale.extend(self.events.iter().map(|id| id.get()));
        self.contexts.clear();
        self.buffers.clear();
        self.programs.clear();
        self.queues.clear();
        self.events.clear();
    }

    fn counts(&self) -> ResourceCounts {
        ResourceCounts {
            contexts: self.contexts.len() as u64,
            buffers: self.buffers.len() as u64,
            programs: self.programs.len() as u64,
            queues: self.queues.len() as u64,
            events: self.events.len() as u64,
        }
    }
}

pub fn fuzz_stateful_commands(data: &[u8]) {
    let data = &data[..data.len().min(MAX_SEQUENCE_BYTES)];
    let mut processor = processor();
    let mut registry = Registry::default();
    let mut request_id = 1_u64;
    let mut namespace = 1_u16;

    for action_bytes in data.chunks(ACTION_BYTES) {
        let mut input = Input::new(action_bytes);
        let action = input.byte() % 18;
        let selector = input.byte();
        let argument = input.u16();
        let entropy = input.u64();

        match action {
            0 => {
                let payload = CreateContextRequest {
                    flags: Le32::new(u32::from(argument & 1)),
                    reserved: Le32::new(u32::from(argument >> 1)),
                };
                let result = execute(
                    &mut processor,
                    request(KnownOpcode::CreateContext, payload.as_bytes(), request_id),
                );
                if result.status == StatusCode::OK {
                    registry.contexts.push(result.object_id());
                }
            }
            1 => {
                let context = choose_context(&registry, selector, entropy);
                let bytes = u64::from(argument % MAX_BUFFER_BYTES as u16) + 1;
                let payload = AllocateBufferRequest {
                    context_id: Le64::new(context),
                    bytes: Le64::new(bytes),
                    alignment: Le64::new(1_u64 << (selector % 7)),
                    memory_domain: 1 + (selector % 3),
                    reserved0: [0; 7],
                    usage: Le32::new(BufferUsage::all().bits()),
                    reserved1: Le32::new(0),
                };
                let result = execute(
                    &mut processor,
                    request(KnownOpcode::AllocateBuffer, payload.as_bytes(), request_id),
                );
                if result.status == StatusCode::OK {
                    registry.buffers.push(Child {
                        id: result.object_id(),
                        context: ObjectId::from_raw(context).unwrap(),
                        retained_bytes: bytes,
                    });
                }
            }
            2 => {
                let context = choose_context(&registry, selector, entropy);
                let artifact = reference::ReferenceArtifact::barrier(0);
                let payload = LoadProgramRequest {
                    context_id: Le64::new(context),
                    format: Le32::new(reference::ARTIFACT_FORMAT.get()),
                    flags: Le32::new(0),
                    target: reference::TARGET_IDENTITY.0.map(Le32::new),
                    payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
                    resident_bytes: Le64::new(reference::RESIDENT_BYTES),
                };
                let mut body = Vec::from(payload.as_bytes());
                body.extend_from_slice(artifact.as_bytes());
                if selector & 0x40 != 0 {
                    let last = body.len() - 1;
                    body[last] ^= 1;
                }
                let result = execute(
                    &mut processor,
                    request(KnownOpcode::LoadProgram, &body, request_id),
                );
                if result.status == StatusCode::OK {
                    registry.programs.push(Child {
                        id: result.object_id(),
                        context: ObjectId::from_raw(context).unwrap(),
                        retained_bytes: reference::RESIDENT_BYTES,
                    });
                }
            }
            3 => {
                let context = choose_context(&registry, selector, entropy);
                let payload = CreateQueueRequest {
                    context_id: Le64::new(context),
                    flags: Le32::new(u32::from(argument & 1)),
                    reserved: Le32::new(u32::from(argument >> 1)),
                };
                let result = execute(
                    &mut processor,
                    request(KnownOpcode::CreateQueue, payload.as_bytes(), request_id),
                );
                if result.status == StatusCode::OK {
                    registry.queues.push(Child {
                        id: result.object_id(),
                        context: ObjectId::from_raw(context).unwrap(),
                        retained_bytes: 0,
                    });
                }
            }
            4 => submit(
                &mut processor,
                &mut registry,
                selector,
                argument,
                entropy,
                request_id,
            ),
            5 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::PollEvent,
                ObjectClass::Event,
                selector,
                entropy,
                request_id,
            ),
            6 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::CancelEvent,
                ObjectClass::Event,
                selector,
                entropy,
                request_id,
            ),
            7 => complete_event(&processor, &registry, selector),
            8 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::DestroyEvent,
                ObjectClass::Event,
                selector,
                entropy,
                request_id,
            ),
            9 | 10 => transfer(
                &mut processor,
                &registry,
                action == 9,
                selector,
                argument,
                entropy,
                request_id,
            ),
            11 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::FreeBuffer,
                ObjectClass::Buffer,
                selector,
                entropy,
                request_id,
            ),
            12 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::UnloadProgram,
                ObjectClass::Program,
                selector,
                entropy,
                request_id,
            ),
            13 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::DestroyQueue,
                ObjectClass::Queue,
                selector,
                entropy,
                request_id,
            ),
            14 => object_command(
                &mut processor,
                &mut registry,
                KnownOpcode::DestroyContext,
                ObjectClass::Context,
                selector,
                entropy,
                request_id,
            ),
            15 => {
                namespace = namespace.saturating_add(1).max(2);
                let report = processor
                    .reset(ObjectNamespace::new(namespace).unwrap())
                    .unwrap();
                assert_eq!(report.disposition, ResetDisposition::BackendReusable);
                assert!(report.quarantined.is_empty());
                assert!(report.quarantined_bytes.is_empty());
                registry.remember_live_as_stale();
            }
            16 => {
                let result = execute(
                    &mut processor,
                    request(KnownOpcode::GetDeviceInfo, &[], request_id),
                );
                assert_eq!(result.status, StatusCode::OK);
            }
            _ => {
                let wrong_kind = choose_context(&registry, selector, entropy);
                let payload = ObjectPayload {
                    object_id: Le64::new(wrong_kind),
                };
                let _ = execute(
                    &mut processor,
                    request(KnownOpcode::FreeBuffer, payload.as_bytes(), request_id),
                );
            }
        }

        assert_invariants(&processor, &registry);
        request_id = request_id.saturating_add(1).max(1);
    }

    namespace = namespace.saturating_add(1).max(2);
    let report = processor
        .reset(ObjectNamespace::new(namespace).unwrap())
        .unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendReusable);
    assert!(processor.state().is_empty());
    assert!(processor.retained_bytes().is_empty());
}

fn processor() -> CommandProcessor<MockAccelerator> {
    let config = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(8),
        max_request_bytes: Le32::new(512),
        max_response_bytes: Le32::new(RESPONSE_BYTES as u32),
    };
    CommandProcessor::new(
        MockAccelerator::default(),
        &config,
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(BUFFER_POLICY_BYTES, PROGRAM_POLICY_BYTES).unwrap(),
    )
    .unwrap()
}

struct ResultFrame {
    status: StatusCode,
    bytes: Vec<u8>,
}

impl ResultFrame {
    fn object_id(&self) -> ObjectId {
        let payload = read_exact::<ObjectPayload>(&self.bytes[16..24]).unwrap();
        ObjectId::from_raw(payload.object_id.get()).unwrap()
    }
}

fn execute(processor: &mut CommandProcessor<MockAccelerator>, frame: Vec<u8>) -> ResultFrame {
    let regions = [
        ChainRegion::readable(frame.len() as u64),
        ChainRegion::writable(RESPONSE_BYTES as u64),
    ];
    let mut response = vec![0xa5_u8; RESPONSE_BYTES];
    let request = frame.as_slice();
    let request = TransportByteSource::new(request);
    let response_sink = response.as_mut_slice();
    let mut response_sink = TransportByteSink::new(response_sink);
    let outcome = processor
        .process(&regions, &request, &mut response_sink)
        .unwrap();
    let CommandOutcome::Response {
        request_id,
        status,
        used,
    } = outcome
    else {
        panic!("generated well-formed frame was unusable");
    };
    assert!(status.is_known());
    let used = used as usize;
    assert!((size_of::<ResponseHeader>()..=RESPONSE_BYTES).contains(&used));
    assert!(response[used..].iter().all(|byte| *byte == 0xa5));
    let header = read_exact::<ResponseHeader>(&response[..size_of::<ResponseHeader>()]).unwrap();
    assert_eq!(header.request_id.get(), request_id);
    assert_eq!(StatusCode(header.status.get()), status);
    assert_eq!(header.flags.get(), 0);
    assert_eq!(
        header.payload_bytes.get() as usize + size_of::<ResponseHeader>(),
        used
    );
    response.truncate(used);
    ResultFrame {
        status,
        bytes: response,
    }
}

fn request(opcode: KnownOpcode, payload: &[u8], request_id: u64) -> Vec<u8> {
    let header = RequestHeader::new(
        opcode,
        RequestFlags::empty(),
        payload.len() as u32,
        request_id,
    );
    let mut frame = Vec::from(header.as_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn submit(
    processor: &mut CommandProcessor<MockAccelerator>,
    registry: &mut Registry,
    selector: u8,
    argument: u16,
    entropy: u64,
    request_id: u64,
) {
    let queue = choose_child(&registry.queues, &registry.stale, selector, entropy);
    let program = choose_child(
        &registry.programs,
        &registry.stale,
        selector.rotate_left(2),
        entropy,
    );
    let buffer = choose_child(
        &registry.buffers,
        &registry.stale,
        selector.rotate_left(4),
        entropy,
    );
    let submit = SubmitRequest {
        queue_id: Le64::new(queue),
        program_id: Le64::new(program),
        binding_count: Le32::new(1),
        flags: Le32::new(0),
        timeout_ns: Le64::new(u64::from(argument)),
    };
    let binding = WireBinding {
        buffer_id: Le64::new(buffer),
        offset: Le64::new(u64::from(selector % 8)),
        bytes: Le64::new(u64::from((argument % MAX_BUFFER_BYTES as u16) + 1)),
        slot: Le32::new(0),
        access: 1 + (selector % 3),
        reserved: [0; 3],
    };
    let mut payload = Vec::from(submit.as_bytes());
    payload.extend_from_slice(binding.as_bytes());
    let result = execute(
        processor,
        request(KnownOpcode::Submit, &payload, request_id),
    );
    if result.status == StatusCode::OK {
        let payload = read_exact::<SubmitResponse>(&result.bytes[16..24]).unwrap();
        registry
            .events
            .push(ObjectId::from_raw(payload.event_id.get()).unwrap());
    }
}

fn transfer(
    processor: &mut CommandProcessor<MockAccelerator>,
    registry: &Registry,
    write: bool,
    selector: u8,
    argument: u16,
    entropy: u64,
    request_id: u64,
) {
    let buffer = choose_child(&registry.buffers, &registry.stale, selector, entropy);
    let bytes = usize::from(argument % MAX_BUFFER_BYTES as u16) + 1;
    let transfer = TransferBufferRequest {
        buffer_id: Le64::new(buffer),
        offset: Le64::new(u64::from(selector % 8)),
        bytes: Le64::new(bytes as u64),
    };
    let mut payload = Vec::from(transfer.as_bytes());
    let opcode = if write {
        payload.extend((0..bytes).map(|index| selector.wrapping_add(index as u8)));
        KnownOpcode::WriteBuffer
    } else {
        KnownOpcode::ReadBuffer
    };
    let _ = execute(processor, request(opcode, &payload, request_id));
}

#[derive(Clone, Copy)]
enum ObjectClass {
    Context,
    Buffer,
    Program,
    Queue,
    Event,
}

fn object_command(
    processor: &mut CommandProcessor<MockAccelerator>,
    registry: &mut Registry,
    opcode: KnownOpcode,
    class: ObjectClass,
    selector: u8,
    entropy: u64,
    request_id: u64,
) {
    let id = match class {
        ObjectClass::Context => choose_id(&registry.contexts, &registry.stale, selector, entropy),
        ObjectClass::Buffer => choose_child(&registry.buffers, &registry.stale, selector, entropy),
        ObjectClass::Program => {
            choose_child(&registry.programs, &registry.stale, selector, entropy)
        }
        ObjectClass::Queue => choose_child(&registry.queues, &registry.stale, selector, entropy),
        ObjectClass::Event => choose_id(&registry.events, &registry.stale, selector, entropy),
    };
    let payload = ObjectPayload {
        object_id: Le64::new(id),
    };
    let result = execute(processor, request(opcode, payload.as_bytes(), request_id));
    if result.status != StatusCode::OK {
        return;
    }
    match class {
        ObjectClass::Context => remove_id(&mut registry.contexts, id),
        ObjectClass::Buffer => remove_child(&mut registry.buffers, id),
        ObjectClass::Program => remove_child(&mut registry.programs, id),
        ObjectClass::Queue => remove_child(&mut registry.queues, id),
        ObjectClass::Event if opcode == KnownOpcode::DestroyEvent => {
            remove_id(&mut registry.events, id)
        }
        ObjectClass::Event => {}
    }
}

fn complete_event(
    processor: &CommandProcessor<MockAccelerator>,
    registry: &Registry,
    selector: u8,
) {
    let Some(id) = registry
        .events
        .get(usize::from(selector) % registry.events.len().max(1))
    else {
        return;
    };
    let Ok(record) = processor.state().event_record(*id) else {
        return;
    };
    let Ok(event) = record.resource() else {
        return;
    };
    match processor.accelerator().complete(event) {
        Ok(()) | Err(BackendError::Busy) => {}
        Err(error) => panic!("mock completion failed unexpectedly: {error:?}"),
    }
}

fn choose_context(registry: &Registry, selector: u8, entropy: u64) -> u64 {
    choose_id(&registry.contexts, &registry.stale, selector, entropy)
}

fn choose_id(live: &[ObjectId], stale: &[u64], selector: u8, entropy: u64) -> u64 {
    if selector & 0x80 == 0 && !live.is_empty() {
        live[usize::from(selector) % live.len()].get()
    } else if !stale.is_empty() {
        stale[usize::from(selector) % stale.len()]
    } else {
        entropy
    }
}

fn choose_child(live: &[Child], stale: &[u64], selector: u8, entropy: u64) -> u64 {
    if selector & 0x80 == 0 && !live.is_empty() {
        live[usize::from(selector) % live.len()].id.get()
    } else if !stale.is_empty() {
        stale[usize::from(selector) % stale.len()]
    } else {
        entropy
    }
}

fn remove_id(ids: &mut Vec<ObjectId>, raw: u64) {
    if let Some(index) = ids.iter().position(|id| id.get() == raw) {
        ids.swap_remove(index);
    }
}

fn remove_child(children: &mut Vec<Child>, raw: u64) {
    if let Some(index) = children.iter().position(|child| child.id.get() == raw) {
        children.swap_remove(index);
    }
}

fn assert_invariants(processor: &CommandProcessor<MockAccelerator>, registry: &Registry) {
    assert_eq!(processor.health(), DeviceHealth::Running);
    assert_eq!(processor.state().resource_counts(), registry.counts());
    let retained = processor.retained_bytes();
    assert_eq!(
        retained.buffer_backing,
        registry
            .buffers
            .iter()
            .map(|child| u128::from(child.retained_bytes))
            .sum::<u128>()
    );
    assert_eq!(
        retained.program_resident,
        registry
            .programs
            .iter()
            .map(|child| u128::from(child.retained_bytes))
            .sum::<u128>()
    );
    assert!(retained.buffer_backing <= u128::from(BUFFER_POLICY_BYTES));
    assert!(retained.program_resident <= u128::from(PROGRAM_POLICY_BYTES));
    for child in registry
        .buffers
        .iter()
        .chain(&registry.programs)
        .chain(&registry.queues)
    {
        assert!(registry.contexts.contains(&child.context));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_sequence_resets_without_leaks() {
        let actions = [0, 1, 2, 3, 4, 7, 5, 8, 11, 12, 13, 14, 15];
        let mut bytes = Vec::new();
        for action in actions {
            bytes.extend_from_slice(&[action, 0, 0, 0, 0, 0, 0, 0]);
        }
        fuzz_stateful_commands(&bytes);
    }
}
