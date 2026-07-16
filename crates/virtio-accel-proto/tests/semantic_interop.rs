use virtio_accel_cleanroom as clean;
use virtio_accel_proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, KnownEventState, KnownOpcode,
    Le16, Le32, Le64, LoadProgramRequest, ObjectPayload, RequestFlags, RequestHeader,
    ResponseHeader, StatusCode, SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding,
    WireConfig, WireDeviceInfo, WireEventState, read_exact,
};
use zerocopy::IntoBytes;

const REQUEST_ID: u64 = 0x1020_3040_5060_7080;
const CONTEXT_ID: u64 = 0x1112_1314_1516_1718;
const BUFFER_ID: u64 = 0x2122_2324_2526_2728;
const PROGRAM_ID: u64 = 0x3132_3334_3536_3738;
const QUEUE_ID: u64 = 0x4142_4344_4546_4748;
const EVENT_ID: u64 = 0x5152_5354_5556_5758;

fn primary_request(opcode: KnownOpcode, payload: &[u8]) -> Vec<u8> {
    let header = RequestHeader::new(
        opcode,
        RequestFlags::empty(),
        u32::try_from(payload.len()).unwrap(),
        REQUEST_ID,
    );
    let mut bytes = Vec::from(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn primary_response(status: StatusCode, payload: &[u8]) -> Vec<u8> {
    let header = ResponseHeader::new(status, u32::try_from(payload.len()).unwrap(), REQUEST_ID);
    let mut bytes = Vec::from(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn clean_request(body: clean::RequestBody<'_>) -> Vec<u8> {
    let request = clean::Request {
        request_id: REQUEST_ID,
        body,
    };
    let mut bytes = vec![0_u8; request.encoded_len().unwrap()];
    let written = request.encode(&mut bytes).unwrap();
    assert_eq!(written, bytes.len());
    bytes
}

fn clean_response(
    status: u16,
    body: clean::ResponseBody<'_>,
    context: clean::ResponseContext,
) -> Vec<u8> {
    let response = clean::Response {
        status,
        request_id: REQUEST_ID,
        body,
    };
    let mut bytes = vec![0_u8; response.encoded_len(context).unwrap()];
    let written = response.encode(context, &mut bytes).unwrap();
    assert_eq!(written, bytes.len());
    bytes
}

fn response_context(opcode: clean::Opcode, read_bytes: Option<usize>) -> clean::ResponseContext {
    clean::ResponseContext {
        opcode: Some(opcode),
        request_id: REQUEST_ID,
        read_bytes,
    }
}

fn assert_primary_header(bytes: &[u8], opcode: KnownOpcode, payload_bytes: usize) {
    let header = read_exact::<RequestHeader>(&bytes[..16]).unwrap();
    assert_eq!(header.opcode.get(), opcode as u16);
    assert_eq!(header.flags.get(), 0);
    assert_eq!(header.payload_bytes.get() as usize, payload_bytes);
    assert_eq!(header.request_id.get(), REQUEST_ID);
}

fn assert_primary_response_header(bytes: &[u8], status: u16, payload_bytes: usize) {
    let header = read_exact::<ResponseHeader>(&bytes[..16]).unwrap();
    assert_eq!(header.status.get(), status);
    assert_eq!(header.flags.get(), 0);
    assert_eq!(header.payload_bytes.get() as usize, payload_bytes);
    assert_eq!(header.request_id.get(), REQUEST_ID);
}

#[test]
fn primary_wire_types_decode_to_independent_semantics_for_every_layout() {
    let config = WireConfig {
        protocol_major: Le16::new(1),
        protocol_minor: Le16::new(0),
        command_queue_count: Le16::new(1),
        max_chain_descriptors: Le16::new(64),
        max_request_bytes: Le32::new(1_048_576),
        max_response_bytes: Le32::new(2_097_152),
    };
    assert_eq!(
        clean::decode_config(config.as_bytes(), 128).unwrap(),
        clean::Config {
            protocol_major: 1,
            protocol_minor: 0,
            command_queue_count: 1,
            max_chain_descriptors: 64,
            max_request_bytes: 1_048_576,
            max_response_bytes: 2_097_152,
        }
    );

    let get_info_bytes = primary_request(KnownOpcode::GetDeviceInfo, &[]);
    let get_info = clean::decode_request(&get_info_bytes).unwrap();
    assert_eq!(get_info.request_id, REQUEST_ID);
    assert_eq!(get_info.body, clean::RequestBody::GetDeviceInfo);

    let create_context = CreateContextRequest {
        flags: Le32::new(0),
        reserved: Le32::new(0),
    };
    assert_eq!(
        clean::decode_request(&primary_request(
            KnownOpcode::CreateContext,
            create_context.as_bytes(),
        ))
        .unwrap()
        .body,
        clean::RequestBody::CreateContext
    );

    let object = ObjectPayload {
        object_id: Le64::new(CONTEXT_ID),
    };
    assert_eq!(
        clean::decode_request(&primary_request(
            KnownOpcode::DestroyContext,
            object.as_bytes(),
        ))
        .unwrap()
        .body,
        clean::RequestBody::DestroyContext {
            context_id: CONTEXT_ID,
        }
    );

    let allocate = AllocateBufferRequest {
        context_id: Le64::new(CONTEXT_ID),
        bytes: Le64::new(65_536),
        alignment: Le64::new(4_096),
        memory_domain: 2,
        reserved0: [0; 7],
        usage: Le32::new(0x15),
        reserved1: Le32::new(0),
    };
    assert_eq!(
        clean::decode_request(&primary_request(
            KnownOpcode::AllocateBuffer,
            allocate.as_bytes(),
        ))
        .unwrap()
        .body,
        clean::RequestBody::AllocateBuffer(clean::AllocateBuffer {
            context_id: CONTEXT_ID,
            bytes: 65_536,
            alignment: 4_096,
            memory_domain: 2,
            usage: 0x15,
        })
    );

    let transfer = TransferBufferRequest {
        buffer_id: Le64::new(BUFFER_ID),
        offset: Le64::new(128),
        bytes: Le64::new(4),
    };
    assert_eq!(
        clean::decode_request(&primary_request(
            KnownOpcode::ReadBuffer,
            transfer.as_bytes(),
        ))
        .unwrap()
        .body,
        clean::RequestBody::ReadBuffer(clean::TransferBuffer {
            buffer_id: BUFFER_ID,
            offset: 128,
            bytes: 4,
        })
    );
    let mut write_payload = Vec::from(transfer.as_bytes());
    write_payload.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    assert_eq!(
        clean::decode_request(&primary_request(KnownOpcode::WriteBuffer, &write_payload,))
            .unwrap()
            .body,
        clean::RequestBody::WriteBuffer {
            transfer: clean::TransferBuffer {
                buffer_id: BUFFER_ID,
                offset: 128,
                bytes: 4,
            },
            data: &[0xde, 0xad, 0xbe, 0xef],
        }
    );

    let load = LoadProgramRequest {
        context_id: Le64::new(CONTEXT_ID),
        format: Le32::new(0x1122_3344),
        flags: Le32::new(0),
        target: core::array::from_fn(|index| Le32::new(100 + index as u32)),
        payload_bytes: Le64::new(3),
        resident_bytes: Le64::new(98_304),
    };
    let mut load_payload = Vec::from(load.as_bytes());
    load_payload.extend_from_slice(&[0xa1, 0xb2, 0xc3]);
    let load_bytes = primary_request(KnownOpcode::LoadProgram, &load_payload);
    let decoded_load = clean::decode_request(&load_bytes).unwrap();
    let clean::RequestBody::LoadProgram(decoded_load) = decoded_load.body else {
        panic!("LOAD_PROGRAM decoded as the wrong independent request");
    };
    assert_eq!(decoded_load.context_id, CONTEXT_ID);
    assert_eq!(decoded_load.format, 0x1122_3344);
    assert_eq!(
        decoded_load.target,
        core::array::from_fn(|index| 100 + index as u32)
    );
    assert_eq!(decoded_load.resident_bytes, 98_304);
    assert_eq!(decoded_load.artifact, &[0xa1, 0xb2, 0xc3]);

    let create_queue = CreateQueueRequest {
        context_id: Le64::new(CONTEXT_ID),
        flags: Le32::new(0),
        reserved: Le32::new(0),
    };
    assert_eq!(
        clean::decode_request(&primary_request(
            KnownOpcode::CreateQueue,
            create_queue.as_bytes(),
        ))
        .unwrap()
        .body,
        clean::RequestBody::CreateQueue {
            context_id: CONTEXT_ID,
        }
    );

    let submit = SubmitRequest {
        queue_id: Le64::new(QUEUE_ID),
        program_id: Le64::new(PROGRAM_ID),
        binding_count: Le32::new(2),
        flags: Le32::new(0),
        timeout_ns: Le64::new(7_500_000),
    };
    let bindings = [
        WireBinding {
            buffer_id: Le64::new(BUFFER_ID),
            offset: Le64::new(64),
            bytes: Le64::new(1_024),
            slot: Le32::new(3),
            access: 1,
            reserved: [0; 3],
        },
        WireBinding {
            buffer_id: Le64::new(BUFFER_ID + 1),
            offset: Le64::new(128),
            bytes: Le64::new(2_048),
            slot: Le32::new(9),
            access: 3,
            reserved: [0; 3],
        },
    ];
    let mut submit_payload = Vec::from(submit.as_bytes());
    for binding in &bindings {
        submit_payload.extend_from_slice(binding.as_bytes());
    }
    let submit_bytes = primary_request(KnownOpcode::Submit, &submit_payload);
    let decoded_submit = clean::decode_request(&submit_bytes).unwrap();
    let clean::RequestBody::Submit(decoded_submit) = decoded_submit.body else {
        panic!("SUBMIT decoded as the wrong independent request");
    };
    assert_eq!(decoded_submit.queue_id, QUEUE_ID);
    assert_eq!(decoded_submit.program_id, PROGRAM_ID);
    assert_eq!(decoded_submit.timeout_ns, 7_500_000);
    assert_eq!(decoded_submit.bindings.count(), Ok(2));
    assert_eq!(
        decoded_submit.bindings.get(0),
        Ok(clean::Binding {
            buffer_id: BUFFER_ID,
            offset: 64,
            bytes: 1_024,
            slot: 3,
            access: 1,
        })
    );
    assert_eq!(
        decoded_submit.bindings.get(1),
        Ok(clean::Binding {
            buffer_id: BUFFER_ID + 1,
            offset: 128,
            bytes: 2_048,
            slot: 9,
            access: 3,
        })
    );

    let device_info = WireDeviceInfo {
        uuid: *b"semantic-interop",
        class: Le16::new(1),
        reserved: Le16::new(0),
        vendor_id: Le32::new(0x1234_5678),
        device_id: Le32::new(0x9abc_def0),
        capabilities: Le64::new(0x1020_3040_5060_7080),
        max_contexts: Le32::new(4),
        max_buffers_per_context: Le32::new(16),
        max_programs_per_context: Le32::new(8),
        max_queues_per_context: Le32::new(2),
        max_events_per_context: Le32::new(32),
        max_bindings_per_submission: Le32::new(64),
        max_buffer_bytes: Le64::new(1 << 28),
        max_artifact_bytes: Le64::new(1 << 20),
    };
    let context = response_context(clean::Opcode::GetDeviceInfo, None);
    let info_bytes = primary_response(StatusCode::OK, device_info.as_bytes());
    let decoded_info = clean::decode_response(&info_bytes, context).unwrap();
    assert_eq!(
        decoded_info.body,
        clean::ResponseBody::DeviceInfo(clean::DeviceInfo {
            uuid: *b"semantic-interop",
            class: 1,
            vendor_id: 0x1234_5678,
            device_id: 0x9abc_def0,
            capabilities: 0x1020_3040_5060_7080,
            max_contexts: 4,
            max_buffers_per_context: 16,
            max_programs_per_context: 8,
            max_queues_per_context: 2,
            max_events_per_context: 32,
            max_bindings_per_submission: 64,
            max_buffer_bytes: 1 << 28,
            max_artifact_bytes: 1 << 20,
        })
    );

    let object_response = ObjectPayload {
        object_id: Le64::new(CONTEXT_ID),
    };
    assert_eq!(
        clean::decode_response(
            &primary_response(StatusCode::OK, object_response.as_bytes()),
            response_context(clean::Opcode::CreateContext, None),
        )
        .unwrap()
        .body,
        clean::ResponseBody::Object(CONTEXT_ID)
    );
    assert_eq!(
        clean::decode_response(
            &primary_response(StatusCode::OK, &[]),
            response_context(clean::Opcode::DestroyContext, None),
        )
        .unwrap()
        .body,
        clean::ResponseBody::Empty
    );
    assert_eq!(
        clean::decode_response(
            &primary_response(StatusCode::OK, &[1, 2, 3, 4]),
            response_context(clean::Opcode::ReadBuffer, Some(4)),
        )
        .unwrap()
        .body,
        clean::ResponseBody::Data(&[1, 2, 3, 4])
    );

    let event = SubmitResponse {
        event_id: Le64::new(EVENT_ID),
    };
    assert_eq!(
        clean::decode_response(
            &primary_response(StatusCode::DEVICE_LOST, event.as_bytes()),
            response_context(clean::Opcode::Submit, None),
        )
        .unwrap()
        .body,
        clean::ResponseBody::EventId(EVENT_ID)
    );

    let event_state = WireEventState {
        state: Le16::new(KnownEventState::Failed as u16),
        error: Le16::new(StatusCode::DEADLINE_EXPIRED.0),
        reserved: Le32::new(0),
    };
    assert_eq!(
        clean::decode_response(
            &primary_response(StatusCode::OK, event_state.as_bytes()),
            response_context(clean::Opcode::PollEvent, None),
        )
        .unwrap()
        .body,
        clean::ResponseBody::EventState(clean::EventState {
            kind: clean::EventKind::Failed,
            error: StatusCode::DEADLINE_EXPIRED.0,
        })
    );
}

#[test]
fn independent_semantics_decode_to_primary_wire_types_for_every_layout() {
    let config = clean::Config {
        protocol_major: 1,
        protocol_minor: 0,
        command_queue_count: 1,
        max_chain_descriptors: 96,
        max_request_bytes: 3_145_728,
        max_response_bytes: 4_194_304,
    };
    let mut config_bytes = [0_u8; clean::Config::ENCODED_BYTES];
    config.encode(128, &mut config_bytes).unwrap();
    let primary_config = read_exact::<WireConfig>(&config_bytes).unwrap();
    assert_eq!(primary_config.protocol_major.get(), 1);
    assert_eq!(primary_config.protocol_minor.get(), 0);
    assert_eq!(primary_config.command_queue_count.get(), 1);
    assert_eq!(primary_config.max_chain_descriptors.get(), 96);
    assert_eq!(primary_config.max_request_bytes.get(), 3_145_728);
    assert_eq!(primary_config.max_response_bytes.get(), 4_194_304);

    let get_info = clean_request(clean::RequestBody::GetDeviceInfo);
    assert_primary_header(&get_info, KnownOpcode::GetDeviceInfo, 0);

    let create_context = clean_request(clean::RequestBody::CreateContext);
    assert_primary_header(&create_context, KnownOpcode::CreateContext, 8);
    let primary_create = read_exact::<CreateContextRequest>(&create_context[16..]).unwrap();
    assert_eq!(primary_create.flags.get(), 0);
    assert_eq!(primary_create.reserved.get(), 0);

    let destroy_context = clean_request(clean::RequestBody::DestroyContext {
        context_id: CONTEXT_ID,
    });
    assert_primary_header(&destroy_context, KnownOpcode::DestroyContext, 8);
    assert_eq!(
        read_exact::<ObjectPayload>(&destroy_context[16..])
            .unwrap()
            .object_id
            .get(),
        CONTEXT_ID
    );

    let allocate = clean_request(clean::RequestBody::AllocateBuffer(clean::AllocateBuffer {
        context_id: CONTEXT_ID,
        bytes: 131_072,
        alignment: 8_192,
        memory_domain: 3,
        usage: 0x0c,
    }));
    assert_primary_header(&allocate, KnownOpcode::AllocateBuffer, 40);
    let primary_allocate = read_exact::<AllocateBufferRequest>(&allocate[16..]).unwrap();
    assert_eq!(primary_allocate.context_id.get(), CONTEXT_ID);
    assert_eq!(primary_allocate.bytes.get(), 131_072);
    assert_eq!(primary_allocate.alignment.get(), 8_192);
    assert_eq!(primary_allocate.memory_domain, 3);
    assert_eq!(primary_allocate.reserved0, [0; 7]);
    assert_eq!(primary_allocate.usage.get(), 0x0c);
    assert_eq!(primary_allocate.reserved1.get(), 0);

    let read = clean_request(clean::RequestBody::ReadBuffer(clean::TransferBuffer {
        buffer_id: BUFFER_ID,
        offset: 256,
        bytes: 8,
    }));
    assert_primary_header(&read, KnownOpcode::ReadBuffer, 24);
    let primary_read = read_exact::<TransferBufferRequest>(&read[16..]).unwrap();
    assert_eq!(primary_read.buffer_id.get(), BUFFER_ID);
    assert_eq!(primary_read.offset.get(), 256);
    assert_eq!(primary_read.bytes.get(), 8);

    let write_data = [8_u8, 7, 6, 5, 4, 3, 2, 1];
    let write = clean_request(clean::RequestBody::WriteBuffer {
        transfer: clean::TransferBuffer {
            buffer_id: BUFFER_ID,
            offset: 512,
            bytes: 8,
        },
        data: &write_data,
    });
    assert_primary_header(&write, KnownOpcode::WriteBuffer, 32);
    let primary_write = read_exact::<TransferBufferRequest>(&write[16..40]).unwrap();
    assert_eq!(primary_write.buffer_id.get(), BUFFER_ID);
    assert_eq!(primary_write.offset.get(), 512);
    assert_eq!(primary_write.bytes.get(), 8);
    assert_eq!(&write[40..], &write_data);

    let target = core::array::from_fn(|index| 200 + index as u32);
    let artifact = [0x10_u8, 0x20, 0x30, 0x40, 0x50];
    let load = clean_request(clean::RequestBody::LoadProgram(clean::LoadProgram {
        context_id: CONTEXT_ID,
        format: 0x5566_7788,
        target,
        resident_bytes: 262_144,
        artifact: &artifact,
    }));
    assert_primary_header(&load, KnownOpcode::LoadProgram, 85);
    let primary_load = read_exact::<LoadProgramRequest>(&load[16..96]).unwrap();
    assert_eq!(primary_load.context_id.get(), CONTEXT_ID);
    assert_eq!(primary_load.format.get(), 0x5566_7788);
    assert_eq!(primary_load.flags.get(), 0);
    assert_eq!(
        primary_load.target.map(|word| word.get()),
        core::array::from_fn(|index| 200 + index as u32)
    );
    assert_eq!(primary_load.payload_bytes.get(), 5);
    assert_eq!(primary_load.resident_bytes.get(), 262_144);
    assert_eq!(&load[96..], &artifact);

    let create_queue = clean_request(clean::RequestBody::CreateQueue {
        context_id: CONTEXT_ID,
    });
    assert_primary_header(&create_queue, KnownOpcode::CreateQueue, 16);
    let primary_queue = read_exact::<CreateQueueRequest>(&create_queue[16..]).unwrap();
    assert_eq!(primary_queue.context_id.get(), CONTEXT_ID);
    assert_eq!(primary_queue.flags.get(), 0);
    assert_eq!(primary_queue.reserved.get(), 0);

    let clean_bindings = [
        clean::Binding {
            buffer_id: BUFFER_ID,
            offset: 32,
            bytes: 512,
            slot: 4,
            access: 1,
        },
        clean::Binding {
            buffer_id: BUFFER_ID + 2,
            offset: 64,
            bytes: 1_024,
            slot: 12,
            access: 2,
        },
    ];
    let submit = clean_request(clean::RequestBody::Submit(clean::Submit {
        queue_id: QUEUE_ID,
        program_id: PROGRAM_ID,
        timeout_ns: 9_000_000,
        bindings: clean::Bindings::Values(&clean_bindings),
    }));
    assert_primary_header(&submit, KnownOpcode::Submit, 96);
    let primary_submit = read_exact::<SubmitRequest>(&submit[16..48]).unwrap();
    assert_eq!(primary_submit.queue_id.get(), QUEUE_ID);
    assert_eq!(primary_submit.program_id.get(), PROGRAM_ID);
    assert_eq!(primary_submit.binding_count.get(), 2);
    assert_eq!(primary_submit.flags.get(), 0);
    assert_eq!(primary_submit.timeout_ns.get(), 9_000_000);
    let primary_binding0 = read_exact::<WireBinding>(&submit[48..80]).unwrap();
    let primary_binding1 = read_exact::<WireBinding>(&submit[80..112]).unwrap();
    assert_eq!(primary_binding0.buffer_id.get(), BUFFER_ID);
    assert_eq!(primary_binding0.offset.get(), 32);
    assert_eq!(primary_binding0.bytes.get(), 512);
    assert_eq!(primary_binding0.slot.get(), 4);
    assert_eq!(primary_binding0.access, 1);
    assert_eq!(primary_binding0.reserved, [0; 3]);
    assert_eq!(primary_binding1.buffer_id.get(), BUFFER_ID + 2);
    assert_eq!(primary_binding1.offset.get(), 64);
    assert_eq!(primary_binding1.bytes.get(), 1_024);
    assert_eq!(primary_binding1.slot.get(), 12);
    assert_eq!(primary_binding1.access, 2);
    assert_eq!(primary_binding1.reserved, [0; 3]);

    let clean_info = clean::DeviceInfo {
        uuid: *b"clean-to-primary",
        class: 3,
        vendor_id: 0xaabb_ccdd,
        device_id: 0x1122_3344,
        capabilities: 0x8877_6655_4433_2211,
        max_contexts: 7,
        max_buffers_per_context: 33,
        max_programs_per_context: 11,
        max_queues_per_context: 5,
        max_events_per_context: 66,
        max_bindings_per_submission: 99,
        max_buffer_bytes: 1 << 29,
        max_artifact_bytes: 1 << 21,
    };
    let info_context = response_context(clean::Opcode::GetDeviceInfo, None);
    let info = clean_response(
        StatusCode::OK.0,
        clean::ResponseBody::DeviceInfo(clean_info),
        info_context,
    );
    assert_primary_response_header(&info, StatusCode::OK.0, 76);
    let primary_info = read_exact::<WireDeviceInfo>(&info[16..]).unwrap();
    assert_eq!(primary_info.uuid, *b"clean-to-primary");
    assert_eq!(primary_info.class.get(), 3);
    assert_eq!(primary_info.reserved.get(), 0);
    assert_eq!(primary_info.vendor_id.get(), 0xaabb_ccdd);
    assert_eq!(primary_info.device_id.get(), 0x1122_3344);
    assert_eq!(primary_info.capabilities.get(), 0x8877_6655_4433_2211);
    assert_eq!(primary_info.max_contexts.get(), 7);
    assert_eq!(primary_info.max_buffers_per_context.get(), 33);
    assert_eq!(primary_info.max_programs_per_context.get(), 11);
    assert_eq!(primary_info.max_queues_per_context.get(), 5);
    assert_eq!(primary_info.max_events_per_context.get(), 66);
    assert_eq!(primary_info.max_bindings_per_submission.get(), 99);
    assert_eq!(primary_info.max_buffer_bytes.get(), 1 << 29);
    assert_eq!(primary_info.max_artifact_bytes.get(), 1 << 21);

    let object_context = response_context(clean::Opcode::CreateContext, None);
    let object = clean_response(
        StatusCode::OK.0,
        clean::ResponseBody::Object(CONTEXT_ID),
        object_context,
    );
    assert_primary_response_header(&object, StatusCode::OK.0, 8);
    assert_eq!(
        read_exact::<ObjectPayload>(&object[16..])
            .unwrap()
            .object_id
            .get(),
        CONTEXT_ID
    );

    let empty_context = response_context(clean::Opcode::DestroyContext, None);
    let empty = clean_response(StatusCode::OK.0, clean::ResponseBody::Empty, empty_context);
    assert_primary_response_header(&empty, StatusCode::OK.0, 0);

    let read_data = [0xc0_u8, 0xff, 0xee];
    let data_context = response_context(clean::Opcode::ReadBuffer, Some(read_data.len()));
    let data = clean_response(
        StatusCode::OK.0,
        clean::ResponseBody::Data(&read_data),
        data_context,
    );
    assert_primary_response_header(&data, StatusCode::OK.0, read_data.len());
    assert_eq!(&data[16..], &read_data);

    let submit_context = response_context(clean::Opcode::Submit, None);
    let event = clean_response(
        StatusCode::DEVICE_LOST.0,
        clean::ResponseBody::EventId(EVENT_ID),
        submit_context,
    );
    assert_primary_response_header(&event, StatusCode::DEVICE_LOST.0, 8);
    assert_eq!(
        read_exact::<SubmitResponse>(&event[16..])
            .unwrap()
            .event_id
            .get(),
        EVENT_ID
    );

    let event_context = response_context(clean::Opcode::PollEvent, None);
    let event_state = clean_response(
        StatusCode::OK.0,
        clean::ResponseBody::EventState(clean::EventState {
            kind: clean::EventKind::Cancelled,
            error: StatusCode::OK.0,
        }),
        event_context,
    );
    assert_primary_response_header(&event_state, StatusCode::OK.0, 8);
    let primary_event = read_exact::<WireEventState>(&event_state[16..]).unwrap();
    assert_eq!(primary_event.state.get(), KnownEventState::Cancelled as u16);
    assert_eq!(primary_event.error.get(), StatusCode::OK.0);
    assert_eq!(primary_event.reserved.get(), 0);
}
