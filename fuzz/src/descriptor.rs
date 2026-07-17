use virtio_accel_core::{Accelerator, TransportByteSink, TransportByteSource};
use virtio_accel_device::{CommandOutcome, CommandProcessor, ObjectNamespace, ResourcePolicy};
use virtio_accel_mock::MockAccelerator;
use virtio_accel_proto::{
    BASELINE_COMMAND_QUEUES, Le16, Le32, PROTOCOL_MAJOR, PROTOCOL_MINOR, ResponseHeader,
    StatusCode, WireConfig, read_exact,
};
use virtio_accel_split_queue::{Descriptor, DriverChain, SplitQueue, VIRTQ_DESC_F_WRITE};
use virtio_accel_transport::{
    DeviceChain, DeviceQueue, DriverChainBuffer, DriverQueue, QueueControl, QueueEpoch, QueueError,
    QueueSize, UsedLength, WritableBytes,
};

use crate::{Input, MAX_FRAME_BYTES};

const QUEUE_SIZE: u16 = 16;
const MAX_DESCRIPTOR_BYTES: usize = 64;
const CANONICAL_MARKER: u8 = 0xa5;

pub fn fuzz_descriptor_end_to_end(data: &[u8]) {
    let data = &data[..data.len().min(MAX_FRAME_BYTES + 32)];
    let mut input = Input::new(data);
    let mode = input.byte();
    let chain = if mode == CANONICAL_MARKER {
        canonical_chain(&mut input)
    } else {
        raw_chain(mode, &mut input)
    };
    let Ok(chain) = chain else {
        return;
    };

    let original_response = chain.validation().ok().and_then(|layout| {
        let len = usize::try_from(layout.writable_bytes()).ok()?;
        let mut bytes = vec![0_u8; len];
        chain.read_device_writable(0, &mut bytes).ok()?;
        Some(bytes)
    });

    let size = QueueSize::new(QUEUE_SIZE).unwrap();
    let mut queue = SplitQueue::new(size, QUEUE_SIZE).unwrap();
    QueueControl::configure(&mut queue, size).unwrap();
    QueueControl::set_ready(&mut queue, true).unwrap();
    if queue.inject_available(chain).is_err() {
        return;
    }
    let Some(mut device_chain) = DeviceQueue::pop_available(&mut queue).unwrap() else {
        panic!("published chain was not available");
    };

    let mut outcome = None;
    let used = match device_chain.io() {
        Err(_) => 0,
        Ok(io) => {
            let (regions, request, response) = io.into_parts();
            let source = TransportByteSource::new(request);
            let mut sink = TransportByteSink::new(response);
            let mut processor = processor();
            let result = processor.process(regions, &source, &mut sink).unwrap();
            let used = match result {
                CommandOutcome::Response {
                    request_id,
                    status,
                    used,
                } => {
                    assert!(status.is_known());
                    outcome = Some((request_id, status, used));
                    used
                }
                CommandOutcome::Unusable(_) => 0,
            };
            assert_eq!(
                processor.health(),
                virtio_accel_device::DeviceHealth::Running
            );
            used
        }
    };

    let capacity = device_chain
        .io()
        .ok()
        .map(|io| io.into_parts().2.len())
        .unwrap_or(0);
    let exceed_used = input.byte() & 1 != 0 && capacity < u64::from(u32::MAX);
    let completion_used = if exceed_used {
        u32::try_from(capacity + 1).unwrap()
    } else {
        used.min(u32::try_from(capacity).unwrap_or(u32::MAX))
    };

    if exceed_used {
        let error =
            DeviceQueue::complete(&mut queue, device_chain, UsedLength::new(completion_used))
                .unwrap_err();
        assert!(matches!(error, QueueError::UsedLengthExceeded { .. }));
        assert!(DriverQueue::pop_used(&mut queue).unwrap().is_none());
        let reclaimed = DriverQueue::reset(&mut queue, QueueEpoch::new(2).unwrap()).unwrap();
        assert_eq!(reclaimed.count(), 1);
        return;
    }

    DeviceQueue::complete(&mut queue, device_chain, UsedLength::new(completion_used)).unwrap();
    let returned = DriverQueue::pop_used(&mut queue)
        .unwrap()
        .expect("completed chain missing from used ring");
    assert_eq!(returned.used().get(), completion_used);

    if let Some(original) = original_response {
        let mut response = vec![0_u8; original.len()];
        returned
            .chain()
            .read_device_writable(0, &mut response)
            .unwrap();
        let used = usize::try_from(completion_used)
            .unwrap()
            .min(response.len());
        assert_eq!(&response[used..], &original[used..]);

        if let Some((request_id, status, outcome_used)) = outcome {
            assert_eq!(outcome_used, completion_used);
            if used >= 16 {
                let header = read_exact::<ResponseHeader>(&response[..16]).unwrap();
                assert_eq!(header.request_id.get(), request_id);
                assert_eq!(StatusCode(header.status.get()), status);
                assert_eq!(header.flags.get(), 0);
                assert_eq!(
                    usize::try_from(header.payload_bytes.get()).unwrap() + 16,
                    used
                );
            }
        } else {
            assert_eq!(used, 0);
            assert_eq!(response, original);
        }
    }
}

fn processor() -> CommandProcessor<MockAccelerator> {
    let config = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(QUEUE_SIZE),
        max_request_bytes: Le32::new(MAX_FRAME_BYTES as u32),
        max_response_bytes: Le32::new(512),
    };
    let accelerator = MockAccelerator::default();
    let _ = accelerator.device_info().unwrap();
    CommandProcessor::new(
        accelerator,
        &config,
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(4_096, 4_096).unwrap(),
    )
    .unwrap()
}

fn canonical_chain(
    input: &mut Input<'_>,
) -> Result<DriverChain, virtio_accel_split_queue::ChainBuildError> {
    let response_bytes = usize::from(input.u16() % 256) + 1;
    let request_controls = [input.byte(), input.byte(), input.byte()];
    let response_controls = [input.byte(), input.byte(), input.byte()];
    let mut frame = input.remaining();
    if frame.is_empty() {
        frame = &[0];
    }
    frame = &frame[..frame.len().min(MAX_FRAME_BYTES)];

    let mut descriptors = Vec::new();
    push_readable_segments(&mut descriptors, frame, &request_controls);
    push_writable_segments(&mut descriptors, response_bytes, &response_controls);
    DriverChain::direct(descriptors)
}

fn raw_chain(
    first: u8,
    input: &mut Input<'_>,
) -> Result<DriverChain, virtio_accel_split_queue::ChainBuildError> {
    let count = usize::from(first & 0x0f) + 1;
    let head = input.u16() % 20;
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::from(input.byte() % (MAX_DESCRIPTOR_BYTES as u8 + 1));
        let flags = input.u16();
        let next = input.u16() % 20;
        if input.byte() & 1 == 0 {
            let mut bytes = Vec::with_capacity(len);
            bytes.extend((0..len).map(|_| input.byte()));
            descriptors.push(Descriptor::raw(bytes, flags, next));
        } else {
            descriptors.push(Descriptor::unmapped(len as u64, flags, next));
        }
    }
    DriverChain::raw(descriptors, head)
}

fn push_readable_segments(descriptors: &mut Vec<Descriptor>, bytes: &[u8], controls: &[u8]) {
    let mut offset = 0;
    for control in controls {
        if offset >= bytes.len() {
            break;
        }
        let remaining = bytes.len() - offset;
        let count = (usize::from(*control) % remaining).saturating_add(1);
        descriptors.push(Descriptor::readable(bytes[offset..offset + count].to_vec()));
        offset += count;
    }
    if offset < bytes.len() {
        descriptors.push(Descriptor::readable(bytes[offset..].to_vec()));
    }
}

fn push_writable_segments(descriptors: &mut Vec<Descriptor>, bytes: usize, controls: &[u8]) {
    let mut remaining = bytes;
    for control in controls {
        if remaining == 0 {
            break;
        }
        let count = (usize::from(*control) % remaining).saturating_add(1);
        descriptors.push(Descriptor::raw(vec![0xa5; count], VIRTQ_DESC_F_WRITE, 0));
        remaining -= count;
    }
    if remaining != 0 {
        descriptors.push(Descriptor::writable(vec![0xa5; remaining]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_proto::{KnownOpcode, RequestFlags, RequestHeader};
    use zerocopy::IntoBytes;

    #[test]
    fn canonical_chain_reaches_command_engine() {
        let request = RequestHeader::new(KnownOpcode::GetDeviceInfo, RequestFlags::empty(), 0, 7);
        let mut data = vec![CANONICAL_MARKER, 92, 0, 1, 2, 3, 4, 5, 6];
        data.extend_from_slice(request.as_bytes());
        fuzz_descriptor_end_to_end(&data);
    }

    #[test]
    fn raw_loop_is_bounded() {
        fuzz_descriptor_end_to_end(&[1, 0, 0, 1, 1, 0, 0, 0, 0]);
    }
}
