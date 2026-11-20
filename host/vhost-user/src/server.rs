use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::AsFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use vhost::vhost_user::GpuBackend;
use vhost::vhost_user::message::*;
use vhost::vhost_user::{BackendReqHandler, Error, Result, VhostUserBackendReqHandlerMut};
use virtio_accel_core::{Accelerator, TransportByteSink, TransportByteSource};
use virtio_accel_device::{
    CommandOutcome, CommandProcessor, DeviceHealth, ObjectNamespace, ResetDisposition,
    ResourcePolicy,
};
use virtio_accel_transport::{ChainRegion, validate_chain_layout};
use virtio_queue::{Queue, QueueOwnedT, QueueT, desc::split::Descriptor};
use vm_memory::{Address, Bytes, FileOffset, GuestAddress, GuestMemory, GuestMemoryMmap};
use zerocopy::IntoBytes;

use crate::memory::{Port, Segment};
use crate::{CONFIG, FEATURES, MAX_DESCRIPTORS, QUEUE_SIZE};

fn invalid() -> Error {
    Error::InvalidParam
}
fn mapped<T, E>(result: std::result::Result<T, E>) -> Result<T> {
    result.map_err(|_| invalid())
}
fn index(value: u32) -> Result<()> {
    if value == 0 { Ok(()) } else { Err(invalid()) }
}
fn protocols() -> VhostUserProtocolFeatures {
    VhostUserProtocolFeatures::CONFIG
        | VhostUserProtocolFeatures::REPLY_ACK
        | VhostUserProtocolFeatures::RESET_DEVICE
}
fn overlaps(a: u64, alen: u64, b: u64, blen: u64) -> bool {
    a < b.saturating_add(blen) && b < a.saturating_add(alen)
}

/// A single virtual device. Native accelerator objects never move to a worker thread.
/// All control operations and queue processing are serialized by `serve`.
pub struct Host<A: Accelerator> {
    processor: CommandProcessor<A>,
    namespace: u16,
    owner: bool,
    negotiated: bool,
    queue: Queue,
    addresses: bool,
    enabled: bool,
    failed: bool,
    memory: Option<GuestMemoryMmap>,
    regions: Vec<VhostUserMemoryRegion>,
    kick: Option<File>,
    call: Option<File>,
    error: Option<File>,
}

impl<A: Accelerator> Host<A> {
    pub fn new(accelerator: A, policy: ResourcePolicy) -> std::result::Result<Self, String> {
        let processor = CommandProcessor::new(
            accelerator,
            &CONFIG,
            ObjectNamespace::new(1).unwrap(),
            policy,
        )
        .map_err(|error| format!("initialize accelerator: {error:?}"))?;
        Ok(Self {
            processor,
            namespace: 1,
            owner: false,
            negotiated: false,
            queue: Queue::new(QUEUE_SIZE).unwrap(),
            addresses: false,
            enabled: false,
            failed: false,
            memory: None,
            regions: Vec::new(),
            kick: None,
            call: None,
            error: None,
        })
    }

    fn stopped(&self) -> Result<()> {
        if self.enabled { Err(invalid()) } else { Ok(()) }
    }

    fn reset(&mut self) -> Result<()> {
        // No command runs concurrently and byte ports never outlive `process`.
        // Retire the ring and all notification FDs BEFORE releasing backend objects.
        self.enabled = false;
        self.failed = true;
        self.queue = Queue::new(QUEUE_SIZE).unwrap();
        self.addresses = false;
        self.kick = None;
        self.call = None;
        self.error = None;
        self.negotiated = false;
        self.namespace = self.namespace.checked_add(1).ok_or_else(invalid)?;
        let namespace = ObjectNamespace::new(self.namespace).ok_or_else(invalid)?;
        let report = mapped(self.processor.reset(namespace))?;
        if report.disposition != ResetDisposition::BackendReusable {
            return Err(invalid());
        }
        self.failed = false;
        Ok(())
    }

    fn translate(&self, address: u64, length: u64) -> Result<u64> {
        self.regions
            .iter()
            .find_map(|region| {
                let offset = address.checked_sub(region.user_addr)?;
                if offset.checked_add(length)? > region.memory_size {
                    return None;
                }
                region.guest_phys_addr.checked_add(offset)
            })
            .ok_or_else(invalid)
    }

    /// Process at most one queue's worth of work, then return to control messages.
    /// `true` asks the event loop to run another bounded batch without awaiting a kick.
    pub fn process_queue(&mut self) -> Result<bool> {
        if !self.enabled {
            return Ok(false);
        }
        let memory = self.memory.as_ref().ok_or_else(invalid)?;
        for _ in 0..QUEUE_SIZE {
            // Unlike pop_descriptor_chain, iter reports invalid producer-index jumps.
            let Some(chain) = mapped(self.queue.iter(memory))?.next() else {
                return Ok(false);
            };
            let head = chain.head_index();
            if head >= self.queue.size() {
                return Err(invalid());
            }
            let used = Self::command(&mut self.processor, &self.queue, memory, head)?;
            mapped(self.queue.add_used(memory, head, used))?;
            // rust-vmm supplies the publication fence but does not yet honor
            // the split ring's legacy NO_INTERRUPT hint itself.
            let notify = mapped(self.queue.needs_notification(memory))?;
            let flags: u16 = mapped(memory.read_obj(GuestAddress(self.queue.avail_ring())))?;
            if notify && u16::from_le(flags) & 1 == 0 {
                self.call
                    .as_ref()
                    .ok_or_else(invalid)?
                    .write_all(&1u64.to_ne_bytes())
                    .map_err(Error::SocketError)?;
            }
            // Generic QEMU test devices cannot faithfully expose backend NEEDS_RESET.
            // End the connection instead of publishing further work after device loss.
            if self.processor.health() != DeviceHealth::Running {
                return Err(invalid());
            }
        }
        Ok(true)
    }

    fn command(
        processor: &mut CommandProcessor<A>,
        queue: &Queue,
        memory: &GuestMemoryMmap,
        head: u16,
    ) -> Result<u32> {
        let mut segments = [Segment::default(); MAX_DESCRIPTORS];
        let mut regions = [ChainRegion::readable(0); MAX_DESCRIPTORS];
        let mut visited = [false; QUEUE_SIZE as usize];
        let rings = [
            (queue.desc_table(), u64::from(queue.size()) * 16),
            (queue.avail_ring(), 6 + u64::from(queue.size()) * 2),
            (queue.used_ring(), 6 + u64::from(queue.size()) * 8),
        ];
        let mut current = head;
        let mut count = 0;
        loop {
            if current >= queue.size() || count == MAX_DESCRIPTORS || visited[current as usize] {
                return Ok(0);
            }
            visited[current as usize] = true;
            let descriptor: Descriptor =
                match memory.read_obj(GuestAddress(rings[0].0 + u64::from(current) * 16)) {
                    Ok(value) => value,
                    Err(_) => return Ok(0),
                };
            let address = descriptor.addr().raw_value();
            let len = u64::from(descriptor.len());
            if descriptor.flags() & !3 != 0
                || len == 0
                || address.checked_add(len).is_none()
                || !memory.check_range(descriptor.addr(), len as usize)
                || rings
                    .iter()
                    .any(|(base, size)| overlaps(address, len, *base, *size))
                || segments[..count]
                    .iter()
                    .any(|s| overlaps(address, len, s.address, s.len))
            {
                return Ok(0);
            }
            segments[count] = Segment { address, len };
            regions[count] = if descriptor.is_write_only() {
                ChainRegion::writable(len)
            } else {
                ChainRegion::readable(len)
            };
            count += 1;
            if !descriptor.has_next() {
                break;
            }
            current = descriptor.next();
        }
        let Ok(layout) = validate_chain_layout(&regions[..count], MAX_DESCRIPTORS as u16) else {
            return Ok(0);
        };
        let split = usize::from(layout.readable_descriptors());
        let request = Port {
            memory,
            segments: &segments[..split],
            len: layout.readable_bytes(),
        };
        let mut response = Port {
            memory,
            segments: &segments[split..count],
            len: layout.writable_bytes(),
        };
        match processor.process(
            &regions[..count],
            &TransportByteSource::new(&request),
            &mut TransportByteSink::new(&mut response),
        ) {
            Ok(CommandOutcome::Response { used, .. }) => Ok(used),
            Ok(CommandOutcome::Unusable(_)) => Ok(0),
            // Preflight failure has no side effects; a response access failure may have.
            Err(virtio_accel_device::CommandProcessError::Preflight(_)) => Ok(0),
            Err(_) => Err(invalid()),
        }
    }
}

impl<A: Accelerator> VhostUserBackendReqHandlerMut for Host<A> {
    fn set_owner(&mut self) -> Result<()> {
        if self.owner {
            return Err(invalid());
        }
        self.owner = true;
        Ok(())
    }
    fn reset_owner(&mut self) -> Result<()> {
        self.reset()?;
        self.owner = false;
        self.memory = None;
        self.regions.clear();
        Ok(())
    }
    fn reset_device(&mut self) -> Result<()> {
        self.reset()
    }
    fn get_features(&mut self) -> Result<u64> {
        Ok(FEATURES)
    }
    fn set_features(&mut self, features: u64) -> Result<()> {
        self.stopped()?;
        if !self.owner || features & !FEATURES != 0 || features & FEATURES != FEATURES {
            return Err(invalid());
        }
        self.negotiated = true;
        Ok(())
    }
    fn set_mem_table(&mut self, regions: &[VhostUserMemoryRegion], files: Vec<File>) -> Result<()> {
        self.stopped()?;
        if !self.owner || regions.is_empty() || regions.len() > 32 || regions.len() != files.len() {
            return Err(invalid());
        }
        let mut ranges = Vec::with_capacity(regions.len());
        for (i, (region, file)) in regions.iter().zip(files).enumerate() {
            let end = region
                .mmap_offset
                .checked_add(region.memory_size)
                .ok_or_else(invalid)?;
            if region.memory_size == 0
                || region.user_addr.checked_add(region.memory_size).is_none()
                || region
                    .guest_phys_addr
                    .checked_add(region.memory_size)
                    .is_none()
                || mapped(file.metadata())?.len() < end
                || regions[..i].iter().any(|other| {
                    overlaps(
                        region.user_addr,
                        region.memory_size,
                        other.user_addr,
                        other.memory_size,
                    ) || overlaps(
                        region.guest_phys_addr,
                        region.memory_size,
                        other.guest_phys_addr,
                        other.memory_size,
                    )
                })
            {
                return Err(invalid());
            }
            ranges.push((
                GuestAddress(region.guest_phys_addr),
                mapped(usize::try_from(region.memory_size))?,
                Some(FileOffset::new(file, region.mmap_offset)),
            ));
        }
        ranges.sort_by_key(|range| range.0);
        let memory = mapped(GuestMemoryMmap::from_ranges_with_files(ranges))?;
        self.memory = Some(memory);
        self.regions = regions.to_vec();
        self.addresses = false;
        Ok(())
    }
    fn set_vring_num(&mut self, which: u32, num: u32) -> Result<()> {
        index(which)?;
        self.stopped()?;
        mapped(self.queue.try_set_size(mapped(u16::try_from(num))?))?;
        self.addresses = false;
        Ok(())
    }
    fn set_vring_addr(
        &mut self,
        which: u32,
        flags: VhostUserVringAddrFlags,
        descriptor: u64,
        used: u64,
        available: u64,
        _log: u64,
    ) -> Result<()> {
        index(which)?;
        self.stopped()?;
        if !flags.is_empty() {
            return Err(invalid());
        }
        let n = u64::from(self.queue.size());
        let d = self.translate(descriptor, n * 16)?;
        let a = self.translate(available, 6 + n * 2)?;
        let u = self.translate(used, 6 + n * 8)?;
        if overlaps(d, n * 16, a, 6 + n * 2)
            || overlaps(d, n * 16, u, 6 + n * 8)
            || overlaps(a, 6 + n * 2, u, 6 + n * 8)
        {
            return Err(invalid());
        }
        let mut queue = mapped(Queue::try_from(self.queue.state()))?;
        mapped(queue.try_set_desc_table_address(GuestAddress(d)))?;
        mapped(queue.try_set_avail_ring_address(GuestAddress(a)))?;
        mapped(queue.try_set_used_ring_address(GuestAddress(u)))?;
        self.queue = queue;
        self.addresses = true;
        Ok(())
    }
    fn set_vring_base(&mut self, which: u32, base: u32) -> Result<()> {
        index(which)?;
        self.stopped()?;
        let base = mapped(u16::try_from(base))?;
        self.queue.set_next_avail(base);
        self.queue.set_next_used(base);
        Ok(())
    }
    fn get_vring_base(&mut self, which: u32) -> Result<VhostUserVringState> {
        index(which)?;
        self.enabled = false;
        self.queue.set_ready(false);
        self.kick = None;
        self.call = None;
        self.error = None;
        Ok(VhostUserVringState::new(
            0,
            u32::from(self.queue.next_avail()),
        ))
    }
    fn set_vring_kick(&mut self, which: u8, fd: Option<File>) -> Result<()> {
        index(which.into())?;
        self.stopped()?;
        self.kick = fd;
        Ok(())
    }
    fn set_vring_call(&mut self, which: u8, fd: Option<File>) -> Result<()> {
        index(which.into())?;
        self.call = fd;
        Ok(())
    }
    fn set_vring_err(&mut self, which: u8, fd: Option<File>) -> Result<()> {
        index(which.into())?;
        self.error = fd;
        Ok(())
    }
    fn get_protocol_features(&mut self) -> Result<VhostUserProtocolFeatures> {
        Ok(protocols())
    }
    fn set_protocol_features(&mut self, features: u64) -> Result<()> {
        if features & !protocols().bits() == 0 {
            Ok(())
        } else {
            Err(invalid())
        }
    }
    fn get_queue_num(&mut self) -> Result<u64> {
        Ok(1)
    }
    fn set_vring_enable(&mut self, which: u32, enable: bool) -> Result<()> {
        index(which)?;
        if enable {
            if self.failed
                || !self.owner
                || !self.negotiated
                || !self.addresses
                || self.kick.is_none()
                || self.call.is_none()
            {
                return Err(invalid());
            }
            self.queue.set_ready(true);
            if !self
                .queue
                .is_valid(self.memory.as_ref().ok_or_else(invalid)?)
            {
                self.queue.set_ready(false);
                return Err(invalid());
            }
            // SET_VRING_BASE restored both indices. This synchronous engine retires
            // every popped chain before processing another control message; never
            // trust a guest write to the device-owned used.idx on resume.
        } else {
            self.queue.set_ready(false);
        }
        self.enabled = enable;
        Ok(())
    }
    fn get_config(
        &mut self,
        offset: u32,
        size: u32,
        _flags: VhostUserConfigFlags,
    ) -> Result<Vec<u8>> {
        let end = offset.checked_add(size).ok_or_else(invalid)?;
        CONFIG
            .as_bytes()
            .get(offset as usize..end as usize)
            .map(|bytes| bytes.to_vec())
            .ok_or_else(invalid)
    }
    fn set_config(
        &mut self,
        _offset: u32,
        _buf: &[u8],
        _flags: VhostUserConfigFlags,
    ) -> Result<()> {
        Err(invalid())
    }
    fn set_gpu_socket(&mut self, _gpu: GpuBackend) -> Result<()> {
        Err(invalid())
    }
    fn get_shared_object(&mut self, _uuid: VhostUserSharedMsg) -> Result<File> {
        Err(invalid())
    }
    fn get_inflight_fd(
        &mut self,
        _inflight: &VhostUserInflight,
    ) -> Result<(VhostUserInflight, File)> {
        Err(invalid())
    }
    fn set_inflight_fd(&mut self, _inflight: &VhostUserInflight, _file: File) -> Result<()> {
        Err(invalid())
    }
    fn get_max_mem_slots(&mut self) -> Result<u64> {
        Err(invalid())
    }
    fn add_mem_region(&mut self, _region: &VhostUserSingleMemoryRegion, _fd: File) -> Result<()> {
        Err(invalid())
    }
    fn remove_mem_region(&mut self, _region: &VhostUserSingleMemoryRegion) -> Result<()> {
        Err(invalid())
    }
    fn set_device_state_fd(
        &mut self,
        _direction: VhostTransferStateDirection,
        _phase: VhostTransferStatePhase,
        _fd: File,
    ) -> Result<Option<File>> {
        Err(invalid())
    }
    fn check_device_state(&mut self) -> Result<()> {
        Err(invalid())
    }
    fn get_shmem_config(&mut self) -> Result<VhostUserShMemConfig> {
        Err(invalid())
    }
    fn set_log_base(&mut self, _log: &VhostUserLog, _file: File) -> Result<()> {
        Err(invalid())
    }
}

/// Serve one QEMU connection. Disconnect terminates this device instance; reconnect and
/// migration are deliberately not advertised. The caller owns the socket directory.
/// No Send/Sync bounds are imposed on the accelerator or any of its native handles.
pub fn serve<A: Accelerator>(
    path: &Path,
    host: Host<A>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let listener = UnixListener::bind(path)?;
    eprintln!("virtio-accel: listening on {}", path.display());
    let (socket, _) = listener.accept()?;
    connected(socket, host)
}

fn connected<A: Accelerator>(
    socket: UnixStream,
    host: Host<A>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let control = socket.try_clone()?;
    // rust-vmm's handler API takes Arc. It is never cloned into another thread.
    #[allow(clippy::arc_with_non_send_sync)]
    let host = Arc::new(Mutex::new(host));
    let mut handler = BackendReqHandler::from_stream(socket, host.clone());
    let mut pending = false;
    let result: std::result::Result<(), Box<dyn std::error::Error>> = (|| {
        loop {
            let state = host
                .lock()
                .map_err(|_| io::Error::other("host lock poisoned"))?;
            let kick = state.kick.as_ref().filter(|_| state.enabled);
            let mut storage = [
                PollFd::new(control.as_fd(), PollFlags::POLLIN),
                PollFd::new(
                    kick.map_or(control.as_fd(), |fd| fd.as_fd()),
                    PollFlags::POLLIN,
                ),
            ];
            let fds = &mut storage[..if kick.is_some() { 2 } else { 1 }];
            match poll(
                fds,
                if pending {
                    PollTimeout::ZERO
                } else {
                    PollTimeout::NONE
                },
            ) {
                Ok(_) => {}
                Err(nix::errno::Errno::EINTR) => continue,
                Err(error) => return Err(error.into()),
            }
            let control_events = fds[0].revents().unwrap_or(PollFlags::empty());
            let kick_events = fds
                .get(1)
                .and_then(|fd| fd.revents())
                .unwrap_or(PollFlags::empty());
            let _ = fds;
            drop(state);
            // Control wins over kicks, so reset/stop cannot race a new batch.
            if control_events.contains(PollFlags::POLLIN) {
                match handler.handle_request() {
                    Ok(()) => {}
                    // EOF between complete messages is normal QEMU shutdown. A
                    // partial header/body still fails through the error arm.
                    Err(Error::Disconnected) => return Ok(()),
                    Err(error) => return Err(error.into()),
                }
                if host
                    .lock()
                    .map_err(|_| io::Error::other("host lock poisoned"))?
                    .failed
                {
                    return Err(io::Error::other("backend reset requires device discard").into());
                }
                pending = true;
                continue;
            }
            if control_events.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                return Err(io::Error::other("control socket failed").into());
            }
            if control_events.contains(PollFlags::POLLHUP) {
                return Ok(());
            }
            if kick_events.intersects(PollFlags::POLLHUP | PollFlags::POLLERR | PollFlags::POLLNVAL)
            {
                return Err(io::Error::other("kick fd failed").into());
            }
            let mut state = host
                .lock()
                .map_err(|_| io::Error::other("host lock poisoned"))?;
            if kick_events.contains(PollFlags::POLLIN) {
                let mut bytes = [0; 8];
                state
                    .kick
                    .as_ref()
                    .ok_or_else(invalid)?
                    .read_exact(&mut bytes)?;
            }
            pending = state.process_queue()?;
        }
    })();
    // Quiesce and attempt dependency-ordered resource teardown even on abrupt disconnect.
    let reset = host
        .lock()
        .map_err(|_| io::Error::other("host lock poisoned"))?
        .reset();
    result?;
    reset?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_mock::MockAccelerator;
    use virtio_accel_proto::{KnownOpcode, RequestFlags, RequestHeader};

    #[test]
    fn disconnect_between_messages_is_clean_but_truncation_fails() {
        let (server, client) = UnixStream::pair().unwrap();
        drop(client);
        assert!(connected(server, host()).is_ok());
        let (server, mut client) = UnixStream::pair().unwrap();
        client.write_all(&[1, 0, 0]).unwrap();
        drop(client);
        assert!(connected(server, host()).is_err());
    }

    fn host() -> Host<MockAccelerator> {
        Host::new(
            MockAccelerator::default(),
            ResourcePolicy::new(1 << 20, 1 << 20).unwrap(),
        )
        .unwrap()
    }

    fn setup() -> Host<MockAccelerator> {
        let mut host = host();
        host.owner = true;
        host.negotiated = true;
        host.memory = Some(GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap());
        host.queue.try_set_size(8).unwrap();
        host.queue
            .try_set_desc_table_address(GuestAddress(0x1000))
            .unwrap();
        host.queue
            .try_set_avail_ring_address(GuestAddress(0x2000))
            .unwrap();
        host.queue
            .try_set_used_ring_address(GuestAddress(0x3000))
            .unwrap();
        host.queue.set_ready(true);
        host
    }

    fn descriptor(
        host: &Host<MockAccelerator>,
        index: u16,
        address: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) {
        host.memory
            .as_ref()
            .unwrap()
            .write_obj(
                Descriptor::new(address, len, flags, next),
                GuestAddress(0x1000 + u64::from(index) * 16),
            )
            .unwrap();
    }

    fn request(host: &Host<MockAccelerator>) {
        let header = RequestHeader::new(KnownOpcode::GetDeviceInfo, RequestFlags::empty(), 0, 7);
        host.memory
            .as_ref()
            .unwrap()
            .write_slice(header.as_bytes(), GuestAddress(0x4000))
            .unwrap();
        descriptor(host, 0, 0x4000, 16, 1, 1);
        descriptor(host, 1, 0x5000, 256, 2, 0);
    }

    fn command(host: &mut Host<MockAccelerator>) -> u32 {
        Host::command(
            &mut host.processor,
            &host.queue,
            host.memory.as_ref().unwrap(),
            0,
        )
        .unwrap()
    }

    #[test]
    fn only_baseline_features_and_read_only_config() {
        let mut h = host();
        assert_eq!(h.get_features().unwrap(), (1 << 32) | (1 << 30));
        assert!(h.set_features(FEATURES).is_err());
        h.set_owner().unwrap();
        assert!(h.set_owner().is_err());
        assert!(h.set_features(FEATURES | 1 << 28).is_err());
        assert!(h.set_features(0).is_err());
        assert!(h.set_features(1 << 32).is_err());
        h.set_features(FEATURES).unwrap();
        assert_eq!(
            h.get_config(0, 16, VhostUserConfigFlags::empty()).unwrap(),
            CONFIG.as_bytes()
        );
        assert!(
            h.get_config(u32::MAX, 2, VhostUserConfigFlags::empty())
                .is_err()
        );
        assert!(
            h.set_config(0, &[0], VhostUserConfigFlags::empty())
                .is_err()
        );
        assert!(h.set_vring_num(1, 8).is_err());
        assert!(h.set_vring_num(0, 7).is_err());
        assert!(h.set_vring_num(0, 2048).is_err());
        assert!(h.set_vring_enable(0, true).is_err());
    }

    #[test]
    fn fragmented_frame_and_response_cross_descriptor_boundaries() {
        let mut h = setup();
        request(&h);
        descriptor(&h, 0, 0x4000, 7, 1, 1);
        descriptor(&h, 1, 0x4007, 9, 1, 2);
        descriptor(&h, 2, 0x5000, 9, 3, 3);
        descriptor(&h, 3, 0x5009, 247, 2, 0);
        assert_eq!(command(&mut h), 92);
        let mut response = [0; 16];
        h.memory
            .as_ref()
            .unwrap()
            .read_slice(&mut response, GuestAddress(0x5000))
            .unwrap();
        assert_eq!(&response[..8], &[0, 0, 0, 0, 76, 0, 0, 0]);
        assert_eq!(u64::from_le_bytes(response[8..].try_into().unwrap()), 7);
    }

    #[test]
    fn malformed_chains_never_dispatch_or_write() {
        for bad in [
            (0x4000, 16, 1, 0),       // loop
            (0x4000, 16, 4, 0),       // indirect
            (u64::MAX - 4, 16, 1, 1), // overflow
            (0x100000, 16, 1, 1),     // unmapped
            (0x4000, 0, 1, 1),        // empty
            (0x4000, 16, 8, 1),       // reserved flags
            (0x4000, 16, 1, 8),       // out-of-range next
        ] {
            let mut h = setup();
            request(&h);
            descriptor(&h, 0, bad.0, bad.1, bad.2, bad.3);
            assert_eq!(command(&mut h), 0);
            assert_eq!(h.processor.health(), DeviceHealth::Running);
            assert_eq!(
                h.memory
                    .as_ref()
                    .unwrap()
                    .read_obj::<u64>(GuestAddress(0x5000))
                    .unwrap(),
                0
            );
            request(&h);
            assert_eq!(command(&mut h), 92);
        }
    }

    #[test]
    fn overlapping_payloads_and_ring_aliases_are_rejected() {
        for address in [0x4000, 0x1000, 0x2000, 0x3000] {
            let mut h = setup();
            request(&h);
            descriptor(&h, 1, address, 256, 2, 0);
            assert_eq!(command(&mut h), 0);
        }
    }

    #[test]
    fn reset_retires_ring_before_processor_and_drops_notifications() {
        let mut h = setup();
        h.enabled = true;
        h.reset_device().unwrap();
        assert!(!h.enabled && !h.queue.ready() && !h.negotiated && !h.addresses);
        assert_eq!(h.namespace, 2);
        assert_eq!(h.processor.health(), DeviceHealth::Running);
        assert!(!h.process_queue().unwrap());
        h.namespace = u16::MAX;
        assert!(h.reset_device().is_err());
        assert!(h.failed);
    }

    #[test]
    fn memory_table_translates_qemu_addresses_and_is_atomic_on_error() {
        let mut h = host();
        h.set_owner().unwrap();
        let file = tempfile::tempfile().unwrap();
        file.set_len(0x10000).unwrap();
        let region = VhostUserMemoryRegion {
            guest_phys_addr: 0x40000000,
            memory_size: 0x10000,
            user_addr: 0x7f000000,
            mmap_offset: 0,
        };
        h.set_mem_table(&[region], vec![file.try_clone().unwrap()])
            .unwrap();
        assert_eq!(h.translate(0x7f002000, 16).unwrap(), 0x40002000);
        assert!(h.translate(0x7f00ffff, 2).is_err());
        assert!(
            h.set_mem_table(
                &[region, region],
                vec![file.try_clone().unwrap(), file.try_clone().unwrap()]
            )
            .is_err()
        );
        assert_eq!(h.regions.len(), 1);
        let oversized = VhostUserMemoryRegion {
            memory_size: 0x20000,
            ..region
        };
        assert!(h.set_mem_table(&[oversized], vec![file]).is_err());
        h.set_vring_num(0, 8).unwrap();
        h.set_vring_addr(
            0,
            VhostUserVringAddrFlags::empty(),
            0x7f001000,
            0x7f003000,
            0x7f002000,
            0,
        )
        .unwrap();
        assert_eq!(h.queue.desc_table(), 0x40001000);
        assert!(
            h.set_vring_addr(
                0,
                VhostUserVringAddrFlags::empty(),
                0x7f001000,
                0x7f001000,
                0x7f002000,
                0
            )
            .is_err()
        );
        h.enabled = true;
        assert!(h.set_mem_table(&[], vec![]).is_err());
        assert!(h.set_vring_num(0, 16).is_err());
    }

    #[test]
    fn wrapping_indices_and_interrupt_suppression_preserve_used_publication() {
        let mut h = setup();
        request(&h);
        h.enabled = true;
        h.call = Some(tempfile::tempfile().unwrap());
        h.queue.set_next_avail(u16::MAX);
        h.queue.set_next_used(u16::MAX);
        let memory = h.memory.as_ref().unwrap();
        memory
            .write_obj(1u16.to_le(), GuestAddress(0x2000))
            .unwrap();
        memory.write_obj(0u16, GuestAddress(0x2002)).unwrap();
        memory
            .write_obj(0u16, GuestAddress(0x2004 + 7 * 2))
            .unwrap();
        assert!(!h.process_queue().unwrap());
        assert_eq!(h.queue.next_avail(), 0);
        assert_eq!(h.queue.next_used(), 0);
        assert_eq!(h.call.as_ref().unwrap().metadata().unwrap().len(), 0);
        let memory = h.memory.as_ref().unwrap();
        assert_eq!(
            u32::from_le(
                memory
                    .read_obj::<u32>(GuestAddress(0x3008 + 7 * 8))
                    .unwrap()
            ),
            92
        );
        memory.write_obj(0u16, GuestAddress(0x2000)).unwrap();
        memory
            .write_obj(1u16.to_le(), GuestAddress(0x2002))
            .unwrap();
        assert!(!h.process_queue().unwrap());
        assert_eq!(h.call.as_ref().unwrap().metadata().unwrap().len(), 8);
    }

    #[test]
    fn invalid_available_index_is_a_transport_fault() {
        let mut h = setup();
        h.enabled = true;
        h.memory
            .as_ref()
            .unwrap()
            .write_obj(9u16.to_le(), GuestAddress(0x2002))
            .unwrap();
        assert!(h.process_queue().is_err());
    }
}
