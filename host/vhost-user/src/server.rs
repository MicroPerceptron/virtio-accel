use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::AsFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use vhost::vhost_user::message::*;
use vhost::vhost_user::{Backend, GpuBackend};
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

use crate::iommu::Iotlb;
use crate::memory::{Port, Segment};
use crate::{ACCESS_PLATFORM, CONFIG, FEATURES, MAX_DESCRIPTORS, QUEUE_SIZE};

fn invalid() -> Error {
    Error::InvalidParam
}
fn mapped<T, E>(result: std::result::Result<T, E>) -> Result<T> {
    result.map_err(|_| invalid())
}
fn index(value: u32) -> Result<()> {
    if value == 0 { Ok(()) } else { Err(invalid()) }
}
/// What one call to [`Host::process_queue`] left to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Nothing is available; wait for a kick.
    Idle,
    /// A bounded batch completed and more may be available.
    More,
    /// The device needs the translation of `iova` before it can go on. Nothing
    /// was consumed: the same work is found again once the frontend answers.
    Miss { iova: u64, write: bool },
}

/// How one chain ended.
enum Outcome {
    /// Completed, with this many bytes written.
    Used(u32),
    /// A buffer is not translated yet.
    Miss { iova: u64, write: bool },
}

/// The address space the queue's addresses name: the IOMMU domain's granted
/// ranges when there is one, all of guest RAM otherwise.
fn view<'a>(
    iotlb: &'a Option<Iotlb>,
    memory: &'a Option<GuestMemoryMmap>,
) -> Result<&'a GuestMemoryMmap> {
    iotlb
        .as_ref()
        .map(Iotlb::view)
        .or(memory.as_ref())
        .ok_or_else(invalid)
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
    /// The device sits behind the platform IOMMU: `VIRTIO_F_ACCESS_PLATFORM`
    /// is offered and required, and every address is an I/O virtual address.
    iommu: bool,
    /// The domain's granted ranges; replaces `memory` when `iommu`.
    iotlb: Option<Iotlb>,
    /// The channel misses are sent on, and a handle to wait for its replies.
    backend: Option<(Backend, UnixStream)>,
    reply_ack: bool,
    /// The address the frontend last answered a miss for. Finding the same
    /// address missing again means it has no translation: asking twice
    /// would ask forever.
    asked: Option<u64>,
    /// A ring address has no translation. The queue stays stopped until the
    /// driver describes it again.
    parked: bool,
    /// Behind an IOMMU the ring addresses arrive exactly as the driver wrote
    /// them. Whether they describe a queue at all is recorded here rather
    /// than answered with an error: see `set_vring_enable`.
    described: bool,
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
            iommu: false,
            iotlb: None,
            backend: None,
            reply_ack: false,
            asked: None,
            parked: false,
            described: false,
        })
    }

    /// Put the device behind the platform IOMMU.
    ///
    /// It then offers `VIRTIO_F_ACCESS_PLATFORM` and refuses to start without
    /// it, so a frontend that would hand it untranslated addresses — no IOMMU
    /// in front of the device, or `iommu_platform=off` — fails closed.
    pub fn behind_iommu(mut self) -> Self {
        self.iommu = true;
        self
    }

    fn features(&self) -> u64 {
        FEATURES | if self.iommu { ACCESS_PLATFORM } else { 0 }
    }

    fn protocols(&self) -> VhostUserProtocolFeatures {
        let base = VhostUserProtocolFeatures::CONFIG
            | VhostUserProtocolFeatures::REPLY_ACK
            | VhostUserProtocolFeatures::RESET_DEVICE;
        if self.iommu {
            base | VhostUserProtocolFeatures::BACKEND_REQ
        } else {
            base
        }
    }

    /// The channel to send a miss on, and the handle to await its reply on.
    fn miss_channel(&self) -> Result<(Backend, UnixStream)> {
        let (backend, replies) = self.backend.as_ref().ok_or_else(invalid)?;
        Ok((
            backend.clone(),
            replies.try_clone().map_err(Error::SocketError)?,
        ))
    }

    /// The frontend answered the miss for `iova`, with a translation or
    /// without one.
    fn answered(&mut self, iova: u64) {
        self.asked = Some(iova);
    }

    /// The first ring address without a translation, if any.
    fn ring_miss(&mut self) -> Option<Step> {
        let iotlb = self.iotlb.as_ref()?;
        let n = u64::from(self.queue.size());
        let (iova, write) = [
            (self.queue.desc_table(), n * 16, false),
            (self.queue.avail_ring(), 6 + n * 2, false),
            (self.queue.used_ring(), 6 + n * 8, true),
        ]
        .into_iter()
        .find_map(|(base, len, write)| Some((iotlb.missing(base, len, write)?, write)))?;
        if self.asked == Some(iova) {
            // The driver described a ring the domain does not map.
            self.asked = None;
            self.parked = true;
            return Some(Step::Idle);
        }
        Some(Step::Miss { iova, write })
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
        self.asked = None;
        self.parked = false;
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
    /// [`Step::More`] asks the event loop to run another bounded batch without awaiting a kick.
    pub fn process_queue(&mut self) -> Result<Step> {
        if !self.enabled || self.parked {
            return Ok(Step::Idle);
        }
        if let Some(step) = self.ring_miss() {
            return Ok(step);
        }
        let memory = view(&self.iotlb, &self.memory)?;
        for _ in 0..QUEUE_SIZE {
            // Unlike pop_descriptor_chain, iter reports invalid producer-index jumps.
            let Some(chain) = mapped(self.queue.iter(memory))?.next() else {
                return Ok(Step::Idle);
            };
            let head = chain.head_index();
            if head >= self.queue.size() {
                return Err(invalid());
            }
            let used = match Self::command(
                &mut self.processor,
                &self.queue,
                memory,
                self.iotlb.as_ref(),
                self.asked,
                head,
            )? {
                Outcome::Used(used) => used,
                Outcome::Miss { iova, write } => {
                    // Hand the chain back: it is taken again, whole, once
                    // the frontend has answered.
                    self.queue.go_to_previous_position();
                    return Ok(Step::Miss { iova, write });
                }
            };
            self.asked = None;
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
        Ok(Step::More)
    }

    /// Run the chain at `head`.
    ///
    /// Behind an IOMMU (`iotlb`), every buffer is checked against the
    /// domain before anything is dispatched, in the direction its descriptor
    /// declares. The first address without a translation is reported as a
    /// miss — unless it is `asked`, the address the frontend has just
    /// answered for, in which case it has none and the chain is unusable
    /// like any other chain naming memory the device cannot reach.
    fn command(
        processor: &mut CommandProcessor<A>,
        queue: &Queue,
        memory: &GuestMemoryMmap,
        iotlb: Option<&Iotlb>,
        asked: Option<u64>,
        head: u16,
    ) -> Result<Outcome> {
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
                return Ok(Outcome::Used(0));
            }
            visited[current as usize] = true;
            let descriptor: Descriptor =
                match memory.read_obj(GuestAddress(rings[0].0 + u64::from(current) * 16)) {
                    Ok(value) => value,
                    Err(_) => return Ok(Outcome::Used(0)),
                };
            let address = descriptor.addr().raw_value();
            let len = u64::from(descriptor.len());
            if descriptor.flags() & !3 != 0
                || len == 0
                || address.checked_add(len).is_none()
                || rings
                    .iter()
                    .any(|(base, size)| overlaps(address, len, *base, *size))
                || segments[..count]
                    .iter()
                    .any(|s| overlaps(address, len, s.address, s.len))
            {
                return Ok(Outcome::Used(0));
            }
            // Reachability last: a chain malformed in itself never asks the
            // frontend for anything.
            match iotlb {
                None if memory.check_range(descriptor.addr(), len as usize) => {}
                None => return Ok(Outcome::Used(0)),
                Some(iotlb) => {
                    let write = descriptor.is_write_only();
                    match iotlb.missing(address, len, write) {
                        None => {}
                        Some(iova) if asked == Some(iova) => return Ok(Outcome::Used(0)),
                        Some(iova) => return Ok(Outcome::Miss { iova, write }),
                    }
                }
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
            return Ok(Outcome::Used(0));
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
            Ok(CommandOutcome::Response { used, .. }) => Ok(Outcome::Used(used)),
            Ok(CommandOutcome::Unusable(_)) => Ok(Outcome::Used(0)),
            // Preflight failure has no side effects; a response access failure may have.
            Err(virtio_accel_device::CommandProcessError::Preflight(_)) => Ok(Outcome::Used(0)),
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
        self.iotlb = None;
        self.regions.clear();
        Ok(())
    }
    fn reset_device(&mut self) -> Result<()> {
        self.reset()
    }
    fn get_features(&mut self) -> Result<u64> {
        Ok(self.features())
    }
    fn set_features(&mut self, features: u64) -> Result<()> {
        self.stopped()?;
        if !self.owner {
            return Err(invalid());
        }
        // Exactly the offered set, or the device does not run. Behind an
        // IOMMU that includes ACCESS_PLATFORM: a frontend strips it when
        // nothing translates for this device.
        //
        // The value is the driver's, and the frontend forwards whatever the
        // driver acknowledged, including nothing at all. That is not a
        // protocol error by the frontend, so it is not answered by ending the
        // connection — which takes the device, and with some frontends the
        // whole machine, away from every other tenant of the guest. The
        // device stays connected and `set_vring_enable` leaves it stopped.
        self.negotiated = features == self.features();
        Ok(())
    }
    fn set_mem_table(&mut self, regions: &[VhostUserMemoryRegion], files: Vec<File>) -> Result<()> {
        self.stopped()?;
        if !self.owner || regions.is_empty() || regions.len() > 32 || regions.len() != files.len() {
            return Err(invalid());
        }
        for (i, (region, file)) in regions.iter().zip(&files).enumerate() {
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
        }
        if self.iommu {
            // Keep the files; map nothing until the domain grants it. A new
            // table starts a new, empty domain view.
            self.iotlb = Some(Iotlb::new(regions, files));
        } else {
            let mut ranges = Vec::with_capacity(regions.len());
            for (region, file) in regions.iter().zip(files) {
                ranges.push((
                    GuestAddress(region.guest_phys_addr),
                    mapped(usize::try_from(region.memory_size))?,
                    Some(FileOffset::new(file, region.mmap_offset)),
                ));
            }
            ranges.sort_by_key(|range| range.0);
            self.memory = Some(mapped(GuestMemoryMmap::from_ranges_with_files(ranges))?);
        }
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
        if self.iommu {
            // Already the device's own address space, and the driver's own
            // numbers: no frontend has mapped or checked them. Whether the
            // domain maps them is found out when the queue first runs.
            let spans = [
                (descriptor, n * 16),
                (available, 6 + n * 2),
                (used, 6 + n * 8),
            ];
            let mut queue = mapped(Queue::try_from(self.queue.state()))?;
            self.described = spans
                .iter()
                .all(|(base, len)| base.checked_add(*len).is_some())
                && !overlaps(spans[0].0, spans[0].1, spans[1].0, spans[1].1)
                && !overlaps(spans[0].0, spans[0].1, spans[2].0, spans[2].1)
                && !overlaps(spans[1].0, spans[1].1, spans[2].0, spans[2].1)
                && queue
                    .try_set_desc_table_address(GuestAddress(descriptor))
                    .is_ok()
                && queue
                    .try_set_avail_ring_address(GuestAddress(available))
                    .is_ok()
                && queue.try_set_used_ring_address(GuestAddress(used)).is_ok();
            if self.described {
                self.queue = queue;
            }
            self.addresses = true;
            return Ok(());
        }
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
        Ok(self.protocols())
    }
    fn set_protocol_features(&mut self, features: u64) -> Result<()> {
        if features & !self.protocols().bits() != 0 {
            return Err(invalid());
        }
        self.reply_ack = features & VhostUserProtocolFeatures::REPLY_ACK.bits() != 0;
        Ok(())
    }
    fn set_backend_req_fd(&mut self, backend: Backend) {
        // Offered only behind an IOMMU, where it carries misses. A handle
        // that cannot be watched leaves the device without the channel, and
        // the first miss then ends the connection.
        backend.set_reply_ack_flag(self.reply_ack);
        self.backend = backend
            .reply_stream()
            .ok()
            .filter(|_| self.iommu)
            .map(|replies| (backend, replies));
    }
    fn iotlb_msg(&mut self, message: &VhostUserIotlb) -> Result<()> {
        match (message.msg_type, self.iotlb.as_mut()) {
            (VhostUserIotlb::UPDATE, Some(iotlb)) => mapped(iotlb.update(message)),
            (VhostUserIotlb::INVALIDATE, Some(iotlb)) => {
                iotlb.invalidate(message.iova, message.size);
                Ok(())
            }
            // The frontend's IOMMU may invalidate before it has sent a
            // memory table; there is nothing to withdraw yet.
            (VhostUserIotlb::INVALIDATE, None) if self.iommu => Ok(()),
            _ => Err(invalid()),
        }
    }
    fn get_queue_num(&mut self) -> Result<u64> {
        Ok(1)
    }
    fn set_vring_enable(&mut self, which: u32, enable: bool) -> Result<()> {
        index(which)?;
        if enable {
            // The frontend's obligations: failing one is a protocol error.
            if self.failed
                || !self.owner
                || !self.addresses
                || self.kick.is_none()
                || self.call.is_none()
            {
                return Err(invalid());
            }
            // The driver's: a feature set this device does not run with, or
            // a queue that is not one. The device stays connected, stopped
            // and untouched until the driver resets and does better.
            if !self.negotiated || (self.iommu && !self.described) {
                self.queue.set_ready(false);
                self.enabled = false;
                return Ok(());
            }
            self.queue.set_ready(true);
            // Behind an IOMMU the rings are I/O virtual addresses, resolved
            // through the domain when the queue first runs (`ring_miss`).
            if self.iotlb.is_none()
                && !self
                    .queue
                    .is_valid(self.memory.as_ref().ok_or_else(invalid)?)
            {
                self.queue.set_ready(false);
                return Err(invalid());
            }
            self.asked = None;
            self.parked = false;
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
            pending = match state.process_queue()? {
                Step::Idle => false,
                Step::More => true,
                Step::Miss { iova, write } => {
                    let (backend, replies) = state.miss_channel()?;
                    drop(state);
                    if resolve(
                        &control,
                        &mut handler,
                        &host,
                        &backend,
                        &replies,
                        iova,
                        write,
                    )?
                    .is_none()
                    {
                        return Ok(());
                    }
                    host.lock()
                        .map_err(|_| io::Error::other("host lock poisoned"))?
                        .answered(iova);
                    true
                }
            };
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

/// Ask the frontend to translate `iova` and wait for its answer: `Some(true)`
/// when it supplied a translation, `Some(false)` when the address has none,
/// `None` when it disconnected.
///
/// The frontend answers in two steps on two sockets. It sends the translation
/// on the control channel and waits there for the acknowledgement, and only
/// then replies to the miss. So the control channel is served while the reply
/// is awaited — a daemon that blocked on the reply alone would deadlock with
/// the frontend — and it is served first, which is also the order the
/// frontend guarantees: by the time the reply is read, the update it refers to
/// has been installed. Any other control message arriving meanwhile (a stop, a
/// reset, an invalidation) is handled the same way; the queue re-derives what
/// it needs when it next runs.
fn resolve<A: Accelerator>(
    control: &UnixStream,
    handler: &mut BackendReqHandler<Mutex<Host<A>>>,
    host: &Mutex<Host<A>>,
    backend: &Backend,
    replies: &UnixStream,
    iova: u64,
    write: bool,
) -> std::result::Result<Option<bool>, Box<dyn std::error::Error>> {
    backend.send_iotlb_miss(
        iova,
        if write {
            VhostUserIotlb::ACCESS_WO
        } else {
            VhostUserIotlb::ACCESS_RO
        },
    )?;
    let failure = PollFlags::POLLERR | PollFlags::POLLNVAL;
    loop {
        let mut fds = [
            PollFd::new(control.as_fd(), PollFlags::POLLIN),
            PollFd::new(replies.as_fd(), PollFlags::POLLIN),
        ];
        match poll(&mut fds, PollTimeout::NONE) {
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(error) => return Err(error.into()),
        }
        let [control_events, reply_events] =
            fds.map(|fd| fd.revents().unwrap_or(PollFlags::empty()));
        if control_events.contains(PollFlags::POLLIN) {
            match handler.handle_request() {
                Ok(()) => {}
                Err(Error::Disconnected) => return Ok(None),
                Err(error) => return Err(error.into()),
            }
            if host
                .lock()
                .map_err(|_| io::Error::other("host lock poisoned"))?
                .failed
            {
                return Err(io::Error::other("backend reset requires device discard").into());
            }
            continue;
        }
        if control_events.intersects(failure) || reply_events.intersects(failure) {
            return Err(io::Error::other("socket failed awaiting a translation").into());
        }
        if reply_events.contains(PollFlags::POLLIN) {
            return Ok(Some(backend.recv_iotlb_miss_reply()?));
        }
        if control_events.contains(PollFlags::POLLHUP) || reply_events.contains(PollFlags::POLLHUP)
        {
            return Ok(None);
        }
    }
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
        match Host::command(
            &mut host.processor,
            &host.queue,
            host.memory.as_ref().unwrap(),
            None,
            None,
            0,
        )
        .unwrap()
        {
            Outcome::Used(used) => used,
            Outcome::Miss { .. } => panic!("a device with no IOMMU cannot miss"),
        }
    }

    #[test]
    fn only_baseline_features_and_read_only_config() {
        let mut h = host();
        assert_eq!(h.get_features().unwrap(), (1 << 32) | (1 << 30));
        assert!(h.set_features(FEATURES).is_err());
        h.set_owner().unwrap();
        assert!(h.set_owner().is_err());
        // A feature set the device does not run with is the driver's doing:
        // recorded, not answered by dropping the frontend.
        for refused in [FEATURES | 1 << 28, 0, 1 << 32, FEATURES | ACCESS_PLATFORM] {
            h.set_features(refused).unwrap();
            assert!(!h.negotiated);
        }
        h.set_features(FEATURES).unwrap();
        assert!(h.negotiated);
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
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
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
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
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
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
        assert_eq!(h.call.as_ref().unwrap().metadata().unwrap().len(), 8);
    }

    /// A device behind an IOMMU, its memory table one 64 KiB file whose page
    /// `n` is filled with byte `n` (pages 1 to 3, which the tests use for
    /// rings, are zero), and a queue of eight at I/O virtual addresses
    /// nothing has granted yet.
    fn behind_iommu() -> Host<MockAccelerator> {
        use std::io::Write as _;
        let mut h = host().behind_iommu();
        h.set_owner().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        for page in 0..16u8 {
            let fill = if (1..=3).contains(&page) { 0 } else { page };
            file.write_all(&[fill; 4096]).unwrap();
        }
        let region = VhostUserMemoryRegion::new(0x4000_0000, 0x10000, 0x7f00_0000, 0);
        h.set_mem_table(&[region], vec![file]).unwrap();
        h.set_features(FEATURES | ACCESS_PLATFORM).unwrap();
        h.set_vring_num(0, 8).unwrap();
        h.set_vring_addr(
            0,
            VhostUserVringAddrFlags::empty(),
            0x1_0000_1000,
            0x1_0000_3000,
            0x1_0000_2000,
            0,
        )
        .unwrap();
        h.kick = Some(tempfile::tempfile().unwrap());
        h.call = Some(tempfile::tempfile().unwrap());
        h.set_vring_enable(0, true).unwrap();
        h
    }

    fn grant(h: &mut Host<MockAccelerator>, iova: u64, page: u64, perm: u8) {
        let mut message = VhostUserIotlb::default();
        message.iova = iova;
        message.size = 4096;
        message.uaddr = 0x7f00_0000 + page * 4096;
        message.perm = perm;
        message.msg_type = VhostUserIotlb::UPDATE;
        h.iotlb_msg(&message).unwrap();
    }

    fn miss(h: &mut Host<MockAccelerator>) -> (u64, bool) {
        match h.process_queue().unwrap() {
            Step::Miss { iova, write } => (iova, write),
            step => panic!("expected a miss, got {step:?}"),
        }
    }

    #[test]
    fn a_device_behind_an_iommu_requires_the_platform_feature() {
        let mut h = host().behind_iommu();
        assert_eq!(h.get_features().unwrap(), FEATURES | ACCESS_PLATFORM);
        assert!(
            h.get_protocol_features()
                .unwrap()
                .contains(VhostUserProtocolFeatures::BACKEND_REQ)
        );
        h.set_owner().unwrap();
        // What a frontend sends when nothing translates for the device.
        h.set_features(FEATURES).unwrap();
        assert!(!h.negotiated);
        // And nothing is mapped by receiving a memory table.
        let file = tempfile::tempfile().unwrap();
        file.set_len(0x10000).unwrap();
        let region = VhostUserMemoryRegion::new(0x4000_0000, 0x10000, 0x7f00_0000, 0);
        h.set_mem_table(&[region], vec![file]).unwrap();
        assert!(h.memory.is_none());
        assert_eq!(h.iotlb.as_ref().unwrap().view().num_regions(), 0);
    }

    #[test]
    fn a_refused_feature_set_leaves_the_device_stopped_not_gone() {
        let mut h = behind_iommu();
        h.set_vring_enable(0, false).unwrap();
        h.set_features(FEATURES).unwrap();
        h.set_vring_enable(0, true).unwrap();
        assert!(!h.enabled && !h.queue.ready());
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
    }

    #[test]
    fn a_queue_that_is_not_one_leaves_the_device_stopped_not_gone() {
        for (descriptor, used, available) in [
            (u64::MAX - 8, 0x1_0000_3000, 0x1_0000_2000),  // wraps
            (0x1_0000_1000, 0x1_0000_1000, 0x1_0000_2000), // aliases
            (0x1_0000_1001, 0x1_0000_3000, 0x1_0000_2000), // misaligned
        ] {
            let mut h = behind_iommu();
            h.set_vring_enable(0, false).unwrap();
            h.set_vring_addr(
                0,
                VhostUserVringAddrFlags::empty(),
                descriptor,
                used,
                available,
                0,
            )
            .unwrap();
            h.set_vring_enable(0, true).unwrap();
            assert!(!h.enabled);
            assert_eq!(h.process_queue().unwrap(), Step::Idle);
        }
    }

    #[test]
    fn rings_and_buffers_are_resolved_through_the_domain_before_anything_runs() {
        let mut h = behind_iommu();
        // Rings first, in the direction each is used.
        assert_eq!(miss(&mut h), (0x1_0000_1000, false));
        grant(&mut h, 0x1_0000_1000, 1, VhostUserIotlb::ACCESS_RO);
        h.answered(0x1_0000_1000);
        assert_eq!(miss(&mut h), (0x1_0000_2000, false));
        grant(&mut h, 0x1_0000_2000, 2, VhostUserIotlb::ACCESS_RO);
        assert_eq!(miss(&mut h), (0x1_0000_3000, true));
        grant(&mut h, 0x1_0000_3000, 3, VhostUserIotlb::ACCESS_RW);
        assert_eq!(h.process_queue().unwrap(), Step::Idle);

        // One chain: a request at 0x…4000 and a response at 0x…5000, neither
        // granted. The table and the available ring are written through the
        // domain's own view, as a driver would through its mapping.
        let header = RequestHeader::new(KnownOpcode::GetDeviceInfo, RequestFlags::empty(), 0, 7);
        grant(&mut h, 0x1_0000_1000, 1, VhostUserIotlb::ACCESS_RW);
        grant(&mut h, 0x1_0000_2000, 2, VhostUserIotlb::ACCESS_RW);
        let view = h.iotlb.as_ref().unwrap().view();
        view.write_obj(
            Descriptor::new(0x1_0000_4000, 16, 1, 1),
            GuestAddress(0x1_0000_1000),
        )
        .unwrap();
        view.write_obj(
            Descriptor::new(0x1_0000_5000, 256, 2, 0),
            GuestAddress(0x1_0000_1010),
        )
        .unwrap();
        view.write_obj(0u16, GuestAddress(0x1_0000_2004)).unwrap();
        view.write_obj(1u16.to_le(), GuestAddress(0x1_0000_2002))
            .unwrap();

        // The request misses, read. Nothing has been consumed.
        assert_eq!(miss(&mut h), (0x1_0000_4000, false));
        assert_eq!(h.queue.next_avail(), 0);
        grant(&mut h, 0x1_0000_4000, 4, VhostUserIotlb::ACCESS_RW);
        h.iotlb
            .as_ref()
            .unwrap()
            .view()
            .write_slice(header.as_bytes(), GuestAddress(0x1_0000_4000))
            .unwrap();
        h.answered(0x1_0000_4000);
        // Then the response, written. Granted read-only it still misses …
        assert_eq!(miss(&mut h), (0x1_0000_5000, true));
        grant(&mut h, 0x1_0000_5000, 5, VhostUserIotlb::ACCESS_RO);
        h.answered(0x1_0000_4000);
        assert_eq!(miss(&mut h), (0x1_0000_5000, true));
        // … and once the frontend has answered for that very address without
        // making it writable, the chain is unusable: completed, empty,
        // nothing dispatched and nothing written.
        h.answered(0x1_0000_5000);
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
        assert_eq!(h.queue.next_avail(), 1);
        let view = h.iotlb.as_ref().unwrap().view();
        assert_eq!(
            view.read_obj::<u32>(GuestAddress(0x1_0000_3008)).unwrap(),
            0
        );
        assert_eq!(view.read_obj::<u8>(GuestAddress(0x1_0000_5000)).unwrap(), 5);

        // The same chain again with the response writable: it runs.
        view.write_obj(2u16.to_le(), GuestAddress(0x1_0000_2002))
            .unwrap();
        view.write_obj(0u16, GuestAddress(0x1_0000_2006)).unwrap();
        grant(&mut h, 0x1_0000_5000, 5, VhostUserIotlb::ACCESS_RW);
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
        let view = h.iotlb.as_ref().unwrap().view();
        assert_eq!(
            u32::from_le(view.read_obj::<u32>(GuestAddress(0x1_0000_3010)).unwrap()),
            92
        );
    }

    #[test]
    fn a_ring_the_domain_does_not_map_parks_the_queue() {
        let mut h = behind_iommu();
        let (iova, _) = miss(&mut h);
        h.answered(iova);
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
        assert!(h.parked);
        // Granting it later does not revive a parked queue; describing the
        // queue again does.
        grant(&mut h, iova, 1, VhostUserIotlb::ACCESS_RW);
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
        h.set_vring_enable(0, false).unwrap();
        h.set_vring_enable(0, true).unwrap();
        assert!(matches!(h.process_queue().unwrap(), Step::Miss { .. }));
    }

    #[test]
    fn an_invalidated_ring_is_asked_for_again() {
        let mut h = behind_iommu();
        for (iova, page) in [(0x1_0000_1000, 1), (0x1_0000_2000, 2), (0x1_0000_3000, 3)] {
            grant(&mut h, iova, page, VhostUserIotlb::ACCESS_RW);
        }
        assert_eq!(h.process_queue().unwrap(), Step::Idle);
        let mut message = VhostUserIotlb::default();
        message.iova = 0x1_0000_2000;
        message.size = 4096;
        message.msg_type = VhostUserIotlb::INVALIDATE;
        h.iotlb_msg(&message).unwrap();
        assert_eq!(miss(&mut h), (0x1_0000_2000, false));
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
