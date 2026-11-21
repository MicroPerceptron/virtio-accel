//! Both halves against each other through memory, against the reference
//! model as oracle, and against a peer that lies.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::AtomicU64;
use std::vec;
use std::vec::Vec;

use virtio_accel_split_queue::{Descriptor, DriverChain as ModelChain, SplitQueue};
use virtio_accel_transport::{
    ByteAccessError, ChainError, ChainLayoutError, ChainRegion, DeviceChain, DeviceQueue,
    DriverChainBuffer, DriverQueue, MalformedChain, NotificationHint, NotificationRecheck,
    PublishErrorKind, QueueConfigError, QueueControl, QueueEpoch, QueueError, QueuePort, QueueSize,
    ReadableBytes, UsedLength, WritableBytes,
};

use super::*;

/// Where the arena sits in the device's address space: nowhere near zero,
/// so an offset mistaken for an address is refused.
const BASE: u64 = 0x1_0000_0000;
const ARENA: usize = 256 * 1024;

/// Shared memory with no `unsafe`: one buffer both halves reach by device
/// address, with ranges that can be made read-only or taken away.
#[derive(Debug)]
struct Arena {
    bytes: RefCell<Vec<u8>>,
    read_only: RefCell<Vec<(u64, u64)>>,
    gone: RefCell<Vec<(u64, u64)>>,
    writes: Cell<u64>,
}

#[derive(Clone, Debug)]
struct Memory(Rc<Arena>);

impl Memory {
    fn new() -> Self {
        Self(Rc::new(Arena {
            bytes: RefCell::new(vec![0; ARENA]),
            read_only: RefCell::default(),
            gone: RefCell::default(),
            writes: Cell::new(0),
        }))
    }

    fn protect(&self, at: u64, len: u64) {
        self.0.read_only.borrow_mut().push((at, len));
    }

    fn remove(&self, at: u64, len: u64) {
        self.0.gone.borrow_mut().push((at, len));
    }

    fn restore(&self) {
        self.0.gone.borrow_mut().clear();
    }

    fn get(&self, at: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0; len];
        self.read(at, &mut out).expect("in the arena");
        out
    }

    fn put(&self, at: u64, bytes: &[u8]) {
        let from = (at - BASE) as usize;
        self.0.bytes.borrow_mut()[from..from + bytes.len()].copy_from_slice(bytes);
    }

    fn offset(&self, at: u64, len: u64, access: Access) -> Result<usize, MemoryFault> {
        if self.admits(at, len, access) {
            Ok((at - BASE) as usize)
        } else {
            Err(MemoryFault::Range)
        }
    }
}

fn touches(ranges: &[(u64, u64)], at: u64, len: u64) -> bool {
    ranges
        .iter()
        .any(|&(start, span)| at < start + span && start < at + len)
}

impl SharedMemory for Memory {
    fn admits(&self, at: u64, len: u64, access: Access) -> bool {
        let Some(end) = at.checked_add(len) else {
            return false;
        };
        at >= BASE
            && end <= BASE + ARENA as u64
            && !touches(&self.0.gone.borrow(), at, len)
            && (access == Access::Read || !touches(&self.0.read_only.borrow(), at, len))
    }

    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), MemoryFault> {
        let from = self.offset(at, target.len() as u64, Access::Read)?;
        target.copy_from_slice(&self.0.bytes.borrow()[from..from + target.len()]);
        Ok(())
    }

    fn write(&self, at: u64, source: &[u8]) -> Result<(), MemoryFault> {
        let from = self.offset(at, source.len() as u64, Access::Write)?;
        self.0.bytes.borrow_mut()[from..from + source.len()].copy_from_slice(source);
        self.0.writes.set(self.0.writes.get() + 1);
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

const N: usize = 256;
const R: usize = 8;
type Driver = DriverRing<Memory, N, R>;
type Device = DeviceRing<Memory, Rc<AtomicU64>, N, R>;
type Chain = DriverChain<Memory, R>;

fn size(value: u16) -> QueueSize {
    QueueSize::new(value).expect("a power of two")
}

/// The queue's three parts at the bottom of the arena; buffers above them.
const BUFFERS: u64 = BASE + 0x1_0000;

/// One memory with a ready queue of `queue` descriptors laid out in it, and
/// both halves over it. Chains are at most `limit` descriptors.
struct Pair {
    memory: Memory,
    at: RingAddresses,
    driver: Driver,
    device: Device,
    /// Where the next buffer goes: buffers are never reused in a test.
    bump: u64,
}

impl Pair {
    fn ready(queue: u16, limit: u16) -> Self {
        let memory = Memory::new();
        let (at, _) = RingAddresses::contiguous(BASE, size(queue)).expect("aligned");
        let mut driver = Driver::new(memory.clone(), size(N as u16), limit).expect("limits fit");
        driver.configure(size(queue)).expect("configured");
        driver.set_addresses(at).expect("addresses");
        driver.set_ready(true).expect("a layout in the arena");
        let mut device =
            Device::new(memory.clone(), Rc::default(), size(N as u16), limit).expect("limits fit");
        device.configure(size(queue)).expect("configured");
        device.set_addresses(at).expect("addresses");
        device.set_ready(true).expect("a layout in the arena");
        Self {
            memory,
            at,
            driver,
            device,
            bump: BUFFERS,
        }
    }

    /// `len` bytes of buffer nothing else names, with a gap after it so no
    /// two buffers are adjacent.
    fn space(&mut self, len: u32) -> Region {
        let address = self.bump;
        self.bump += u64::from(len) + 24;
        Region { address, len }
    }

    /// A chain of fresh buffers: readable lengths, then writable ones.
    fn chain(&mut self, readable: &[u32], writable: &[u32]) -> Chain {
        let readable: Vec<Region> = readable.iter().map(|len| self.space(*len)).collect();
        let writable: Vec<Region> = writable.iter().map(|len| self.space(*len)).collect();
        Chain::new(self.memory.clone(), &readable, &writable).expect("a chain")
    }

    /// The device takes one chain, answers it with `response` and uses it.
    fn serve(&mut self, response: &[u8]) -> (Vec<u8>, NotificationHint) {
        let mut chain = self
            .device
            .pop_available()
            .expect("the ring")
            .expect("a chain");
        let request = {
            let (_, request, sink) = chain.io().expect("well formed").into_parts();
            let mut bytes = vec![0; request.len() as usize];
            request.read_at(0, &mut bytes).expect("readable");
            sink.write_at(0, response).expect("writable");
            bytes
        };
        let hint = self
            .device
            .complete(chain, UsedLength::new(response.len() as u32))
            .expect("used");
        (request, hint)
    }

    // What a driver written by hand would do, for the tests that need a
    // driver no `DriverRing` would be.

    fn descriptor(&self, index: u16, address: u64, len: u32, flags: u16, next: u16) {
        self.memory.put(
            self.at.descriptors + u64::from(index) * 16,
            &layout::descriptor(address, len, flags, next),
        );
    }

    fn offer(&self, slot: u16, head: u16, index: u16) {
        self.memory.put(
            self.at.available + 4 + u64::from(slot) * 2,
            &head.to_le_bytes(),
        );
        self.memory.put(self.at.available + 2, &index.to_le_bytes());
    }

    // And what a device written by hand would do.

    fn use_entry(&self, slot: u16, id: u32, len: u32, index: u16) {
        let entry = self.at.used + 4 + u64::from(slot) * 8;
        self.memory.put(entry, &id.to_le_bytes());
        self.memory.put(entry + 4, &len.to_le_bytes());
        self.memory.put(self.at.used + 2, &index.to_le_bytes());
    }
}

// -- layout -------------------------------------------------------------------

#[test]
fn the_three_parts_are_sized_and_placed_as_the_layout_says() {
    let queue = size(8);
    assert_eq!(RingAddresses::descriptor_table_bytes(queue), 128);
    assert_eq!(RingAddresses::available_ring_bytes(queue), 20);
    assert_eq!(RingAddresses::used_ring_bytes(queue), 68);
    let (at, bytes) = RingAddresses::contiguous(0x1000, queue).unwrap();
    assert_eq!(
        (at.descriptors, at.available, at.used),
        (0x1000, 0x1080, 0x1094)
    );
    assert_eq!(bytes, 0x94 + 68);
    assert_eq!(
        RingAddresses::contiguous(0x1008, queue),
        None,
        "a misaligned table"
    );
    assert_eq!(
        RingAddresses::contiguous(u64::MAX - 15, queue),
        None,
        "the end wraps"
    );
    // The largest queue there is still fits its arithmetic.
    let (at, bytes) = RingAddresses::contiguous(0, size(32_768)).unwrap();
    assert_eq!(at.used % 4, 0);
    assert_eq!(bytes, at.used + 4 + 32_768 * 8);
}

#[test]
fn a_queue_set_up_wrongly_is_refused_by_both_halves() {
    let memory = Memory::new();
    let good = RingAddresses::contiguous(BASE, size(8)).unwrap().0;
    // Each misplaced part is moved well clear of the other two, so that it
    // is refused for the one thing wrong with it.
    let far = BASE + 0x8000;
    let cases = [
        (
            "a misaligned descriptor table",
            RingAddresses {
                descriptors: far + 8,
                ..good
            },
        ),
        (
            "a misaligned available ring",
            RingAddresses {
                available: far + 1,
                ..good
            },
        ),
        (
            "a misaligned used ring",
            RingAddresses {
                used: far + 2,
                ..good
            },
        ),
        (
            "a part outside the memory",
            RingAddresses {
                used: BASE + ARENA as u64,
                ..good
            },
        ),
        (
            "a part that runs off the end",
            RingAddresses {
                used: BASE + ARENA as u64 - 64,
                ..good
            },
        ),
        (
            "a part below the memory",
            RingAddresses {
                descriptors: 0,
                ..good
            },
        ),
        (
            "two parts on the same bytes",
            RingAddresses {
                available: good.descriptors + 16,
                ..good
            },
        ),
        (
            "the used ring inside the table",
            RingAddresses {
                used: good.descriptors + 32,
                ..good
            },
        ),
        (
            "a part that ends inside the next",
            RingAddresses {
                used: good.available + 16,
                ..good
            },
        ),
        (
            "a part that wraps",
            RingAddresses {
                used: u64::MAX - 3,
                ..good
            },
        ),
    ];
    for (name, at) in cases {
        let mut driver = Driver::new(memory.clone(), size(8), 4).unwrap();
        driver.configure(size(8)).unwrap();
        driver.set_addresses(at).unwrap();
        assert_eq!(
            driver.set_ready(true),
            Err(QueueError::Transport(DriverRingError::Layout)),
            "{name}"
        );
        let mut device = Device::new(memory.clone(), Rc::default(), size(8), 4).unwrap();
        device.configure(size(8)).unwrap();
        device.set_addresses(at).unwrap();
        assert_eq!(
            device.set_ready(true),
            Err(QueueError::Transport(DeviceRingError::Layout)),
            "{name}"
        );
    }

    // A device that cannot write its own ring cannot run the queue; a
    // driver's memory is its own, so it needs all three writable.
    memory.protect(good.used, 68);
    let mut device = Device::new(memory.clone(), Rc::default(), size(8), 4).unwrap();
    device.configure(size(8)).unwrap();
    device.set_addresses(good).unwrap();
    assert_eq!(
        device.set_ready(true),
        Err(QueueError::Transport(DeviceRingError::Layout))
    );

    // The limits and the order of configuration.
    assert!(
        Driver::new(memory.clone(), size(8), 1).is_none(),
        "a chain is two descriptors"
    );
    assert!(
        Driver::new(memory.clone(), size(8), 9).is_none(),
        "longer than the chain type"
    );
    assert!(
        Driver::new(memory.clone(), size(4), 8).is_none(),
        "longer than the queue"
    );
    assert!(
        DriverRing::<Memory, 4, 8>::new(memory.clone(), size(8), 2).is_none(),
        "more than N"
    );
    assert!(Device::new(memory.clone(), Rc::default(), size(8), 1).is_none());
    assert!(
        DeviceRing::<Memory, Rc<AtomicU64>, 4, 8>::new(memory.clone(), Rc::default(), size(8), 2)
            .is_none()
    );
    let mut driver = Driver::new(Memory::new(), size(16), 4).unwrap();
    assert_eq!(
        driver.set_ready(true),
        Err(QueueError::InvalidConfiguration(
            QueueConfigError::ReadyWithoutSize
        ))
    );
    assert_eq!(
        driver.configure(size(32)),
        Err(QueueError::InvalidConfiguration(
            QueueConfigError::SizeExceedsMaximum
        ))
    );
    assert_eq!(
        driver.configure(size(2)),
        Err(QueueError::Transport(
            DriverRingError::ChainLimitExceedsQueue
        ))
    );
    driver.configure(size(8)).unwrap();
    driver.set_addresses(good).unwrap();
    driver.set_ready(true).unwrap();
    assert_eq!(
        driver.configure(size(8)),
        Err(QueueError::Transport(DriverRingError::Busy))
    );
    assert_eq!(
        driver.set_addresses(good),
        Err(QueueError::Transport(DriverRingError::Busy))
    );
}

// -- through memory ----------------------------------------------------------

#[test]
fn a_chain_goes_out_and_comes_back_through_memory() {
    let mut pair = Pair::ready(8, 4);
    let mut chain = pair.chain(&[5, 11], &[3, 29]);
    chain.write_device_readable(0, b"sixteen bytes in").unwrap();
    let published = pair.driver.publish(chain).expect("room");
    assert_eq!(published.notification(), NotificationHint::Notify);
    assert_eq!(
        (pair.driver.outstanding(), pair.driver.free_descriptors()),
        (1, 4)
    );

    let mut taken = pair.device.pop_available().unwrap().expect("published");
    assert_eq!(taken.id(), published.id());
    assert_eq!(pair.device.in_flight(), 1);
    {
        let (regions, request, response) = taken.io().unwrap().into_parts();
        assert_eq!(
            regions,
            [
                ChainRegion::readable(5),
                ChainRegion::readable(11),
                ChainRegion::writable(3),
                ChainRegion::writable(29),
            ]
        );
        let mut bytes = [0; 16];
        request.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"sixteen bytes in");
        // Across the seam between two descriptors, in both directions.
        let mut seam = [0; 4];
        request.read_at(3, &mut seam).unwrap();
        assert_eq!(&seam, b"teen");
        response.write_at(0, b"twenty bytes answer!").unwrap();
        assert_eq!(
            request.read_at(13, &mut seam),
            Err(ByteAccessError::OutOfBounds)
        );
        assert_eq!(
            response.write_at(30, &[0; 3]),
            Err(ByteAccessError::OutOfBounds)
        );
    }
    assert_eq!(
        pair.device.complete(taken, UsedLength::new(20)),
        Ok(NotificationHint::Notify)
    );
    assert_eq!(pair.device.in_flight(), 0);

    let used = pair.driver.pop_used().unwrap().expect("used");
    assert_eq!(
        (used.id(), used.used()),
        (published.id(), UsedLength::new(20))
    );
    let (_, _, chain) = used.into_parts();
    let mut answer = [0; 20];
    chain.read_device_writable(0, &mut answer).unwrap();
    assert_eq!(&answer, b"twenty bytes answer!");
    assert_eq!(
        (pair.driver.outstanding(), pair.driver.free_descriptors()),
        (0, 8)
    );
    assert!(pair.driver.pop_used().unwrap().is_none());
}

/// A driver in C and a device in another language meet in these bytes, so
/// they are held to the specification's and not to each other.
#[test]
fn what_is_in_memory_is_the_specifications_layout() {
    let mut pair = Pair::ready(8, 4);
    let chain = pair.chain(&[24], &[32, 8]);
    let regions: Vec<Region> = chain
        .readable()
        .iter()
        .chain(chain.writable())
        .copied()
        .collect();
    let head = pair.driver.publish(chain).unwrap().id().token() as u16;

    // Descriptors: addr u64, len u32, flags u16, next u16, little-endian.
    let mut index = head;
    for (position, region) in regions.iter().enumerate() {
        let bytes = pair
            .memory
            .get(pair.at.descriptors + u64::from(index) * 16, 16);
        assert_eq!(bytes[..8], region.address.to_le_bytes());
        assert_eq!(bytes[8..12], region.len.to_le_bytes());
        let flags = u16::from_le_bytes([bytes[12], bytes[13]]);
        let last = position == regions.len() - 1;
        assert_eq!(flags & VIRTQ_DESC_F_NEXT != 0, !last);
        assert_eq!(flags & VIRTQ_DESC_F_WRITE != 0, position > 0);
        index = u16::from_le_bytes([bytes[14], bytes[15]]);
    }
    // Available ring: flags u16, idx u16, ring[u16].
    assert_eq!(
        pair.memory.get(pair.at.available, 6),
        [0, 0, 1, 0, head as u8, 0]
    );

    pair.serve(&[7; 9]);
    // Used ring: flags u16, idx u16, ring[{id u32, len u32}].
    assert_eq!(
        pair.memory.get(pair.at.used, 12),
        [0, 0, 1, 0, head as u8, 0, 0, 0, 9, 0, 0, 0]
    );
}

#[test]
fn a_borrowed_memory_is_a_memory() {
    let memory = Memory::new();
    let (at, _) = RingAddresses::contiguous(BASE, size(4)).unwrap();
    let mut driver = DriverRing::<&Memory, 4, 2>::new(&memory, size(4), 2).unwrap();
    driver.configure(size(4)).unwrap();
    driver.set_addresses(at).unwrap();
    driver.set_ready(true).unwrap();
    let chain = DriverChain::<&Memory, 2>::new(
        &memory,
        &[Region {
            address: BUFFERS,
            len: 8,
        }],
        &[Region {
            address: BUFFERS + 64,
            len: 8,
        }],
    )
    .unwrap();
    assert!(driver.publish(chain).is_ok());
    assert_eq!(memory.get(at.available + 2, 2), [1, 0]);
}

// -- the oracle --------------------------------------------------------------

/// A chain's shape as plain numbers: `(length, flags, next)` per descriptor.
type Shape = &'static [(u32, u16, u16)];

const NEXT: u16 = VIRTQ_DESC_F_NEXT;
const WRITE: u16 = VIRTQ_DESC_F_WRITE;

/// Shapes a driver may publish, good and bad, with what the protocol calls
/// each. The chain limit in these tests is four descriptors.
const SHAPES: &[(&str, Shape, u16, Option<MalformedChain>)] = &[
    (
        "one readable, one writable",
        &[(24, NEXT, 1), (32, WRITE, 0)],
        0,
        None,
    ),
    (
        "two and two",
        &[
            (8, NEXT, 1),
            (8, NEXT, 2),
            (8, NEXT | WRITE, 3),
            (8, WRITE, 0),
        ],
        0,
        None,
    ),
    (
        "a head that is not descriptor zero",
        &[(16, WRITE, 0), (16, NEXT, 0)],
        1,
        None,
    ),
    (
        "no writable descriptor",
        &[(8, NEXT, 1), (8, 0, 0)],
        0,
        Some(MalformedChain::Direction),
    ),
    (
        "no readable descriptor",
        &[(8, NEXT | WRITE, 1), (8, WRITE, 0)],
        0,
        Some(MalformedChain::Direction),
    ),
    (
        "readable after writable",
        &[(8, NEXT, 1), (8, NEXT | WRITE, 2), (8, 0, 0)],
        0,
        Some(MalformedChain::Direction),
    ),
    (
        "a single descriptor",
        &[(8, WRITE, 0)],
        0,
        Some(MalformedChain::DescriptorCount),
    ),
    (
        "longer than the chain limit",
        &[
            (8, NEXT, 1),
            (8, NEXT, 2),
            (8, NEXT, 3),
            (8, NEXT | WRITE, 4),
            (8, WRITE, 0),
        ],
        0,
        Some(MalformedChain::DescriptorCount),
    ),
    (
        "a zero-length descriptor",
        &[(0, NEXT, 1), (8, WRITE, 0)],
        0,
        Some(MalformedChain::ZeroLength),
    ),
    (
        "a loop",
        &[(8, NEXT, 1), (8, NEXT | WRITE, 0)],
        0,
        Some(MalformedChain::DescriptorLoop),
    ),
    (
        "a descriptor that follows itself",
        &[(8, NEXT, 0)],
        0,
        Some(MalformedChain::DescriptorLoop),
    ),
    (
        "an indirect descriptor",
        &[(16, VIRTQ_DESC_F_INDIRECT, 0)],
        0,
        Some(MalformedChain::IndirectUnsupported),
    ),
    (
        "a flag the profile does not have",
        &[(8, NEXT | 0x80, 1), (8, WRITE, 0)],
        0,
        Some(MalformedChain::DescriptorFlags),
    ),
];

/// Descriptors as a driver writes them: `(address, length, flags, next)`.
type Placed = &'static [(u64, u32, u16, u16)];

/// What a consumed chain says of itself: its regions, or why it is malformed.
type Verdict = Result<Vec<ChainRegion>, MalformedChain>;

fn verdict<C: DeviceChain>(chain: &mut C, name: &str) -> Verdict
where
    C::Error: core::fmt::Debug,
{
    match chain.io() {
        Ok(io) => Ok(io.into_parts().0.to_vec()),
        Err(ChainError::Malformed(malformed)) => Err(malformed),
        Err(other) => panic!("{name}: {other:?}"),
    }
}

fn model(queue: u16, limit: u16) -> SplitQueue {
    let mut model = SplitQueue::new(size(queue), limit).unwrap();
    QueueControl::configure(&mut model, size(queue)).unwrap();
    QueueControl::set_ready(&mut model, true).unwrap();
    model
}

/// Every shape gets the same answer from this ring as from the protocol's
/// own in-memory queue, and a malformed one is handed up, says why, and is
/// used with nothing written.
#[test]
fn every_chain_shape_is_judged_as_the_reference_model_judges_it() {
    for &(name, shape, head, expected) in SHAPES {
        let mut pair = Pair::ready(8, 4);
        for (index, &(len, flags, next)) in shape.iter().enumerate() {
            let at = pair.space(len).address;
            pair.descriptor(index as u16, at, len, flags, next);
        }
        pair.offer(0, head, 1);
        let mut chain = pair.device.pop_available().unwrap().expect(name);
        let ours = verdict(&mut chain, name);
        assert_eq!(ours.as_ref().err().copied(), expected, "{name}");

        let mut reference = model(8, 4);
        let descriptors = shape
            .iter()
            .map(|&(len, flags, next)| Descriptor::raw(vec![0; len as usize], flags, next))
            .collect();
        let raw = ModelChain::raw(descriptors, head).expect(name);
        reference.inject_available(raw).map_err(|_| name).unwrap();
        let mut modelled = DeviceQueue::pop_available(&mut reference)
            .unwrap()
            .expect(name);
        assert_eq!(
            ours,
            verdict(&mut modelled, name),
            "{name}: this ring against the model"
        );

        let writes = pair.memory.0.writes.get();
        let used = u32::from(ours.is_ok());
        pair.device
            .complete(chain, UsedLength::new(used))
            .expect(name);
        // Two stores: the entry and the index. No buffer was written.
        assert_eq!(pair.memory.0.writes.get(), writes + 2, "{name}");
        assert_eq!(
            pair.memory.get(pair.at.used + 4, 8)[4..],
            used.to_le_bytes(),
            "{name}"
        );
    }
}

/// The driver's half and the model, given the same publications and the
/// same completions: they accept and refuse the same things, and return the
/// same lengths.
#[test]
fn publication_is_refused_and_returned_as_the_reference_model_does() {
    let mut pair = Pair::ready(8, 4);
    let mut reference = model(8, 4);
    // (readable regions, writable regions) of each chain, eight bytes each.
    let script: &[(usize, usize)] = &[(1, 1), (2, 2), (1, 1), (1, 1), (1, 2), (1, 1)];
    let mut outcomes = Vec::new();
    for &(readable, writable) in script {
        let chain = pair.chain(&vec![8; readable], &vec![8; writable]);
        let ours = pair
            .driver
            .publish(chain)
            .map(|_| ())
            .map_err(|error| core::mem::discriminant(&error.into_parts().1.map_transport()));
        let mut descriptors: Vec<Descriptor> = (0..readable)
            .map(|_| Descriptor::readable(vec![0; 8]))
            .collect();
        descriptors.extend((0..writable).map(|_| Descriptor::writable(vec![0; 8])));
        let theirs = DriverQueue::publish(&mut reference, ModelChain::direct(descriptors).unwrap())
            .map(|_| ())
            .map_err(|error| core::mem::discriminant(&error.into_parts().1.map_transport()));
        assert_eq!(ours, theirs, "{readable} readable, {writable} writable");
        outcomes.push(ours.is_ok());
    }
    // Eight descriptors: 2 + 4 + 2 fit, and nothing after them does.
    assert_eq!(outcomes, [true, true, true, false, false, false]);

    // Complete all three, the middle one first, and both drivers get the
    // same lengths back for the same number of chains.
    for used in [3u32, 1, 2] {
        let chain = pair.device.pop_available().unwrap().unwrap();
        pair.device.complete(chain, UsedLength::new(used)).unwrap();
        let chain = DeviceQueue::pop_available(&mut reference).unwrap().unwrap();
        DeviceQueue::complete(&mut reference, chain, UsedLength::new(used)).unwrap();
    }
    for _ in 0..3 {
        let ours = pair.driver.pop_used().unwrap().unwrap().used();
        let theirs = DriverQueue::pop_used(&mut reference)
            .unwrap()
            .unwrap()
            .used();
        assert_eq!(ours, theirs);
    }
    assert!(pair.driver.pop_used().unwrap().is_none());
    assert!(DriverQueue::pop_used(&mut reference).unwrap().is_none());
    // And room again, in both.
    assert!(pair.chain(&[8], &[8]).descriptors() == 2);
    let chain = pair.chain(&[8, 8], &[8, 8]);
    assert!(pair.driver.publish(chain).is_ok());
}

/// Comparing refusals across two queues whose transport errors differ: keep
/// the classification and drop the payload.
trait Classified {
    fn map_transport(self) -> PublishErrorKind<()>;
}

impl<E> Classified for PublishErrorKind<E> {
    fn map_transport(self) -> PublishErrorKind<()> {
        match self {
            Self::NotReady => PublishErrorKind::NotReady,
            Self::QueueFull => PublishErrorKind::QueueFull,
            Self::InsufficientDescriptors => PublishErrorKind::InsufficientDescriptors,
            Self::ResetRace { operation, current } => {
                PublishErrorKind::ResetRace { operation, current }
            }
            Self::Transport(_) => PublishErrorKind::Transport(()),
        }
    }
}

// -- a device, and a driver that lies ----------------------------------------

#[test]
fn indices_and_addresses_outside_what_is_shared_are_malformed_chains() {
    const OUTSIDE: u64 = BASE + ARENA as u64;
    let cases: &[(&str, Placed, u16, MalformedChain)] = &[
        (
            "a head past the queue",
            &[(BUFFERS, 8, NEXT, 1), (BUFFERS + 64, 8, WRITE, 0)],
            9,
            MalformedChain::DescriptorIndex,
        ),
        (
            "a next past the queue",
            &[(BUFFERS, 8, NEXT, 8), (BUFFERS + 64, 8, WRITE, 0)],
            0,
            MalformedChain::DescriptorIndex,
        ),
        (
            "a request OUTSIDE",
            &[(OUTSIDE, 8, NEXT, 1), (BUFFERS, 8, WRITE, 0)],
            0,
            MalformedChain::Address,
        ),
        (
            "a response that runs off the end",
            &[(BUFFERS, 8, NEXT, 1), (OUTSIDE - 4, 8, WRITE, 0)],
            0,
            MalformedChain::Address,
        ),
        (
            "an address that wraps",
            &[(BUFFERS, 8, NEXT, 1), (u64::MAX - 3, 8, WRITE, 0)],
            0,
            MalformedChain::Address,
        ),
        (
            "an address below the memory",
            &[(0x1000, 8, NEXT, 1), (BUFFERS, 8, WRITE, 0)],
            0,
            MalformedChain::Address,
        ),
    ];
    for &(name, descriptors, head, expected) in cases {
        let mut pair = Pair::ready(8, 4);
        for (index, &(at, len, flags, next)) in descriptors.iter().enumerate() {
            pair.descriptor(index as u16, at, len, flags, next);
        }
        pair.offer(0, head, 1);
        let mut chain = pair.device.pop_available().unwrap().expect(name);
        assert_eq!(
            chain.io().err(),
            Some(ChainError::Malformed(expected)),
            "{name}"
        );
        assert_eq!(
            pair.device.complete(chain, UsedLength::new(0)),
            Ok(NotificationHint::Notify),
            "{name}"
        );
        // The driver is told the head it published, whatever it was.
        assert_eq!(
            pair.memory.get(pair.at.used + 4, 4),
            u32::from(head).to_le_bytes(),
            "{name}"
        );
    }

    // Reachable is not enough: the direction has to be allowed too.
    let mut pair = Pair::ready(8, 4);
    pair.memory.protect(BUFFERS + 0x100, 64);
    pair.descriptor(0, BUFFERS, 8, NEXT, 1);
    pair.descriptor(1, BUFFERS + 0x100, 8, WRITE, 0);
    pair.offer(0, 0, 1);
    let mut chain = pair.device.pop_available().unwrap().unwrap();
    assert_eq!(
        chain.io().err(),
        Some(ChainError::Malformed(MalformedChain::Address))
    );
    pair.device.complete(chain, UsedLength::new(0)).unwrap();
    // The same bytes as a request are fine.
    pair.descriptor(2, BUFFERS + 0x100, 8, NEXT, 3);
    pair.descriptor(3, BUFFERS + 0x200, 8, WRITE, 0);
    pair.offer(1, 2, 2);
    assert!(pair.device.pop_available().unwrap().unwrap().io().is_ok());
}

#[test]
fn a_descriptor_rewritten_after_publication_changes_nothing() {
    let mut pair = Pair::ready(8, 4);
    let (request, response, decoy) = (BUFFERS, BUFFERS + 0x100, BUFFERS + 0x200);
    pair.memory.put(request, b"the real request");
    pair.memory.put(decoy, &[0xaa; 64]);
    pair.descriptor(0, request, 16, NEXT, 1);
    pair.descriptor(1, response, 16, WRITE, 0);
    pair.offer(0, 0, 1);
    let mut chain = pair.device.pop_available().unwrap().unwrap();
    // The driver now points both descriptors somewhere else, and longer.
    pair.descriptor(0, decoy, 64, NEXT, 1);
    pair.descriptor(1, decoy, 64, WRITE, 0);
    {
        let (regions, source, sink) = chain.io().unwrap().into_parts();
        assert_eq!(
            regions,
            [ChainRegion::readable(16), ChainRegion::writable(16)]
        );
        let mut bytes = [0; 16];
        source.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"the real request");
        sink.write_at(0, &[0x55; 16]).unwrap();
        assert_eq!(
            sink.write_at(0, &[0; 17]),
            Err(ByteAccessError::OutOfBounds)
        );
    }
    assert_eq!(
        pair.device.complete(chain, UsedLength::new(17)),
        Err(QueueError::UsedLengthExceeded {
            used: UsedLength::new(17),
            capacity: 16
        })
    );
    assert_eq!(pair.memory.get(response, 16), [0x55; 16]);
    assert_eq!(pair.memory.get(decoy, 64), [0xaa; 64]);
}

#[test]
fn a_driver_that_lies_about_the_ring_stops_the_queue() {
    // An available index further ahead than the queue is long.
    let mut pair = Pair::ready(8, 4);
    pair.offer(0, 0, 9);
    assert_eq!(
        pair.device.pop_available().err(),
        Some(QueueError::Transport(DeviceRingError::AvailableIndex))
    );

    // A head the device already holds, published again.
    let mut pair = Pair::ready(8, 4);
    pair.descriptor(0, BUFFERS, 8, NEXT, 1);
    pair.descriptor(1, BUFFERS + 64, 8, WRITE, 0);
    pair.offer(0, 0, 1);
    let held = pair.device.pop_available().unwrap().unwrap();
    pair.offer(1, 0, 2);
    for _ in 0..2 {
        // Refused, and refused again: the entry is not consumed.
        assert_eq!(
            pair.device.pop_available().err(),
            Some(QueueError::Transport(DeviceRingError::HeadInFlight))
        );
    }
    assert_eq!(pair.device.in_flight(), 1);
    pair.device.complete(held, UsedLength::new(0)).unwrap();
    // Once it is used the head is the driver's to publish again.
    assert!(pair.device.pop_available().unwrap().is_some());

    // The ring taken away under a running queue is the queue's error, not
    // a chain's.
    let mut pair = Pair::ready(8, 4);
    let chain = pair.chain(&[8], &[8]);
    pair.driver.publish(chain).unwrap();
    pair.memory.remove(pair.at.descriptors, 16);
    assert_eq!(
        pair.device.pop_available().err(),
        Some(QueueError::Transport(DeviceRingError::Memory(
            MemoryFault::Range
        )))
    );
    pair.memory.restore();
    assert!(
        pair.device.pop_available().unwrap().is_some(),
        "nothing was consumed by the fault"
    );
}

#[test]
fn a_reset_silences_every_chain_the_device_still_holds() {
    let mut pair = Pair::ready(8, 4);
    let chain = pair.chain(&[8], &[8]);
    let response = chain.writable()[0].address;
    pair.driver.publish(chain).unwrap();
    let mut held = pair.device.pop_available().unwrap().unwrap();
    let next = QueueEpoch::INITIAL.checked_next().unwrap();
    assert_eq!(
        DeviceQueue::reset(&mut pair.device, QueueEpoch::INITIAL),
        Err(QueueError::InvalidConfiguration(
            QueueConfigError::NonIncreasingEpoch
        ))
    );
    {
        // Ports taken before the reset refuse after it.
        let (_, source, sink) = held.io().unwrap().into_parts();
        DeviceQueue::reset(&mut pair.device, next).unwrap();
        assert_eq!(source.read_at(0, &mut [0; 8]), Err(ByteAccessError::Reset));
        assert_eq!(sink.write_at(0, &[0xff; 8]), Err(ByteAccessError::Reset));
    }
    assert_eq!(
        held.io().err(),
        Some(ChainError::ResetRace {
            chain: QueueEpoch::INITIAL,
            current: next
        })
    );
    let writes = pair.memory.0.writes.get();
    assert_eq!(
        pair.device.complete(held, UsedLength::new(0)),
        Err(QueueError::ResetRace {
            operation: QueueEpoch::INITIAL,
            current: next
        })
    );
    assert_eq!(
        pair.memory.0.writes.get(),
        writes,
        "a stale completion writes nothing"
    );
    assert_eq!(pair.memory.get(response, 8), [0; 8]);
    assert_eq!(
        (pair.device.in_flight(), pair.device.state().epoch()),
        (0, next)
    );
    assert_eq!(
        pair.device.pop_available().err(),
        Some(QueueError::NotReady)
    );
}

#[test]
fn suppression_is_honoured_and_loses_no_work() {
    let mut pair = Pair::ready(8, 4);
    // The device asks not to be notified, and is not.
    pair.device.disable_available_notifications().unwrap();
    let chain = pair.chain(&[8], &[8]);
    assert_eq!(
        pair.driver.publish(chain).unwrap().notification(),
        NotificationHint::Suppressed
    );
    // Turning notifications back on finds what was published meanwhile.
    assert_eq!(
        pair.device.enable_available_notifications(),
        Ok(NotificationRecheck::WorkPending)
    );
    // The driver asks not to be interrupted, and is not.
    pair.driver.disable_used_notifications().unwrap();
    assert_eq!(pair.serve(&[1; 8]).1, NotificationHint::Suppressed);
    assert_eq!(
        pair.device.enable_available_notifications(),
        Ok(NotificationRecheck::Idle)
    );
    assert_eq!(
        pair.driver.enable_used_notifications(),
        Ok(NotificationRecheck::WorkPending)
    );
    assert!(pair.driver.pop_used().unwrap().is_some());
    assert_eq!(
        pair.driver.enable_used_notifications(),
        Ok(NotificationRecheck::Idle)
    );
    let chain = pair.chain(&[8], &[8]);
    assert_eq!(
        pair.driver.publish(chain).unwrap().notification(),
        NotificationHint::Notify
    );
    assert_eq!(pair.serve(&[1; 8]).1, NotificationHint::Notify);
}

// -- a driver, and a device that lies ----------------------------------------

#[test]
fn a_used_entry_is_checked_before_it_is_believed() {
    // (what the device wrote: id, length, index) for a queue holding one
    // chain at head 0 with eight writable bytes.
    let lies: &[(&str, u32, u32, u16, DriverRingError)] = &[
        ("a head past the queue", 8, 0, 1, DriverRingError::UsedEntry),
        (
            "a head past the queue's own storage",
            300,
            0,
            1,
            DriverRingError::UsedEntry,
        ),
        (
            "a head past sixteen bits",
            0x1_0000,
            0,
            1,
            DriverRingError::UsedEntry,
        ),
        (
            "a head it does not hold",
            3,
            0,
            1,
            DriverRingError::UsedEntry,
        ),
        (
            "more bytes than the chain has room for",
            0,
            9,
            1,
            DriverRingError::UsedEntry,
        ),
        (
            "more entries than chains",
            0,
            8,
            2,
            DriverRingError::UsedIndex,
        ),
    ];
    for &(name, id, len, index, expected) in lies {
        let mut pair = Pair::ready(8, 4);
        let chain = pair.chain(&[8], &[8]);
        let published = pair.driver.publish(chain).unwrap();
        assert_eq!(published.id().token(), 0);
        pair.use_entry(0, id, len, index);
        assert_eq!(
            pair.driver.pop_used().err(),
            Some(QueueError::Transport(expected)),
            "{name}"
        );
        // The device has stopped being one: nothing more goes either way.
        assert_eq!(
            pair.driver.pop_used().err(),
            Some(QueueError::Transport(expected)),
            "{name}"
        );
        let chain = pair.chain(&[8], &[8]);
        let refused = pair.driver.publish(chain).expect_err(name).into_parts().1;
        assert_eq!(refused, PublishErrorKind::Transport(expected), "{name}");
        assert_eq!(
            pair.driver.disable_used_notifications(),
            Err(QueueError::Transport(expected)),
            "{name}"
        );
        // The reset the driver owes the device returns what the device held.
        let next = QueueEpoch::INITIAL.checked_next().unwrap();
        let reclaimed: Vec<_> = DriverQueue::reset(&mut pair.driver, next)
            .unwrap()
            .collect();
        assert_eq!(reclaimed.len(), 1, "{name}");
        assert_eq!(reclaimed[0].id(), published.id(), "{name}");
    }

    // An honest entry for the same chain, for contrast.
    let mut pair = Pair::ready(8, 4);
    let chain = pair.chain(&[8], &[8]);
    pair.driver.publish(chain).unwrap();
    pair.use_entry(0, 0, 8, 1);
    assert_eq!(
        pair.driver.pop_used().unwrap().unwrap().used(),
        UsedLength::new(8)
    );
    // The same entry a second time is a head the device no longer holds.
    pair.use_entry(1, 0, 8, 2);
    assert_eq!(
        pair.driver.pop_used().err(),
        Some(QueueError::Transport(DriverRingError::UsedIndex))
    );
}

/// The table is never read back, so a device that rewrites it changes
/// nothing: every chain comes back whole and no two ever share a descriptor.
#[test]
fn chains_never_share_a_descriptor_whatever_the_device_writes() {
    let mut pair = Pair::ready(32, 4);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    // What the device holds: (head, descriptors, the chain's first region).
    let mut held: Vec<(u16, u16, u64)> = Vec::new();
    let mut owned = [false; 32];
    for step in 0..20_000 {
        let publish = held.is_empty() || (held.len() < 12 && random(3) != 0);
        if publish {
            let (readable, writable) = (1 + random(2) as usize, 1 + random(2) as usize);
            if pair.bump > BASE + ARENA as u64 - 0x1000 {
                pair.bump = BUFFERS;
            }
            let chain = pair.chain(&vec![8; readable], &vec![8; writable]);
            let (count, first) = (chain.descriptors(), chain.readable()[0].address);
            match pair.driver.publish(chain) {
                Ok(published) => {
                    let head = published.id().token() as u16;
                    // Walk the table as the device would and claim each.
                    let mut index = head;
                    for _ in 0..count {
                        assert!(
                            !owned[usize::from(index)],
                            "step {step}: descriptor {index} given twice"
                        );
                        owned[usize::from(index)] = true;
                        index = u16::from_le_bytes(
                            pair.memory
                                .get(pair.at.descriptors + u64::from(index) * 16 + 14, 2)
                                .try_into()
                                .unwrap(),
                        );
                    }
                    held.push((head, count, first));
                    // The device scribbles on the whole table.
                    for scribble in 0..32u16 {
                        if !owned[usize::from(scribble)] {
                            pair.descriptor(scribble, 0xdead_0000, 0xffff, NEXT, scribble);
                        }
                    }
                }
                Err(error) => {
                    let used: u16 = held.iter().map(|chain| chain.1).sum();
                    assert!(32 - used < count, "step {step}: refused with room");
                    assert_eq!(
                        error.into_parts().1,
                        PublishErrorKind::InsufficientDescriptors
                    );
                }
            }
        } else {
            let (head, count, first) = held.swap_remove(random(held.len() as u64) as usize);
            // Release what the device walked before it rewrites the links.
            let mut index = head;
            for _ in 0..count {
                owned[usize::from(index)] = false;
                let at = pair.at.descriptors + u64::from(index) * 16;
                let next = u16::from_le_bytes(pair.memory.get(at + 14, 2).try_into().unwrap());
                pair.descriptor(index, 0xdead_0000, 0xffff, NEXT, index);
                index = next;
            }
            let slot = pair.memory.get(pair.at.used + 2, 2);
            let slot = u16::from_le_bytes([slot[0], slot[1]]);
            pair.use_entry(slot % 32, u32::from(head), 8, slot.wrapping_add(1));
            let used = pair
                .driver
                .pop_used()
                .unwrap()
                .expect("the entry just written");
            assert_eq!(used.id().token(), u64::from(head), "step {step}");
            let chain = used.into_parts().2;
            assert_eq!(
                (chain.descriptors(), chain.readable()[0].address),
                (count, first),
                "step {step}"
            );
        }
        let used: u16 = held.iter().map(|chain| chain.1).sum();
        assert_eq!(pair.driver.free_descriptors(), 32 - used, "step {step}");
        assert_eq!(
            usize::from(pair.driver.outstanding()),
            held.len(),
            "step {step}"
        );
    }
}

#[test]
fn a_publication_that_cannot_be_written_publishes_nothing() {
    let mut pair = Pair::ready(8, 4);
    for (name, at, len) in [
        ("the descriptor table", pair.at.descriptors, 16),
        ("the available ring's entry", pair.at.available + 4, 2),
        ("the available index", pair.at.available + 2, 2),
    ] {
        pair.memory.remove(at, len);
        let chain = pair.chain(&[8, 8], &[8]);
        let (chain, kind) = pair.driver.publish(chain).expect_err(name).into_parts();
        assert_eq!(
            kind,
            PublishErrorKind::Transport(DriverRingError::Memory(MemoryFault::Range)),
            "{name}"
        );
        assert_eq!(chain.descriptors(), 3, "{name}: the chain comes back");
        assert_eq!(
            (pair.driver.outstanding(), pair.driver.free_descriptors()),
            (0, 8),
            "{name}"
        );
        pair.memory.restore();
        assert_eq!(
            pair.memory.get(pair.at.available + 2, 2),
            [0, 0],
            "{name}: nothing was published"
        );
        assert!(pair.device.pop_available().unwrap().is_none(), "{name}");
    }
    // A suppression flag that cannot be read is not a failed publication:
    // the chain is the device's, and the device is told.
    pair.memory.remove(pair.at.used, 2);
    let chain = pair.chain(&[8], &[8]);
    assert_eq!(
        pair.driver.publish(chain).unwrap().notification(),
        NotificationHint::Notify
    );
    pair.memory.restore();
    assert!(pair.device.pop_available().unwrap().is_some());
}

#[test]
fn a_full_queue_pushes_back_and_takes_chains_again_when_some_return() {
    let mut pair = Pair::ready(8, 4);
    let too_long = Chain::new(
        pair.memory.clone(),
        &[Region {
            address: BUFFERS,
            len: 8,
        }; 3],
        &[Region {
            address: BUFFERS + 64,
            len: 8,
        }; 2],
    )
    .unwrap();
    let (chain, kind) = pair.driver.publish(too_long).unwrap_err().into_parts();
    assert_eq!(
        kind,
        PublishErrorKind::InsufficientDescriptors,
        "longer than the chain limit"
    );
    assert_eq!(chain.descriptors(), 5);
    for _ in 0..4 {
        let chain = pair.chain(&[8], &[8]);
        pair.driver.publish(chain).unwrap();
    }
    let chain = pair.chain(&[8], &[8]);
    let (chain, kind) = pair.driver.publish(chain).unwrap_err().into_parts();
    assert_eq!(kind, PublishErrorKind::InsufficientDescriptors);
    pair.serve(&[0; 8]);
    assert!(pair.driver.pop_used().unwrap().is_some());
    assert!(pair.driver.publish(chain).is_ok());

    let mut idle = Driver::new(Memory::new(), size(8), 4).unwrap();
    let chain = pair.chain(&[8], &[8]);
    assert_eq!(
        idle.publish(chain).unwrap_err().into_parts().1,
        PublishErrorKind::NotReady
    );
    assert_eq!(idle.pop_used().err(), Some(QueueError::NotReady));
}

#[test]
fn a_driver_reset_returns_every_chain_and_starts_over() {
    let mut pair = Pair::ready(8, 4);
    let mut ids = Vec::new();
    for _ in 0..3 {
        let chain = pair.chain(&[8], &[8]);
        ids.push(pair.driver.publish(chain).unwrap().id());
    }
    pair.serve(&[0; 8]);
    assert_eq!(
        DriverQueue::reset(&mut pair.driver, QueueEpoch::INITIAL).err(),
        Some(QueueError::InvalidConfiguration(
            QueueConfigError::NonIncreasingEpoch
        ))
    );
    let next = QueueEpoch::INITIAL.checked_next().unwrap();
    // All three: a used entry the driver had not taken is not taken now.
    let reclaimed: Vec<_> = DriverQueue::reset(&mut pair.driver, next)
        .unwrap()
        .map(|chain| chain.id())
        .collect();
    assert_eq!(reclaimed.len(), 3);
    assert!(ids.iter().all(|id| reclaimed.contains(id)));
    assert_eq!(pair.driver.state().epoch(), next);
    assert_eq!(
        (pair.driver.outstanding(), pair.driver.free_descriptors()),
        (0, 0)
    );
    assert_eq!(pair.driver.pop_used().err(), Some(QueueError::NotReady));

    // The same memory, laid out again, is a clean queue in the new epoch.
    pair.driver.configure(size(8)).unwrap();
    pair.driver.set_addresses(pair.at).unwrap();
    pair.driver.set_ready(true).unwrap();
    assert_eq!(
        pair.memory.get(pair.at.used, 4),
        [0; 4],
        "the stale used index is gone"
    );
    DeviceQueue::reset(&mut pair.device, next).unwrap();
    pair.device.configure(size(8)).unwrap();
    pair.device.set_addresses(pair.at).unwrap();
    pair.device.set_ready(true).unwrap();
    let chain = pair.chain(&[8], &[8]);
    let published = pair.driver.publish(chain).unwrap();
    assert_eq!(published.id().epoch(), next);
    pair.serve(&[0; 4]);
    assert_eq!(
        pair.driver.pop_used().unwrap().unwrap().id(),
        published.id()
    );
}

/// Not ready is a pause: nothing moves while it lasts, nothing is cleared
/// when it ends, and what the device did meanwhile is found.
#[test]
fn a_paused_queue_resumes_where_it_was() {
    let mut pair = Pair::ready(8, 4);
    let chain = pair.chain(&[8], &[8]);
    let published = pair.driver.publish(chain).unwrap();
    pair.driver.set_ready(false).unwrap();
    assert_eq!(pair.driver.pop_used().err(), Some(QueueError::NotReady));
    let chain = pair.chain(&[8], &[8]);
    let (chain, kind) = pair.driver.publish(chain).unwrap_err().into_parts();
    assert_eq!(kind, PublishErrorKind::NotReady);
    // The device holds a chain: the queue can be neither moved nor resized.
    assert_eq!(
        pair.driver.set_addresses(pair.at),
        Err(QueueError::Transport(DriverRingError::Busy))
    );
    assert_eq!(
        pair.driver.configure(size(8)),
        Err(QueueError::Transport(DriverRingError::Busy))
    );
    pair.serve(&[9; 8]);

    pair.driver.set_ready(true).unwrap();
    assert_eq!(
        pair.memory.get(pair.at.available + 2, 2),
        [1, 0],
        "the published index survives"
    );
    assert_eq!(
        pair.driver.pop_used().unwrap().unwrap().id(),
        published.id()
    );
    // The next chain goes in the next slot, not the first again.
    pair.driver.publish(chain).unwrap();
    assert_eq!(pair.memory.get(pair.at.available + 2, 2), [2, 0]);
    assert_eq!(pair.serve(&[9; 8]).0, [0; 8]);
    assert!(pair.driver.pop_used().unwrap().is_some());

    // Sizing it again is not a pause. It is a new queue in the same memory,
    // and it starts from nothing on both sides.
    pair.driver.set_ready(false).unwrap();
    pair.driver.configure(size(8)).unwrap();
    pair.driver.set_ready(true).unwrap();
    assert_eq!(pair.memory.get(pair.at.available, 4), [0; 4]);
    assert_eq!(pair.memory.get(pair.at.used, 4), [0; 4]);
    let next = QueueEpoch::INITIAL.checked_next().unwrap();
    DeviceQueue::reset(&mut pair.device, next).unwrap();
    pair.device.configure(size(8)).unwrap();
    pair.device.set_addresses(pair.at).unwrap();
    pair.device.set_ready(true).unwrap();
    assert!(pair.driver.pop_used().unwrap().is_none());
    let chain = pair.chain(&[8], &[8]);
    pair.driver.publish(chain).unwrap();
    assert_eq!(pair.memory.get(pair.at.available + 2, 2), [1, 0]);
    pair.serve(&[9; 8]);
    assert!(pair.driver.pop_used().unwrap().is_some());
}

// -- chains ------------------------------------------------------------------

#[test]
fn a_chain_is_a_shape_the_protocol_accepts() {
    let memory = Memory::new();
    let region = |len| Region {
        address: BUFFERS,
        len,
    };
    let shape = |readable: &[Region], writable: &[Region]| {
        Chain::new(memory.clone(), readable, writable).err()
    };
    assert_eq!(shape(&[region(8)], &[region(8)]), None);
    assert_eq!(
        shape(&[region(8); 5], &[region(8); 4]),
        Some(ChainShapeError::Capacity)
    );
    assert_eq!(
        shape(&[region(8)], &[]),
        Some(ChainShapeError::Layout(ChainLayoutError::DescriptorCount))
    );
    assert_eq!(
        shape(&[region(8); 2], &[]),
        Some(ChainShapeError::Layout(ChainLayoutError::Direction))
    );
    assert_eq!(
        shape(&[], &[region(8); 2]),
        Some(ChainShapeError::Layout(ChainLayoutError::Direction))
    );
    assert_eq!(
        shape(&[region(0)], &[region(8)]),
        Some(ChainShapeError::Layout(ChainLayoutError::ZeroLength))
    );
    assert_eq!(
        shape(
            &[Region {
                address: u64::MAX - 3,
                len: 8
            }],
            &[region(8)]
        ),
        Some(ChainShapeError::AddressOverflow)
    );

    // Its bytes are reached across its regions, and only inside them.
    let (a, b, c) = (BUFFERS, BUFFERS + 0x40, BUFFERS + 0x80);
    let mut chain = Chain::new(
        memory.clone(),
        &[Region { address: a, len: 4 }, Region { address: b, len: 4 }],
        &[Region { address: c, len: 6 }],
    )
    .unwrap();
    assert_eq!(
        (chain.device_readable_len(), chain.device_writable_len()),
        (8, 6)
    );
    chain.write_device_readable(2, b"wxyz").unwrap();
    assert_eq!(memory.get(a, 4), [0, 0, b'w', b'x']);
    assert_eq!(memory.get(b, 4), [b'y', b'z', 0, 0]);
    assert_eq!(
        chain.write_device_readable(5, &[0; 4]),
        Err(MemoryFault::Range)
    );
    memory.put(c, b"answer");
    let mut out = [0; 3];
    chain.read_device_writable(3, &mut out).unwrap();
    assert_eq!(&out, b"wer");
    assert_eq!(
        chain.read_device_writable(4, &mut out),
        Err(MemoryFault::Range)
    );
    assert_eq!(
        chain.read_device_writable(u64::MAX, &mut out),
        Err(MemoryFault::Range)
    );
    assert_eq!(
        memory.get(c + 6, 2),
        [0, 0],
        "nothing past a region is touched"
    );
}

// -- long lives --------------------------------------------------------------

/// Both sixteen-bit indices wrap, the device's visited set outlives its own
/// counter, and the queue goes on.
#[test]
fn the_indices_wrap_and_the_queue_goes_on() {
    let mut pair = Pair::ready(4, 2);
    let (request, response) = (pair.space(8), pair.space(8));
    for round in 0..70_000u32 {
        let mut chain = Chain::new(pair.memory.clone(), &[request], &[response]).unwrap();
        chain
            .write_device_readable(0, &u64::from(round).to_le_bytes())
            .unwrap();
        pair.driver.publish(chain).unwrap();
        let (seen, _) = pair.serve(&round.to_le_bytes());
        assert_eq!(seen, u64::from(round).to_le_bytes(), "round {round}");
        let used = pair.driver.pop_used().unwrap().expect("used");
        assert_eq!(used.used(), UsedLength::new(4), "round {round}");
        let mut answer = [0; 4];
        used.chain().read_device_writable(0, &mut answer).unwrap();
        assert_eq!(answer, round.to_le_bytes(), "round {round}");
    }
    assert_eq!(
        pair.memory.get(pair.at.used + 2, 2),
        (70_000u32 as u16).to_le_bytes()
    );
}

/// A chain as long as the queue allows, on a queue far past sixty-four
/// descriptors, and a loop among that many.
#[test]
fn a_long_chain_on_a_large_queue_is_walked_once() {
    let mut pair = Pair::ready(256, 8);
    let chain = pair.chain(&[1, 2, 3, 4], &[5, 6, 7, 8]);
    pair.driver.publish(chain).unwrap();
    let mut taken = pair.device.pop_available().unwrap().unwrap();
    assert_eq!(taken.io().unwrap().into_parts().0.len(), 8);
    pair.device.complete(taken, UsedLength::new(26)).unwrap();
    assert_eq!(
        pair.driver.pop_used().unwrap().unwrap().used(),
        UsedLength::new(26)
    );

    // Two hundred descriptors in a ring, entered at the far end.
    let mut pair = Pair::ready(256, 8);
    for index in 0..200u16 {
        pair.descriptor(index, BUFFERS, 8, NEXT, (index + 1) % 200);
    }
    pair.offer(0, 199, 1);
    let mut looped = pair.device.pop_available().unwrap().unwrap();
    assert_eq!(
        looped.io().err(),
        Some(ChainError::Malformed(MalformedChain::DescriptorLoop))
    );
    // And two hundred in a line: too long, and said to be.
    pair.descriptor(199, BUFFERS, 8, WRITE, 0);
    pair.offer(1, 0, 2);
    pair.device.complete(looped, UsedLength::new(0)).unwrap();
    let mut long = pair.device.pop_available().unwrap().unwrap();
    assert_eq!(
        long.io().err(),
        Some(ChainError::Malformed(MalformedChain::DescriptorCount))
    );
}
