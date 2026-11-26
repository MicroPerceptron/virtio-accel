//! Guest memory as a device behind the platform IOMMU sees it.
//!
//! With `VIRTIO_F_ACCESS_PLATFORM` every address the driver hands the device —
//! the three ring addresses and each descriptor's buffer — is an I/O virtual
//! address in the device's IOMMU domain. The frontend owns the translation and
//! shares it one range at a time (vhost-user `IOTLB_MSG`): the backend asks for
//! what it lacks (a *miss*), the frontend answers with an *update* or with a
//! refusal, and withdraws ranges with an *invalidate*.
//!
//! # Only what is granted is mapped
//!
//! In this mode the daemon does not map guest RAM when it receives the memory
//! table. It keeps the region files, and maps exactly the ranges the IOMMU has
//! granted, at the moment they are granted. An invalidation unmaps. Nothing in
//! this process can reach guest memory the guest's domain does not name, by
//! address arithmetic or by a bug: there is no mapping for it to land in.
//!
//! The result is an ordinary [`GuestMemoryMmap`] whose guest addresses are I/O
//! virtual addresses, so the queue and the byte ports use it unchanged.

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::Arc;

use vhost::vhost_user::message::{VhostUserIotlb, VhostUserMemoryRegion};
use vm_memory::{FileOffset, GuestAddress, GuestMemoryMmap, GuestRegionMmap, MmapRegion};

/// Ranges held at once. Each is one mapping; the bound keeps a guest from
/// growing the daemon's mapping count without limit. A full table is emptied
/// rather than aged: entries come back on demand, and the largest chain this
/// device accepts needs far fewer ranges than this, so a chain can always be
/// satisfied by the table it refills.
pub(crate) const MAX_RANGES: usize = 8192;

/// One region of the frontend's memory table, kept as the file it came in.
struct Source {
    user_addr: u64,
    size: u64,
    offset: u64,
    file: Arc<File>,
}

struct Range {
    last: u64,
    read: bool,
    write: bool,
    /// Which source, and where in its file, the range is mapped from.
    backing: (usize, u64),
    region: Arc<GuestRegionMmap>,
}

/// Why an update was not installed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// Empty, wrapping, or not a message this side receives.
    Malformed,
    /// The frontend address is not inside one region of its memory table.
    Unbacked,
    /// The range could not be mapped.
    Mapping,
}

pub(crate) struct Iotlb {
    sources: Vec<Source>,
    ranges: BTreeMap<u64, Range>,
    view: GuestMemoryMmap,
}

impl Iotlb {
    /// A domain with nothing granted, over the memory table's files.
    pub(crate) fn new(regions: &[VhostUserMemoryRegion], files: Vec<File>) -> Self {
        Self {
            sources: regions
                .iter()
                .zip(files)
                .map(|(region, file)| Source {
                    user_addr: region.user_addr,
                    size: region.memory_size,
                    offset: region.mmap_offset,
                    file: Arc::new(file),
                })
                .collect(),
            ranges: BTreeMap::new(),
            view: GuestMemoryMmap::new(),
        }
    }

    /// Everything currently granted, addressed by I/O virtual address.
    pub(crate) fn view(&self) -> &GuestMemoryMmap {
        &self.view
    }

    /// The first address in `[address, address + len)` the device may not
    /// access as asked, or `None` when the whole range is granted.
    ///
    /// `len` is nonzero and the range does not wrap: callers have checked.
    pub(crate) fn missing(&self, address: u64, len: u64, write: bool) -> Option<u64> {
        let last = address + (len - 1);
        let mut at = address;
        loop {
            let Some((_, range)) = self.ranges.range(..=at).next_back() else {
                return Some(at);
            };
            let allowed = if write { range.write } else { range.read };
            if range.last < at || !allowed {
                return Some(at);
            }
            if range.last >= last {
                return None;
            }
            at = range.last + 1;
        }
    }

    /// Install a translation.
    ///
    /// A frontend may grant a range one direction at a time, answering each
    /// miss with exactly the access that miss asked for. A grant that repeats
    /// an installed range — same addresses, same backing — therefore adds
    /// its access to what the range already allows: both statements came
    /// from the frontend and no invalidation separates them. Anything else
    /// replaces what it overlaps.
    pub(crate) fn update(&mut self, message: &VhostUserIotlb) -> Result<(), Refused> {
        let last = message
            .size
            .checked_sub(1)
            .and_then(|span| message.iova.checked_add(span))
            .ok_or(Refused::Malformed)?;
        let (read, write) = match message.perm {
            VhostUserIotlb::ACCESS_RO => (true, false),
            VhostUserIotlb::ACCESS_WO => (false, true),
            VhostUserIotlb::ACCESS_RW => (true, true),
            _ => return Err(Refused::Malformed),
        };
        let (index, start) = self
            .sources
            .iter()
            .enumerate()
            .find_map(|(index, source)| {
                let within = message.uaddr.checked_sub(source.user_addr)?;
                (within.checked_add(message.size)? <= source.size)
                    .then(|| source.offset.checked_add(within))
                    .flatten()
                    .map(|start| (index, start))
            })
            .ok_or(Refused::Unbacked)?;
        if let Some(range) = self
            .ranges
            .get_mut(&message.iova)
            .filter(|range| range.last == last && range.backing == (index, start))
        {
            range.read |= read;
            range.write |= write;
            return Ok(());
        }
        let size = usize::try_from(message.size).map_err(|_| Refused::Mapping)?;
        let file = self.sources[index].file.clone();
        let region = MmapRegion::from_file(FileOffset::from_arc(file, start), size)
            .ok()
            .and_then(|mapping| GuestRegionMmap::new(mapping, GuestAddress(message.iova)))
            .ok_or(Refused::Mapping)?;
        self.withdraw(message.iova, last);
        if self.ranges.len() >= MAX_RANGES {
            self.ranges.clear();
        }
        self.ranges.insert(
            message.iova,
            Range {
                last,
                read,
                write,
                backing: (index, start),
                region: Arc::new(region),
            },
        );
        self.publish();
        Ok(())
    }

    /// Withdraw every translation touching `[iova, iova + size)`.
    ///
    /// The end is computed the way the kernel's vhost does, wrapping: a
    /// frontend invalidating a whole domain sends a size of zero, which names
    /// everything from `iova` up. A range only partly covered is withdrawn
    /// whole; what is still valid of it returns on the next miss.
    pub(crate) fn invalidate(&mut self, iova: u64, size: u64) {
        let last = match iova.wrapping_add(size).wrapping_sub(1) {
            last if last >= iova => last,
            _ => u64::MAX,
        };
        if self.withdraw(iova, last) {
            self.publish();
        }
    }

    fn withdraw(&mut self, first: u64, last: u64) -> bool {
        let before = self.ranges.len();
        self.ranges
            .retain(|start, range| *start > last || range.last < first);
        self.ranges.len() != before
    }

    fn publish(&mut self) {
        // Sorted and disjoint by construction; an empty table is the empty space.
        self.view = GuestMemoryMmap::from_arc_regions(
            self.ranges
                .values()
                .map(|range| range.region.clone())
                .collect(),
        )
        .unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use vm_memory::Bytes;

    const PAGE: u64 = 4096;
    const BASE: u64 = 0x7f00_0000_0000;

    /// Sixteen pages of "guest RAM", page `n` filled with byte `n`.
    fn domain() -> Iotlb {
        let mut file = tempfile::tempfile().unwrap();
        for page in 0..16u8 {
            file.write_all(&[page; PAGE as usize]).unwrap();
        }
        let region = VhostUserMemoryRegion::new(0x4000_0000, 16 * PAGE, BASE, 0);
        Iotlb::new(&[region], vec![file])
    }

    fn update(iova: u64, pages: u64, page: u64, perm: u8) -> VhostUserIotlb {
        let mut message = VhostUserIotlb::default();
        message.iova = iova;
        message.size = pages * PAGE;
        message.uaddr = BASE + page * PAGE;
        message.perm = perm;
        message.msg_type = VhostUserIotlb::UPDATE;
        message
    }

    fn byte(domain: &Iotlb, iova: u64) -> Option<u8> {
        domain.view().read_obj(GuestAddress(iova)).ok()
    }

    #[test]
    fn nothing_is_reachable_until_granted() {
        let mut domain = domain();
        assert_eq!(domain.missing(0x10_0000, 1, false), Some(0x10_0000));
        assert_eq!(byte(&domain, 0x10_0000), None);
        // The I/O virtual address is the driver's choice, not the page's own.
        domain
            .update(&update(0x10_0000, 1, 5, VhostUserIotlb::ACCESS_RW))
            .unwrap();
        assert_eq!(domain.missing(0x10_0000, PAGE, true), None);
        assert_eq!(byte(&domain, 0x10_0000), Some(5));
        // The page's guest-physical and frontend addresses name nothing.
        assert_eq!(byte(&domain, 0x4000_5000), None);
        assert_eq!(byte(&domain, BASE + 5 * PAGE), None);
        // One byte past the grant is the first missing address.
        assert_eq!(
            domain.missing(0x10_0000, PAGE + 1, false),
            Some(0x10_0000 + PAGE)
        );
    }

    #[test]
    fn a_range_spans_grants_that_are_not_contiguous_in_the_guest() {
        let mut domain = domain();
        domain
            .update(&update(0x20_0000, 1, 9, VhostUserIotlb::ACCESS_RW))
            .unwrap();
        domain
            .update(&update(0x20_1000, 1, 2, VhostUserIotlb::ACCESS_RW))
            .unwrap();
        assert_eq!(domain.missing(0x20_0000, 2 * PAGE, true), None);
        let mut bytes = [0u8; 2];
        domain
            .view()
            .read_slice(&mut bytes, GuestAddress(0x20_0fff))
            .unwrap();
        assert_eq!(bytes, [9, 2]);
    }

    #[test]
    fn permission_is_part_of_the_grant() {
        let mut domain = domain();
        domain
            .update(&update(0x30_0000, 1, 1, VhostUserIotlb::ACCESS_RO))
            .unwrap();
        assert_eq!(domain.missing(0x30_0000, 8, false), None);
        assert_eq!(domain.missing(0x30_0000, 8, true), Some(0x30_0000));
        // The same range granted the other direction: both now hold.
        domain
            .update(&update(0x30_0000, 1, 1, VhostUserIotlb::ACCESS_WO))
            .unwrap();
        assert_eq!(domain.missing(0x30_0000, 8, false), None);
        assert_eq!(domain.missing(0x30_0000, 8, true), None);
        // The same address over a different page is a different mapping. It
        // replaces the old one and inherits nothing from it.
        domain
            .update(&update(0x30_0000, 1, 2, VhostUserIotlb::ACCESS_RO))
            .unwrap();
        assert_eq!(byte(&domain, 0x30_0000), Some(2));
        assert_eq!(domain.missing(0x30_0000, 8, true), Some(0x30_0000));
        // And an invalidation ends every grant, in both directions.
        domain
            .update(&update(0x30_0000, 1, 2, VhostUserIotlb::ACCESS_WO))
            .unwrap();
        domain.invalidate(0x30_0000, PAGE);
        domain
            .update(&update(0x30_0000, 1, 2, VhostUserIotlb::ACCESS_RO))
            .unwrap();
        assert_eq!(domain.missing(0x30_0000, 8, true), Some(0x30_0000));
    }

    #[test]
    fn invalidation_unmaps() {
        let mut domain = domain();
        for page in 0..4 {
            domain
                .update(&update(
                    0x40_0000 + page * PAGE,
                    1,
                    page,
                    VhostUserIotlb::ACCESS_RW,
                ))
                .unwrap();
        }
        domain.invalidate(0x40_1000, PAGE);
        assert_eq!(byte(&domain, 0x40_0000), Some(0));
        assert_eq!(byte(&domain, 0x40_1000), None);
        assert_eq!(domain.missing(0x40_0000, 4 * PAGE, false), Some(0x40_1000));
        // A size of zero is the frontend's "the whole domain".
        domain.invalidate(0, 0);
        assert_eq!(byte(&domain, 0x40_0000), None);
        assert_eq!(byte(&domain, 0x40_3000), None);
    }

    #[test]
    fn a_partly_invalidated_range_is_withdrawn_whole() {
        let mut domain = domain();
        domain
            .update(&update(0x50_0000, 4, 4, VhostUserIotlb::ACCESS_RW))
            .unwrap();
        domain.invalidate(0x50_2000, PAGE);
        assert_eq!(domain.missing(0x50_0000, 1, false), Some(0x50_0000));
        assert_eq!(byte(&domain, 0x50_0000), None);
    }

    #[test]
    fn a_grant_must_lie_in_the_memory_table() {
        let mut domain = domain();
        let mut outside = update(0x60_0000, 1, 16, VhostUserIotlb::ACCESS_RW);
        assert_eq!(domain.update(&outside), Err(Refused::Unbacked));
        outside.uaddr = BASE + 15 * PAGE;
        outside.size = 2 * PAGE;
        assert_eq!(domain.update(&outside), Err(Refused::Unbacked));
        let mut empty = update(0x60_0000, 1, 0, VhostUserIotlb::ACCESS_RW);
        empty.size = 0;
        assert_eq!(domain.update(&empty), Err(Refused::Malformed));
        let mut wrapping = update(u64::MAX - 0xfff, 2, 0, VhostUserIotlb::ACCESS_RW);
        wrapping.perm = VhostUserIotlb::ACCESS_RW;
        assert_eq!(domain.update(&wrapping), Err(Refused::Malformed));
        assert_eq!(byte(&domain, 0x60_0000), None);
    }

    #[test]
    fn a_full_table_is_emptied_and_refills() {
        let mut domain = domain();
        for index in 0..MAX_RANGES as u64 {
            domain
                .update(&update(
                    index * PAGE,
                    1,
                    index % 16,
                    VhostUserIotlb::ACCESS_RO,
                ))
                .unwrap();
        }
        assert_eq!(byte(&domain, 0), Some(0));
        domain
            .update(&update(0x1_0000_0000, 1, 7, VhostUserIotlb::ACCESS_RO))
            .unwrap();
        assert_eq!(byte(&domain, 0x1_0000_0000), Some(7));
        assert_eq!(byte(&domain, 0), None);
        assert_eq!(domain.ranges.len(), 1);
    }
}
