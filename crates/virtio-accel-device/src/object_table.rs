use alloc::vec::Vec;
use core::num::{NonZeroU16, NonZeroU64};

const KIND_MASK: u32 = 0b111;
const GENERATION_BITS: u32 = 13;
const GENERATION_MASK: u16 = (1 << GENERATION_BITS) - 1;
const GENERATION_SHIFT: u32 = 3;
const NAMESPACE_SHIFT: u32 = GENERATION_SHIFT + GENERATION_BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjectKind {
    Context = 1,
    Buffer = 2,
    Program = 3,
    Queue = 4,
    Event = 5,
}

impl ObjectKind {
    const fn tag(self) -> u32 {
        self as u32
    }
}

/// Device-instance namespace encoded into every object ID.
///
/// A transport integration assigns a distinct nonzero namespace to each device reset epoch and
/// does not reuse it while an ID from that epoch could still be presented. IDs from different
/// devices or reset epochs can therefore never resolve even when their slot, kind, and generation
/// are otherwise identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ObjectNamespace(NonZeroU16);

impl ObjectNamespace {
    pub const fn new(value: u16) -> Option<Self> {
        match NonZeroU16::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Opaque guest-visible identifier. Its encoding is device-private, not part of the wire ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ObjectId(NonZeroU64);

impl ObjectId {
    pub const fn from_raw(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(raw) => Some(Self(raw)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    const fn new(index: u32, namespace: u16, generation: u16, kind: ObjectKind) -> Self {
        let token = ((namespace as u32) << NAMESPACE_SHIFT)
            | ((generation as u32) << GENERATION_SHIFT)
            | kind.tag();
        let raw = ((token as u64) << 32) | (index as u64 + 1);
        match NonZeroU64::new(raw) {
            Some(raw) => Self(raw),
            None => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectTableError {
    InvalidId,
    WrongKind,
    StaleId,
    Full,
    AllocationFailed,
}

struct Slot<T> {
    generation: u16,
    value: Option<T>,
    retired: bool,
}

/// Bounded generational object table for one resource kind.
///
/// Kind tags, generations, and device namespaces occupy separate token fields. A slot is retired
/// before generation overflow, so an old ID cannot become valid again after wraparound.
pub struct ObjectTable<T> {
    kind: ObjectKind,
    namespace: u16,
    max_slots: u32,
    live: u32,
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> ObjectTable<T> {
    pub const fn new(kind: ObjectKind, max_slots: u32) -> Self {
        Self {
            kind,
            namespace: 0,
            max_slots,
            live: 0,
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub const fn with_namespace(
        kind: ObjectKind,
        max_slots: u32,
        namespace: ObjectNamespace,
    ) -> Self {
        Self {
            kind,
            namespace: namespace.get(),
            max_slots,
            live: 0,
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub const fn len(&self) -> u32 {
        self.live
    }

    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn insert(&mut self, value: T) -> Result<ObjectId, ObjectTableError> {
        self.try_reserve_insert()?;
        Ok(self.insert_prepared(value))
    }

    /// Reserve all capacity required by the next insertion without changing table state.
    pub fn try_reserve_insert(&mut self) -> Result<(), ObjectTableError> {
        if !self.free.is_empty() {
            return Ok(());
        }
        if self.slots.len() >= self.max_slots as usize {
            return Err(ObjectTableError::Full);
        }
        self.slots
            .try_reserve(1)
            .map_err(|_| ObjectTableError::AllocationFailed)?;
        let new_slot_count = self.slots.len() + 1;
        if self.free.capacity() < new_slot_count {
            self.free
                .try_reserve(new_slot_count - self.free.len())
                .map_err(|_| ObjectTableError::AllocationFailed)?;
        }
        Ok(())
    }

    pub(crate) fn insert_prepared(&mut self, value: T) -> ObjectId {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none() && !slot.retired);
            slot.value = Some(value);
            self.live += 1;
            return ObjectId::new(index, self.namespace, slot.generation, self.kind);
        }

        debug_assert!(self.slots.len() < self.max_slots as usize);
        debug_assert!(self.slots.len() < self.slots.capacity());
        let new_slot_count = self.slots.len() + 1;
        debug_assert!(self.free.capacity() >= new_slot_count);
        let index = self.slots.len() as u32;
        let generation = 0;
        self.slots.push(Slot {
            generation,
            value: Some(value),
            retired: false,
        });
        self.live += 1;
        ObjectId::new(index, self.namespace, generation, self.kind)
    }

    pub fn get(&self, id: ObjectId) -> Result<&T, ObjectTableError> {
        let index = self.locate(id)?;
        self.slots[index]
            .value
            .as_ref()
            .ok_or(ObjectTableError::StaleId)
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Result<&mut T, ObjectTableError> {
        let index = self.locate(id)?;
        self.slots[index]
            .value
            .as_mut()
            .ok_or(ObjectTableError::StaleId)
    }

    pub fn remove(&mut self, id: ObjectId) -> Result<T, ObjectTableError> {
        let index = self.locate(id)?;
        let slot = &mut self.slots[index];
        let value = slot.value.take().ok_or(ObjectTableError::StaleId)?;
        self.live -= 1;

        if slot.generation == GENERATION_MASK {
            slot.retired = true;
        } else {
            slot.generation += 1;
            self.free.push(index as u32);
        }
        Ok(value)
    }

    pub(crate) fn next_id_from(&self, start: usize) -> Option<(usize, ObjectId)> {
        self.slots
            .iter()
            .enumerate()
            .skip(start)
            .find_map(|(index, slot)| {
                if slot.retired || slot.value.is_none() {
                    return None;
                }
                Some((
                    index + 1,
                    ObjectId::new(index as u32, self.namespace, slot.generation, self.kind),
                ))
            })
    }

    fn locate(&self, id: ObjectId) -> Result<usize, ObjectTableError> {
        let raw = id.get();
        let slot_number = raw as u32;
        if slot_number == 0 {
            return Err(ObjectTableError::InvalidId);
        }
        let token = (raw >> 32) as u32;
        if token & KIND_MASK != self.kind.tag() {
            return Err(ObjectTableError::WrongKind);
        }
        if (token >> NAMESPACE_SHIFT) as u16 != self.namespace {
            return Err(ObjectTableError::StaleId);
        }
        let generation = ((token >> GENERATION_SHIFT) as u16) & GENERATION_MASK;
        let index = (slot_number - 1) as usize;
        let slot = self.slots.get(index).ok_or(ObjectTableError::StaleId)?;
        if slot.generation != generation || slot.retired || slot.value.is_none() {
            return Err(ObjectTableError::StaleId);
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_ids_never_resolve_after_slot_reuse() {
        let mut table = ObjectTable::new(ObjectKind::Buffer, 1);
        let old = table.insert(10).unwrap();
        assert_eq!(table.remove(old), Ok(10));
        assert_eq!(table.get(old), Err(ObjectTableError::StaleId));

        let new = table.insert(20).unwrap();
        assert_ne!(old, new);
        assert_eq!(table.get(new), Ok(&20));
        assert_eq!(table.get(old), Err(ObjectTableError::StaleId));
    }

    #[test]
    fn exhausted_generations_retire_the_slot_before_an_id_can_revive() {
        let mut table = ObjectTable::new(ObjectKind::Buffer, 1);
        let first = table.insert(()).unwrap();
        table.remove(first).unwrap();

        for _ in 1..=GENERATION_MASK {
            let current = table.insert(()).unwrap();
            assert_ne!(current, first);
            assert_eq!(table.get(first), Err(ObjectTableError::StaleId));
            table.remove(current).unwrap();
        }

        assert_eq!(table.insert(()), Err(ObjectTableError::Full));
        assert_eq!(table.get(first), Err(ObjectTableError::StaleId));
    }

    #[test]
    fn kind_tags_prevent_cross_table_aliasing() {
        let mut contexts = ObjectTable::new(ObjectKind::Context, 1);
        let id = contexts.insert(()).unwrap();
        let buffers = ObjectTable::<()>::new(ObjectKind::Buffer, 1);
        assert_eq!(buffers.get(id), Err(ObjectTableError::WrongKind));
    }

    #[test]
    fn namespaces_prevent_cross_device_or_reset_epoch_aliasing() {
        let first_namespace = ObjectNamespace::new(1).unwrap();
        let second_namespace = ObjectNamespace::new(2).unwrap();
        let mut first = ObjectTable::with_namespace(ObjectKind::Context, 1, first_namespace);
        let id = first.insert(()).unwrap();
        let second = ObjectTable::<()>::with_namespace(ObjectKind::Context, 1, second_namespace);
        assert_eq!(second.get(id), Err(ObjectTableError::StaleId));
    }

    #[test]
    fn limits_are_enforced_before_growth() {
        let mut table = ObjectTable::new(ObjectKind::Event, 1);
        table.insert(1).unwrap();
        assert_eq!(table.insert(2), Err(ObjectTableError::Full));
        assert_eq!(table.len(), 1);
    }
}
