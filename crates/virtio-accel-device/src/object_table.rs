use alloc::vec::Vec;
use core::num::NonZeroU64;

const KIND_MASK: u32 = 0b111;
const GENERATION_STRIDE: u32 = KIND_MASK + 1;

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

    const fn new(index: u32, generation: u32) -> Self {
        let raw = ((generation as u64) << 32) | (index as u64 + 1);
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
    generation: u32,
    value: Option<T>,
    retired: bool,
}

/// Bounded generational object table for one resource kind.
///
/// Kind tags occupy the low three generation bits. A slot is retired before generation overflow,
/// so an old ID cannot become valid again after wraparound.
pub struct ObjectTable<T> {
    kind: ObjectKind,
    max_slots: u32,
    live: u32,
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> ObjectTable<T> {
    pub const fn new(kind: ObjectKind, max_slots: u32) -> Self {
        Self {
            kind,
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
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none() && !slot.retired);
            slot.value = Some(value);
            self.live += 1;
            return Ok(ObjectId::new(index, slot.generation));
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
        let index = self.slots.len() as u32;
        let generation = self.kind.tag();
        self.slots.push(Slot {
            generation,
            value: Some(value),
            retired: false,
        });
        self.live += 1;
        Ok(ObjectId::new(index, generation))
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

        match slot.generation.checked_add(GENERATION_STRIDE) {
            Some(next) => {
                slot.generation = next;
                self.free.push(index as u32);
            }
            None => slot.retired = true,
        }
        Ok(value)
    }

    fn locate(&self, id: ObjectId) -> Result<usize, ObjectTableError> {
        let raw = id.get();
        let slot_number = raw as u32;
        if slot_number == 0 {
            return Err(ObjectTableError::InvalidId);
        }
        let generation = (raw >> 32) as u32;
        if generation & KIND_MASK != self.kind.tag() {
            return Err(ObjectTableError::WrongKind);
        }
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
    fn kind_tags_prevent_cross_table_aliasing() {
        let mut contexts = ObjectTable::new(ObjectKind::Context, 1);
        let id = contexts.insert(()).unwrap();
        let buffers = ObjectTable::<()>::new(ObjectKind::Buffer, 1);
        assert_eq!(buffers.get(id), Err(ObjectTableError::WrongKind));
    }

    #[test]
    fn limits_are_enforced_before_growth() {
        let mut table = ObjectTable::new(ObjectKind::Event, 1);
        table.insert(1).unwrap();
        assert_eq!(table.insert(2), Err(ObjectTableError::Full));
        assert_eq!(table.len(), 1);
    }
}
