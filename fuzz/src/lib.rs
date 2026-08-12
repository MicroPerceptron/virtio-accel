#![forbid(unsafe_code)]

mod descriptor;
mod guest;
mod protocol;
mod stateful;
mod tosa;

pub use descriptor::fuzz_descriptor_end_to_end;
pub use guest::fuzz_guest_client;
pub use protocol::fuzz_protocol_decode;
pub use stateful::fuzz_stateful_commands;
pub use tosa::fuzz_tosa_parse;

pub(crate) const MAX_FRAME_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Input<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Input<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    pub(crate) fn byte(&mut self) -> u8 {
        let value = self.bytes.get(self.cursor).copied().unwrap_or(0);
        self.cursor = self.cursor.saturating_add(1);
        value
    }

    pub(crate) fn u16(&mut self) -> u16 {
        u16::from_le_bytes([self.byte(), self.byte()])
    }

    pub(crate) fn u64(&mut self) -> u64 {
        u64::from_le_bytes([
            self.byte(),
            self.byte(),
            self.byte(),
            self.byte(),
            self.byte(),
            self.byte(),
            self.byte(),
            self.byte(),
        ])
    }

    pub(crate) fn remaining(&self) -> &'a [u8] {
        self.bytes.get(self.cursor..).unwrap_or_default()
    }
}
