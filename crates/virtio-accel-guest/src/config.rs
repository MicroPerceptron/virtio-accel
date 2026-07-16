use virtio_accel_proto::{ConfigError, WireConfig};
use virtio_accel_transport::{QueueSize, QueueState};

/// Validated protocol configuration and bounded guest tracking limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestConfig {
    wire: WireConfig,
    max_inflight: u16,
}

impl GuestConfig {
    /// Validate the device-specific configuration and offered feature bits.
    ///
    /// Protocol 1.0 accepts no device-specific transport feature bits, including unknown bits.
    pub fn new(
        wire: WireConfig,
        offered_features: u64,
        max_inflight: u16,
    ) -> Result<Self, GuestConfigError> {
        wire.validate().map_err(GuestConfigError::Wire)?;
        if offered_features != 0 {
            return Err(GuestConfigError::Features);
        }
        if max_inflight == 0 {
            return Err(GuestConfigError::InflightLimit);
        }
        Ok(Self { wire, max_inflight })
    }

    /// Validated wire configuration.
    pub const fn wire(self) -> WireConfig {
        self.wire
    }

    /// Maximum requests retained by this client.
    pub const fn max_inflight(self) -> u16 {
        self.max_inflight
    }

    pub(crate) fn validate_queue(self, state: QueueState) -> Result<(), GuestConfigError> {
        let size = state.size().ok_or(GuestConfigError::QueueNotConfigured)?;
        if !state.ready() {
            return Err(GuestConfigError::QueueNotReady);
        }
        self.validate_size(size)
    }

    pub(crate) fn validate_size(self, size: QueueSize) -> Result<(), GuestConfigError> {
        if self.wire.max_chain_descriptors.get() > size.get() {
            return Err(GuestConfigError::QueueSize);
        }
        if self.max_inflight > size.get() {
            return Err(GuestConfigError::InflightLimit);
        }
        Ok(())
    }
}

/// Invalid guest-visible configuration or negotiation result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestConfigError {
    /// Device-specific configuration violates protocol 1.0.
    Wire(ConfigError),
    /// Protocol 1.0 cannot accept any offered device-specific feature bit.
    Features,
    /// The command queue has no configured size.
    QueueNotConfigured,
    /// The command queue is not ready.
    QueueNotReady,
    /// The queue is smaller than the advertised descriptor-chain limit.
    QueueSize,
    /// The tracking limit is zero or exceeds queue capacity.
    InflightLimit,
}
