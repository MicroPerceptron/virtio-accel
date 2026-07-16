//! Transport-neutral flattened descriptor metadata.

/// Direction of one flattened descriptor-backed byte region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionDirection {
    /// Bytes are readable by the device and contain request data.
    DeviceReadable,
    /// Bytes are writable by the device and contain response data.
    DeviceWritable,
}

/// Direction and length of one flattened chain region, without an address or transport identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainRegion {
    /// Direction in which the device may access this region.
    pub direction: RegionDirection,
    /// Nonzero region length in bytes.
    pub bytes: u64,
}

impl ChainRegion {
    /// Construct a device-readable region.
    pub const fn readable(bytes: u64) -> Self {
        Self {
            direction: RegionDirection::DeviceReadable,
            bytes,
        }
    }

    /// Construct a device-writable region.
    pub const fn writable(bytes: u64) -> Self {
        Self {
            direction: RegionDirection::DeviceWritable,
            bytes,
        }
    }
}

/// Failure to validate the device-specific flattened chain layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainLayoutError {
    /// The chain contains fewer than two or more than the configured maximum descriptors.
    DescriptorCount,
    /// A descriptor has zero length.
    ZeroLength,
    /// A readable descriptor follows a writable descriptor, or one direction is missing.
    Direction,
    /// A readable or writable byte total overflowed `u64`.
    LengthOverflow,
    /// The mapped byte ports do not exactly match the flattened descriptor totals.
    PortLengthMismatch,
}

/// Validated counts and byte totals for one flattened descriptor chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainLayout {
    descriptor_count: u16,
    readable_descriptors: u16,
    writable_descriptors: u16,
    readable_bytes: u64,
    writable_bytes: u64,
}

impl ChainLayout {
    /// Total flattened descriptor count.
    pub const fn descriptor_count(self) -> u16 {
        self.descriptor_count
    }

    /// Number of device-readable descriptors.
    pub const fn readable_descriptors(self) -> u16 {
        self.readable_descriptors
    }

    /// Number of device-writable descriptors.
    pub const fn writable_descriptors(self) -> u16 {
        self.writable_descriptors
    }

    /// Total device-readable bytes.
    pub const fn readable_bytes(self) -> u64 {
        self.readable_bytes
    }

    /// Total device-writable bytes.
    pub const fn writable_bytes(self) -> u64 {
        self.writable_bytes
    }

    /// Validate mapped byte-port lengths against this layout.
    pub const fn validate_port_lengths(
        self,
        request_bytes: u64,
        response_bytes: u64,
    ) -> Result<(), ChainLayoutError> {
        if request_bytes != self.readable_bytes || response_bytes != self.writable_bytes {
            return Err(ChainLayoutError::PortLengthMismatch);
        }
        Ok(())
    }
}

/// Validate transport-neutral descriptor metadata before frame bytes are decoded.
///
/// This operation is nonblocking, performs no allocation, and visits each region once. Guest
/// addresses, descriptor indices, and mapping details remain owned by the transport adapter.
pub fn validate_chain_layout(
    regions: &[ChainRegion],
    max_descriptors: u16,
) -> Result<ChainLayout, ChainLayoutError> {
    if regions.len() < 2 || regions.len() > usize::from(max_descriptors) {
        return Err(ChainLayoutError::DescriptorCount);
    }

    let mut readable_descriptors = 0_u16;
    let mut writable_descriptors = 0_u16;
    let mut readable_bytes = 0_u64;
    let mut writable_bytes = 0_u64;
    let mut saw_writable = false;

    for region in regions {
        if region.bytes == 0 {
            return Err(ChainLayoutError::ZeroLength);
        }
        match region.direction {
            RegionDirection::DeviceReadable => {
                if saw_writable {
                    return Err(ChainLayoutError::Direction);
                }
                readable_descriptors += 1;
                readable_bytes = readable_bytes
                    .checked_add(region.bytes)
                    .ok_or(ChainLayoutError::LengthOverflow)?;
            }
            RegionDirection::DeviceWritable => {
                saw_writable = true;
                writable_descriptors += 1;
                writable_bytes = writable_bytes
                    .checked_add(region.bytes)
                    .ok_or(ChainLayoutError::LengthOverflow)?;
            }
        }
    }

    if readable_descriptors == 0 || writable_descriptors == 0 {
        return Err(ChainLayoutError::Direction);
    }

    Ok(ChainLayout {
        descriptor_count: regions.len() as u16,
        readable_descriptors,
        writable_descriptors,
        readable_bytes,
        writable_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_layout_rejects_direction_and_length_errors() {
        assert_eq!(
            validate_chain_layout(&[ChainRegion::readable(16)], 8),
            Err(ChainLayoutError::DescriptorCount)
        );
        assert_eq!(
            validate_chain_layout(
                &[
                    ChainRegion::readable(16),
                    ChainRegion::writable(16),
                    ChainRegion::readable(1),
                ],
                8,
            ),
            Err(ChainLayoutError::Direction)
        );
        assert_eq!(
            validate_chain_layout(&[ChainRegion::readable(0), ChainRegion::writable(16)], 8),
            Err(ChainLayoutError::ZeroLength)
        );
        assert_eq!(
            validate_chain_layout(
                &[
                    ChainRegion::readable(u64::MAX),
                    ChainRegion::readable(1),
                    ChainRegion::writable(16),
                ],
                8,
            ),
            Err(ChainLayoutError::LengthOverflow)
        );

        let layout =
            validate_chain_layout(&[ChainRegion::readable(16), ChainRegion::writable(16)], 8)
                .unwrap();
        assert_eq!(
            layout.validate_port_lengths(15, 16),
            Err(ChainLayoutError::PortLengthMismatch)
        );
    }
}
