use core::fmt;

use crate::Version;
use virtio_accel_core::{ArtifactFormat, TargetIdentity};

/// Raw TOSA FlatBuffer payload (`"TOSA"` as a big-endian four-character code).
pub const ARTIFACT_FORMAT: ArtifactFormat = match ArtifactFormat::new(0x544f_5341) {
    Some(format) => format,
    None => panic!("TOSA artifact format must be nonzero"),
};

const TARGET_MAGIC: u32 = 0x544f_5341;
const TARGET_ABI: u32 = 1;

/// Set of TOSA base profiles implemented by a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ProfileSet(u32);

impl ProfileSet {
    pub const INTEGER: Self = Self(1 << 0);
    pub const FLOATING_POINT: Self = Self(1 << 1);
    pub const ALL: Self = Self(Self::INTEGER.0 | Self::FLOATING_POINT.0);

    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits != 0 && bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

/// TOSA implementation level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Level {
    /// No finite level is claimed; provider-specific limits still apply.
    Unbounded = 0,
    /// TOSA Level 8K.
    Level8K = 1,
}

/// Argument ceilings assigned by a TOSA 1.0 implementation level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelLimits {
    pub max_rank: usize,
    pub max_kernel: i32,
    pub max_stride: i32,
    pub max_scale: i32,
    pub max_log2_size: u32,
    pub max_nesting: usize,
    pub max_tensor_list_size: usize,
}

impl Level {
    pub const fn limits(self) -> LevelLimits {
        match self {
            Self::Unbounded => LevelLimits {
                max_rank: 32,
                max_kernel: i32::MAX,
                max_stride: i32::MAX,
                max_scale: 2_048,
                max_log2_size: 63,
                max_nesting: 256,
                max_tensor_list_size: 256,
            },
            Self::Level8K => LevelLimits {
                max_rank: 6,
                max_kernel: 8_192,
                max_stride: 8_192,
                max_scale: 256,
                max_log2_size: 31,
                max_nesting: 6,
                max_tensor_list_size: 64,
            },
        }
    }
}

impl TryFrom<u32> for Level {
    type Error = TargetError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unbounded),
            1 => Ok(Self::Level8K),
            _ => Err(TargetError::UnknownLevel(value)),
        }
    }
}

/// TOSA 1.0 profile-extension bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ExtensionSet(u64);

impl ExtensionSet {
    pub const NONE: Self = Self(0);
    pub const INT16: Self = Self(1 << 0);
    pub const INT4: Self = Self(1 << 1);
    pub const BF16: Self = Self(1 << 2);
    pub const FP8E4M3: Self = Self(1 << 3);
    pub const FP8E5M2: Self = Self(1 << 4);
    pub const FFT: Self = Self(1 << 5);
    pub const VARIABLE: Self = Self(1 << 6);
    pub const CONTROL_FLOW: Self = Self(1 << 7);
    pub const DYNAMIC: Self = Self(1 << 8);
    pub const DOUBLE_ROUND: Self = Self(1 << 9);
    pub const INEXACT_ROUND: Self = Self(1 << 10);
    pub const ALL: Self = Self((1 << 11) - 1);

    pub const fn from_bits(bits: u64) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

/// Device-neutral target requirements carried in `virtio-accel`'s opaque target words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub version: Version,
    pub profiles: ProfileSet,
    pub level: Level,
    pub extensions: ExtensionSet,
}

impl Target {
    pub const fn new(
        version: Version,
        profiles: ProfileSet,
        level: Level,
        extensions: ExtensionSet,
    ) -> Self {
        Self {
            version,
            profiles,
            level,
            extensions,
        }
    }

    pub const fn to_identity(self) -> TargetIdentity {
        let extensions = self.extensions.bits();
        TargetIdentity([
            TARGET_MAGIC,
            TARGET_ABI,
            self.version.major as u32,
            self.version.minor as u32,
            self.version.patch as u32,
            self.profiles.bits(),
            self.level as u32,
            extensions as u32,
            (extensions >> 32) as u32,
            0,
            0,
            0,
        ])
    }

    /// Check that every extension is paired with one of its permitted base profiles.
    pub fn validate(self) -> Result<Self, TargetError> {
        let integer_only = ExtensionSet::INT16
            .union(ExtensionSet::INT4)
            .union(ExtensionSet::DOUBLE_ROUND)
            .union(ExtensionSet::INEXACT_ROUND);
        let floating_only = ExtensionSet::BF16
            .union(ExtensionSet::FP8E4M3)
            .union(ExtensionSet::FP8E5M2)
            .union(ExtensionSet::FFT);
        if self.extensions.intersects(integer_only)
            && !self.profiles.intersects(ProfileSet::INTEGER)
        {
            return Err(TargetError::ExtensionProfileMismatch {
                extensions: ExtensionSet(self.extensions.bits() & integer_only.bits()),
                required_profiles: ProfileSet::INTEGER,
            });
        }
        if self.extensions.intersects(floating_only)
            && !self.profiles.intersects(ProfileSet::FLOATING_POINT)
        {
            return Err(TargetError::ExtensionProfileMismatch {
                extensions: ExtensionSet(self.extensions.bits() & floating_only.bits()),
                required_profiles: ProfileSet::FLOATING_POINT,
            });
        }
        Ok(self)
    }

    pub fn from_identity(identity: TargetIdentity) -> Result<Self, TargetError> {
        let words = identity.0;
        if words[0] != TARGET_MAGIC {
            return Err(TargetError::WrongMagic(words[0]));
        }
        if words[1] != TARGET_ABI {
            return Err(TargetError::UnknownAbi(words[1]));
        }
        if words[2] > u16::MAX as u32 || words[3] > u16::MAX as u32 || words[4] > u16::MAX as u32 {
            return Err(TargetError::VersionOutOfRange);
        }
        let profiles =
            ProfileSet::from_bits(words[5]).ok_or(TargetError::InvalidProfiles(words[5]))?;
        let level = Level::try_from(words[6])?;
        let extension_bits = u64::from(words[7]) | (u64::from(words[8]) << 32);
        let extensions = ExtensionSet::from_bits(extension_bits)
            .ok_or(TargetError::UnknownExtensions(extension_bits))?;
        if words[9..].iter().any(|word| *word != 0) {
            return Err(TargetError::ReservedWords);
        }
        Self {
            version: Version::new(words[2] as u16, words[3] as u16, words[4] as u16),
            profiles,
            level,
            extensions,
        }
        .validate()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetError {
    WrongMagic(u32),
    UnknownAbi(u32),
    VersionOutOfRange,
    VersionMismatch {
        target: Version,
        model: Version,
    },
    InvalidProfiles(u32),
    UnknownLevel(u32),
    UnknownExtensions(u64),
    ExtensionProfileMismatch {
        extensions: ExtensionSet,
        required_profiles: ProfileSet,
    },
    ReservedWords,
}

impl fmt::Display for TargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    VersionMismatch { model: Version, target: Version },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
