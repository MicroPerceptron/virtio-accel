#[cfg(any(target_os = "macos", test))]
use std::collections::BTreeSet;

const MAGIC: [u8; 4] = *b"CMLP";
const VERSION_MAJOR: u8 = 1;
const VERSION_MINOR: u8 = 0;
const HEADER_BYTES: usize = 16;
const ENTRY_HEADER_BYTES: usize = 8;
pub(crate) const MAX_ARTIFACT_BYTES: u64 = 64 * 1024;
const MAX_MODEL_PATH_BYTES: usize = 4 * 1024;
const MAX_FEATURE_NAME_BYTES: usize = 1024;
const MAX_MAPPINGS: usize = 256;

/// A Core ML model feature's role in one virtio-accel binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FeatureRole {
    Input = 1,
    Output = 2,
}

impl FeatureRole {
    #[cfg(any(target_os = "macos", test))]
    fn from_wire(value: u8) -> Result<Self, DecodeError> {
        match value {
            1 => Ok(Self::Input),
            2 => Ok(Self::Output),
            _ => Err(DecodeError::Invalid),
        }
    }
}

/// Invalid Core ML path artifact construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactBuildError {
    EmptyModelPath,
    ModelPathTooLong,
    EmptyFeatureName,
    FeatureNameTooLong,
    DuplicateFeature,
    EmptyMappings,
    TooManyMappings,
    ArtifactTooLarge,
}

impl std::fmt::Display for ArtifactBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ArtifactBuildError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FeatureMapping {
    pub slot: u32,
    pub role: FeatureRole,
    pub name: String,
}

/// Builder for the provider-owned Core ML path artifact.
///
/// The model path is relative to the host-selected model root. Every nonoptional model input and
/// output must be mapped exactly once. Mapping one input and one compatible output to the same slot
/// creates an in-place `ReadWrite` binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreMlArtifact {
    model_path: String,
    mappings: Vec<FeatureMapping>,
}

impl CoreMlArtifact {
    pub fn new(model_path: impl Into<String>) -> Result<Self, ArtifactBuildError> {
        let model_path = model_path.into();
        if model_path.is_empty() {
            return Err(ArtifactBuildError::EmptyModelPath);
        }
        if model_path.len() > MAX_MODEL_PATH_BYTES || model_path.len() > u16::MAX as usize {
            return Err(ArtifactBuildError::ModelPathTooLong);
        }
        Ok(Self {
            model_path,
            mappings: Vec::new(),
        })
    }

    pub fn map_input(
        self,
        slot: u32,
        feature_name: impl Into<String>,
    ) -> Result<Self, ArtifactBuildError> {
        self.map(slot, FeatureRole::Input, feature_name.into())
    }

    pub fn map_output(
        self,
        slot: u32,
        feature_name: impl Into<String>,
    ) -> Result<Self, ArtifactBuildError> {
        self.map(slot, FeatureRole::Output, feature_name.into())
    }

    pub fn encode(&self) -> Result<Vec<u8>, ArtifactBuildError> {
        if self.mappings.is_empty() {
            return Err(ArtifactBuildError::EmptyMappings);
        }
        if self.mappings.len() > MAX_MAPPINGS || self.mappings.len() > u16::MAX as usize {
            return Err(ArtifactBuildError::TooManyMappings);
        }
        let mut bytes = Vec::with_capacity(
            HEADER_BYTES
                + self.model_path.len()
                + self
                    .mappings
                    .iter()
                    .map(|mapping| ENTRY_HEADER_BYTES + mapping.name.len())
                    .sum::<usize>(),
        );
        bytes.extend_from_slice(&MAGIC);
        bytes.push(VERSION_MAJOR);
        bytes.push(VERSION_MINOR);
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&(self.model_path.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&(self.mappings.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(self.model_path.as_bytes());
        for mapping in &self.mappings {
            bytes.extend_from_slice(&mapping.slot.to_le_bytes());
            bytes.push(mapping.role as u8);
            bytes.push(0);
            bytes.extend_from_slice(&(mapping.name.len() as u16).to_le_bytes());
            bytes.extend_from_slice(mapping.name.as_bytes());
        }
        if bytes.len() as u64 > MAX_ARTIFACT_BYTES {
            return Err(ArtifactBuildError::ArtifactTooLarge);
        }
        Ok(bytes)
    }

    fn map(
        mut self,
        slot: u32,
        role: FeatureRole,
        name: String,
    ) -> Result<Self, ArtifactBuildError> {
        if name.is_empty() {
            return Err(ArtifactBuildError::EmptyFeatureName);
        }
        if name.len() > MAX_FEATURE_NAME_BYTES || name.len() > u16::MAX as usize {
            return Err(ArtifactBuildError::FeatureNameTooLong);
        }
        if self
            .mappings
            .iter()
            .any(|mapping| mapping.role == role && mapping.name == name)
        {
            return Err(ArtifactBuildError::DuplicateFeature);
        }
        if self.mappings.len() == MAX_MAPPINGS {
            return Err(ArtifactBuildError::TooManyMappings);
        }
        self.mappings.push(FeatureMapping { slot, role, name });
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(target_os = "macos", test))]
pub(crate) enum DecodeError {
    Invalid,
    OutOfBounds,
    ResourceLimit,
}

#[derive(Debug)]
#[cfg(any(target_os = "macos", test))]
pub(crate) struct DecodedArtifact {
    pub model_path: String,
    pub mappings: Vec<FeatureMapping>,
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn decode<S: virtio_accel_core::ByteSource + ?Sized>(
    source: &S,
) -> Result<DecodedArtifact, DecodeError> {
    if source.len() > MAX_ARTIFACT_BYTES {
        return Err(DecodeError::ResourceLimit);
    }
    if source.len() < HEADER_BYTES as u64 {
        return Err(DecodeError::Invalid);
    }

    let mut header = [0; HEADER_BYTES];
    source
        .read_at(0, &mut header)
        .map_err(|_| DecodeError::OutOfBounds)?;
    if header[..4] != MAGIC
        || header[4] != VERSION_MAJOR
        || header[5] != VERSION_MINOR
        || header[6..8] != [0, 0]
        || header[12..16] != [0, 0, 0, 0]
    {
        return Err(DecodeError::Invalid);
    }
    let path_len = u16::from_le_bytes([header[8], header[9]]) as usize;
    let mapping_count = u16::from_le_bytes([header[10], header[11]]) as usize;
    if path_len == 0 || path_len > MAX_MODEL_PATH_BYTES || mapping_count > MAX_MAPPINGS {
        return Err(DecodeError::Invalid);
    }

    let mut cursor = HEADER_BYTES as u64;
    let mut path = vec![0; path_len];
    source
        .read_at(cursor, &mut path)
        .map_err(|_| DecodeError::OutOfBounds)?;
    cursor += path_len as u64;
    let model_path = String::from_utf8(path).map_err(|_| DecodeError::Invalid)?;

    let mut mappings = Vec::new();
    mappings
        .try_reserve_exact(mapping_count)
        .map_err(|_| DecodeError::ResourceLimit)?;
    let mut features = BTreeSet::new();
    for _ in 0..mapping_count {
        let mut entry = [0; ENTRY_HEADER_BYTES];
        source
            .read_at(cursor, &mut entry)
            .map_err(|_| DecodeError::OutOfBounds)?;
        cursor += ENTRY_HEADER_BYTES as u64;
        if entry[5] != 0 {
            return Err(DecodeError::Invalid);
        }
        let slot = u32::from_le_bytes(entry[..4].try_into().unwrap());
        let role = FeatureRole::from_wire(entry[4])?;
        let name_len = u16::from_le_bytes([entry[6], entry[7]]) as usize;
        if name_len == 0 || name_len > MAX_FEATURE_NAME_BYTES {
            return Err(DecodeError::Invalid);
        }
        let mut name = vec![0; name_len];
        source
            .read_at(cursor, &mut name)
            .map_err(|_| DecodeError::OutOfBounds)?;
        cursor += name_len as u64;
        let name = String::from_utf8(name).map_err(|_| DecodeError::Invalid)?;
        if !features.insert((role as u8, name.clone())) {
            return Err(DecodeError::Invalid);
        }
        mappings.push(FeatureMapping { slot, role, name });
    }
    if cursor != source.len() || mappings.is_empty() {
        return Err(DecodeError::Invalid);
    }
    Ok(DecodedArtifact {
        model_path,
        mappings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_artifact_round_trips() {
        let artifact = CoreMlArtifact::new("models/twice.mlmodel")
            .unwrap()
            .map_input(7, "x")
            .unwrap()
            .map_output(7, "y")
            .unwrap();
        let bytes = artifact.encode().unwrap();
        let decoded = decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded.model_path, "models/twice.mlmodel");
        assert_eq!(decoded.mappings, artifact.mappings);
    }

    #[test]
    fn malformed_envelopes_are_rejected() {
        assert_eq!(
            CoreMlArtifact::new("model.mlmodel").unwrap().encode(),
            Err(ArtifactBuildError::EmptyMappings)
        );
        let artifact = CoreMlArtifact::new("model.mlmodel")
            .unwrap()
            .map_input(0, "x")
            .unwrap();
        let mut bytes = artifact.encode().unwrap();
        bytes[6] = 1;
        assert_eq!(decode(bytes.as_slice()).unwrap_err(), DecodeError::Invalid);

        let duplicate = CoreMlArtifact {
            model_path: "model.mlmodel".into(),
            mappings: vec![
                FeatureMapping {
                    slot: 0,
                    role: FeatureRole::Input,
                    name: "x".into(),
                },
                FeatureMapping {
                    slot: 1,
                    role: FeatureRole::Input,
                    name: "x".into(),
                },
            ],
        }
        .encode()
        .unwrap();
        assert_eq!(
            decode(duplicate.as_slice()).unwrap_err(),
            DecodeError::Invalid
        );
    }
}
