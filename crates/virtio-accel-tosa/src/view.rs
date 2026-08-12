use crate::artifact::{ArtifactError, Target};
use crate::generated::tosa as wire;
use crate::{AttributeKind, DType, Op, OpAttributes, Stats, Version};
use flatbuffers::{ForwardsUOffset, Vector};
use virtio_accel_core::{ArtifactRef, BackendError, ByteSource};

type WireRegions<'a> = Vector<'a, ForwardsUOffset<wire::TosaRegion<'a>>>;
type WireBlocks<'a> = Vector<'a, ForwardsUOffset<wire::TosaBasicBlock<'a>>>;
type WireTensors<'a> = Vector<'a, ForwardsUOffset<wire::TosaTensor<'a>>>;
type WireShapes<'a> = Vector<'a, ForwardsUOffset<wire::TosaShape<'a>>>;
type WireOperators<'a> = Vector<'a, ForwardsUOffset<wire::TosaOperator<'a>>>;
type WireStrings<'a> = Vector<'a, ForwardsUOffset<&'a str>>;

/// A verified, zero-copy TOSA graph.
#[derive(Clone, Copy)]
pub struct Model<'a> {
    pub(crate) graph: wire::TosaGraph<'a>,
    pub(crate) bytes: &'a [u8],
    pub(crate) version: Version,
    pub(crate) stats: Stats,
    pub(crate) source: SliceSource<'a>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SliceSource<'a>(pub(crate) &'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        ByteSource::read_at(self.0, offset, target)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
}

impl core::fmt::Debug for Model<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Model")
            .field("version", &self.version)
            .field("stats", &self.stats)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl<'a> Model<'a> {
    pub const fn version(&self) -> Version {
        self.version
    }

    pub const fn stats(&self) -> Stats {
        self.stats
    }

    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn regions(&self) -> Regions<'a> {
        Regions::new(self.graph.regions())
    }

    /// Run an operator- or provider-specific validation pass over this safe graph view.
    pub fn validate_with<V: ModelValidator + ?Sized>(
        &self,
        validator: &mut V,
    ) -> Result<(), V::Error> {
        validator.validate(self)
    }

    /// Wrap these exact validated bytes in the standard `virtio-accel` artifact envelope.
    pub fn artifact_ref(
        &self,
        target: Target,
        resident_bytes: u64,
    ) -> Result<ArtifactRef<'_>, ArtifactError> {
        if target.version != self.version {
            return Err(ArtifactError::VersionMismatch {
                model: self.version,
                target: target.version,
            });
        }
        Ok(ArtifactRef {
            format: crate::ARTIFACT_FORMAT,
            target: target.to_identity(),
            payload: &self.source,
            resident_bytes,
        })
    }

    /// Apply the complete stable TOSA semantic pass for a device-neutral target.
    pub fn validate_for(&self, target: Target) -> Result<(), crate::SemanticError> {
        crate::validate_semantics(self, target)
    }

    /// Validate once and build a compact provider-neutral lowering plan over this borrowed model.
    pub fn analyze_for(
        &self,
        target: Target,
    ) -> Result<crate::TosaAnalysis<'a>, crate::AnalysisError> {
        crate::TosaAnalysis::build(self, target)
    }
}

/// Extension point for semantic, profile, extension, or backend-capability validators.
pub trait ModelValidator {
    type Error;

    fn validate(&mut self, model: &Model<'_>) -> Result<(), Self::Error>;
}

macro_rules! table_iter {
    ($iterator:ident, $item:ident, $wire:ty, $wrapper:expr) => {
        #[derive(Clone)]
        pub struct $iterator<'a> {
            vector: Option<$wire>,
            index: usize,
        }

        impl<'a> $iterator<'a> {
            fn new(vector: Option<$wire>) -> Self {
                Self { vector, index: 0 }
            }
        }

        impl<'a> Iterator for $iterator<'a> {
            type Item = $item<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                let vector = self.vector?;
                if self.index >= vector.len() {
                    return None;
                }
                let item = vector.get(self.index);
                self.index += 1;
                Some($wrapper(item))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let remaining = self
                    .vector
                    .map_or(0, |vector| vector.len().saturating_sub(self.index));
                (remaining, Some(remaining))
            }
        }

        impl ExactSizeIterator for $iterator<'_> {}
    };
}

/// Borrowed region view.
#[derive(Clone, Copy, Debug)]
pub struct Region<'a>(wire::TosaRegion<'a>);

impl<'a> Region<'a> {
    pub fn name(&self) -> &'a str {
        self.0.name().expect("validated region name")
    }

    pub fn blocks(&self) -> BasicBlocks<'a> {
        BasicBlocks::new(self.0.blocks())
    }
}

table_iter!(Regions, Region, WireRegions<'a>, Region);

/// Borrowed basic-block view.
#[derive(Clone, Copy, Debug)]
pub struct BasicBlock<'a>(wire::TosaBasicBlock<'a>);

impl<'a> BasicBlock<'a> {
    pub fn name(&self) -> &'a str {
        self.0.name().expect("validated block name")
    }

    pub fn tensors(&self) -> Tensors<'a> {
        Tensors::new(self.0.tensors())
    }

    pub fn shapes(&self) -> Shapes<'a> {
        Shapes::new(self.0.shapes())
    }

    pub fn operators(&self) -> Operators<'a> {
        Operators::new(self.0.operators())
    }

    pub fn inputs(&self) -> StringList<'a> {
        StringList::new(self.0.inputs())
    }

    pub fn outputs(&self) -> StringList<'a> {
        StringList::new(self.0.outputs())
    }
}

table_iter!(BasicBlocks, BasicBlock, WireBlocks<'a>, BasicBlock);

/// Borrowed tensor view.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a>(wire::TosaTensor<'a>);

impl<'a> Tensor<'a> {
    pub fn name(&self) -> &'a str {
        self.0.name().expect("validated tensor name")
    }

    pub fn dtype(&self) -> DType {
        DType::new(self.0.type_().0)
    }

    pub fn rank(&self) -> Option<usize> {
        if self.0.is_unranked() {
            None
        } else {
            Some(self.0.shape().map_or(0, |shape| shape.len()))
        }
    }

    pub fn dimensions(&self) -> impl Iterator<Item = i32> + 'a {
        self.0.shape().into_iter().flat_map(|shape| shape.iter())
    }

    /// One ranked dimension without constructing or advancing an iterator.
    pub fn dimension(&self, index: usize) -> Option<i32> {
        self.0
            .shape()
            .filter(|_| !self.0.is_unranked())
            .and_then(|shape| (index < shape.len()).then(|| shape.get(index)))
    }

    pub fn data(&self) -> &'a [u8] {
        self.0.data().map_or(&[], |data| data.bytes())
    }

    pub fn is_variable(&self) -> bool {
        self.0.variable()
    }

    pub fn variable_name(&self) -> Option<&'a str> {
        self.0.variable_name()
    }

    pub fn external_data_range(&self) -> Option<(u64, u64)> {
        let size = self.0.size();
        (size != 0).then_some((self.0.offset(), size))
    }
}

table_iter!(Tensors, Tensor, WireTensors<'a>, Tensor);

/// Borrowed shape-value view.
#[derive(Clone, Copy, Debug)]
pub struct Shape<'a>(wire::TosaShape<'a>);

impl<'a> Shape<'a> {
    pub fn name(&self) -> &'a str {
        self.0.name().expect("validated shape name")
    }

    pub fn rank(&self) -> u32 {
        self.0.rank()
    }

    pub fn data(&self) -> &'a [u8] {
        self.0.data().map_or(&[], |data| data.bytes())
    }

    /// Decoded shape values, or `None` for a nonempty intermediate shape without constant data.
    /// A rank-zero shape is the constant empty list and therefore returns an empty iterator.
    pub fn values(&self) -> Option<ShapeValues<'a>> {
        let data = self.data();
        (!data.is_empty() || self.rank() == 0).then_some(ShapeValues {
            data,
            index: 0,
            len: self.rank() as usize,
        })
    }
}

table_iter!(Shapes, Shape, WireShapes<'a>, Shape);

/// Exact-size iterator over little-endian `i64` shape values.
#[derive(Clone)]
pub struct ShapeValues<'a> {
    data: &'a [u8],
    index: usize,
    len: usize,
}

impl Iterator for ShapeValues<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let start = self.index * core::mem::size_of::<i64>();
        let bytes: [u8; 8] = self.data[start..start + 8]
            .try_into()
            .expect("validated shape data");
        self.index += 1;
        Some(i64::from_le_bytes(bytes))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ShapeValues<'_> {}

/// Borrowed operator view.
#[derive(Clone, Copy, Debug)]
pub struct Operator<'a>(wire::TosaOperator<'a>);

impl<'a> Operator<'a> {
    pub fn op(&self) -> Op {
        Op::new(self.0.op().0)
    }

    pub fn attribute_kind(&self) -> AttributeKind {
        AttributeKind::new(self.0.attribute_type().0)
    }

    /// Safe view of every field in this operator's stable TOSA 1.0 attribute table.
    pub fn attributes(&self) -> OpAttributes<'a> {
        OpAttributes::from_wire(self.0)
    }

    pub fn inputs(&self) -> StringList<'a> {
        StringList::new(self.0.inputs())
    }

    pub fn outputs(&self) -> StringList<'a> {
        StringList::new(self.0.outputs())
    }

    pub fn location(&self) -> Option<&'a str> {
        self.0.location().and_then(|location| location.text())
    }
}

table_iter!(Operators, Operator, WireOperators<'a>, Operator);

/// Iterator over borrowed TOSA symbol names.
#[derive(Clone)]
pub struct StringList<'a> {
    vector: Option<WireStrings<'a>>,
    index: usize,
}

impl<'a> StringList<'a> {
    fn new(vector: Option<WireStrings<'a>>) -> Self {
        Self { vector, index: 0 }
    }
}

impl<'a> Iterator for StringList<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let vector = self.vector?;
        if self.index >= vector.len() {
            return None;
        }
        let item = vector.get(self.index);
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .vector
            .map_or(0, |vector| vector.len().saturating_sub(self.index));
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for StringList<'_> {}
