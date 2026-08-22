use crate::{DType, Op, Target};

macro_rules! flags {
    ($name:ident, $bits:ty) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name($bits);

        impl $name {
            pub const NONE: Self = Self(0);

            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            pub const fn bits(self) -> $bits {
                self.0
            }
        }
    };
}

flags!(ValueRoles, u8);

impl ValueRoles {
    /// Graph-visible block input.
    pub const INPUT: Self = Self(1 << 0);
    /// Graph-visible block output.
    pub const OUTPUT: Self = Self(1 << 1);
    /// Serialized constant tensor.
    pub const CONSTANT: Self = Self(1 << 2);
    /// Non-boundary value produced and consumed inside a graph.
    pub const INTERMEDIATE: Self = Self(1 << 3);
    /// Every tensor role.
    pub const ALL: Self =
        Self(Self::INPUT.0 | Self::OUTPUT.0 | Self::CONSTANT.0 | Self::INTERMEDIATE.0);
}

flags!(DTypeConstraints, u8);

impl DTypeConstraints {
    /// This dtype is accepted only when a constant is consumed as a compile-time operator
    /// parameter and does not become a graph-visible provider tensor.
    pub const PARAMETER_ONLY: Self = Self(1 << 0);
}

flags!(OperatorConstraints, u16);

impl OperatorConstraints {
    /// Every NaN-mode attribute on this operator must select propagating NaNs.
    pub const PROPAGATING_NAN: Self = Self(1 << 0);
    /// Pool padding values must all be zero.
    pub const ZERO_PADDING: Self = Self(1 << 1);
    /// Shape or permutation operands must be serialized compile-time constants.
    pub const CONSTANT_PARAMETERS: Self = Self(1 << 2);
    /// TOSA zero-point operands must be serialized zeros.
    pub const ZERO_ZERO_POINTS: Self = Self(1 << 3);
    /// The TOSA `MUL` shift operand must be a serialized zero.
    pub const ZERO_SHIFT: Self = Self(1 << 4);
}

/// One dtype admitted in explicitly listed graph roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DTypeCapability {
    pub dtype: DType,
    pub roles: ValueRoles,
    pub constraints: DTypeConstraints,
}

impl DTypeCapability {
    pub const fn new(dtype: DType, roles: ValueRoles) -> Self {
        Self {
            dtype,
            roles,
            constraints: DTypeConstraints::NONE,
        }
    }

    pub const fn constrained(
        dtype: DType,
        roles: ValueRoles,
        constraints: DTypeConstraints,
    ) -> Self {
        Self {
            dtype,
            roles,
            constraints,
        }
    }
}

/// One implemented operator and the conservative restrictions a scheduler must preserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorCapability {
    pub op: Op,
    pub constraints: OperatorConstraints,
}

impl OperatorCapability {
    pub const fn new(op: Op) -> Self {
        Self {
            op,
            constraints: OperatorConstraints::NONE,
        }
    }

    pub const fn constrained(op: Op, constraints: OperatorConstraints) -> Self {
        Self { op, constraints }
    }
}

/// Provider treatment of semantic runtime conditions derived during TOSA analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeConditionSupport {
    /// No analyzed runtime conditions are accepted at program admission.
    None,
    /// Advisory `REQUIRE` conditions may remain, but mandatory dynamic conditions are rejected.
    AdvisoryOnly,
}

/// Whole-graph structural boundary that is not reducible to an operator bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphCapabilities {
    /// Maximum regions accepted in one artifact.
    pub max_regions: usize,
    /// Maximum basic blocks accepted across all regions.
    pub max_blocks: usize,
    /// Whether graph-visible tensor dimensions may remain dynamic at load time.
    pub dynamic_shapes: bool,
    /// Runtime conditions the provider may retain after semantic analysis.
    pub runtime_conditions: RuntimeConditionSupport,
}

/// Conservative semantic admission descriptor for one exact TOSA target.
///
/// A positive query means that scheduling a `load_program` attempt is lawful. It does not promise
/// that concrete shapes, cross-operand relationships, resource availability, native compilation,
/// or runtime device state will succeed. Program admission remains authoritative.
#[derive(Clone, Copy, Debug)]
pub struct CapabilityDescriptor {
    pub target: Target,
    pub dtypes: &'static [DTypeCapability],
    pub operators: &'static [OperatorCapability],
    pub graph: GraphCapabilities,
}

impl CapabilityDescriptor {
    pub const fn dtype(self, dtype: DType) -> Option<DTypeCapability> {
        let mut index = 0;
        while index < self.dtypes.len() {
            let capability = self.dtypes[index];
            if capability.dtype.get() == dtype.get() {
                return Some(capability);
            }
            index += 1;
        }
        None
    }

    pub const fn supports_dtype(self, dtype: DType, role: ValueRoles) -> bool {
        match self.dtype(dtype) {
            Some(capability) => capability.roles.contains(role),
            None => false,
        }
    }

    pub const fn operator(self, op: Op) -> Option<OperatorCapability> {
        let mut index = 0;
        while index < self.operators.len() {
            let capability = self.operators[index];
            if capability.op.get() == op.get() {
                return Some(capability);
            }
            index += 1;
        }
        None
    }

    pub const fn supports_operator(self, op: Op) -> bool {
        self.operator(op).is_some()
    }
}

/// Optional host-side interface implemented by concrete TOSA providers.
///
/// Providers return an empty slice when their native runtime/device is unavailable. Each entry is
/// an exact target/profile tier; consumers must not combine fields from different descriptors.
pub trait TosaCapabilityProvider {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExtensionSet, Level, ProfileSet, Version};

    const DTYPES: &[DTypeCapability] = &[
        DTypeCapability::new(DType::FP32, ValueRoles::ALL),
        DTypeCapability::constrained(
            DType::INT8,
            ValueRoles::CONSTANT,
            DTypeConstraints::PARAMETER_ONLY,
        ),
    ];
    const OPERATORS: &[OperatorCapability] = &[
        OperatorCapability::new(Op::IDENTITY),
        OperatorCapability::constrained(Op::MAXIMUM, OperatorConstraints::PROPAGATING_NAN),
    ];
    const DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
        target: Target::new(
            Version::TOSA_1_0,
            ProfileSet::FLOATING_POINT,
            Level::Level8K,
            ExtensionSet::NONE,
        ),
        dtypes: DTYPES,
        operators: OPERATORS,
        graph: GraphCapabilities {
            max_regions: 1,
            max_blocks: 1,
            dynamic_shapes: false,
            runtime_conditions: RuntimeConditionSupport::None,
        },
    };

    #[test]
    fn role_queries_do_not_turn_parameter_constants_into_boundaries() {
        assert!(DESCRIPTOR.supports_dtype(DType::FP32, ValueRoles::INPUT));
        assert!(DESCRIPTOR.supports_dtype(DType::INT8, ValueRoles::CONSTANT));
        assert!(!DESCRIPTOR.supports_dtype(DType::INT8, ValueRoles::INPUT));
        assert!(
            DESCRIPTOR
                .dtype(DType::INT8)
                .unwrap()
                .constraints
                .contains(DTypeConstraints::PARAMETER_ONLY)
        );
    }

    #[test]
    fn operator_queries_return_scheduler_visible_restrictions() {
        assert!(DESCRIPTOR.supports_operator(Op::IDENTITY));
        assert!(!DESCRIPTOR.supports_operator(Op::ERF));
        assert!(
            DESCRIPTOR
                .operator(Op::MAXIMUM)
                .unwrap()
                .constraints
                .contains(OperatorConstraints::PROPAGATING_NAN)
        );
    }
}
