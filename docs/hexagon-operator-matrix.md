# Qualcomm Hexagon TOSA operator matrix

This matrix records the TOSA 1.0 floating-point surface validated on Snapdragon X126100, Hexagon
HTP v73, driver `30.0.222.0`, and QAIRT `2.49.0.260730`. “HTP” means the checked-in numerical
fixture passed through the QNN HTP backend; “portable” means lowering is also covered without the
SDK. The floating target admits FP16, BOOL conditions/results, and required INT32 indexing results.
The separate integer target remains limited to INT8 `IDENTITY` and zero-point-aware INT8 `MATMUL`
with INT32 output.

| TOSA operator | QNN representation | Validated restriction | Evidence |
|---|---|---|---|
| `ARGMAX` | `ArgMax` | FP16 input, INT32 output, one static axis, `keep_dims=false` | portable + HTP |
| `MAX_POOL2D` | `Gather` + `ElementWiseMaximum` | FP16 NHWC, positive static kernel/stride, zero padding | portable + HTP |
| `MATMUL` | `MatMul` | FP16, or exact INT8/INT32 tier; no transpose | portable + HTP |
| `CLAMP` | `ReluMinMax` | FP16 bounds, NaN propagate | portable + HTP |
| `ERF` | none | blocked: QAIRT 2.49 public `QnnOpDef.h` defines no ERF operation | parity exception |
| `SIGMOID` | `Sigmoid` | FP16 | portable + HTP |
| `TANH` | `Tanh` | FP16 | portable + HTP |
| `ADD` | `ElementWiseAdd` | FP16 broadcasting | portable + HTP |
| `LOGICAL_AND` | `ElementWiseAnd` | BOOL | portable + HTP |
| `LOGICAL_OR` | `ElementWiseOr` | BOOL | portable + HTP |
| `LOGICAL_XOR` | `ElementWiseXor` | BOOL | portable + HTP |
| `MAXIMUM` | `ElementWiseMaximum` | FP16 broadcasting, NaN propagate | portable + HTP |
| `MINIMUM` | `ElementWiseMinimum` | FP16 broadcasting, NaN propagate | portable + HTP |
| `MUL` | `ElementWiseMultiply` | FP16 broadcasting, validated zero shift | portable + HTP |
| `POW` | `ElementWisePower` | FP16 broadcasting; TOSA domain requirement remains caller-visible | portable + HTP |
| `SUB` | `ElementWiseSubtract` | FP16 broadcasting | portable + HTP |
| `ABS` | `ElementWiseAbs` | FP16 | portable + HTP |
| `CEIL` | `ElementWiseCeil` | FP16 | portable + HTP |
| `COS` | `ElementWiseCos` | FP16, 8-ULP oracle bound | portable + HTP |
| `EXP` | `ElementWiseExp` | FP16 | portable + HTP |
| `FLOOR` | `ElementWiseFloor` | FP16 | portable + HTP |
| `LOG` | `ElementWiseLog` | positive FP16 fixture domain | portable + HTP |
| `LOGICAL_NOT` | `ElementWiseNot` | BOOL | portable + HTP |
| `NEGATE` | `ElementWiseNeg` | FP16, zero input/output zero points | portable + HTP |
| `RECIPROCAL` | `ElementWiseUnary(RECIPROCAL)` | nonzero FP16 fixture domain | portable + HTP |
| `RSQRT` | `ElementWiseRsqrt` | positive FP16 fixture domain | portable + HTP |
| `SIN` | `ElementWiseSin` | FP16, 8-ULP oracle bound | portable + HTP |
| `SELECT` | `ElementWiseSelect` | BOOL condition, FP16 values | portable + HTP |
| `EQUAL` | `ElementWiseEqual` | FP16 input, BOOL output | portable + HTP |
| `GREATER` | `ElementWiseGreater` | FP16 input, BOOL output | portable + HTP |
| `GREATER_EQUAL` | `ElementWiseGreaterEqual` | FP16 input, BOOL output | portable + HTP |
| `REDUCE_MAX` | `ReduceMax` | FP16, one static axis, `keep_dims=true` | portable + HTP |
| `REDUCE_MIN` | `ReduceMin` | FP16, one static axis, `keep_dims=true` | portable + HTP |
| `REDUCE_PRODUCT` | `Gather` + `ElementWiseMultiply` | FP16, one static axis; public `ReduceProd` rejected by HTP v73 | portable + HTP |
| `REDUCE_SUM` | `ReduceSum` | FP16, one static axis, `keep_dims=true` | portable + HTP |
| `CONCAT` | `Concat` | FP16, static valid axis | portable + HTP |
| `RESHAPE` | `Reshape` | FP16, compile-time `CONST_SHAPE` | portable + HTP |
| `REVERSE` | descending static indices + `Gather` | FP16, one static valid axis | portable + HTP |
| `TRANSPOSE` | `Transpose` | FP16, static complete permutation | portable + HTP |
| `CONST` | owned QNN static tensor | internal FP16 constants with exact byte length | portable + HTP |
| `CONST_SHAPE` | consumed during lowering | static valid reshape shape | portable + HTP |
| `IDENTITY` | `Reshape` | FP16 or exact INT8 tier | portable + HTP |

Every attribute, dtype, rank, axis, permutation, shape, and constant payload first passes the shared
bounded TOSA verifier and semantic analyzer. The Hexagon planner then owns constants, uses checked
shape/byte arithmetic, and rejects unsupported graphs before entering QNN. The native bridge repeats
descriptor arity, pointer, element-size, constant-size, reference, and generated-tensor resource
checks before calling the provider.

Reproduce the portable and hardware evidence with the commands in the
[Hexagon README](../crates/virtio-accel-hexagon/README.md#manual-hardware-test).
