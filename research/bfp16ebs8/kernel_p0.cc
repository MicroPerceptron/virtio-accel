// P0 encoding/layout probe (issue #146).
//
// Converts 64 caller-chosen FP32 values to one v64bfp16ebs8 with the core's
// to_v64bfp16ebs8 conversion, then writes three things the host can decode:
//
//   bytes [0..64)    the mantissa plane, one signed byte per element, element order
//   bytes [64..72)   the exponent plane, one byte per 8-element block, block order
//   bytes [72..144)  the native v64bfp16ebs8 struct stored to memory as-is
//                    (the compiler/DMA-facing byte layout, which the register
//                    plane split above does not reveal)
//   bytes [144..148) the core's rounding-mode register at kernel entry (LE u32)
//
// The plane dump freezes the element encoding (H1); the struct store freezes the
// in-memory layout (H2); the mode word records the default rounding (H3 input).
#include <cstdint>

#include <aie_api/aie.hpp>

extern "C" void probe_p0(const float *__restrict input, uint32_t *__restrict output) {
    uint8_t *bytes = reinterpret_cast<uint8_t *>(output);

    aie::vector<float, 64> values = aie::load_v<64>(input);
    aie::accum<accfloat, 64> acc;
    acc.from_vector(values);
    v64bfp16ebs8 encoded = ::to_v64bfp16ebs8(acc);

    // Register-level planes. Spilling through the stack is fine in a probe.
    v64int8 mantissa = encoded.mantissa;
    const uint8_t *mantissa_bytes = reinterpret_cast<const uint8_t *>(&mantissa);
    for (unsigned i = 0; i < 64; ++i) {
        bytes[i] = mantissa_bytes[i];
    }
    v8int8 exponent = encoded.exponent;
    const uint8_t *exponent_bytes = reinterpret_cast<const uint8_t *>(&exponent);
    for (unsigned i = 0; i < 8; ++i) {
        bytes[64 + i] = exponent_bytes[i];
    }

    // Native in-memory layout of the block-vector type itself.
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(&encoded);
    for (unsigned i = 0; i < sizeof(v64bfp16ebs8); ++i) {
        bytes[72 + i] = raw[i];
    }

    output[36] = static_cast<uint32_t>(::get_rnd());
}
