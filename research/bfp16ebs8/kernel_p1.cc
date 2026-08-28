// P1 rounding probe (issue #146).
//
// Converts the same 64 caller-chosen FP32 values under every crrnd rounding mode via
// to_v64bfp16ebs8_conf, dumping the mantissa+exponent planes per mode:
//
//   bytes [k*72 .. k*72+64)     mantissa plane under mode MODES[k]
//   bytes [k*72+64 .. k*72+72)  exponent plane under mode MODES[k]
//   bytes [720..724)            get_rnd() at kernel entry (LE u32)
//
// MODES = {0,1,2,3,8,9,10,11,12,13} = {floor, ceil, sym_floor, sym_ceil, neg_inf, pos_inf,
// sym_zero, sym_inf, conv_even, conv_odd} per aie2p_defines.h. The host chooses tie-case
// inputs; the kernel is input-agnostic.
#include <cstdint>

#include <aie_api/aie.hpp>

extern "C" void probe_p1(const float *__restrict input, uint32_t *__restrict output) {
    uint8_t *bytes = reinterpret_cast<uint8_t *>(output);

    aie::vector<float, 64> values = aie::load_v<64>(input);
    aie::accum<accfloat, 64> acc;
    acc.from_vector(values);

    output[180] = static_cast<uint32_t>(::get_rnd());

    const unsigned modes[10] = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13};
    for (unsigned k = 0; k < 10; ++k) {
        v64bfp16ebs8 encoded = ::to_v64bfp16ebs8_conf(acc, static_cast<crrnd_t>(modes[k]));
        uint8_t *slot = bytes + k * 72;
        v64int8 mantissa = encoded.mantissa;
        const uint8_t *mantissa_bytes = reinterpret_cast<const uint8_t *>(&mantissa);
        for (unsigned i = 0; i < 64; ++i) {
            slot[i] = mantissa_bytes[i];
        }
        v8int8 exponent = encoded.exponent;
        const uint8_t *exponent_bytes = reinterpret_cast<const uint8_t *>(&exponent);
        for (unsigned i = 0; i < 8; ++i) {
            slot[64 + i] = exponent_bytes[i];
        }
    }
}
