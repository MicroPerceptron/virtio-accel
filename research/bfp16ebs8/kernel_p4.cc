// P4/P5 MMUL-contract probe (issue #146).
//
// The host crafts RAW bfp16ebs8 planes (layout pinned by P0: 64 mantissa bytes then 8 exponent
// bytes per v64), bypassing the converter entirely. The kernel chains the native 8x8x8T MMUL
// over K = 32 (4 sub-blocks) and returns the FP32 accumulator:
//
//   input  [0..288)    A: four v64bfp16ebs8 chunks (8x32 operand, chunk k = columns 8k..8k+8)
//   input  [288..576)  B: four v64bfp16ebs8 chunks (32x8 operand, transposed-B layout)
//   output [0..256)    C: 64 FP32 accumulator lanes, stored in lane order
//
// The host discovers the actual element->matrix-position mapping empirically (single-entry
// operands), so this kernel makes no layout claim beyond "chunk k multiplies chunk k".
#include <cstdint>

#include <aie_api/aie.hpp>

extern "C" void probe_p4(const uint8_t *__restrict input, float *__restrict output) {
    v64bfp16ebs8 a[4];
    v64bfp16ebs8 b[4];
    uint8_t *a_bytes = reinterpret_cast<uint8_t *>(a);
    uint8_t *b_bytes = reinterpret_cast<uint8_t *>(b);
    for (unsigned i = 0; i < 288; ++i) {
        a_bytes[i] = input[i];
        b_bytes[i] = input[288 + i];
    }

    v64accfloat acc = ::mul_8x8_8x8T(a[0], b[0]);
    for (unsigned k = 1; k < 4; ++k) {
        acc = ::mac_8x8_8x8T(a[k], b[k], acc);
    }

    aie::accum<accfloat, 64> result(acc);
    aie::store_v(output, result.template to_vector<float>());
}
