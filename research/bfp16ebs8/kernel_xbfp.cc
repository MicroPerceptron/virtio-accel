// XBFP flavor-1 prototype kernel (issue #148, design step 1).
//
// C[8,8] (FP32) = A[8,K] . B[8,K]^T on bfp16ebs8 planes, K = @K@ (multiple of 8).
// Chunk c holds columns 8c..8c+8 as one v64bfp16ebs8 unit (64 mantissas then 8 exponents,
// the P0-pinned layout). Accumulation contract: ascending-c mul/mac chain, FP32.
//
//   input  [0 .. 9*K)        A: K/8 units of 72 bytes
//   input  [9*K .. 18*K)     B: K/8 units of 72 bytes
//   output [0..256)          C: 64 FP32 lanes, lane i*8+j
#include <cstdint>

#include <aie_api/aie.hpp>

extern "C" void probe_xbfp(const uint8_t *__restrict input, float *__restrict output) {
    constexpr unsigned K = @K@;
    constexpr unsigned CHUNKS = K / 8;
    constexpr unsigned UNIT = sizeof(v64bfp16ebs8);
    static_assert(UNIT == 72, "layout contract");

    const uint8_t *a_bytes = input;
    const uint8_t *b_bytes = input + CHUNKS * UNIT;

    v64bfp16ebs8 a;
    v64bfp16ebs8 b;
    uint8_t *a_load = reinterpret_cast<uint8_t *>(&a);
    uint8_t *b_load = reinterpret_cast<uint8_t *>(&b);

    for (unsigned i = 0; i < UNIT; ++i) {
        a_load[i] = a_bytes[i];
        b_load[i] = b_bytes[i];
    }
    v64accfloat acc = ::mul_8x8_8x8T(a, b);
    for (unsigned c = 1; c < CHUNKS; ++c) {
        for (unsigned i = 0; i < UNIT; ++i) {
            a_load[i] = a_bytes[c * UNIT + i];
            b_load[i] = b_bytes[c * UNIT + i];
        }
        acc = ::mac_8x8_8x8T(a, b, acc);
    }

    aie::accum<accfloat, 64> result(acc);
    aie::store_v(output, result.template to_vector<float>());
}
