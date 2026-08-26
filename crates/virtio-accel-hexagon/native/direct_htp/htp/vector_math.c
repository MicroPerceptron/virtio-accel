// SPDX-License-Identifier: MIT OR Apache-2.0

#include "vector_math.h"

#include <qhmath_hvx_vector.h>

__attribute__((noinline)) HVX_Vector va_hvx_atan_f32(HVX_Vector value) {
    return qhmath_hvx_atan_vf(value);
}

__attribute__((noinline)) HVX_Vector va_hvx_log_f32(HVX_Vector value) {
    return qhmath_hvx_log_vf(value);
}
