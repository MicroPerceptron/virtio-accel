// SPDX-License-Identifier: MIT OR Apache-2.0

#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "va_htp.h"
#include "vector_math.h"
#include "worker_pool.h"

#define VA_HTP_WORKERS 4u
#define VA_HVX_FLOAT_LANES 32u

enum {
    VA_HTP_OP_IDENTITY = 1,
    VA_HTP_OP_ADD = 2,
    VA_HTP_OP_MULTIPLY = 3,
    VA_HTP_OP_RECIPROCAL = 4,
    VA_HTP_OP_RSQRT = 5,
    VA_HTP_OP_MATMUL = 6,
    VA_HTP_OP_WORMHOLE_TRACE = 16,
    VA_HTP_OP_KERR_TRACE = 17,
    VA_HTP_OP_KERR_FRAME = 18,
    VA_HTP_OP_KERR_SHADE = 19,
};

#define VA_KERR_FRAME_MAGIC 0x4d52464bu
#define VA_KERR_FRAME_HEADER_WORDS 8u
#define VA_KERR_SCENE_MAGIC 0x4543534bu
#define VA_KERR_SCENE_ABI 1u
#define VA_KERR_SCENE_HEADER_BYTES 160u
#define VA_KERR_BOUNDARY_WORDS 16u
#define VA_FRAME_FLAG_POWER_APP 0x01u
#define VA_FRAME_FLAG_POWER_HVX 0x02u
#define VA_FRAME_FLAG_POWER_DCVS 0x04u
#define VA_FRAME_FLAG_POWER_CORE 0x08u
#define VA_FRAME_FLAG_POWER_BUS 0x10u
#define VA_FRAME_FLAG_UDMA 0x100u

struct wormhole_parameters {
    float rho;
    float a;
    float m;
    float step_size;
    uint32_t max_steps;
    float escape_ell;
};

struct kerr_parameters {
    float mass;
    float spin;
    float step_size;
    uint32_t max_steps;
    float gradient_epsilon;
    float escape_radius;
    float disk_inner_radius;
    float disk_outer_radius;
};

struct kerr_frame_parameters {
    struct kerr_parameters trace;
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_pixel;
    float tan_half_fov;
    float camera_position[3];
    float camera_time[4];
    float camera_right[4];
    float camera_up[4];
    float camera_forward[4];
};

struct kerr_scene_header {
    float time;
    uint32_t magic;
    uint32_t abi;
    uint32_t pixels;
    uint32_t base_spp;
    uint32_t refinement_spp;
    uint32_t storage;
    uint32_t plasma_mode;
    uint32_t dim_x;
    uint32_t dim_y;
    uint32_t dim_z;
    uint32_t sky_width;
    uint32_t sky_height;
    uint32_t base_offset;
    uint32_t refinement_lookup_offset;
    uint32_t refinement_offset;
    uint32_t extinction_offset;
    uint32_t source_offset;
    uint32_t temperature_offset;
    uint32_t surface_temperature_offset;
    uint32_t surface_density_offset;
    uint32_t surface_transfer_offset;
    uint32_t blackbody_offset;
    uint32_t sky_offset;
    uint32_t total_bytes;
    float plasma_time_scale;
    float sky_speed;
    float exposure;
    float plasma_extinction;
    float plasma_emission;
    float xy_half_extent;
    float half_thickness;
    float kinetic_reference;
    float disk_outer_radius;
    uint32_t surface_temperature_bins;
    uint32_t surface_redshift_bins;
    uint32_t blackbody_bins;
    uint32_t reserved[3];
};

struct kerr_boundary_record {
    uint32_t kind;
    float value[15];
};

struct matmul_parameters {
    uint32_t rows;
    uint32_t inner;
    uint32_t columns;
};

struct kerr_ray {
    float x[4];
    float p[4];
};

struct va_htp_context {
    unsigned char *arena;
    uint32_t arena_size;
    int arena_fd;
    va_worker_pool_t worker_pool;
    unsigned char *vtcm;
    uint32_t vtcm_size;
    uint32_t vtcm_resource;
    void *power_context;
    uint32_t power_flags;
};

static void va_htp_release_power(struct va_htp_context *ctx) {
    if (!ctx->power_context) return;
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = FALSE;
    (void)HAP_power_set(ctx->power_context, &request);
    (void)HAP_power_destroy(ctx->power_context);
    HAP_utils_destroy_context(ctx->power_context);
    ctx->power_context = NULL;
    ctx->power_flags = 0;
}

static void va_htp_request_power(struct va_htp_context *ctx) {
    ctx->power_context = HAP_utils_create_context();
    if (!ctx->power_context) return;
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set(ctx->power_context, &request) == 0)
        ctx->power_flags |= VA_FRAME_FLAG_POWER_APP;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    if (HAP_power_set(ctx->power_context, &request) == 0)
        ctx->power_flags |= VA_FRAME_FLAG_POWER_HVX;
    if (HAP_power_set_dcvs_option(ctx->power_context, TRUE,
            HAP_DCVS_V2_PERFORMANCE_MODE) == 0)
        ctx->power_flags |= VA_FRAME_FLAG_POWER_DCVS;
    if (HAP_power_set_core_corner(ctx->power_context, HAP_DCVS_VCORNER_TURBO,
            HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO) == 0)
        ctx->power_flags |= VA_FRAME_FLAG_POWER_CORE;
    if (HAP_power_set_bus_corner(ctx->power_context, HAP_DCVS_VCORNER_TURBO,
            HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO) == 0)
        ctx->power_flags |= VA_FRAME_FLAG_POWER_BUS;
    (void)HAP_power_set_sleep_mode(ctx->power_context, HAP_DCVS_LPM_LEVEL1);
}

static AEEResult va_htp_init_vtcm(struct va_htp_context *ctx) {
    unsigned int size = 8u * 1024u * 1024u;
    HAP_compute_res_query_VTCM(0, &size, NULL, NULL, NULL);
    if (!size) return AEE_ENOMEMORY;
    compute_res_attr_t attributes;
    HAP_compute_res_attr_init(&attributes);
    HAP_compute_res_attr_set_serialize(&attributes, 0);
    HAP_compute_res_attr_set_cache_mode(&attributes, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attributes, size, size, size);
    const uint32_t resource = HAP_compute_res_acquire(&attributes, 1000000u);
    if (!resource) return AEE_ENOMEMORY;
    void *base = NULL;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attributes, &base, &size) != 0 || !base) {
        HAP_compute_res_release(resource);
        return AEE_ENOMEMORY;
    }
    ctx->vtcm = (unsigned char *)base;
    ctx->vtcm_size = size;
    ctx->vtcm_resource = resource;
    return AEE_SUCCESS;
}

AEEResult va_htp_open(const char *uri, remote_handle64 *handle) {
    (void) uri;
    if (!handle) return AEE_EBADPARM;
    struct va_htp_context *ctx = (struct va_htp_context *) calloc(1, sizeof(*ctx));
    if (!ctx) return AEE_ENOMEMORY;
    ctx->arena_fd = -1;
    AEEResult result = va_worker_pool_init(&ctx->worker_pool, VA_HTP_WORKERS);
    if (result != AEE_SUCCESS) {
        free(ctx);
        return result;
    }
    result = va_htp_init_vtcm(ctx);
    if (result != AEE_SUCCESS) {
        va_worker_pool_release(&ctx->worker_pool);
        free(ctx);
        return result;
    }
    va_htp_request_power(ctx);
    *handle = (remote_handle64) ctx;
    return AEE_SUCCESS;
}

AEEResult va_htp_close(remote_handle64 handle) {
    struct va_htp_context *ctx = (struct va_htp_context *) handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->arena) HAP_munmap(ctx->arena, ctx->arena_size);
    va_worker_pool_release(&ctx->worker_pool);
    if (ctx->vtcm_resource) HAP_compute_res_release(ctx->vtcm_resource);
    va_htp_release_power(ctx);
    free(ctx);
    return AEE_SUCCESS;
}

AEEResult va_htp_hwinfo(remote_handle64 handle, uint32_t *arch, uint32_t *hvx_units, uint32_t *vtcm_bytes) {
    if (!handle || !arch || !hvx_units || !vtcm_bytes) return AEE_EBADPARM;
    *arch = __HVX_ARCH__;
    *hvx_units = 4;
    *vtcm_bytes = ((struct va_htp_context *)handle)->vtcm_size;
    return AEE_SUCCESS;
}

AEEResult va_htp_map_arena(remote_handle64 handle, uint32_t fd, uint32_t size) {
    struct va_htp_context *ctx = (struct va_htp_context *) handle;
    if (!ctx || !size || ctx->arena) return AEE_EBADPARM;
    void *base = HAP_mmap(NULL, size, HAP_PROT_READ | HAP_PROT_WRITE, 0, (int) fd, 0);
    if (base == (void *) -1) return AEE_EFAILED;
    ctx->arena = (unsigned char *) base;
    ctx->arena_size = size;
    ctx->arena_fd = (int) fd;
    return AEE_SUCCESS;
}

AEEResult va_htp_unmap_arena(remote_handle64 handle) {
    struct va_htp_context *ctx = (struct va_htp_context *) handle;
    if (!ctx) return AEE_EBADPARM;
    if (!ctx->arena) return AEE_SUCCESS;
    HAP_munmap(ctx->arena, ctx->arena_size);
    ctx->arena = NULL;
    ctx->arena_size = 0;
    ctx->arena_fd = -1;
    return AEE_SUCCESS;
}

static AEEResult vector_binary(
    uint32_t opcode,
    uint32_t lanes,
    const float *lhs,
    const float *rhs,
    float *out) {
    uint32_t i = 0;
    for (; i + 32 <= lanes; i += 32) {
        HVX_Vector a = *(const HVX_Vector *)(lhs + i);
        if (opcode == VA_HTP_OP_IDENTITY) {
            *(HVX_Vector *)(out + i) = a;
            continue;
        }
        HVX_Vector b = *(const HVX_Vector *)(rhs + i);
        *(HVX_Vector *)(out + i) = opcode == VA_HTP_OP_ADD
            ? Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
            : Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
    }
    for (; i < lanes; ++i) {
        out[i] = opcode == VA_HTP_OP_IDENTITY ? lhs[i]
            : opcode == VA_HTP_OP_ADD ? lhs[i] + rhs[i]
            : lhs[i] * rhs[i];
    }
    return AEE_SUCCESS;
}

static AEEResult vector_unary(
    uint32_t opcode,
    uint32_t lanes,
    const float *input,
    float *output) {
    for (uint32_t i = 0; i < lanes; ++i) {
        output[i] = opcode == VA_HTP_OP_RECIPROCAL
            ? 1.0f / input[i]
            : 1.0f / sqrtf(input[i]);
    }
    return AEE_SUCCESS;
}

static AEEResult matrix_multiply(
    const struct matmul_parameters *p,
    const float *lhs,
    const float *rhs,
    float *output) {
    for (uint32_t row = 0; row < p->rows; ++row) {
        for (uint32_t column = 0; column < p->columns; ++column) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < p->inner; ++k) {
                sum += lhs[row * p->inner + k] * rhs[k * p->columns + column];
            }
            output[row * p->columns + column] = sum;
        }
    }
    return AEE_SUCCESS;
}

static inline float wormhole_radius(const struct wormhole_parameters *p, float ell) {
    const float exterior = fabsf(ell) - p->a;
    if (exterior <= 0.0f) return p->rho;
    const float two_over_pi = 0.63661977236758134308f;
    const float x = two_over_pi * exterior / p->m;
    return p->rho + p->m * (x * atanf(x) - 0.5f * log1pf(x * x));
}

static inline float wormhole_force(
    const struct wormhole_parameters *p,
    float ell,
    float impact) {
    const float exterior = fabsf(ell) - p->a;
    const float r = wormhole_radius(p, ell);
    float derivative = 0.0f;
    if (exterior > 0.0f) {
        const float x = 0.63661977236758134308f * exterior / p->m;
        const float magnitude = 0.63661977236758134308f * atanf(x);
        derivative = ell < 0.0f ? -magnitude : magnitude;
    }
    return impact * impact * derivative / (r * r * r);
}

static inline int wormhole_terminal(
    const struct wormhole_parameters *p,
    float ell,
    float p_ell) {
    return (ell <= -p->escape_ell && p_ell < 0.0f) ||
        (ell >= p->escape_ell && p_ell > 0.0f);
}

static AEEResult wormhole_trace_scalar(
    uint32_t count,
    uint32_t input_stride,
    uint32_t output_stride,
    const struct wormhole_parameters *p,
    const float *input,
    float *output) {
    const float *ell_in = input;
    const float *phi_in = input + input_stride;
    const float *p_ell_in = input + 2 * input_stride;
    const float *impact_in = input + 3 * input_stride;
    const float *active_in = input + 4 * input_stride;
    float *ell_out = output;
    float *phi_out = output + output_stride;
    float *p_ell_out = output + 2 * output_stride;
    float *active_out = output + 3 * output_stride;

    for (uint32_t lane = 0; lane < count; ++lane) {
        float ell = ell_in[lane];
        float phi = phi_in[lane];
        float p_ell = p_ell_in[lane];
        const float impact = impact_in[lane];
        int active = active_in[lane] != 0.0f;
        if (active && wormhole_terminal(p, ell, p_ell)) active = 0;
        for (uint32_t step = 0; active && step < p->max_steps; ++step) {
            const float r_old = wormhole_radius(p, ell);
            const float half = 0.5f * p->step_size;
            p_ell += half * wormhole_force(p, ell, impact);
            ell += p->step_size * p_ell;
            const float r_new = wormhole_radius(p, ell);
            p_ell += half * wormhole_force(p, ell, impact);
            phi += half * impact * (1.0f / (r_old * r_old) + 1.0f / (r_new * r_new));
            if (wormhole_terminal(p, ell, p_ell)) active = 0;
        }
        ell_out[lane] = ell;
        phi_out[lane] = phi;
        p_ell_out[lane] = p_ell;
        active_out[lane] = active ? 1.0f : 0.0f;
    }
    return AEE_SUCCESS;
}

static inline float kerr_radius_squared(const struct kerr_parameters *p, const float x[3]) {
    const float a2 = p->spin * p->spin;
    const float s = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - a2;
    return 0.5f * (s + sqrtf(s * s + 4.0f * a2 * x[2] * x[2]));
}

static inline float kerr_radius(const struct kerr_parameters *p, const struct kerr_ray *ray) {
    const float point[3] = {ray->x[1], ray->x[2], ray->x[3]};
    return sqrtf(kerr_radius_squared(p, point));
}

static inline void kerr_field(
    const struct kerr_parameters *p,
    const struct kerr_ray *ray,
    float k_up[4],
    float *field) {
    const float point[3] = {ray->x[1], ray->x[2], ray->x[3]};
    const float r2 = kerr_radius_squared(p, point);
    const float r = sqrtf(r2);
    const float a = p->spin;
    const float denominator = r2 + a * a;
    k_up[0] = -1.0f;
    k_up[1] = (r * point[0] + a * point[1]) / denominator;
    k_up[2] = (r * point[1] - a * point[0]) / denominator;
    k_up[3] = point[2] / r;
    *field = 2.0f * p->mass * r2 * r / (r2 * r2 + a * a * point[2] * point[2]);
}

static inline float kerr_hamiltonian(
    const struct kerr_parameters *parameters,
    const struct kerr_ray *ray) {
    float k[4], field;
    kerr_field(parameters, ray, k, &field);
    const float eta_p2 = -ray->p[0] * ray->p[0] + ray->p[1] * ray->p[1] +
        ray->p[2] * ray->p[2] + ray->p[3] * ray->p[3];
    const float kp = k[0] * ray->p[0] + k[1] * ray->p[1] +
        k[2] * ray->p[2] + k[3] * ray->p[3];
    return 0.5f * (eta_p2 - field * kp * kp);
}

static inline void kerr_position_rate(
    const struct kerr_parameters *parameters,
    const struct kerr_ray *ray,
    float rate[4]) {
    float k[4], field;
    kerr_field(parameters, ray, k, &field);
    const float kp = k[0] * ray->p[0] + k[1] * ray->p[1] +
        k[2] * ray->p[2] + k[3] * ray->p[3];
    rate[0] = -ray->p[0] - field * k[0] * kp;
    rate[1] = ray->p[1] - field * k[1] * kp;
    rate[2] = ray->p[2] - field * k[2] * kp;
    rate[3] = ray->p[3] - field * k[3] * kp;
}

static inline void kerr_rhs(
    const struct kerr_parameters *parameters,
    const struct kerr_ray *ray,
    struct kerr_ray *rate) {
    kerr_position_rate(parameters, ray, rate->x);
    rate->p[0] = 0.0f;
    const float epsilon = parameters->gradient_epsilon * fmaxf(kerr_radius(parameters, ray), 1.0f);
    for (int axis = 1; axis < 4; ++axis) {
        struct kerr_ray plus = *ray;
        struct kerr_ray minus = *ray;
        plus.x[axis] += epsilon;
        minus.x[axis] -= epsilon;
        rate->p[axis] = -(kerr_hamiltonian(parameters, &plus) -
            kerr_hamiltonian(parameters, &minus)) / (2.0f * epsilon);
    }
}

static inline void kerr_midpoint(
    const struct kerr_parameters *parameters,
    struct kerr_ray *ray) {
    const float conserved_p_t = ray->p[0];
    struct kerr_ray first, midpoint, second;
    kerr_rhs(parameters, ray, &first);
    for (int i = 0; i < 4; ++i) {
        midpoint.x[i] = ray->x[i] + 0.5f * parameters->step_size * first.x[i];
        midpoint.p[i] = ray->p[i] + 0.5f * parameters->step_size * first.p[i];
    }
    kerr_rhs(parameters, &midpoint, &second);
    for (int i = 0; i < 4; ++i) {
        ray->x[i] += parameters->step_size * second.x[i];
        ray->p[i] += parameters->step_size * second.p[i];
    }
    ray->p[0] = conserved_p_t;
}

static inline int kerr_outward(
    const struct kerr_parameters *parameters,
    const struct kerr_ray *ray) {
    float rate[4];
    kerr_position_rate(parameters, ray, rate);
    return ray->x[1] * rate[1] + ray->x[2] * rate[2] + ray->x[3] * rate[3] > 0.0f;
}

/* V73 represents vector FP32 arithmetic through QFloat32 accumulators. */
static inline HVX_Vector va_qadd(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

static inline HVX_Vector va_qsub(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

static inline HVX_Vector va_qmul(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

static inline HVX_Vector va_qdot3(
    HVX_Vector a0, HVX_Vector b0,
    HVX_Vector a1, HVX_Vector b1,
    HVX_Vector a2, HVX_Vector b2) {
    HVX_Vector sum = Q6_Vqf32_vmpy_VsfVsf(a0, b0);
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(a1, b1));
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(a2, b2));
    return Q6_Vsf_equals_Vqf32(sum);
}

static inline HVX_Vector va_qdot4(
    HVX_Vector a0, HVX_Vector b0,
    HVX_Vector a1, HVX_Vector b1,
    HVX_Vector a2, HVX_Vector b2,
    HVX_Vector a3, HVX_Vector b3) {
    HVX_Vector sum = Q6_Vqf32_vmpy_VsfVsf(a0, b0);
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(a1, b1));
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(a2, b2));
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(a3, b3));
    return Q6_Vsf_equals_Vqf32(sum);
}

static inline HVX_Vector va_splat(float value) {
    union { float f; uint32_t u; } bits = { .f = value };
    return Q6_V_vsplat_R(bits.u);
}

static inline HVX_Vector va_qinverse(HVX_Vector value) {
    const HVX_Vector two = va_splat(2.0f);
    const HVX_Vector two_qf = Q6_Vqf32_vadd_VsfVsf(two, Q6_V_vzero());
    HVX_Vector estimate = Q6_Vqf32_vadd_VsfVsf(
        Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7eeeeBB3), value), Q6_V_vzero());
    const HVX_Vector value_qf = Q6_Vqf32_vadd_VsfVsf(value, Q6_V_vzero());
    for (int iteration = 0; iteration < 2; ++iteration) {
        const HVX_Vector product = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, value_qf);
        const HVX_Vector correction = Q6_Vqf32_vsub_Vqf32Vqf32(two_qf, product);
        estimate = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, correction);
    }
    return Q6_Vsf_equals_Vqf32(estimate);
}

static inline HVX_Vector va_qdiv(HVX_Vector numerator, HVX_Vector denominator) {
    return va_qmul(numerator, va_qinverse(denominator));
}

static inline HVX_Vector va_qinverse_seeded(HVX_Vector value, HVX_Vector seed) {
    const HVX_Vector two_qf = Q6_Vqf32_vadd_VsfVsf(va_splat(2.0f), Q6_V_vzero());
    const HVX_Vector value_qf = Q6_Vqf32_vadd_VsfVsf(value, Q6_V_vzero());
    HVX_Vector estimate = Q6_Vqf32_vadd_VsfVsf(seed, Q6_V_vzero());
    const HVX_Vector product = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, value_qf);
    const HVX_Vector correction = Q6_Vqf32_vsub_Vqf32Vqf32(two_qf, product);
    estimate = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, correction);
    return Q6_Vsf_equals_Vqf32(estimate);
}

static inline HVX_Vector va_qrsqrt(HVX_Vector value) {
    const HVX_Vector three_halves = va_splat(1.5f);
    const HVX_Vector three_halves_qf = Q6_Vqf32_vadd_VsfVsf(three_halves, Q6_V_vzero());
    HVX_Vector estimate = Q6_Vqf32_vadd_VsfVsf(
        Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5f3759df), Q6_Vw_vasr_VwR(value, 1)),
        Q6_V_vzero());
    const HVX_Vector x_half = Q6_Vqf32_vmpy_VsfVsf(value, va_splat(0.5f));
    for (int iteration = 0; iteration < 2; ++iteration) {
        const HVX_Vector square = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, estimate);
        const HVX_Vector product = Q6_Vqf32_vmpy_Vqf32Vqf32(x_half, square);
        const HVX_Vector correction = Q6_Vqf32_vsub_Vqf32Vqf32(
            three_halves_qf, product);
        estimate = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, correction);
    }
    return Q6_Vsf_equals_Vqf32(estimate);
}

static inline HVX_Vector va_qsqrt(HVX_Vector value) {
    return va_qmul(value, va_qrsqrt(value));
}

static inline HVX_Vector va_qrsqrt_seeded(
    HVX_Vector value,
    HVX_Vector seed,
    int refinements) {
    const HVX_Vector three_halves_qf = Q6_Vqf32_vadd_VsfVsf(va_splat(1.5f),
        Q6_V_vzero());
    const HVX_Vector x_half = Q6_Vqf32_vmpy_VsfVsf(value, va_splat(0.5f));
    HVX_Vector estimate = Q6_Vqf32_vadd_VsfVsf(seed, Q6_V_vzero());
    for (int iteration = 0; iteration < refinements; ++iteration) {
        const HVX_Vector square = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, estimate);
        const HVX_Vector product = Q6_Vqf32_vmpy_Vqf32Vqf32(x_half, square);
        const HVX_Vector correction = Q6_Vqf32_vsub_Vqf32Vqf32(
            three_halves_qf, product);
        estimate = Q6_Vqf32_vmpy_Vqf32Vqf32(estimate, correction);
    }
    return Q6_Vsf_equals_Vqf32(estimate);
}

static inline HVX_Vector va_qmax(HVX_Vector a, HVX_Vector b) {
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(a, b), a, b);
}

static inline HVX_Vector va_qneg(HVX_Vector value) {
    return Q6_V_vxor_VV(value, Q6_V_vsplat_R((int)0x80000000u));
}

struct kerr_ray_vector {
    HVX_Vector x[4];
    HVX_Vector p[4];
};

static inline HVX_Vector kerr_radius_squared_and_s_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    HVX_Vector *s_out) {
    const HVX_Vector a2 = va_splat(parameters->spin * parameters->spin);
    HVX_Vector s = va_qdot3(ray->x[1], ray->x[1], ray->x[2], ray->x[2],
        ray->x[3], ray->x[3]);
    s = va_qsub(s, a2);
    *s_out = s;
    const HVX_Vector discriminant = va_qadd(va_qmul(s, s),
        va_qmul(va_splat(4.0f * parameters->spin * parameters->spin),
            va_qmul(ray->x[3], ray->x[3])));
    return va_qmul(va_splat(0.5f), va_qadd(s, va_qsqrt(discriminant)));
}

static inline HVX_Vector kerr_radius_squared_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray) {
    HVX_Vector s;
    return kerr_radius_squared_and_s_vector(parameters, ray, &s);
}

static inline HVX_Vector kerr_radius_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray) {
    return va_qsqrt(kerr_radius_squared_vector(parameters, ray));
}

struct kerr_geometry_seed_vector {
    HVX_Vector inverse_discriminant_root;
    HVX_Vector inverse_radius;
    HVX_Vector inverse_direction_denominator;
    HVX_Vector inverse_field_denominator;
};

static inline void kerr_field_and_radius_seed_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    HVX_Vector k[4],
    HVX_Vector *field,
    HVX_Vector *radius,
    struct kerr_geometry_seed_vector *seed) {
    const HVX_Vector a2 = va_splat(parameters->spin * parameters->spin);
    HVX_Vector s = va_qdot3(ray->x[1], ray->x[1], ray->x[2], ray->x[2],
        ray->x[3], ray->x[3]);
    s = va_qsub(s, a2);
    const HVX_Vector discriminant = va_qadd(va_qmul(s, s),
        va_qmul(va_splat(4.0f * parameters->spin * parameters->spin),
            va_qmul(ray->x[3], ray->x[3])));
    seed->inverse_discriminant_root = va_qrsqrt(discriminant);
    const HVX_Vector r2 = va_qmul(va_splat(0.5f),
        va_qadd(s, va_qmul(discriminant, seed->inverse_discriminant_root)));
    seed->inverse_radius = va_qrsqrt(r2);
    const HVX_Vector inverse_r = seed->inverse_radius;
    const HVX_Vector r = va_qmul(r2, inverse_r);
    *radius = r;
    const HVX_Vector spin = va_splat(parameters->spin);
    const HVX_Vector denominator = va_qadd(r2, a2);
    seed->inverse_direction_denominator = va_qinverse(denominator);
    const HVX_Vector inverse_denominator = seed->inverse_direction_denominator;
    k[0] = va_splat(-1.0f);
    k[1] = va_qmul(va_qadd(va_qmul(r, ray->x[1]), va_qmul(spin, ray->x[2])),
        inverse_denominator);
    k[2] = va_qmul(va_qsub(va_qmul(r, ray->x[2]), va_qmul(spin, ray->x[1])),
        inverse_denominator);
    k[3] = va_qmul(ray->x[3], inverse_r);
    /* From r^4 - s*r^2 - a^2*z^2 = 0:
       r^4 + a^2*z^2 = r^2*(2*r^2 - s).  Cancelling r^2 keeps the exact
       Kerr-Schild field while removing four vector multiplies. */
    const HVX_Vector field_denominator = va_qsub(va_qadd(r2, r2), s);
    seed->inverse_field_denominator = va_qinverse(field_denominator);
    *field = va_qmul(va_qmul(va_splat(2.0f * parameters->mass), r),
        seed->inverse_field_denominator);
}

static inline void kerr_field_and_radius_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    HVX_Vector k[4],
    HVX_Vector *field,
    HVX_Vector *radius) {
    struct kerr_geometry_seed_vector seed;
    kerr_field_and_radius_seed_vector(parameters, ray, k, field, radius, &seed);
}

static inline void kerr_field_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    HVX_Vector k[4],
    HVX_Vector *field) {
    HVX_Vector radius;
    kerr_field_and_radius_vector(parameters, ray, k, field, &radius);
}

static inline void kerr_field_vector_seeded(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    const struct kerr_geometry_seed_vector *seed,
    HVX_Vector k[4],
    HVX_Vector *field) {
    const HVX_Vector a2 = va_splat(parameters->spin * parameters->spin);
    HVX_Vector s = va_qdot3(ray->x[1], ray->x[1], ray->x[2], ray->x[2],
        ray->x[3], ray->x[3]);
    s = va_qsub(s, a2);
    const HVX_Vector discriminant = va_qadd(va_qmul(s, s),
        va_qmul(va_splat(4.0f * parameters->spin * parameters->spin),
            va_qmul(ray->x[3], ray->x[3])));
    const HVX_Vector inverse_root = va_qrsqrt_seeded(discriminant,
        seed->inverse_discriminant_root, 2);
    const HVX_Vector r2 = va_qmul(va_splat(0.5f),
        va_qadd(s, va_qmul(discriminant, inverse_root)));
    const HVX_Vector inverse_r = va_qrsqrt_seeded(r2, seed->inverse_radius, 1);
    const HVX_Vector r = va_qmul(r2, inverse_r);
    const HVX_Vector inverse_denominator = va_qinverse_seeded(va_qadd(r2, a2),
        seed->inverse_direction_denominator);
    const HVX_Vector spin = va_splat(parameters->spin);
    k[0] = va_splat(-1.0f);
    k[1] = va_qmul(va_qadd(va_qmul(r, ray->x[1]), va_qmul(spin, ray->x[2])),
        inverse_denominator);
    k[2] = va_qmul(va_qsub(va_qmul(r, ray->x[2]), va_qmul(spin, ray->x[1])),
        inverse_denominator);
    k[3] = va_qmul(ray->x[3], inverse_r);
    const HVX_Vector inverse_field = va_qinverse_seeded(
        va_qsub(va_qadd(r2, r2), s), seed->inverse_field_denominator);
    *field = va_qmul(va_qmul(va_splat(2.0f * parameters->mass), r), inverse_field);
}

static inline HVX_Vector kerr_hamiltonian_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray) {
    HVX_Vector k[4], field;
    kerr_field_vector(parameters, ray, k, &field);
    HVX_Vector eta_p2 = va_qdot4(va_qneg(ray->p[0]), ray->p[0],
        ray->p[1], ray->p[1], ray->p[2], ray->p[2], ray->p[3], ray->p[3]);
    HVX_Vector kp = va_qdot4(k[0], ray->p[0], k[1], ray->p[1],
        k[2], ray->p[2], k[3], ray->p[3]);
    return va_qmul(va_splat(0.5f), va_qsub(eta_p2, va_qmul(field, va_qmul(kp, kp))));
}

static inline void kerr_position_rate_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    HVX_Vector rate[4]) {
    HVX_Vector k[4], field;
    kerr_field_vector(parameters, ray, k, &field);
    HVX_Vector kp = va_qdot4(k[0], ray->p[0], k[1], ray->p[1],
        k[2], ray->p[2], k[3], ray->p[3]);
    const HVX_Vector field_kp = va_qmul(field, kp);
    rate[0] = va_qsub(va_qneg(ray->p[0]), va_qmul(field_kp, k[0]));
    rate[1] = va_qsub(ray->p[1], va_qmul(field_kp, k[1]));
    rate[2] = va_qsub(ray->p[2], va_qmul(field_kp, k[2]));
    rate[3] = va_qsub(ray->p[3], va_qmul(field_kp, k[3]));
}

static inline void kerr_position_rate_and_radius_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    HVX_Vector rate[4],
    HVX_Vector *radius,
    struct kerr_geometry_seed_vector *seed) {
    HVX_Vector k[4], field;
    kerr_field_and_radius_seed_vector(parameters, ray, k, &field, radius, seed);
    HVX_Vector kp = va_qdot4(k[0], ray->p[0], k[1], ray->p[1],
        k[2], ray->p[2], k[3], ray->p[3]);
    const HVX_Vector field_kp = va_qmul(field, kp);
    rate[0] = va_qsub(va_qneg(ray->p[0]), va_qmul(field_kp, k[0]));
    rate[1] = va_qsub(ray->p[1], va_qmul(field_kp, k[1]));
    rate[2] = va_qsub(ray->p[2], va_qmul(field_kp, k[2]));
    rate[3] = va_qsub(ray->p[3], va_qmul(field_kp, k[3]));
}

static inline void kerr_position_metric_term_vector_pair(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray0,
    const struct kerr_ray_vector *ray1,
    const struct kerr_geometry_seed_vector *seed,
    HVX_Vector *term0,
    HVX_Vector *term1) {
    const struct kerr_ray_vector *rays[2] = {ray0, ray1};
    HVX_Vector *terms[2] = {term0, term1};
    const HVX_Vector a2 = va_splat(parameters->spin * parameters->spin);
    const HVX_Vector spin = va_splat(parameters->spin);
    const HVX_Vector twice_mass = va_splat(2.0f * parameters->mass);
    for (int index = 0; index < 2; ++index) {
        const struct kerr_ray_vector *ray = rays[index];
        HVX_Vector s = va_qdot3(ray->x[1], ray->x[1], ray->x[2], ray->x[2],
            ray->x[3], ray->x[3]);
        s = va_qsub(s, a2);
        const HVX_Vector discriminant = va_qadd(va_qmul(s, s),
            va_qmul(va_splat(4.0f * parameters->spin * parameters->spin),
                va_qmul(ray->x[3], ray->x[3])));
        const HVX_Vector inverse_root = va_qrsqrt_seeded(discriminant,
            seed->inverse_discriminant_root, 2);
        const HVX_Vector r2 = va_qmul(va_splat(0.5f),
            va_qadd(s, va_qmul(discriminant, inverse_root)));
        const HVX_Vector inverse_r = va_qrsqrt_seeded(r2, seed->inverse_radius, 1);
        const HVX_Vector radius = va_qmul(r2, inverse_r);
        const HVX_Vector inverse_denominator = va_qinverse_seeded(va_qadd(r2, a2),
            seed->inverse_direction_denominator);
        const HVX_Vector radial = va_qadd(va_qmul(ray->x[1], ray->p[1]),
            va_qmul(ray->x[2], ray->p[2]));
        const HVX_Vector angular = va_qsub(va_qmul(ray->x[2], ray->p[1]),
            va_qmul(ray->x[1], ray->p[2]));
        const HVX_Vector planar = va_qmul(va_qadd(va_qmul(radius, radial),
            va_qmul(spin, angular)), inverse_denominator);
        const HVX_Vector vertical = va_qmul(va_qmul(ray->x[3], ray->p[3]), inverse_r);
        const HVX_Vector kp = va_qadd(va_qsub(planar, ray->p[0]), vertical);
        const HVX_Vector inverse_field = va_qinverse_seeded(
            va_qsub(va_qadd(r2, r2), s), seed->inverse_field_denominator);
        const HVX_Vector field = va_qmul(va_qmul(twice_mass, radius), inverse_field);
        *terms[index] = va_qmul(field, va_qmul(kp, kp));
    }
}

struct kerr_fixed_rhs_vector {
    HVX_Vector k[4];
    HVX_Vector field;
    HVX_Vector radius;
    HVX_Vector inverse_two_epsilon;
    HVX_Vector plus_k[3][4];
    HVX_Vector minus_k[3][4];
    HVX_Vector plus_field[3];
    HVX_Vector minus_field[3];
};

static inline void kerr_fixed_rhs_vector_init(
    const struct kerr_parameters *parameters,
    const float position[3],
    struct kerr_fixed_rhs_vector *fixed) {
    struct kerr_ray_vector ray;
    ray.x[0] = Q6_V_vzero();
    ray.p[0] = ray.p[1] = ray.p[2] = ray.p[3] = Q6_V_vzero();
    for (int i = 0; i < 3; ++i) ray.x[i + 1] = va_splat(position[i]);
    struct kerr_geometry_seed_vector seed;
    kerr_field_and_radius_seed_vector(parameters, &ray, fixed->k, &fixed->field,
        &fixed->radius, &seed);
    const HVX_Vector epsilon = va_qmul(va_splat(parameters->gradient_epsilon),
        va_qmax(fixed->radius, va_splat(1.0f)));
    fixed->inverse_two_epsilon = va_qinverse(va_qadd(epsilon, epsilon));
    for (int axis = 1; axis < 4; ++axis) {
        struct kerr_ray_vector plus = ray;
        struct kerr_ray_vector minus = ray;
        plus.x[axis] = va_qadd(plus.x[axis], epsilon);
        minus.x[axis] = va_qsub(minus.x[axis], epsilon);
        kerr_field_vector_seeded(parameters, &plus, &seed,
            fixed->plus_k[axis - 1], &fixed->plus_field[axis - 1]);
        kerr_field_vector_seeded(parameters, &minus, &seed,
            fixed->minus_k[axis - 1], &fixed->minus_field[axis - 1]);
    }
}

static inline void kerr_rhs_fixed_position_vector(
    const struct kerr_fixed_rhs_vector *fixed,
    const struct kerr_ray_vector *ray,
    struct kerr_ray_vector *rate) {
    const HVX_Vector kp = va_qdot4(fixed->k[0], ray->p[0], fixed->k[1], ray->p[1],
        fixed->k[2], ray->p[2], fixed->k[3], ray->p[3]);
    const HVX_Vector field_kp = va_qmul(fixed->field, kp);
    rate->x[0] = va_qsub(va_qneg(ray->p[0]), va_qmul(field_kp, fixed->k[0]));
    rate->x[1] = va_qsub(ray->p[1], va_qmul(field_kp, fixed->k[1]));
    rate->x[2] = va_qsub(ray->p[2], va_qmul(field_kp, fixed->k[2]));
    rate->x[3] = va_qsub(ray->p[3], va_qmul(field_kp, fixed->k[3]));
    rate->p[0] = Q6_V_vzero();
    const HVX_Vector half = va_splat(0.5f);
    for (int axis = 0; axis < 3; ++axis) {
        const HVX_Vector plus_kp = va_qdot4(
            fixed->plus_k[axis][0], ray->p[0], fixed->plus_k[axis][1], ray->p[1],
            fixed->plus_k[axis][2], ray->p[2], fixed->plus_k[axis][3], ray->p[3]);
        const HVX_Vector minus_kp = va_qdot4(
            fixed->minus_k[axis][0], ray->p[0], fixed->minus_k[axis][1], ray->p[1],
            fixed->minus_k[axis][2], ray->p[2], fixed->minus_k[axis][3], ray->p[3]);
        const HVX_Vector plus_term = va_qmul(fixed->plus_field[axis],
            va_qmul(plus_kp, plus_kp));
        const HVX_Vector minus_term = va_qmul(fixed->minus_field[axis],
            va_qmul(minus_kp, minus_kp));
        rate->p[axis + 1] = va_qmul(va_qmul(half, va_qsub(plus_term, minus_term)),
            fixed->inverse_two_epsilon);
    }
}

static inline void kerr_rhs_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray,
    struct kerr_ray_vector *rate) {
    HVX_Vector radius;
    struct kerr_geometry_seed_vector seed;
    kerr_position_rate_and_radius_vector(parameters, ray, rate->x, &radius, &seed);
    rate->p[0] = Q6_V_vzero();
    const HVX_Vector epsilon = va_qmul(va_splat(parameters->gradient_epsilon),
        va_qmax(radius, va_splat(1.0f)));
    const HVX_Vector two_epsilon = va_qadd(epsilon, epsilon);
    const HVX_Vector inverse_two_epsilon = va_qinverse(two_epsilon);
    const HVX_Vector half = va_splat(0.5f);
    for (int axis = 1; axis < 4; ++axis) {
        struct kerr_ray_vector plus = *ray;
        struct kerr_ray_vector minus = *ray;
        plus.x[axis] = va_qadd(plus.x[axis], epsilon);
        minus.x[axis] = va_qsub(minus.x[axis], epsilon);
        /* The Minkowski momentum term is position-independent and therefore
           identical on both sides of the centered difference. Eliding it is
           backend CSE; the finite-difference operation itself is unchanged. */
        HVX_Vector plus_term, minus_term;
        kerr_position_metric_term_vector_pair(parameters, &plus, &minus, &seed,
            &plus_term, &minus_term);
        rate->p[axis] = va_qmul(va_qmul(half, va_qsub(plus_term, minus_term)),
            inverse_two_epsilon);
    }
}

static inline struct kerr_ray_vector kerr_midpoint_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray) {
    struct kerr_ray_vector first, midpoint, second, result = *ray;
    kerr_rhs_vector(parameters, ray, &first);
    const HVX_Vector half_step = va_splat(0.5f * parameters->step_size);
    const HVX_Vector step = va_splat(parameters->step_size);
    for (int i = 0; i < 4; ++i) {
        midpoint.x[i] = va_qadd(ray->x[i], va_qmul(half_step, first.x[i]));
        midpoint.p[i] = va_qadd(ray->p[i], va_qmul(half_step, first.p[i]));
    }
    kerr_rhs_vector(parameters, &midpoint, &second);
    for (int i = 0; i < 4; ++i) {
        result.x[i] = va_qadd(ray->x[i], va_qmul(step, second.x[i]));
        result.p[i] = va_qadd(ray->p[i], va_qmul(step, second.p[i]));
    }
    result.p[0] = ray->p[0];
    return result;
}

static inline void kerr_midpoint_vector_fixed_first(
    const struct kerr_parameters *parameters,
    const struct kerr_fixed_rhs_vector *fixed,
    struct kerr_ray_vector *ray) {
    const HVX_Vector conserved_p_t = ray->p[0];
    struct kerr_ray_vector first, second;
    kerr_rhs_fixed_position_vector(fixed, ray, &first);
    const HVX_Vector half_step = va_splat(0.5f * parameters->step_size);
    const HVX_Vector step = va_splat(parameters->step_size);
    for (int i = 0; i < 4; ++i) {
        first.x[i] = va_qadd(ray->x[i], va_qmul(half_step, first.x[i]));
        first.p[i] = va_qadd(ray->p[i], va_qmul(half_step, first.p[i]));
    }
    kerr_rhs_vector(parameters, &first, &second);
    for (int i = 0; i < 4; ++i) {
        ray->x[i] = va_qadd(ray->x[i], va_qmul(step, second.x[i]));
        ray->p[i] = va_qadd(ray->p[i], va_qmul(step, second.p[i]));
    }
    ray->p[0] = conserved_p_t;
}

static inline HVX_VectorPred kerr_outward_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_ray_vector *ray) {
    HVX_Vector rate[4];
    kerr_position_rate_vector(parameters, ray, rate);
    HVX_Vector radial = va_qadd(va_qadd(va_qmul(ray->x[1], rate[1]),
        va_qmul(ray->x[2], rate[2])), va_qmul(ray->x[3], rate[3]));
    return Q6_Q_vcmp_gt_VsfVsf(radial, Q6_V_vzero());
}

static inline HVX_Vector va_select(HVX_VectorPred predicate, HVX_Vector yes, HVX_Vector no) {
    return Q6_V_vmux_QVV(predicate, yes, no);
}

static inline int va_predicate_any(HVX_VectorPred predicate) {
    union {
        HVX_Vector vector;
        uint32_t words[VA_HVX_FLOAT_LANES];
    } mask = { .vector = va_select(predicate, va_splat(1.0f), Q6_V_vzero()) };
    uint32_t any = 0;
    for (int lane = 0; lane < VA_HVX_FLOAT_LANES; ++lane) any |= mask.words[lane];
    return any != 0;
}

static inline HVX_VectorPred wormhole_terminal_vector(
    const struct wormhole_parameters *parameters,
    HVX_Vector ell,
    HVX_Vector p_ell) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector negative_escape = va_splat(-parameters->escape_ell);
    const HVX_Vector positive_escape = va_splat(parameters->escape_ell);
    const HVX_VectorPred left = Q6_Q_and_QQ(
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(ell, negative_escape)),
        Q6_Q_vcmp_gt_VsfVsf(zero, p_ell));
    const HVX_VectorPred right = Q6_Q_and_QQ(
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(positive_escape, ell)),
        Q6_Q_vcmp_gt_VsfVsf(p_ell, zero));
    return Q6_Q_or_QQ(left, right);
}

static inline void wormhole_geometry_vector(
    const struct wormhole_parameters *parameters,
    HVX_Vector ell,
    HVX_Vector *radius,
    HVX_Vector *derivative) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector exterior = va_qsub(
        Q6_V_vand_VV(ell, Q6_V_vsplat_R(0x7fffffff)), va_splat(parameters->a));
    const HVX_VectorPred outside = Q6_Q_vcmp_gt_VsfVsf(exterior, zero);
    const HVX_Vector x = va_qmul(exterior,
        va_splat(0.63661977236758134308f / parameters->m));
    const HVX_Vector atan_x = va_hvx_atan_f32(x);
    const HVX_Vector log_one_plus_x2 = va_hvx_log_f32(
        va_qadd(va_splat(1.0f), va_qmul(x, x)));
    const HVX_Vector exterior_radius = va_qadd(va_splat(parameters->rho),
        va_qmul(va_splat(parameters->m), va_qsub(va_qmul(x, atan_x),
            va_qmul(va_splat(0.5f), log_one_plus_x2))));
    *radius = va_select(outside, exterior_radius, va_splat(parameters->rho));
    const HVX_Vector magnitude = va_qmul(va_splat(0.63661977236758134308f), atan_x);
    const HVX_Vector signed_magnitude = va_select(Q6_Q_vcmp_gt_VsfVsf(zero, ell),
        va_qneg(magnitude), magnitude);
    *derivative = va_select(outside, signed_magnitude, zero);
}

static void wormhole_trace_hvx(
    uint32_t count,
    uint32_t input_stride,
    uint32_t output_stride,
    const struct wormhole_parameters *parameters,
    const float *input,
    float *output) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one = va_splat(1.0f);
    const HVX_Vector half_step = va_splat(0.5f * parameters->step_size);
    const HVX_Vector step = va_splat(parameters->step_size);
    for (uint32_t lane = 0; lane < count; lane += VA_HVX_FLOAT_LANES) {
        HVX_Vector ell = *(const HVX_Vector *)(input + lane);
        HVX_Vector phi = *(const HVX_Vector *)(input + input_stride + lane);
        HVX_Vector p_ell = *(const HVX_Vector *)(input + 2 * input_stride + lane);
        const HVX_Vector impact = *(const HVX_Vector *)(input + 3 * input_stride + lane);
        const HVX_Vector input_active = *(const HVX_Vector *)(input + 4 * input_stride + lane);
        HVX_VectorPred active = Q6_Q_not_Q(Q6_Q_vcmp_eq_VwVw(input_active, zero));
        active = Q6_Q_and_QQ(active,
            Q6_Q_not_Q(wormhole_terminal_vector(parameters, ell, p_ell)));
        for (uint32_t step_index = 0; step_index < parameters->max_steps; ++step_index) {
            if (!va_predicate_any(active)) break;
            HVX_Vector old_radius, old_derivative;
            wormhole_geometry_vector(parameters, ell, &old_radius, &old_derivative);
            const HVX_Vector impact_squared = va_qmul(impact, impact);
            const HVX_Vector old_force = va_qdiv(va_qmul(impact_squared, old_derivative),
                va_qmul(old_radius, va_qmul(old_radius, old_radius)));
            HVX_Vector next_p_ell = va_qadd(p_ell, va_qmul(half_step, old_force));
            HVX_Vector next_ell = va_qadd(ell, va_qmul(step, next_p_ell));
            HVX_Vector new_radius, new_derivative;
            wormhole_geometry_vector(parameters, next_ell, &new_radius, &new_derivative);
            const HVX_Vector new_force = va_qdiv(va_qmul(impact_squared, new_derivative),
                va_qmul(new_radius, va_qmul(new_radius, new_radius)));
            next_p_ell = va_qadd(next_p_ell, va_qmul(half_step, new_force));
            const HVX_Vector inverse_old_r2 = va_qinverse(va_qmul(old_radius, old_radius));
            const HVX_Vector inverse_new_r2 = va_qinverse(va_qmul(new_radius, new_radius));
            const HVX_Vector next_phi = va_qadd(phi, va_qmul(half_step,
                va_qmul(impact, va_qadd(inverse_old_r2, inverse_new_r2))));
            ell = va_select(active, next_ell, ell);
            phi = va_select(active, next_phi, phi);
            p_ell = va_select(active, next_p_ell, p_ell);
            active = Q6_Q_and_QQ(active,
                Q6_Q_not_Q(wormhole_terminal_vector(parameters, ell, p_ell)));
        }
        *(HVX_Vector *)(output + lane) = ell;
        *(HVX_Vector *)(output + output_stride + lane) = phi;
        *(HVX_Vector *)(output + 2 * output_stride + lane) = p_ell;
        *(HVX_Vector *)(output + 3 * output_stride + lane) = va_select(active, one, zero);
    }
}

static void kerr_trace_hvx(
    uint32_t count,
    uint32_t input_stride,
    uint32_t output_stride,
    const struct kerr_parameters *parameters,
    const float *input,
    float *output) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one = va_splat(1.0f);
    const HVX_Vector event_capture = va_splat(1.0f);
    const HVX_Vector event_disk = va_splat(2.0f);
    const HVX_Vector event_escape = va_splat(3.0f);
    const HVX_Vector horizon = va_splat(parameters->mass + sqrtf(
        parameters->mass * parameters->mass - parameters->spin * parameters->spin));
    const HVX_Vector escape_radius = va_splat(parameters->escape_radius);
    const HVX_Vector disk_inner = va_splat(parameters->disk_inner_radius);
    const HVX_Vector disk_outer = va_splat(parameters->disk_outer_radius);

    for (uint32_t lane = 0; lane < count; lane += VA_HVX_FLOAT_LANES) {
        struct kerr_ray_vector ray;
        for (int i = 0; i < 4; ++i) {
            ray.x[i] = *(const HVX_Vector *)(input + i * input_stride + lane);
            ray.p[i] = *(const HVX_Vector *)(input + (4 + i) * input_stride + lane);
        }
        const HVX_Vector input_active = *(const HVX_Vector *)(input + 8 * input_stride + lane);
        HVX_VectorPred active = Q6_Q_not_Q(Q6_Q_vcmp_eq_VwVw(input_active, zero));
        HVX_Vector event = zero;
        HVX_Vector disk_x = zero, disk_y = zero, disk_radius = zero;

        for (uint32_t step_index = 0; step_index < parameters->max_steps; ++step_index) {
            if (!va_predicate_any(active)) break;
            HVX_Vector radius = kerr_radius_vector(parameters, &ray);
            HVX_VectorPred capture = Q6_Q_and_QQ(active,
                Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(radius, horizon)));
            event = va_select(capture, event_capture, event);
            active = Q6_Q_and_QQ(active, Q6_Q_not_Q(capture));

            const HVX_VectorPred false_predicate = Q6_Q_vcmp_gt_VsfVsf(zero, zero);
            const HVX_VectorPred escape_candidates = Q6_Q_and_QQ(active,
                Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(escape_radius, radius)));
            HVX_VectorPred escape = false_predicate;
            if (va_predicate_any(escape_candidates))
                escape = Q6_Q_and_QQ(escape_candidates, kerr_outward_vector(parameters, &ray));
            event = va_select(escape, event_escape, event);
            active = Q6_Q_and_QQ(active, Q6_Q_not_Q(escape));

            if (!va_predicate_any(active)) break;

            const struct kerr_ray_vector previous = ray;
            const struct kerr_ray_vector advanced = kerr_midpoint_vector(parameters, &ray);
            for (int i = 0; i < 4; ++i) {
                ray.x[i] = va_select(active, advanced.x[i], ray.x[i]);
                ray.p[i] = va_select(active, advanced.p[i], ray.p[i]);
            }

            radius = kerr_radius_vector(parameters, &ray);
            capture = Q6_Q_and_QQ(active,
                Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(radius, horizon)));
            event = va_select(capture, event_capture, event);
            active = Q6_Q_and_QQ(active, Q6_Q_not_Q(capture));

            const HVX_VectorPred z0_positive = Q6_Q_vcmp_gt_VsfVsf(previous.x[3], zero);
            const HVX_VectorPred z0_negative = Q6_Q_vcmp_gt_VsfVsf(zero, previous.x[3]);
            const HVX_VectorPred z1_positive = Q6_Q_vcmp_gt_VsfVsf(ray.x[3], zero);
            const HVX_VectorPred z1_negative = Q6_Q_vcmp_gt_VsfVsf(zero, ray.x[3]);
            HVX_VectorPred crossed = Q6_Q_or_QQ(
                Q6_Q_and_QQ(z0_positive, Q6_Q_not_Q(z1_positive)),
                Q6_Q_and_QQ(z0_negative, Q6_Q_not_Q(z1_negative)));
            crossed = Q6_Q_and_QQ(active, crossed);
            if (va_predicate_any(crossed)) {
                const HVX_Vector fraction = va_qdiv(previous.x[3],
                    va_qsub(previous.x[3], ray.x[3]));
                const HVX_Vector hit_x = va_qadd(previous.x[1],
                    va_qmul(fraction, va_qsub(ray.x[1], previous.x[1])));
                const HVX_Vector hit_y = va_qadd(previous.x[2],
                    va_qmul(fraction, va_qsub(ray.x[2], previous.x[2])));
                struct kerr_ray_vector hit = ray;
                hit.x[1] = hit_x;
                hit.x[2] = hit_y;
                hit.x[3] = zero;
                const HVX_Vector hit_radius = kerr_radius_vector(parameters, &hit);
                disk_x = va_select(crossed, hit_x, disk_x);
                disk_y = va_select(crossed, hit_y, disk_y);
                disk_radius = va_select(crossed, hit_radius, disk_radius);
                HVX_VectorPred disk = Q6_Q_and_QQ(crossed,
                    Q6_Q_and_QQ(Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(disk_inner, hit_radius)),
                        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(hit_radius, disk_outer))));
                event = va_select(disk, event_disk, event);
                for (int i = 0; i < 4; ++i) {
                    ray.p[i] = va_select(disk, va_qadd(previous.p[i],
                        va_qmul(fraction, va_qsub(ray.p[i], previous.p[i]))), ray.p[i]);
                }
                ray.x[1] = va_select(disk, hit_x, ray.x[1]);
                ray.x[2] = va_select(disk, hit_y, ray.x[2]);
                ray.x[3] = va_select(disk, zero, ray.x[3]);
                active = Q6_Q_and_QQ(active, Q6_Q_not_Q(disk));
            }

            const HVX_VectorPred final_escape_candidates = Q6_Q_and_QQ(active,
                Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(escape_radius, radius)));
            escape = false_predicate;
            if (va_predicate_any(final_escape_candidates))
                escape = Q6_Q_and_QQ(final_escape_candidates,
                    kerr_outward_vector(parameters, &ray));
            event = va_select(escape, event_escape, event);
            active = Q6_Q_and_QQ(active, Q6_Q_not_Q(escape));
        }

        for (int i = 0; i < 4; ++i) {
            *(HVX_Vector *)(output + i * output_stride + lane) = ray.x[i];
            *(HVX_Vector *)(output + (4 + i) * output_stride + lane) = ray.p[i];
        }
        *(HVX_Vector *)(output + 8 * output_stride + lane) = event;
        *(HVX_Vector *)(output + 9 * output_stride + lane) = disk_x;
        *(HVX_Vector *)(output + 10 * output_stride + lane) = disk_y;
        *(HVX_Vector *)(output + 11 * output_stride + lane) = disk_radius;
        (void)one;
    }
}

static AEEResult kerr_trace_scalar(
    uint32_t count,
    uint32_t input_stride,
    uint32_t output_stride,
    const struct kerr_parameters *parameters,
    const float *input,
    float *output) {
    const float horizon = parameters->mass + sqrtf(
        parameters->mass * parameters->mass - parameters->spin * parameters->spin);
    for (uint32_t lane = 0; lane < count; ++lane) {
        struct kerr_ray ray;
        for (int i = 0; i < 4; ++i) {
            ray.x[i] = input[i * input_stride + lane];
            ray.p[i] = input[(4 + i) * input_stride + lane];
        }
        const int input_active = input[8 * input_stride + lane] != 0.0f;
        uint32_t event = 0;
        float disk_x = 0.0f, disk_y = 0.0f, disk_radius = 0.0f;
        for (uint32_t step = 0; input_active && event == 0 && step < parameters->max_steps; ++step) {
            const float current_radius = kerr_radius(parameters, &ray);
            if (current_radius <= horizon) { event = 1; break; }
            if (current_radius >= parameters->escape_radius && kerr_outward(parameters, &ray)) {
                event = 3;
                break;
            }
            const struct kerr_ray previous = ray;
            kerr_midpoint(parameters, &ray);
            const float next_radius = kerr_radius(parameters, &ray);
            if (next_radius <= horizon) { event = 1; break; }
            const float z0 = previous.x[3], z1 = ray.x[3];
            const int crossed = (z0 > 0.0f && z1 <= 0.0f) || (z0 < 0.0f && z1 >= 0.0f);
            if (crossed) {
                const float fraction = z0 / (z0 - z1);
                disk_x = previous.x[1] + fraction * (ray.x[1] - previous.x[1]);
                disk_y = previous.x[2] + fraction * (ray.x[2] - previous.x[2]);
                struct kerr_ray hit = ray;
                hit.x[1] = disk_x;
                hit.x[2] = disk_y;
                hit.x[3] = 0.0f;
                disk_radius = kerr_radius(parameters, &hit);
                if (disk_radius >= parameters->disk_inner_radius &&
                    disk_radius <= parameters->disk_outer_radius) {
                    event = 2;
                    ray.x[1] = disk_x;
                    ray.x[2] = disk_y;
                    ray.x[3] = 0.0f;
                    for (int i = 0; i < 4; ++i)
                        ray.p[i] = previous.p[i] + fraction * (ray.p[i] - previous.p[i]);
                    break;
                }
            }
            if (next_radius >= parameters->escape_radius && kerr_outward(parameters, &ray)) {
                event = 3;
                break;
            }
        }
        for (int i = 0; i < 4; ++i) {
            output[i * output_stride + lane] = ray.x[i];
            output[(4 + i) * output_stride + lane] = ray.p[i];
        }
        output[8 * output_stride + lane] = (float) event;
        output[9 * output_stride + lane] = disk_x;
        output[10 * output_stride + lane] = disk_y;
        output[11 * output_stride + lane] = disk_radius;
    }
    return AEE_SUCCESS;
}

static uint32_t va_hash_u32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

static float va_value_noise(float x, float y, uint32_t seed) {
    const int32_t x0 = (int32_t)floorf(x);
    const int32_t y0 = (int32_t)floorf(y);
    const float tx = x - (float)x0;
    const float ty = y - (float)y0;
    const float sx = tx * tx * (3.0f - 2.0f * tx);
    const float sy = ty * ty * (3.0f - 2.0f * ty);
#define VA_NOISE_SAMPLE(dx, dy) ((float)va_hash_u32( \
    ((uint32_t)(x0 + (dx)) * 0x9e3779b9u) ^ \
    ((uint32_t)(y0 + (dy)) * 0x85ebca6bu) ^ seed) / (float)UINT32_MAX)
    const float s00 = VA_NOISE_SAMPLE(0, 0);
    const float s10 = VA_NOISE_SAMPLE(1, 0);
    const float s01 = VA_NOISE_SAMPLE(0, 1);
    const float s11 = VA_NOISE_SAMPLE(1, 1);
#undef VA_NOISE_SAMPLE
    const float low = s00 + sx * (s10 - s00);
    const float high = s01 + sx * (s11 - s01);
    return low + sy * (high - low);
}

static float va_fbm(float x, float y, uint32_t seed) {
    float sum = 0.0f, weight = 0.5f, normalization = 0.0f;
    for (uint32_t octave = 0; octave < 5; ++octave) {
        sum += weight * va_value_noise(x, y, seed + octave * 0x27d4eb2du);
        normalization += weight;
        const float next_x = 1.73f * x - 1.11f * y + 7.3f;
        y = 1.11f * x + 1.73f * y - 4.1f;
        x = next_x;
        weight *= 0.5f;
    }
    return sum / normalization;
}

static void va_kerr_covariant_metric(
    const struct kerr_parameters *parameters,
    const float point[3],
    float metric[4][4]) {
    struct kerr_ray ray = { .x = {0.0f, point[0], point[1], point[2]} };
    float k_up[4], field;
    kerr_field(parameters, &ray, k_up, &field);
    const float k_cov[4] = {1.0f, k_up[1], k_up[2], k_up[3]};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const float eta = row == column ? (row == 0 ? -1.0f : 1.0f) : 0.0f;
            metric[row][column] = eta + field * k_cov[row] * k_cov[column];
        }
    }
}

static void va_kerr_camera_ray(
    const struct kerr_frame_parameters *parameters,
    float metric[4][4],
    uint32_t sample,
    struct kerr_ray *ray) {
    const uint32_t pixel = sample;
    const uint32_t x = pixel % parameters->width;
    const uint32_t y = pixel / parameters->width;
    const float jitter_x = 0.5f;
    const float jitter_y = 0.5f;
    const float aspect = (float)parameters->width / (float)parameters->height;
    const float screen_x = (((float)x + jitter_x) / (float)parameters->width * 2.0f - 1.0f) *
        parameters->tan_half_fov * aspect;
    const float screen_y = -(((float)y + jitter_y) / (float)parameters->height * 2.0f - 1.0f) *
        parameters->tan_half_fov;
    const float inverse_norm = 1.0f / sqrtf(1.0f + screen_x * screen_x + screen_y * screen_y);
    float tangent[4];
    for (int component = 0; component < 4; ++component) {
        tangent[component] = parameters->camera_time[component] + inverse_norm *
            (parameters->camera_forward[component] + screen_x * parameters->camera_right[component] +
                screen_y * parameters->camera_up[component]);
    }
    for (int row = 0; row < 4; ++row) {
        ray->p[row] = 0.0f;
        for (int column = 0; column < 4; ++column)
            ray->p[row] += metric[row][column] * tangent[column];
    }
    ray->x[0] = 0.0f;
    ray->x[1] = parameters->camera_position[0];
    ray->x[2] = parameters->camera_position[1];
    ray->x[3] = parameters->camera_position[2];
}

static void va_sky_color(const float direction[3], float color[3]) {
    const float cloud = va_fbm(3.2f * direction[0] + 1.7f * direction[2],
        3.2f * direction[1] - 1.3f * direction[2], 0x51f15e3du);
    const float dust = va_fbm(11.0f * direction[0] - 4.0f * direction[1],
        11.0f * direction[2] + 3.0f * direction[1], 0xa14f2c91u);
    const float band_coordinate = direction[2] * 0.79f + direction[0] * 0.24f -
        direction[1] * 0.18f + 0.10f * (cloud - 0.5f);
    const float band = expf(-34.0f * band_coordinate * band_coordinate);
    const float dust_offset = band_coordinate + 0.018f * (dust - 0.5f);
    const float dust_lane = 1.0f - 0.82f * expf(-260.0f * dust_offset * dust_offset) *
        (0.35f + 0.65f * dust);
    const int32_t cell_x = (int32_t)floorf(direction[0] * 900.0f);
    const int32_t cell_y = (int32_t)floorf(direction[1] * 900.0f);
    const int32_t cell_z = (int32_t)floorf(direction[2] * 900.0f);
    const float random = (float)va_hash_u32((uint32_t)cell_x * 0x9e3779b9u ^
        (uint32_t)cell_y * 0x85ebca6bu ^ (uint32_t)cell_z * 0xc2b2ae35u) /
        (float)UINT32_MAX;
    float star = 0.0f;
    if (random > 0.9972f) {
        const float base = (random - 0.9972f) / 0.0028f;
        star = base * base * base * base * base * base * 5.0f;
    }
    color[0] = 0.0015f + 0.11f * band * dust_lane * (0.25f + 0.75f * cloud) + star;
    color[1] = 0.0022f + 0.15f * band * dust_lane * (0.35f + 0.65f * cloud) + star * 0.91f;
    color[2] = 0.0050f + 0.23f * band * dust_lane * (0.45f + 0.55f * cloud) + star * 0.80f;
}

static void va_blackbody_rgb(float temperature_kelvin, float color[3]) {
    const double second_radiation = 0.01438776877;
    const double temperature = (double)fminf(fmaxf(temperature_kelvin, 900.0f), 30000.0f);
    const double wavelengths[3] = {610.0e-9, 550.0e-9, 460.0e-9};
    double values[3];
    for (int i = 0; i < 3; ++i) {
        const double exponent = second_radiation / (wavelengths[i] * temperature);
        values[i] = 1.0 / (pow(wavelengths[i], 5.0) * expm1(exponent));
    }
    const double maximum = fmax(values[0], fmax(values[1], values[2]));
    for (int i = 0; i < 3; ++i) color[i] = (float)(values[i] / maximum);
}

static float va_disk_turbulence(float x, float y, float radius, float inner_radius) {
    const float winding = -1.7f * logf(radius / inner_radius);
    const float sin_winding = sinf(winding), cos_winding = cosf(winding);
    const float rotated_x = 0.52f * (cos_winding * x - sin_winding * y);
    const float rotated_y = 0.52f * (sin_winding * x + cos_winding * y);
    const float broad = va_fbm(rotated_x, rotated_y, 0xc13fa9a9u);
    const float fine = va_fbm(2.7f * rotated_x + 5.0f, 0.65f * rotated_y - 3.0f,
        0x91e10da5u);
    return 0.62f + 0.58f * broad + 0.16f * (fine - 0.5f);
}

static float va_disk_flux(float radius, float inner_radius) {
    if (radius < inner_radius) return 0.0f;
    const float inverse = inner_radius / radius;
    const float inverse_cubed = inverse * inverse * inverse;
    const float edge = 1.0f - sqrtf(inverse);
    const float peak_inverse = 36.0f / 49.0f;
    const float peak = peak_inverse * peak_inverse * peak_inverse *
        (1.0f - sqrtf(peak_inverse));
    return fmaxf(inverse_cubed * edge / peak, 0.0f);
}

static float va_metric_inner(float metric[4][4], const float left[4], const float right[4]) {
    float value = 0.0f;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            value += metric[row][column] * left[row] * right[column];
    return value;
}

static float va_disk_redshift(
    const struct kerr_frame_parameters *parameters,
    float x,
    float y,
    float radius,
    const float momentum[4]) {
    const float root_mass = sqrtf(parameters->trace.mass);
    const float denominator = radius * sqrtf(radius) + parameters->trace.spin * root_mass;
    if (!isfinite(denominator) || fabsf(denominator) <= 1.0e-12f) return 1.0f;
    const float angular_velocity = root_mass / denominator;
    float emitter[4] = {1.0f, -angular_velocity * y, angular_velocity * x, 0.0f};
    const float point[3] = {x, y, 0.0f};
    float metric[4][4];
    va_kerr_covariant_metric(&parameters->trace, point, metric);
    const float tangent_norm = va_metric_inner(metric, emitter, emitter);
    if (!isfinite(tangent_norm) || tangent_norm >= -1.0e-12f) return 1.0f;
    const float normalization = 1.0f / sqrtf(-tangent_norm);
    for (int i = 0; i < 4; ++i) emitter[i] *= normalization;
    float observed = 0.0f, emitted = 0.0f;
    for (int i = 0; i < 4; ++i) {
        observed -= momentum[i] * parameters->camera_time[i];
        emitted -= momentum[i] * emitter[i];
    }
    if (!isfinite(observed) || !isfinite(emitted) || observed <= 1.0e-12f || emitted <= 1.0e-12f)
        return 1.0f;
    return fminf(fmaxf(observed / emitted, 0.12f), 3.0f);
}

static void va_disk_color(
    const struct kerr_frame_parameters *parameters,
    float x,
    float y,
    float radius,
    const float momentum[4],
    float color[3]) {
    const float local_flux = va_disk_flux(radius, parameters->trace.disk_inner_radius);
    const float turbulence = va_disk_turbulence(x, y, radius,
        parameters->trace.disk_inner_radius);
    const float redshift = va_disk_redshift(parameters, x, y, radius, momentum);
    const float local_temperature = 2200.0f + 8800.0f * powf(local_flux, 0.25f);
    va_blackbody_rgb(local_temperature * redshift, color);
    const float redshift_squared = redshift * redshift;
    const float intensity = 4.5f * local_flux * turbulence * redshift_squared * redshift_squared;
    for (int i = 0; i < 3; ++i) color[i] *= intensity;
}

static uint32_t va_pack_aces(const float color[3]) {
    uint32_t encoded[3];
    for (int channel = 0; channel < 3; ++channel) {
        const float linear = fmaxf(color[channel], 0.0f);
        float mapped = (linear * (2.51f * linear + 0.03f)) /
            (linear * (2.43f * linear + 0.59f) + 0.14f);
        mapped = fminf(fmaxf(mapped, 0.0f), 1.0f);
        const float srgb = mapped <= 0.0031308f ? 12.92f * mapped :
            1.055f * powf(mapped, 1.0f / 2.4f) - 0.055f;
        encoded[channel] = (uint32_t)(srgb * 255.0f + 0.5f);
    }
    return 0xff000000u | (encoded[0] << 16) | (encoded[1] << 8) | encoded[2];
}

static uint32_t va_shade_kerr_lane(
    const struct kerr_frame_parameters *parameters,
    const float *trace_output,
    uint32_t stride,
    uint32_t lane,
    uint32_t *event_out) {
    const uint32_t event = (uint32_t)trace_output[8 * stride + lane];
    float color[3];
    if (event == 1) {
        *event_out = event;
        return 0xff000000u;
    } else if (event == 2) {
        float momentum[4];
        for (int i = 0; i < 4; ++i) momentum[i] = trace_output[(4 + i) * stride + lane];
        va_disk_color(parameters, trace_output[9 * stride + lane],
            trace_output[10 * stride + lane], trace_output[11 * stride + lane], momentum, color);
    } else if (event == 3) {
        struct kerr_ray ray;
        for (int i = 0; i < 4; ++i) {
            ray.x[i] = trace_output[i * stride + lane];
            ray.p[i] = trace_output[(4 + i) * stride + lane];
        }
        float rate[4];
        kerr_position_rate(&parameters->trace, &ray, rate);
        const float norm = fmaxf(sqrtf(rate[1] * rate[1] + rate[2] * rate[2] +
            rate[3] * rate[3]), 1.0e-12f);
        const float direction[3] = {rate[1] / norm, rate[2] / norm, rate[3] / norm};
        va_sky_color(direction, color);
    } else {
        *event_out = 0;
        return 0xff550f21u;
    }
    *event_out = event >= 1 && event <= 3 ? event : 0;
    return va_pack_aces(color);
}

struct kerr_trace_vector_result {
    struct kerr_ray_vector ray;
    HVX_Vector event;
    HVX_Vector disk_x;
    HVX_Vector disk_y;
    HVX_Vector disk_radius;
};

/* Fast common-path probe for the one-step frame program.  A 1080x760 camera
   frame normally has no terminal events after one integration step.  Keep
   only the three previous position vectors needed to detect a disk crossing;
   if any lane does terminate, the caller reruns the full trace below to
   produce the exact event state used by shading. */
static int kerr_trace_one_vector_has_event(
    const struct kerr_parameters *parameters,
    const struct kerr_fixed_rhs_vector *fixed,
    const struct kerr_ray_vector *input_ray,
    HVX_VectorPred active) {
    struct kerr_ray_vector ray = *input_ray;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector horizon = va_splat(parameters->mass + sqrtf(
        parameters->mass * parameters->mass - parameters->spin * parameters->spin));
    const HVX_Vector escape_radius = va_splat(parameters->escape_radius);
    const HVX_Vector disk_inner = va_splat(parameters->disk_inner_radius);
    const HVX_Vector disk_outer = va_splat(parameters->disk_outer_radius);

    HVX_Vector radius = fixed->radius;
    HVX_VectorPred capture = Q6_Q_and_QQ(active,
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(radius, horizon)));
    if (va_predicate_any(capture)) return 1;

    const HVX_VectorPred escape_candidates = Q6_Q_and_QQ(active,
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(escape_radius, radius)));
    if (va_predicate_any(escape_candidates) &&
        va_predicate_any(Q6_Q_and_QQ(escape_candidates,
            kerr_outward_vector(parameters, &ray)))) return 1;

    const HVX_Vector previous_x1 = ray.x[1];
    const HVX_Vector previous_x2 = ray.x[2];
    const HVX_Vector previous_x3 = ray.x[3];
    kerr_midpoint_vector_fixed_first(parameters, fixed, &ray);

    radius = kerr_radius_vector(parameters, &ray);
    capture = Q6_Q_and_QQ(active,
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(radius, horizon)));
    if (va_predicate_any(capture)) return 1;
    active = Q6_Q_and_QQ(active, Q6_Q_not_Q(capture));

    const HVX_VectorPred z0_positive = Q6_Q_vcmp_gt_VsfVsf(previous_x3, zero);
    const HVX_VectorPred z0_negative = Q6_Q_vcmp_gt_VsfVsf(zero, previous_x3);
    const HVX_VectorPred z1_positive = Q6_Q_vcmp_gt_VsfVsf(ray.x[3], zero);
    const HVX_VectorPred z1_negative = Q6_Q_vcmp_gt_VsfVsf(zero, ray.x[3]);
    HVX_VectorPred crossed = Q6_Q_or_QQ(
        Q6_Q_and_QQ(z0_positive, Q6_Q_not_Q(z1_positive)),
        Q6_Q_and_QQ(z0_negative, Q6_Q_not_Q(z1_negative)));
    crossed = Q6_Q_and_QQ(active, crossed);
    if (va_predicate_any(crossed)) {
        const HVX_Vector fraction = va_qdiv(previous_x3,
            va_qsub(previous_x3, ray.x[3]));
        struct kerr_ray_vector hit = ray;
        hit.x[1] = va_qadd(previous_x1,
            va_qmul(fraction, va_qsub(ray.x[1], previous_x1)));
        hit.x[2] = va_qadd(previous_x2,
            va_qmul(fraction, va_qsub(ray.x[2], previous_x2)));
        hit.x[3] = zero;
        const HVX_Vector hit_radius = kerr_radius_vector(parameters, &hit);
        const HVX_VectorPred disk = Q6_Q_and_QQ(crossed,
            Q6_Q_and_QQ(Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(disk_inner, hit_radius)),
                Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(hit_radius, disk_outer))));
        if (va_predicate_any(disk)) return 1;
    }

    const HVX_VectorPred final_escape_candidates = Q6_Q_and_QQ(active,
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(escape_radius, radius)));
    return va_predicate_any(final_escape_candidates) &&
        va_predicate_any(Q6_Q_and_QQ(final_escape_candidates,
            kerr_outward_vector(parameters, &ray)));
}

/* Execute the exact one-step Kerr trace directly in HVX registers.  This is
   the same event ordering and centered finite-difference midpoint operation as
   kerr_trace_hvx, but avoids materializing nine input and twelve output planes
   for the fused frame program. */
static __attribute__((noinline)) int kerr_trace_one_vector(
    const struct kerr_parameters *parameters,
    const struct kerr_fixed_rhs_vector *fixed,
    const struct kerr_ray_vector *input_ray,
    HVX_VectorPred active,
    struct kerr_trace_vector_result *result_out) {
    struct kerr_ray_vector ray = *input_ray;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector event_capture = va_splat(1.0f);
    const HVX_Vector event_disk = va_splat(2.0f);
    const HVX_Vector event_escape = va_splat(3.0f);
    const HVX_Vector horizon = va_splat(parameters->mass + sqrtf(
        parameters->mass * parameters->mass - parameters->spin * parameters->spin));
    const HVX_Vector escape_radius = va_splat(parameters->escape_radius);
    const HVX_Vector disk_inner = va_splat(parameters->disk_inner_radius);
    const HVX_Vector disk_outer = va_splat(parameters->disk_outer_radius);
    const HVX_VectorPred false_predicate = Q6_Q_vcmp_gt_VsfVsf(zero, zero);
    HVX_Vector event = zero;
    HVX_Vector disk_x = zero, disk_y = zero, disk_radius = zero;

    HVX_Vector radius = fixed->radius;
    HVX_VectorPred capture = Q6_Q_and_QQ(active,
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(radius, horizon)));
    event = va_select(capture, event_capture, event);
    active = Q6_Q_and_QQ(active, Q6_Q_not_Q(capture));

    const HVX_VectorPred escape_candidates = Q6_Q_and_QQ(active,
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(escape_radius, radius)));
    HVX_VectorPred escape = false_predicate;
    if (va_predicate_any(escape_candidates))
        escape = Q6_Q_and_QQ(escape_candidates, kerr_outward_vector(parameters, &ray));
    event = va_select(escape, event_escape, event);
    active = Q6_Q_and_QQ(active, Q6_Q_not_Q(escape));

    if (va_predicate_any(active)) {
        const struct kerr_ray_vector previous = ray;
        kerr_midpoint_vector_fixed_first(parameters, fixed, &ray);
        for (int i = 0; i < 4; ++i) {
            ray.x[i] = va_select(active, ray.x[i], previous.x[i]);
            ray.p[i] = va_select(active, ray.p[i], previous.p[i]);
        }

        radius = kerr_radius_vector(parameters, &ray);
        capture = Q6_Q_and_QQ(active,
            Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(radius, horizon)));
        event = va_select(capture, event_capture, event);
        active = Q6_Q_and_QQ(active, Q6_Q_not_Q(capture));

        const HVX_VectorPred z0_positive = Q6_Q_vcmp_gt_VsfVsf(previous.x[3], zero);
        const HVX_VectorPred z0_negative = Q6_Q_vcmp_gt_VsfVsf(zero, previous.x[3]);
        const HVX_VectorPred z1_positive = Q6_Q_vcmp_gt_VsfVsf(ray.x[3], zero);
        const HVX_VectorPred z1_negative = Q6_Q_vcmp_gt_VsfVsf(zero, ray.x[3]);
        HVX_VectorPred crossed = Q6_Q_or_QQ(
            Q6_Q_and_QQ(z0_positive, Q6_Q_not_Q(z1_positive)),
            Q6_Q_and_QQ(z0_negative, Q6_Q_not_Q(z1_negative)));
        crossed = Q6_Q_and_QQ(active, crossed);
        if (va_predicate_any(crossed)) {
            const HVX_Vector fraction = va_qdiv(previous.x[3],
                va_qsub(previous.x[3], ray.x[3]));
            const HVX_Vector hit_x = va_qadd(previous.x[1],
                va_qmul(fraction, va_qsub(ray.x[1], previous.x[1])));
            const HVX_Vector hit_y = va_qadd(previous.x[2],
                va_qmul(fraction, va_qsub(ray.x[2], previous.x[2])));
            struct kerr_ray_vector hit = ray;
            hit.x[1] = hit_x;
            hit.x[2] = hit_y;
            hit.x[3] = zero;
            const HVX_Vector hit_radius = kerr_radius_vector(parameters, &hit);
            disk_x = va_select(crossed, hit_x, disk_x);
            disk_y = va_select(crossed, hit_y, disk_y);
            disk_radius = va_select(crossed, hit_radius, disk_radius);
            const HVX_VectorPred disk = Q6_Q_and_QQ(crossed,
                Q6_Q_and_QQ(Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(disk_inner, hit_radius)),
                    Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(hit_radius, disk_outer))));
            event = va_select(disk, event_disk, event);
            for (int i = 0; i < 4; ++i) {
                ray.p[i] = va_select(disk, va_qadd(previous.p[i],
                    va_qmul(fraction, va_qsub(ray.p[i], previous.p[i]))), ray.p[i]);
            }
            ray.x[1] = va_select(disk, hit_x, ray.x[1]);
            ray.x[2] = va_select(disk, hit_y, ray.x[2]);
            ray.x[3] = va_select(disk, zero, ray.x[3]);
            active = Q6_Q_and_QQ(active, Q6_Q_not_Q(disk));
        }

        const HVX_VectorPred final_escape_candidates = Q6_Q_and_QQ(active,
            Q6_Q_not_Q(Q6_Q_vcmp_gt_VsfVsf(escape_radius, radius)));
        escape = false_predicate;
        if (va_predicate_any(final_escape_candidates))
            escape = Q6_Q_and_QQ(final_escape_candidates,
                kerr_outward_vector(parameters, &ray));
        event = va_select(escape, event_escape, event);
    }

    if (!va_predicate_any(Q6_Q_not_Q(Q6_Q_vcmp_eq_VwVw(event, zero)))) return 0;
    *result_out = (struct kerr_trace_vector_result){
        .ray = ray,
        .event = event,
        .disk_x = disk_x,
        .disk_y = disk_y,
        .disk_radius = disk_radius,
    };
    return 1;
}

struct kerr_camera_covariant_basis {
    float time[4];
    float right[4];
    float up[4];
    float forward[4];
};

static void va_kerr_camera_covariant_basis_init(
    const struct kerr_frame_parameters *parameters,
    const float metric[4][4],
    struct kerr_camera_covariant_basis *basis) {
    for (int row = 0; row < 4; ++row) {
        basis->time[row] = 0.0f;
        basis->right[row] = 0.0f;
        basis->up[row] = 0.0f;
        basis->forward[row] = 0.0f;
        for (int column = 0; column < 4; ++column) {
            basis->time[row] += metric[row][column] * parameters->camera_time[column];
            basis->right[row] += metric[row][column] * parameters->camera_right[column];
            basis->up[row] += metric[row][column] * parameters->camera_up[column];
            basis->forward[row] += metric[row][column] * parameters->camera_forward[column];
        }
    }
}

static inline struct kerr_ray_vector va_kerr_camera_ray_vector(
    const struct kerr_frame_parameters *parameters,
    const struct kerr_camera_covariant_basis *basis,
    HVX_Vector screen_x,
    HVX_Vector screen_y) {
    const HVX_Vector one = va_splat(1.0f);
    const HVX_Vector inverse_norm = va_qrsqrt(va_qadd(one,
        va_qadd(va_qmul(screen_x, screen_x), va_qmul(screen_y, screen_y))));
    struct kerr_ray_vector ray;
    ray.x[0] = Q6_V_vzero();
    for (int i = 0; i < 3; ++i) ray.x[i + 1] = va_splat(parameters->camera_position[i]);
    for (int row = 0; row < 4; ++row) {
        const HVX_Vector direction = va_qadd(va_splat(basis->forward[row]),
            va_qadd(va_qmul(screen_x, va_splat(basis->right[row])),
                va_qmul(screen_y, va_splat(basis->up[row]))));
        ray.p[row] = va_qadd(va_splat(basis->time[row]), va_qmul(inverse_norm, direction));
    }
    return ray;
}

struct kerr_frame_job {
    struct va_htp_context *context;
    const struct kerr_frame_parameters *parameters;
    uint32_t lanes;
    uint32_t *pixels;
    atomic_uint worker_mask;
    atomic_uint concurrent;
    atomic_uint peak_concurrent;
    atomic_uint events[4];
    atomic_uint flags;
};

static void va_update_peak(atomic_uint *peak, unsigned int value) {
    unsigned int current = atomic_load(peak);
    while (current < value && !atomic_compare_exchange_weak(peak, &current, value)) {
    }
}

static void va_pack_kerr_vector(
    const struct kerr_frame_parameters *parameters,
    const struct kerr_trace_vector_result *result,
    uint32_t count,
    uint32_t *packed,
    uint32_t events[4]) {
    const HVX_VectorPred has_event = Q6_Q_not_Q(
        Q6_Q_vcmp_eq_VwVw(result->event, Q6_V_vzero()));
    if (!va_predicate_any(has_event)) {
        *(HVX_Vector *)packed = Q6_V_vsplat_R((int)0xff550f21u);
        events[0] += count;
        return;
    }

    float trace[12][VA_HVX_FLOAT_LANES] __attribute__((aligned(128)));
    for (int i = 0; i < 4; ++i) {
        *(HVX_Vector *)trace[i] = result->ray.x[i];
        *(HVX_Vector *)trace[4 + i] = result->ray.p[i];
    }
    *(HVX_Vector *)trace[8] = result->event;
    *(HVX_Vector *)trace[9] = result->disk_x;
    *(HVX_Vector *)trace[10] = result->disk_y;
    *(HVX_Vector *)trace[11] = result->disk_radius;
    for (uint32_t lane = 0; lane < count; ++lane) {
        uint32_t event;
        packed[lane] = va_shade_kerr_lane(parameters, &trace[0][0],
            VA_HVX_FLOAT_LANES, lane, &event);
        ++events[event];
    }
}

/* Keep the rare event materialization and scalar shading frame out of the
   event-free worker hot path. */
static __attribute__((noinline)) void kerr_trace_and_pack_one_vector_event(
    const struct kerr_frame_parameters *parameters,
    const struct kerr_fixed_rhs_vector *fixed,
    const struct kerr_ray_vector *ray,
    HVX_VectorPred active,
    uint32_t count,
    uint32_t *packed,
    uint32_t events[4]) {
    struct kerr_trace_vector_result result;
    if (kerr_trace_one_vector(&parameters->trace, fixed, ray, active, &result)) {
        va_pack_kerr_vector(parameters, &result, count, packed, events);
    } else {
        *(HVX_Vector *)packed = Q6_V_vsplat_R((int)0xff550f21u);
        events[0] += count;
    }
}

/* The one-step performance gate is row-regular, so keep the entire fused
   camera/trace/pack path in HVX registers and use VTCM only for reusable
   horizontal coordinates and two DMA-overlapped packed output rows. */
static void kerr_frame_one_step_worker(
    unsigned int workers,
    unsigned int worker,
    void *opaque) {
    struct kerr_frame_job *job = (struct kerr_frame_job *)opaque;
    atomic_fetch_or(&job->worker_mask, 1u << worker);
    const unsigned int concurrent = atomic_fetch_add(&job->concurrent, 1) + 1;
    va_update_peak(&job->peak_concurrent, concurrent);

    const uint32_t width = job->parameters->width;
    const uint32_t height = job->parameters->height;
    const uint32_t groups = (width + VA_HVX_FLOAT_LANES - 1u) / VA_HVX_FLOAT_LANES;
    const uint32_t row_slot_bytes = groups * sizeof(HVX_Vector);
    const uint32_t slice_bytes = (job->context->vtcm_size / workers) & ~127u;
    unsigned char *slice = job->context->vtcm + worker * slice_bytes;
    const uint32_t fixed_bytes = (sizeof(struct kerr_fixed_rhs_vector) + 127u) & ~127u;
    struct kerr_fixed_rhs_vector *fixed_rhs = (struct kerr_fixed_rhs_vector *)slice;
    HVX_Vector *screen_x = (HVX_Vector *)(slice + fixed_bytes);
    uint32_t *packed_rows[2] = {
        (uint32_t *)(slice + fixed_bytes + row_slot_bytes),
        (uint32_t *)(slice + fixed_bytes + 2u * row_slot_bytes),
    };
    if (fixed_bytes + 3u * row_slot_bytes > slice_bytes) {
        atomic_fetch_sub(&job->concurrent, 1);
        return;
    }

    const float aspect = (float)width / (float)height;
    for (uint32_t group = 0; group < groups; ++group) {
        union {
            HVX_Vector vector;
            float lane[VA_HVX_FLOAT_LANES];
        } coordinates;
        for (uint32_t lane = 0; lane < VA_HVX_FLOAT_LANES; ++lane) {
            const uint32_t x = group * VA_HVX_FLOAT_LANES + lane;
            coordinates.lane[lane] = (((float)x + 0.5f) / (float)width * 2.0f - 1.0f) *
                job->parameters->tan_half_fov * aspect;
        }
        screen_x[group] = coordinates.vector;
    }

    float camera_metric[4][4];
    va_kerr_covariant_metric(&job->parameters->trace,
        job->parameters->camera_position, camera_metric);
    struct kerr_camera_covariant_basis camera_basis;
    va_kerr_camera_covariant_basis_init(job->parameters, camera_metric, &camera_basis);
    kerr_fixed_rhs_vector_init(&job->parameters->trace,
        job->parameters->camera_position, fixed_rhs);
    uint32_t local_events[4] = {0, 0, 0, 0};
    hexagon_udma_descriptor_type0_t descriptors[2] __attribute__((aligned(16)));
    int dma_active = 0;
    uint32_t slot = 0;
    const uint32_t first_row = height * worker / workers;
    const uint32_t last_row = height * (worker + 1u) / workers;
    const HVX_Vector full_active = va_splat(1.0f);
    const HVX_VectorPred full_active_predicate = Q6_Q_not_Q(
        Q6_Q_vcmp_eq_VwVw(full_active, Q6_V_vzero()));
    HVX_VectorPred tail_active_predicate = full_active_predicate;
    const uint32_t tail_count = width - (groups - 1u) * VA_HVX_FLOAT_LANES;
    if (tail_count != VA_HVX_FLOAT_LANES) {
        union {
            HVX_Vector vector;
            float lane[VA_HVX_FLOAT_LANES];
        } mask;
        for (uint32_t lane = 0; lane < VA_HVX_FLOAT_LANES; ++lane)
            mask.lane[lane] = lane < tail_count ? 1.0f : 0.0f;
        tail_active_predicate = Q6_Q_not_Q(
            Q6_Q_vcmp_eq_VwVw(mask.vector, Q6_V_vzero()));
    }

    for (uint32_t y = first_row; y < last_row; ++y, slot ^= 1u) {
        const float screen_y_scalar = -(((float)y + 0.5f) / (float)height * 2.0f - 1.0f) *
            job->parameters->tan_half_fov;
        const HVX_Vector screen_y = va_splat(screen_y_scalar);
        uint32_t *packed = packed_rows[slot];
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t x = group * VA_HVX_FLOAT_LANES;
            const uint32_t count = width - x < VA_HVX_FLOAT_LANES ?
                width - x : VA_HVX_FLOAT_LANES;
            const HVX_VectorPred active = group + 1u == groups ?
                tail_active_predicate : full_active_predicate;
            const struct kerr_ray_vector ray = va_kerr_camera_ray_vector(
                job->parameters, &camera_basis, screen_x[group], screen_y);
            if (kerr_trace_one_vector_has_event(&job->parameters->trace, fixed_rhs,
                    &ray, active)) {
                kerr_trace_and_pack_one_vector_event(job->parameters, fixed_rhs,
                    &ray, active, count, packed + x, local_events);
            } else {
                *(HVX_Vector *)(packed + x) = Q6_V_vsplat_R((int)0xff550f21u);
                local_events[0] += count;
            }
        }

        if (dma_active) (void)Q6_R_dmwait();
        descriptors[slot] = (hexagon_udma_descriptor_type0_t){
            .next = NULL,
            .length = width * sizeof(uint32_t),
            .desctype = HEXAGON_UDMA_DESC_DESCTYPE_TYPE0,
            .dstcomp = HEXAGON_UDMA_DESC_COMP_NONE,
            .srccomp = HEXAGON_UDMA_DESC_COMP_NONE,
            .dstbypass = HEXAGON_UDMA_DESC_BYPASS_OFF,
            .srcbypass = HEXAGON_UDMA_DESC_BYPASS_OFF,
            .order = HEXAGON_UDMA_DESC_ORDER_ORDER,
            .dstate = HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE,
            .src = packed,
            .dst = job->pixels + y * width,
        };
        Q6_dmstart_A(&descriptors[slot]);
        dma_active = 1;
    }
    if (dma_active) (void)Q6_R_dmwait();
    if (dma_active) atomic_fetch_or(&job->flags, VA_FRAME_FLAG_UDMA);
    for (int event = 0; event < 4; ++event)
        atomic_fetch_add(&job->events[event], local_events[event]);
    atomic_fetch_sub(&job->concurrent, 1);
}

static void kerr_frame_worker(unsigned int workers, unsigned int worker, void *opaque) {
    struct kerr_frame_job *job = (struct kerr_frame_job *)opaque;
    atomic_fetch_or(&job->worker_mask, 1u << worker);
    const unsigned int concurrent = atomic_fetch_add(&job->concurrent, 1) + 1;
    va_update_peak(&job->peak_concurrent, concurrent);
    const uint32_t vector_groups = (job->lanes + VA_HVX_FLOAT_LANES - 1u) /
        VA_HVX_FLOAT_LANES;
    const uint32_t first = (vector_groups * worker / workers) * VA_HVX_FLOAT_LANES;
    uint32_t last = (vector_groups * (worker + 1) / workers) * VA_HVX_FLOAT_LANES;
    if (last > job->lanes) last = job->lanes;
    const uint32_t slice_bytes = (job->context->vtcm_size / workers) & ~127u;
    unsigned char *slice = job->context->vtcm + worker * slice_bytes;
    const uint32_t slot_bytes = (slice_bytes / 2u) & ~127u;
    uint32_t capacity = slot_bytes / (22u * sizeof(float));
    capacity &= ~(VA_HVX_FLOAT_LANES - 1u);
    if (capacity < VA_HVX_FLOAT_LANES) {
        atomic_fetch_sub(&job->concurrent, 1);
        return;
    }
    uint32_t local_events[4] = {0, 0, 0, 0};
    float camera_metric[4][4];
    va_kerr_covariant_metric(&job->parameters->trace,
        job->parameters->camera_position, camera_metric);
    hexagon_udma_descriptor_type0_t descriptors[2] __attribute__((aligned(16)));
    int dma_active = 0;
    uint32_t slot = 0;
    for (uint32_t offset = first; offset < last; slot ^= 1u) {
        const uint32_t count = last - offset < capacity ? last - offset : capacity;
        const uint32_t padded = (count + VA_HVX_FLOAT_LANES - 1u) &
            ~(VA_HVX_FLOAT_LANES - 1u);
        float *input = (float *)(slice + slot * slot_bytes);
        float *trace_output = input + 9u * capacity;
        uint32_t *packed = (uint32_t *)(trace_output + 12u * capacity);
        struct kerr_ray last_ray;
        for (uint32_t lane = 0; lane < count; ++lane) {
            struct kerr_ray ray;
            va_kerr_camera_ray(job->parameters, camera_metric, offset + lane, &ray);
            last_ray = ray;
            for (int i = 0; i < 4; ++i) {
                input[i * capacity + lane] = ray.x[i];
                input[(4 + i) * capacity + lane] = ray.p[i];
            }
            input[8u * capacity + lane] = 1.0f;
        }
        for (uint32_t lane = count; lane < padded; ++lane) {
            for (int i = 0; i < 4; ++i) {
                input[i * capacity + lane] = last_ray.x[i];
                input[(4 + i) * capacity + lane] = last_ray.p[i];
            }
            input[8u * capacity + lane] = 0.0f;
        }
        kerr_trace_hvx(padded, capacity, capacity, &job->parameters->trace, input, trace_output);
        for (uint32_t lane = 0; lane < count; ++lane) {
            uint32_t event;
            packed[lane] = va_shade_kerr_lane(job->parameters, trace_output, capacity, lane, &event);
            ++local_events[event];
        }
        if (dma_active) (void)Q6_R_dmwait();
        descriptors[slot] = (hexagon_udma_descriptor_type0_t){
            .next = NULL,
            .length = count * sizeof(uint32_t),
            .desctype = HEXAGON_UDMA_DESC_DESCTYPE_TYPE0,
            .dstcomp = HEXAGON_UDMA_DESC_COMP_NONE,
            .srccomp = HEXAGON_UDMA_DESC_COMP_NONE,
            .dstbypass = HEXAGON_UDMA_DESC_BYPASS_OFF,
            .srcbypass = HEXAGON_UDMA_DESC_BYPASS_OFF,
            .order = HEXAGON_UDMA_DESC_ORDER_ORDER,
            .dstate = HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE,
            .src = packed,
            .dst = job->pixels + offset,
        };
        Q6_dmstart_A(&descriptors[slot]);
        dma_active = 1;
        atomic_fetch_or(&job->flags, VA_FRAME_FLAG_UDMA);
        offset += count;
    }
    if (dma_active) (void)Q6_R_dmwait();
    for (int event = 0; event < 4; ++event)
        atomic_fetch_add(&job->events[event], local_events[event]);
    atomic_fetch_sub(&job->concurrent, 1);
}

static AEEResult kerr_frame_parallel(
    struct va_htp_context *context,
    uint32_t lanes,
    const struct kerr_frame_parameters *parameters,
    uint32_t *output) {
    if (!context->vtcm_resource ||
        HAP_compute_res_acquire_cached(context->vtcm_resource, 1000000u) != 0)
        return AEE_ENOMEMORY;
    struct kerr_frame_job job = {
        .context = context,
        .parameters = parameters,
        .lanes = lanes,
        .pixels = output + VA_KERR_FRAME_HEADER_WORDS,
    };
    atomic_init(&job.worker_mask, 0);
    atomic_init(&job.concurrent, 0);
    atomic_init(&job.peak_concurrent, 0);
    atomic_init(&job.flags, context->power_flags);
    for (int event = 0; event < 4; ++event) atomic_init(&job.events[event], 0);
    const unsigned int vector_groups =
        (lanes + VA_HVX_FLOAT_LANES - 1u) / VA_HVX_FLOAT_LANES;
    const unsigned int workers = vector_groups < VA_HTP_WORKERS ?
        vector_groups : VA_HTP_WORKERS;
    va_worker_callback_t callback = parameters->trace.max_steps == 1u ?
        kerr_frame_one_step_worker : kerr_frame_worker;
    AEEResult result = va_worker_pool_run(context->worker_pool, callback, &job, workers);
    HAP_compute_res_release_cached(context->vtcm_resource);
    output[0] = VA_KERR_FRAME_MAGIC;
    output[1] = atomic_load(&job.events[1]);
    output[2] = atomic_load(&job.events[2]);
    output[3] = atomic_load(&job.events[3]);
    output[4] = atomic_load(&job.events[0]);
    output[5] = atomic_load(&job.worker_mask);
    output[6] = atomic_load(&job.peak_concurrent);
    output[7] = atomic_load(&job.flags);
    return result;
}

static float va_decode_plasma(uint32_t storage, uint8_t bits) {
    const uint32_t exponent = storage == 1u ? bits >> 4 : (bits >> 3) & 0x1fu;
    const uint32_t mantissa = storage == 1u ? bits & 0x0fu : bits & 0x07u;
    if (storage == 0u) {
        const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
        const uint32_t signed_exponent = (bits >> 3) & 0x0fu;
        if (signed_exponent == 0x0fu && mantissa == 0x07u) return NAN;
        if (signed_exponent == 0u) return sign * (float)mantissa * 0.001953125f;
        return sign * ldexpf(1.0f + (float)mantissa * 0.125f,
            (int)signed_exponent - 7);
    }
    if ((storage == 1u && exponent == 0x0fu) ||
        (storage == 2u && exponent == 0x1fu))
        return mantissa == 0u ? INFINITY : NAN;
    if (exponent == 0u)
        return (float)mantissa * (storage == 1u ? 0.0009765625f : 0.00000762939453125f);
    return ldexpf(1.0f + (float)mantissa * (storage == 1u ? 0.0625f : 0.125f),
        (int)exponent - (storage == 1u ? 7 : 15));
}

static float va_lerp(float left, float right, float amount) {
    return left + amount * (right - left);
}

static float va_sample_trilinear_u8(
    const struct kerr_scene_header *scene,
    const uint8_t *values,
    const float point[3]) {
    const uint32_t dims[3] = {scene->dim_x, scene->dim_y, scene->dim_z};
    for (int axis = 0; axis < 3; ++axis)
        if (!isfinite(point[axis]) || point[axis] < 0.0f || point[axis] > (float)dims[axis])
            return 0.0f;
    uint32_t low[3], high[3];
    float fraction[3];
    for (int axis = 0; axis < 3; ++axis) {
        const float centered = fminf(fmaxf(point[axis] - 0.5f, 0.0f), (float)dims[axis] - 1.0f);
        low[axis] = (uint32_t)floorf(centered);
        high[axis] = low[axis] + 1u < dims[axis] ? low[axis] + 1u : low[axis];
        fraction[axis] = centered - (float)low[axis];
    }
#define VA_SAMPLE3(x, y, z) va_decode_plasma(scene->storage, \
    values[(x) + scene->dim_x * ((y) + scene->dim_y * (z))])
    const float x00 = va_lerp(VA_SAMPLE3(low[0], low[1], low[2]),
        VA_SAMPLE3(high[0], low[1], low[2]), fraction[0]);
    const float x10 = va_lerp(VA_SAMPLE3(low[0], high[1], low[2]),
        VA_SAMPLE3(high[0], high[1], low[2]), fraction[0]);
    const float x01 = va_lerp(VA_SAMPLE3(low[0], low[1], high[2]),
        VA_SAMPLE3(high[0], low[1], high[2]), fraction[0]);
    const float x11 = va_lerp(VA_SAMPLE3(low[0], high[1], high[2]),
        VA_SAMPLE3(high[0], high[1], high[2]), fraction[0]);
#undef VA_SAMPLE3
    return va_lerp(va_lerp(x00, x10, fraction[1]),
        va_lerp(x01, x11, fraction[1]), fraction[2]);
}

static float va_sample_surface(
    const struct kerr_scene_header *scene,
    const float *values,
    const float point[2]) {
    if (!isfinite(point[0]) || !isfinite(point[1]) || point[0] < 0.0f || point[1] < 0.0f ||
        point[0] > (float)scene->dim_x || point[1] > (float)scene->dim_y)
        return 0.0f;
    const float centered_x = fminf(fmaxf(point[0] - 0.5f, 0.0f), (float)scene->dim_x - 1.0f);
    const float centered_y = fminf(fmaxf(point[1] - 0.5f, 0.0f), (float)scene->dim_y - 1.0f);
    const uint32_t x0 = (uint32_t)floorf(centered_x), y0 = (uint32_t)floorf(centered_y);
    const uint32_t x1 = x0 + 1u < scene->dim_x ? x0 + 1u : x0;
    const uint32_t y1 = y0 + 1u < scene->dim_y ? y0 + 1u : y0;
    const float fx = centered_x - (float)x0, fy = centered_y - (float)y0;
    const float top = va_lerp(values[x0 + scene->dim_x * y0],
        values[x1 + scene->dim_x * y0], fx);
    const float bottom = va_lerp(values[x0 + scene->dim_x * y1],
        values[x1 + scene->dim_x * y1], fx);
    return va_lerp(top, bottom, fy);
}

static void va_sample_surface_transfer(
    const struct kerr_scene_header *scene,
    const float *values,
    float temperature,
    float redshift,
    float color[3]) {
    const float tp = (fminf(fmaxf(temperature, 250000.0f), 1500000.0f) - 250000.0f) /
        1250000.0f * (float)(scene->surface_temperature_bins - 1u);
    const float rp = (fminf(fmaxf(redshift, 0.12f), 3.0f) - 0.12f) / 2.88f *
        (float)(scene->surface_redshift_bins - 1u);
    const uint32_t tl = (uint32_t)floorf(tp), rl = (uint32_t)floorf(rp);
    const uint32_t th = tl + 1u < scene->surface_temperature_bins ? tl + 1u : tl;
    const uint32_t rh = rl + 1u < scene->surface_redshift_bins ? rl + 1u : rl;
    const float ta = tp - (float)tl, ra = rp - (float)rl;
    for (uint32_t channel = 0; channel < 3u; ++channel) {
#define VA_SURFACE_LUT(t, r) values[3u * ((t) + scene->surface_temperature_bins * (r)) + channel]
        const float low = va_lerp(VA_SURFACE_LUT(tl, rl), VA_SURFACE_LUT(th, rl), ta);
        const float high = va_lerp(VA_SURFACE_LUT(tl, rh), VA_SURFACE_LUT(th, rh), ta);
#undef VA_SURFACE_LUT
        color[channel] = va_lerp(low, high, ra);
    }
}

static float va_display_temperature(float effective) {
    const float floor_log = logf(250000.0f), peak_log = logf(1500000.0f);
    const float amount = fminf(fmaxf((logf(fmaxf(effective, 250000.0f)) - floor_log) /
        (peak_log - floor_log), 0.0f), 1.0f);
    return 1600.0f + powf(amount, 0.90f) * 3200.0f;
}

static void va_sample_blackbody(
    const struct kerr_scene_header *scene,
    const float *values,
    float temperature,
    float color[3]) {
    const float position = (fminf(fmaxf(temperature, 900.0f), 30000.0f) - 900.0f) /
        29100.0f * (float)(scene->blackbody_bins - 1u);
    const uint32_t low = (uint32_t)floorf(position);
    const uint32_t high = low + 1u < scene->blackbody_bins ? low + 1u : low;
    const float amount = position - (float)low;
    for (uint32_t channel = 0; channel < 3u; ++channel)
        color[channel] = va_lerp(values[4u * low + channel], values[4u * high + channel], amount);
}

static void va_sample_sky(
    const struct kerr_scene_header *scene,
    const float *values,
    float u,
    float v,
    float color[3]) {
    const float x = u * (float)scene->sky_width - 0.5f;
    const float y = v * (float)scene->sky_height - 0.5f;
    const int32_t x0 = (int32_t)floorf(x), y0 = (int32_t)floorf(y);
    const float fx = x - (float)x0, fy = y - (float)y0;
    int32_t xs[2] = {x0, x0 + 1}, ys[2] = {y0, y0 + 1};
    for (int i = 0; i < 2; ++i) {
        xs[i] %= (int32_t)scene->sky_width;
        if (xs[i] < 0) xs[i] += (int32_t)scene->sky_width;
        if (ys[i] < 0) ys[i] = 0;
        if (ys[i] >= (int32_t)scene->sky_height) ys[i] = (int32_t)scene->sky_height - 1;
    }
    for (uint32_t channel = 0; channel < 3u; ++channel) {
#define VA_SKY(xi, yi) values[3u * ((uint32_t)xs[(xi)] + scene->sky_width * (uint32_t)ys[(yi)]) + channel]
        const float upper = va_lerp(VA_SKY(0, 0), VA_SKY(1, 0), fx);
        const float lower = va_lerp(VA_SKY(0, 1), VA_SKY(1, 1), fx);
#undef VA_SKY
        color[channel] = va_lerp(upper, lower, fy);
    }
}

static void va_march_emission(
    const struct kerr_scene_header *scene,
    const uint8_t *extinction,
    const uint8_t *source,
    const float origin[3],
    const float direction[3],
    float t_max,
    float color[3],
    float *transmittance) {
    float d[3], t0 = 0.0f, t1 = t_max;
    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = fabsf(direction[axis]) < 1.0e-12f ? copysignf(1.0e-12f, direction[axis]) : direction[axis];
        const float inverse = 1.0f / d[axis];
        const float ta = -origin[axis] * inverse;
        const float dims = (float)(axis == 0 ? scene->dim_x : axis == 1 ? scene->dim_y : scene->dim_z);
        const float tb = (dims - origin[axis]) * inverse;
        t0 = fmaxf(t0, fminf(ta, tb));
        t1 = fminf(t1, fmaxf(ta, tb));
    }
    color[0] = color[1] = color[2] = 0.0f;
    *transmittance = 1.0f;
    if (t0 > t1 || t_max <= 0.0f) return;
    int32_t cell[3], step[3];
    float t_next[3], t_delta[3];
    const uint32_t dims[3] = {scene->dim_x, scene->dim_y, scene->dim_z};
    for (int axis = 0; axis < 3; ++axis) {
        const float position = origin[axis] + t0 * d[axis];
        int32_t coordinate = (int32_t)position;
        if (coordinate < 0) coordinate = 0;
        if (coordinate >= (int32_t)dims[axis]) coordinate = (int32_t)dims[axis] - 1;
        cell[axis] = coordinate;
        step[axis] = d[axis] > 0.0f ? 1 : -1;
        t_delta[axis] = 1.0f / fabsf(d[axis]);
        const float boundary = d[axis] > 0.0f ? (float)coordinate + 1.0f : (float)coordinate;
        t_next[axis] = t0 + (boundary - position) / d[axis];
    }
    const int32_t strides[3] = {1, (int32_t)scene->dim_x,
        (int32_t)(scene->dim_x * scene->dim_y)};
    int32_t index = cell[0] + (int32_t)scene->dim_x *
        (cell[1] + (int32_t)scene->dim_y * cell[2]);
    float t = t0;
    for (;;) {
        const int axis = t_next[0] <= t_next[1] && t_next[0] <= t_next[2] ? 0 :
            (t_next[1] <= t_next[2] ? 1 : 2);
        const float t_exit = fminf(t_next[axis], t1);
        const float segment = fmaxf(t_exit - t, 0.0f);
        const float sigma = fmaxf(va_decode_plasma(scene->storage, extinction[index]), 0.0f);
        if (sigma > 0.0f && segment > 0.0f) {
            const float absorbed = -expm1f(-sigma * scene->plasma_extinction * segment);
            const float before = *transmittance;
            const float after = before * (1.0f - absorbed);
            const int cutoff = after <= 0.01f;
            const float applied = cutoff ? 1.0f - 0.01f / before : absorbed;
            const float weight = before * applied;
            for (int channel = 0; channel < 3; ++channel)
                color[channel] += weight * fmaxf(va_decode_plasma(scene->storage,
                    source[3 * index + channel]), 0.0f) * scene->plasma_emission;
            *transmittance = cutoff ? 0.01f : after;
            if (cutoff) return;
        }
        t = t_next[axis];
        if (t > t1) return;
        cell[axis] += step[axis];
        if (cell[axis] < 0 || cell[axis] >= (int32_t)dims[axis]) return;
        index += step[axis] * strides[axis];
        t_next[axis] += t_delta[axis];
    }
}

static void va_shade_reference_disk(
    const struct kerr_scene_header *scene,
    const unsigned char *bytes,
    const struct kerr_boundary_record *record,
    float color[3]) {
    const float *v = record->value;
    const float phase = v[10] * (scene->time * scene->plasma_time_scale - v[11]);
    const float sine = sinf(-phase), cosine = cosf(-phase);
    const float point[3] = {cosine * v[0] - sine * v[1], sine * v[0] + cosine * v[1], v[2]};
    const float direction[3] = {cosine * v[4] - sine * v[5], sine * v[4] + cosine * v[5], v[6]};
    const float redshift = fminf(fmaxf(v[7], 0.12f), 3.0f);
    const float position[3] = {
        (point[0] + scene->xy_half_extent) * (float)scene->dim_x / (2.0f * scene->xy_half_extent),
        (point[1] + scene->xy_half_extent) * (float)scene->dim_y / (2.0f * scene->xy_half_extent),
        (point[2] + scene->half_thickness) * (float)scene->dim_z / (2.0f * scene->half_thickness),
    };
    const float outer_amount = fminf(fmaxf((scene->disk_outer_radius - v[3]) /
        (0.16f * scene->disk_outer_radius), 0.0f), 1.0f);
    const float outer_taper = outer_amount * outer_amount * (3.0f - 2.0f * outer_amount);
    float temperature, density;
    if (scene->plasma_mode == 1u) {
        const float *temperature_field = (const float *)(bytes + scene->surface_temperature_offset);
        const float *density_field = (const float *)(bytes + scene->surface_density_offset);
        temperature = va_sample_surface(scene, temperature_field, position);
        density = va_sample_surface(scene, density_field, position);
    } else {
        temperature = 10000.0f * va_sample_trilinear_u8(scene,
            bytes + scene->temperature_offset, position);
        density = va_sample_trilinear_u8(scene, bytes + scene->extinction_offset, position);
    }
    if (temperature < 900.0f) {
        const float kinetic = fminf(fmaxf(v[9] / scene->kinetic_reference, 0.0f), 1.5f);
        const float heating = fminf(fmaxf(v[8] * (0.58f + 0.42f * kinetic), 0.0f), 1.0f);
        const double floor2 = 250000.0 * 250000.0, peak2 = 1500000.0 * 1500000.0;
        temperature = (float)sqrt(sqrt(floor2 * floor2 + (double)heating *
            (peak2 * peak2 - floor2 * floor2)));
    }
    const float fill = fminf(fmaxf(density / 0.30f, 0.0f), 1.0f);
    const float surface_scale = outer_taper * (0.06f + 0.64f * fill);
    if (scene->plasma_mode == 1u) {
        va_sample_surface_transfer(scene,
            (const float *)(bytes + scene->surface_transfer_offset), temperature, redshift, color);
        for (int channel = 0; channel < 3; ++channel) color[channel] *= surface_scale;
        return;
    }
    const float *blackbody = (const float *)(bytes + scene->blackbody_offset);
    float reference[3], shifted[3], surface[3];
    va_sample_blackbody(scene, blackbody, va_display_temperature(temperature), reference);
    va_sample_blackbody(scene, blackbody, va_display_temperature(temperature * redshift), shifted);
    const float gain = 13.0f * sqrtf(temperature / 1000000.0f);
    for (int channel = 0; channel < 3; ++channel)
        surface[channel] = shifted[channel] * gain * surface_scale;
    const float span = 2.5f * scene->xy_half_extent + 2.0f * scene->half_thickness;
    const float scale[3] = {(float)scene->dim_x / (2.0f * scene->xy_half_extent),
        (float)scene->dim_y / (2.0f * scene->xy_half_extent),
        (float)scene->dim_z / (2.0f * scene->half_thickness)};
    float origin[3], grid_direction[3];
    for (int axis = 0; axis < 3; ++axis) {
        const float half = axis == 2 ? scene->half_thickness : scene->xy_half_extent;
        origin[axis] = (point[axis] - span * direction[axis] + half) * scale[axis];
        grid_direction[axis] = direction[axis] * scale[axis];
    }
    float volume[3], transmittance;
    va_march_emission(scene, bytes + scene->extinction_offset,
        bytes + scene->source_offset, origin, grid_direction, 2.0f * span,
        volume, &transmittance);
    for (int channel = 0; channel < 3; ++channel) {
        const float spectral_shift = shifted[channel] / fmaxf(reference[channel], 1.0e-4f);
        color[channel] = volume[channel] * spectral_shift + transmittance * surface[channel];
    }
}

static void va_shade_reference_sample(
    const struct kerr_scene_header *scene,
    const unsigned char *bytes,
    const struct kerr_boundary_record *record,
    float color[3]) {
    if (record->kind == 1u || record->kind == 5u) {
        color[0] = color[1] = color[2] = 0.0f;
    } else if (record->kind == 2u) {
        va_shade_reference_disk(scene, bytes, record, color);
    } else if (record->kind == 3u) {
        const float frame_u = scene->time * scene->sky_speed * 0.15915494309189535f;
        va_sample_sky(scene, (const float *)(bytes + scene->sky_offset),
            record->value[5] + frame_u, record->value[6], color);
    } else {
        color[0] = 0.08f; color[1] = 0.012f; color[2] = 0.025f;
    }
}

static uint32_t va_shade_reference_pixel(
    const struct kerr_scene_header *scene,
    const unsigned char *bytes,
    uint32_t pixel) {
    const struct kerr_boundary_record *base =
        (const struct kerr_boundary_record *)(bytes + scene->base_offset);
    const uint32_t *lookup = (const uint32_t *)(bytes + scene->refinement_lookup_offset);
    const struct kerr_boundary_record *refined =
        (const struct kerr_boundary_record *)(bytes + scene->refinement_offset);
    const struct kerr_boundary_record *samples = base + pixel * scene->base_spp;
    uint32_t count = scene->base_spp;
    if (scene->refinement_spp && lookup[pixel] != UINT32_MAX) {
        samples = refined + lookup[pixel] * scene->refinement_spp;
        count = scene->refinement_spp;
    }
    float color[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t sample = 0; sample < count; ++sample) {
        float sample_color[3];
        va_shade_reference_sample(scene, bytes, samples + sample, sample_color);
        for (int channel = 0; channel < 3; ++channel) color[channel] += sample_color[channel];
    }
    const float scale = scene->exposure / (float)count;
    for (int channel = 0; channel < 3; ++channel) color[channel] *= scale;
    return va_pack_aces(color);
}

struct kerr_shade_job {
    struct va_htp_context *context;
    const struct kerr_scene_header *scene;
    const unsigned char *bytes;
    uint32_t *output;
};

static void kerr_shade_worker(unsigned int workers, unsigned int worker, void *opaque) {
    struct kerr_shade_job *job = (struct kerr_shade_job *)opaque;
    const uint32_t first = job->scene->pixels * worker / workers;
    const uint32_t last = job->scene->pixels * (worker + 1u) / workers;
    const uint32_t slice_bytes = (job->context->vtcm_size / workers) & ~127u;
    const uint32_t slot_bytes = (slice_bytes / 2u) & ~127u;
    const uint32_t capacity = slot_bytes / sizeof(uint32_t);
    unsigned char *slice = job->context->vtcm + worker * slice_bytes;
    hexagon_udma_descriptor_type0_t descriptors[2] __attribute__((aligned(16)));
    int dma_active = 0;
    uint32_t slot = 0;
    for (uint32_t offset = first; offset < last; slot ^= 1u) {
        const uint32_t count = last - offset < capacity ? last - offset : capacity;
        uint32_t *packed = (uint32_t *)(slice + slot * slot_bytes);
        for (uint32_t lane = 0; lane < count; ++lane)
            packed[lane] = va_shade_reference_pixel(job->scene, job->bytes, offset + lane);
        if (dma_active) (void)Q6_R_dmwait();
        descriptors[slot] = (hexagon_udma_descriptor_type0_t){
            .next = NULL,
            .length = count * sizeof(uint32_t),
            .desctype = HEXAGON_UDMA_DESC_DESCTYPE_TYPE0,
            .dstcomp = HEXAGON_UDMA_DESC_COMP_NONE,
            .srccomp = HEXAGON_UDMA_DESC_COMP_NONE,
            .dstbypass = HEXAGON_UDMA_DESC_BYPASS_OFF,
            .srcbypass = HEXAGON_UDMA_DESC_BYPASS_OFF,
            .order = HEXAGON_UDMA_DESC_ORDER_ORDER,
            .dstate = HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE,
            .src = packed,
            .dst = job->output + offset,
        };
        Q6_dmstart_A(&descriptors[slot]);
        dma_active = 1;
        offset += count;
    }
    if (dma_active) (void)Q6_R_dmwait();
}

static int va_scene_offset_valid(uint32_t offset, uint32_t bytes, uint32_t total) {
    return offset >= VA_KERR_SCENE_HEADER_BYTES && offset <= total && bytes <= total - offset;
}

static AEEResult kerr_shade_parallel(
    struct va_htp_context *context,
    const struct kerr_scene_header *scene,
    const unsigned char *bytes,
    uint32_t *output) {
    if (!context->vtcm_resource ||
        HAP_compute_res_acquire_cached(context->vtcm_resource, 1000000u) != 0)
        return AEE_ENOMEMORY;
    struct kerr_shade_job job = {.context = context, .scene = scene, .bytes = bytes, .output = output};
    const unsigned int workers = scene->pixels >= VA_HTP_WORKERS ? VA_HTP_WORKERS : 1u;
    AEEResult result = va_worker_pool_run(context->worker_pool, kerr_shade_worker, &job, workers);
    HAP_compute_res_release_cached(context->vtcm_resource);
    return result;
}

struct trace_job {
    struct va_htp_context *context;
    uint32_t opcode;
    uint32_t lanes;
    const void *parameters;
    const float *input;
    float *output;
    int use_vtcm;
};

static void trace_worker(unsigned int workers, unsigned int worker, void *opaque) {
    struct trace_job *job = (struct trace_job *)opaque;
    const uint32_t vector_groups = job->lanes / VA_HVX_FLOAT_LANES;
    const uint32_t first = (vector_groups * worker / workers) * VA_HVX_FLOAT_LANES;
    uint32_t last = (vector_groups * (worker + 1) / workers) * VA_HVX_FLOAT_LANES;
    if (worker + 1 == workers) last = job->lanes;
    const uint32_t input_planes = job->opcode == VA_HTP_OP_KERR_TRACE ? 9u : 5u;
    const uint32_t output_planes = job->opcode == VA_HTP_OP_KERR_TRACE ? 12u : 4u;

    uint32_t tile_capacity = last - first;
    unsigned char *scratch = NULL;
    if (job->use_vtcm) {
        const uint32_t slice_bytes = (job->context->vtcm_size / workers) & ~127u;
        scratch = job->context->vtcm + worker * slice_bytes;
        tile_capacity = slice_bytes / ((input_planes + output_planes) * sizeof(float));
        tile_capacity &= ~(VA_HVX_FLOAT_LANES - 1u);
    }
    if (!tile_capacity) tile_capacity = last - first;

    uint32_t offset = first;
    while (offset < last) {
        uint32_t count = last - offset;
        if (count > tile_capacity) count = tile_capacity;
        if (scratch && count >= VA_HVX_FLOAT_LANES) {
            count &= ~(VA_HVX_FLOAT_LANES - 1u);
            float *tile_input = (float *)scratch;
            float *tile_output = tile_input + input_planes * count;
            for (uint32_t plane = 0; plane < input_planes; ++plane) {
                memcpy(tile_input + plane * count,
                    job->input + plane * job->lanes + offset, count * sizeof(float));
            }
            if (job->opcode == VA_HTP_OP_KERR_TRACE) {
                kerr_trace_hvx(count, count, count,
                    (const struct kerr_parameters *)job->parameters, tile_input, tile_output);
            } else {
                wormhole_trace_hvx(count, count, count,
                    (const struct wormhole_parameters *)job->parameters, tile_input, tile_output);
            }
            for (uint32_t plane = 0; plane < output_planes; ++plane) {
                memcpy(job->output + plane * job->lanes + offset,
                    tile_output + plane * count, count * sizeof(float));
            }
        } else {
            if (job->opcode == VA_HTP_OP_KERR_TRACE) {
                kerr_trace_scalar(count, job->lanes, job->lanes,
                    (const struct kerr_parameters *)job->parameters,
                    job->input + offset, job->output + offset);
            } else {
                wormhole_trace_scalar(count, job->lanes, job->lanes,
                    (const struct wormhole_parameters *)job->parameters,
                    job->input + offset, job->output + offset);
            }
        }
        offset += count;
    }
}

static AEEResult trace_parallel(
    struct va_htp_context *context,
    uint32_t opcode,
    uint32_t lanes,
    const void *parameters,
    const float *input,
    float *output) {
    struct trace_job job = {
        .context = context,
        .opcode = opcode,
        .lanes = lanes,
        .parameters = parameters,
        .input = input,
        .output = output,
        .use_vtcm = 0,
    };
    const uint32_t max_steps = opcode == VA_HTP_OP_KERR_TRACE ?
        ((const struct kerr_parameters *)parameters)->max_steps :
        ((const struct wormhole_parameters *)parameters)->max_steps;
    /* One-step dispatches stream aligned shared DDR directly. Reused multi-step
       state is staged through worker-private VTCM tiles. */
    if (max_steps > 1 && context->vtcm_resource &&
        HAP_compute_res_acquire_cached(context->vtcm_resource, 1000000u) == 0) {
        job.use_vtcm = 1;
    }
    AEEResult result = va_worker_pool_run(context->worker_pool, trace_worker, &job,
        lanes >= VA_HTP_WORKERS * VA_HVX_FLOAT_LANES ? VA_HTP_WORKERS : 1u);
    if (job.use_vtcm) HAP_compute_res_release_cached(context->vtcm_resource);
    return result;
}

AEEResult va_htp_execute(
    remote_handle64 handle,
    uint32_t opcode,
    uint32_t lanes,
    const unsigned char *parameters,
    int parameters_len,
    const unsigned char *input0,
    int input0_len,
    const unsigned char *input1,
    int input1_len,
    unsigned char *output,
    int output_len,
    uint64_t *elapsed_cycles) {
    struct va_htp_context *ctx = (struct va_htp_context *) handle;
    if (!ctx || (parameters_len && !parameters) || !input0 || !output || !elapsed_cycles || !lanes)
        return AEE_EBADPARM;
    const uint32_t bytes = lanes * sizeof(float);
    if (opcode == VA_HTP_OP_KERR_SHADE) {
        if (parameters_len != (int)sizeof(uint32_t) || input0_len < 0 || output_len < 0 ||
            *(const uint32_t *)parameters != (uint32_t)input0_len ||
            (uint64_t)output_len < (uint64_t)lanes * sizeof(uint32_t) ||
            (uint32_t)input0_len < VA_KERR_SCENE_HEADER_BYTES)
            return AEE_EBADPARM;
        const struct kerr_scene_header *scene = (const struct kerr_scene_header *)input0;
        const uint64_t voxels = (uint64_t)scene->dim_x * scene->dim_y * scene->dim_z;
        const uint64_t base_records = (uint64_t)scene->pixels * scene->base_spp;
        const uint64_t surface_values = (uint64_t)scene->dim_x * scene->dim_y;
        const uint64_t transfer_values = (uint64_t)scene->surface_temperature_bins *
            scene->surface_redshift_bins * 3u;
        const uint64_t sky_values = (uint64_t)scene->sky_width * scene->sky_height * 3u;
        if (scene->magic != VA_KERR_SCENE_MAGIC || scene->abi != VA_KERR_SCENE_ABI ||
            scene->pixels != lanes || !scene->base_spp || scene->storage > 2u ||
            scene->plasma_mode > 1u || !scene->dim_x || !scene->dim_y || !scene->dim_z ||
            !scene->sky_width || !scene->sky_height || !scene->surface_temperature_bins ||
            !scene->surface_redshift_bins || !scene->blackbody_bins ||
            scene->total_bytes != (uint32_t)input0_len ||
            !isfinite(scene->time) || !isfinite(scene->plasma_time_scale) ||
            !isfinite(scene->sky_speed) || !isfinite(scene->exposure) || scene->exposure <= 0.0f ||
            voxels > UINT32_MAX || base_records > UINT32_MAX || surface_values > UINT32_MAX ||
            transfer_values > UINT32_MAX || sky_values > UINT32_MAX ||
            !va_scene_offset_valid(scene->base_offset,
                (uint32_t)base_records * sizeof(struct kerr_boundary_record), scene->total_bytes) ||
            !va_scene_offset_valid(scene->refinement_lookup_offset,
                scene->pixels * sizeof(uint32_t), scene->total_bytes) ||
            scene->refinement_offset < scene->refinement_lookup_offset ||
            scene->refinement_offset > scene->extinction_offset ||
            !va_scene_offset_valid(scene->extinction_offset, (uint32_t)voxels, scene->total_bytes) ||
            !va_scene_offset_valid(scene->source_offset, (uint32_t)voxels * 3u, scene->total_bytes) ||
            !va_scene_offset_valid(scene->temperature_offset, (uint32_t)voxels, scene->total_bytes) ||
            !va_scene_offset_valid(scene->surface_temperature_offset,
                (uint32_t)surface_values * sizeof(float), scene->total_bytes) ||
            !va_scene_offset_valid(scene->surface_density_offset,
                (uint32_t)surface_values * sizeof(float), scene->total_bytes) ||
            !va_scene_offset_valid(scene->surface_transfer_offset,
                (uint32_t)transfer_values * sizeof(float), scene->total_bytes) ||
            !va_scene_offset_valid(scene->blackbody_offset,
                scene->blackbody_bins * 4u * sizeof(float), scene->total_bytes) ||
            !va_scene_offset_valid(scene->sky_offset,
                (uint32_t)sky_values * sizeof(float), scene->total_bytes))
            return AEE_EBADPARM;
        const uint64_t start = HAP_perf_get_qtimer_count();
        AEEResult result = kerr_shade_parallel(ctx, scene, input0, (uint32_t *)output);
        *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
        return result;
    }
    if (opcode == VA_HTP_OP_KERR_FRAME) {
        if (parameters_len != (int)sizeof(struct kerr_frame_parameters) ||
            input0_len < (int)sizeof(uint32_t) || output_len < 0 ||
            (uint64_t)output_len < (uint64_t)VA_KERR_FRAME_HEADER_WORDS * sizeof(uint32_t) + bytes)
            return AEE_EBADPARM;
        const struct kerr_frame_parameters *p =
            (const struct kerr_frame_parameters *)parameters;
        if (!p->width || !p->height || p->samples_per_pixel != 1 ||
            (uint64_t)p->width * p->height != lanes)
            return AEE_EBADPARM;
        const uint64_t start = HAP_perf_get_qtimer_count();
        AEEResult result = kerr_frame_parallel(ctx, lanes, p, (uint32_t *)output);
        *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
        return result;
    }
    if (opcode == VA_HTP_OP_WORMHOLE_TRACE) {
        if (parameters_len != (int) sizeof(struct wormhole_parameters) ||
            (uint32_t) input0_len < bytes * 5 || (uint32_t) output_len < bytes * 4)
            return AEE_EBADPARM;
        const uint64_t start = HAP_perf_get_qtimer_count();
        AEEResult result = trace_parallel(ctx, opcode, lanes, parameters,
            (const float *)input0, (float *)output);
        *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
        return result;
    }
    if (opcode == VA_HTP_OP_KERR_TRACE) {
        if (parameters_len != (int) sizeof(struct kerr_parameters) ||
            (uint32_t) input0_len < bytes * 9 || (uint32_t) output_len < bytes * 12)
            return AEE_EBADPARM;
        const uint64_t start = HAP_perf_get_qtimer_count();
        AEEResult result = trace_parallel(ctx, opcode, lanes, parameters,
            (const float *)input0, (float *)output);
        *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
        return result;
    }
    if (opcode == VA_HTP_OP_MATMUL) {
        if (parameters_len != (int) sizeof(struct matmul_parameters) || !input1)
            return AEE_EBADPARM;
        const struct matmul_parameters *p = (const struct matmul_parameters *) parameters;
        const uint64_t lhs_bytes = (uint64_t) p->rows * p->inner * sizeof(float);
        const uint64_t rhs_bytes = (uint64_t) p->inner * p->columns * sizeof(float);
        const uint64_t result_bytes = (uint64_t) p->rows * p->columns * sizeof(float);
        if (!p->rows || !p->inner || !p->columns ||
            input0_len < 0 || input1_len < 0 || output_len < 0 ||
            (uint64_t) input0_len < lhs_bytes || (uint64_t) input1_len < rhs_bytes ||
            (uint64_t) output_len < result_bytes)
            return AEE_EBADPARM;
        const uint64_t start = HAP_perf_get_qtimer_count();
        AEEResult result = matrix_multiply(p, (const float *) input0,
            (const float *) input1, (float *) output);
        *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
        return result;
    }
    if (opcode == VA_HTP_OP_RECIPROCAL || opcode == VA_HTP_OP_RSQRT) {
        if (parameters_len != 0 || (uint32_t) input0_len < bytes ||
            (uint32_t) output_len < bytes)
            return AEE_EBADPARM;
        const uint64_t start = HAP_perf_get_qtimer_count();
        AEEResult result = vector_unary(opcode, lanes, (const float *) input0,
            (float *) output);
        *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
        return result;
    }
    if (parameters_len != 0 || (uint32_t) input0_len < bytes || (uint32_t) output_len < bytes ||
        (opcode != VA_HTP_OP_IDENTITY && opcode != VA_HTP_OP_ADD &&
            opcode != VA_HTP_OP_MULTIPLY) ||
        (opcode != VA_HTP_OP_IDENTITY && (!input1 || (uint32_t) input1_len < bytes)))
        return AEE_EBADPARM;
    const uint64_t start = HAP_perf_get_qtimer_count();
    const float *lhs = (const float *) input0;
    const float *rhs = opcode == VA_HTP_OP_IDENTITY ? NULL : (const float *) input1;
    float *out = (float *) output;
    AEEResult result = vector_binary(opcode, lanes, lhs, rhs, out);
    *elapsed_cycles = HAP_perf_get_qtimer_count() - start;
    return result;
}
