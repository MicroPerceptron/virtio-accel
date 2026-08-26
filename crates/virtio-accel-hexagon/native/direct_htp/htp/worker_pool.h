// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef VA_HTP_WORKER_POOL_H
#define VA_HTP_WORKER_POOL_H

#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stdint.h>

typedef void (*va_worker_callback_t)(unsigned int workers, unsigned int worker, void *data);
typedef void *va_worker_pool_t;

AEEResult va_worker_pool_init(va_worker_pool_t *pool, uint32_t threads);
void va_worker_pool_release(va_worker_pool_t *pool);
AEEResult va_worker_pool_run(va_worker_pool_t pool, va_worker_callback_t callback,
    void *data, unsigned int jobs);

#endif
