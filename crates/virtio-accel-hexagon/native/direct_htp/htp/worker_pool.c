// SPDX-License-Identifier: MIT OR Apache-2.0
// QuRT worker-pool design adapted from ggml-hexagon's MIT-licensed worker pool.

#include "worker_pool.h"

#include <qurt.h>
#include <stdatomic.h>
#include <stdlib.h>

#define VA_MAX_WORKERS 4
#define VA_WORKER_STACK_SIZE (8 * 16384)

struct va_pool;

struct va_worker {
    struct va_pool *pool;
    unsigned int id;
};

struct va_pool {
    va_worker_callback_t callback;
    void *data;
    qurt_thread_t threads[VA_MAX_WORKERS];
    struct va_worker workers[VA_MAX_WORKERS];
    void *stacks[VA_MAX_WORKERS];
    unsigned int thread_count;
    atomic_uint sequence;
    atomic_uint next_job;
    atomic_uint pending;
    atomic_uint jobs;
    atomic_bool killed;
};

static void va_worker_main(void *opaque) {
    struct va_worker *worker = (struct va_worker *)opaque;
    struct va_pool *pool = worker->pool;
    unsigned int previous = 0;
    while (!atomic_load(&pool->killed)) {
        const unsigned int sequence = atomic_load(&pool->sequence);
        if (sequence == previous) {
            qurt_futex_wait(&pool->sequence, previous);
            continue;
        }
        previous = sequence;
        const unsigned int jobs = atomic_load(&pool->jobs);
        const unsigned int job = atomic_fetch_add(&pool->next_job, 1);
        if (job >= jobs) continue;
        pool->callback(jobs, job, pool->data);
        atomic_fetch_sub(&pool->pending, 1);
    }
}

AEEResult va_worker_pool_init(va_worker_pool_t *result, uint32_t threads) {
    if (!result || !threads || threads > VA_MAX_WORKERS) return AEE_EBADPARM;
    const size_t stacks_size = VA_WORKER_STACK_SIZE * threads;
    unsigned char *allocation = (unsigned char *)malloc(stacks_size + sizeof(struct va_pool));
    if (!allocation) return AEE_ENOMEMORY;
    struct va_pool *pool = (struct va_pool *)(allocation + stacks_size);
    *pool = (struct va_pool){0};
    pool->thread_count = threads;

    qurt_thread_attr_t attributes;
    qurt_thread_attr_init(&attributes);
    int priority = qurt_thread_get_priority(qurt_thread_get_id());
    if (priority < 1) priority = 1;
    if (priority > 254) priority = 254;
    qurt_thread_attr_set_priority(&attributes, priority);
    qurt_thread_attr_set_bus_priority(&attributes, 1);
    qurt_thread_attr_set_stack_size(&attributes, VA_WORKER_STACK_SIZE);
    qurt_thread_attr_set_name(&attributes, "va-hvx-worker");

    for (unsigned int i = 0; i < threads; ++i) {
        pool->stacks[i] = allocation + i * VA_WORKER_STACK_SIZE;
        pool->workers[i].pool = pool;
        pool->workers[i].id = i;
        qurt_thread_attr_set_stack_addr(&attributes, pool->stacks[i]);
        if (qurt_thread_create(&pool->threads[i], &attributes, va_worker_main,
                &pool->workers[i]) != 0) {
            pool->thread_count = i;
            va_worker_pool_t partial = pool;
            va_worker_pool_release(&partial);
            return AEE_EQURTTHREADCREATE;
        }
    }
    *result = pool;
    return AEE_SUCCESS;
}

void va_worker_pool_release(va_worker_pool_t *opaque) {
    if (!opaque || !*opaque) return;
    struct va_pool *pool = (struct va_pool *)*opaque;
    atomic_store(&pool->killed, 1);
    atomic_fetch_add(&pool->sequence, 1);
    qurt_futex_wake(&pool->sequence, pool->thread_count);
    for (unsigned int i = 0; i < pool->thread_count; ++i) {
        int status;
        (void)qurt_thread_join(pool->threads[i], &status);
    }
    free(pool->stacks[0]);
    *opaque = NULL;
}

AEEResult va_worker_pool_run(va_worker_pool_t opaque, va_worker_callback_t callback,
    void *data, unsigned int jobs) {
    struct va_pool *pool = (struct va_pool *)opaque;
    if (!pool || !callback || !jobs || jobs > pool->thread_count) return AEE_EBADPARM;
    pool->callback = callback;
    pool->data = data;
    atomic_store(&pool->next_job, 0);
    atomic_store(&pool->jobs, jobs);
    atomic_store(&pool->pending, jobs);
    atomic_fetch_add(&pool->sequence, 1);
    qurt_futex_wake(&pool->sequence, jobs);
    /* Keep the comparatively small FastRPC dispatch stack out of deep vector kernels. */
    while (atomic_load(&pool->pending)) {
    }
    return AEE_SUCCESS;
}
