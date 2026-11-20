/* SPDX-License-Identifier: MIT */
#ifndef _UAPI_VIRTIO_ACCEL_H
#define _UAPI_VIRTIO_ACCEL_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* Experimental Linux ABI, independent of the frozen device wire protocol.
 * All fields are native endian. No embedded pointers; compat layout is identical.
 * The argument allocation must contain this header followed by
 * max(request_bytes, response_bytes) bytes. Request is copied before submission;
 * response overwrites that tail. response_bytes becomes the actual used length.
 * flags/reserved must be zero. A failed submitted exchange invalidates the fd.
 */
struct va_exchange {
	__u32 request_bytes;
	__u32 response_bytes;
	__u32 timeout_ms;
	__u32 flags;
	__u32 reserved[4];
};

struct va_info {
	__u32 abi;
	__u32 max_request_bytes;
	__u32 max_response_bytes;
	__u32 protocol_major;
	__u32 protocol_minor;
	__u32 reserved[3];
};

/* Locally experimental ioctl numbers, not an upstream allocation. */
#define VA_EXCHANGE _IOWR(0xb7, 0, struct va_exchange)
#define VA_INFO _IOR(0xb7, 1, struct va_info)

#endif
