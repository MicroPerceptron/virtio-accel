// SPDX-License-Identifier: MIT
/* Experimental single-session Linux adapter for protocol 1.0. */
#include <linux/compat.h>
#include <linux/completion.h>
#include <linux/fs.h>
#include <linux/idr.h>
#include <linux/kref.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/scatterlist.h>
#include <linux/uaccess.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/vmalloc.h>
#include "virtio_accel.h"

/* Development MMIO device only. This is NOT an allocated Virtio device ID. */
#define VA_EXPERIMENTAL_ID 56
#define VA_MAX_BYTES (1024U * 1024U)

struct va_config {
	__le16 major, minor, queues, descriptors;
	__le32 request_bytes, response_bytes;
};

static_assert(sizeof(struct va_config) == 16);
static_assert(sizeof(struct va_info) == 32);
static_assert(sizeof(struct va_exchange) == 32);

struct va_device {
	struct virtio_device *vdev;
	struct virtqueue *vq;
	struct miscdevice misc;
	struct kref refs;
	struct mutex lock; /* serializes all queue operations and session state */
	struct completion done; /* callback never manipulates the queue */
	atomic_t removed;
	bool opened;
	int index;
	struct va_info info;
	void *request, *response;
	struct scatterlist *out_sg, *in_sg;
};

static DEFINE_IDA(va_indices);

static void va_done(struct virtqueue *vq)
{
	struct va_device *d = vq->vdev->priv;

	complete(&d->done);
}

/* Reset synchronizes with the device before any published storage is freed.
 * del_vqs synchronizes callbacks before the device reference can be dropped.
 */
static void va_stop(struct va_device *d)
{
	if (!d->vq)
		return;
	virtio_reset_device(d->vdev);
	while (virtqueue_detach_unused_buf(d->vq))
		;
	d->vdev->config->del_vqs(d->vdev);
	d->vq = NULL;
}

static int va_queue_start(struct va_device *d, bool reinit)
{
	struct virtio_device *v = d->vdev;
	int ret;

	if (reinit) {
		v->config->set_status(v, VIRTIO_CONFIG_S_ACKNOWLEDGE);
		virtio_add_status(v, VIRTIO_CONFIG_S_DRIVER);
		ret = v->config->finalize_features(v);
		if (ret)
			goto fail;
		virtio_add_status(v, VIRTIO_CONFIG_S_FEATURES_OK);
		if (!(v->config->get_status(v) & VIRTIO_CONFIG_S_FEATURES_OK)) {
			ret = -ENODEV;
			goto fail;
		}
	}
	d->vq = virtio_find_single_vq(v, va_done, "command");
	if (IS_ERR(d->vq)) {
		ret = PTR_ERR(d->vq);
		d->vq = NULL;
		goto fail;
	}
	virtio_device_ready(v);
	return 0;
fail:
	virtio_reset_device(v);
	return ret;
}

static void va_free(struct kref *ref)
{
	struct va_device *d = container_of(ref, struct va_device, refs);

	ida_free(&va_indices, d->index);
	kfree(d->misc.name);
	kfree(d);
}

static void va_free_buffers(struct va_device *d)
{
	vfree(d->request);
	vfree(d->response);
	kfree(d->out_sg);
	kfree(d->in_sg);
	d->request = d->response = NULL;
	d->out_sg = d->in_sg = NULL;
}

static int va_open(struct inode *inode, struct file *file)
{
	struct va_device *d = container_of(file->private_data, struct va_device, misc);
	int ret = 0;

	mutex_lock(&d->lock);
	if (atomic_read(&d->removed)) {
		ret = -ENODEV;
		goto out;
	}
	if (d->opened) {
		ret = -EBUSY;
		goto out;
	}
	d->request = vzalloc(d->info.max_request_bytes);
	d->response = vzalloc(d->info.max_response_bytes);
	d->out_sg = kcalloc(DIV_ROUND_UP(d->info.max_request_bytes, PAGE_SIZE),
			    sizeof(*d->out_sg), GFP_KERNEL);
	d->in_sg = kcalloc(DIV_ROUND_UP(d->info.max_response_bytes, PAGE_SIZE),
			   sizeof(*d->in_sg), GFP_KERNEL);
	if (!d->request || !d->response || !d->out_sg || !d->in_sg) {
		ret = -ENOMEM;
		goto free;
	}
	if (!d->vq) {
		ret = va_queue_start(d, true);
		if (ret)
			goto free;
	}
	d->opened = true;
	kref_get(&d->refs);
	file->private_data = d;
	nonseekable_open(inode, file);
	goto out;
free:
	va_free_buffers(d);
out:
	mutex_unlock(&d->lock);
	return ret;
}

static int va_release(struct inode *inode, struct file *file)
{
	struct va_device *d = file->private_data;

	mutex_lock(&d->lock);
	va_stop(d);
	va_free_buffers(d);
	d->opened = false;
	mutex_unlock(&d->lock);
	kref_put(&d->refs, va_free);
	return 0;
}

static void va_sg(struct scatterlist *sg, void *buffer, unsigned int bytes)
{
	unsigned int i, n = DIV_ROUND_UP(bytes, PAGE_SIZE);

	sg_init_table(sg, n);
	for (i = 0; i < n; i++) {
		unsigned int chunk = min_t(unsigned int, bytes, PAGE_SIZE);

		sg_set_page(&sg[i], vmalloc_to_page(buffer), chunk, 0);
		buffer += PAGE_SIZE;
		bytes -= chunk;
	}
}

static long va_exchange(struct va_device *d, void __user *arg)
{
	struct va_exchange x;
	struct scatterlist *sgs[] = { d->out_sg, d->in_sg };
	unsigned int length = 0;
	unsigned long deadline, remaining;
	void *token;
	long ret;

	if (copy_from_user(&x, arg, sizeof(x)))
		return -EFAULT;
	if (x.flags || memchr_inv(x.reserved, 0, sizeof(x.reserved)) ||
	    x.request_bytes < 16 || x.request_bytes > d->info.max_request_bytes ||
	    x.response_bytes < 16 || x.response_bytes > d->info.max_response_bytes ||
	    !x.timeout_ms || x.timeout_ms > 60000)
		return -EINVAL;
	if (copy_from_user(d->request, arg + sizeof(x), x.request_bytes))
		return -EFAULT;
	/* Never expose a previous response even if the device lies about used length. */
	memset(d->response, 0, x.response_bytes);
	va_sg(d->out_sg, d->request, x.request_bytes);
	va_sg(d->in_sg, d->response, x.response_bytes);
	reinit_completion(&d->done);
	if (atomic_read(&d->removed))
		return -ENODEV;
	ret = virtqueue_add_sgs(d->vq, sgs, 1, 1, d, GFP_KERNEL);
	if (ret)
		return ret;
	if (!virtqueue_kick(d->vq)) {
		ret = -EIO;
		goto poison;
	}
	/* Uninterruptible bounded wait: a signal cannot cause userspace to retry an
	 * already accepted mutation through the libc/CPython EINTR retry machinery.
	 */
	deadline = jiffies + msecs_to_jiffies(x.timeout_ms);
	do {
		unsigned long now = jiffies;

		remaining = time_before(now, deadline) ? deadline - now : 0;
		wait_for_completion_timeout(&d->done, remaining);
		if (atomic_read(&d->removed)) {
			ret = -ENODEV;
			goto poison;
		}
		token = virtqueue_get_buf(d->vq, &length);
		if (!token && (time_after_eq(jiffies, deadline) || virtqueue_is_broken(d->vq))) {
			ret = virtqueue_is_broken(d->vq) ? -EIO : -ETIMEDOUT;
			goto poison;
		}
		/* A delayed/spurious IRQ is only a wake-up hint. It does not prove
		 * that this request has a used entry, especially on SMP guests.
		 */
	} while (!token);
	if (token != d || length < 16 || length > x.response_bytes) {
		ret = -EPROTO;
		goto poison;
	}
	x.response_bytes = length;
	if (copy_to_user(arg + sizeof(x), d->response, length) ||
	    copy_to_user(arg, &x, sizeof(x))) {
		ret = -EFAULT;
		goto poison;
	}
	return 0;
poison:
	va_stop(d);
	return ret;
}

static long va_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct va_device *d = file->private_data;
	void __user *ptr = (void __user *)arg;
	long ret;

	mutex_lock(&d->lock);
	if (atomic_read(&d->removed) || !d->vq) {
		ret = -ENODEV;
		goto out;
	}
	switch (cmd) {
	case VA_INFO:
		ret = copy_to_user(ptr, &d->info, sizeof(d->info)) ? -EFAULT : 0;
		break;
	case VA_EXCHANGE:
		ret = va_exchange(d, ptr);
		break;
	default:
		ret = -ENOTTY;
	}
out:
	mutex_unlock(&d->lock);
	return ret;
}

static const struct file_operations va_fops = {
	.owner = THIS_MODULE,
	.open = va_open,
	.release = va_release,
	.unlocked_ioctl = va_ioctl,
	.compat_ioctl = compat_ptr_ioctl,
};

static int va_probe(struct virtio_device *v)
{
	struct va_config config;
	struct va_device *d;
	unsigned int capacity;
	int ret;

	if (!virtio_has_feature(v, VIRTIO_F_VERSION_1))
		return -ENODEV;
	virtio_cread_bytes(v, 0, &config, sizeof(config));
	if (le16_to_cpu(config.major) != 1 || le16_to_cpu(config.minor) != 0 ||
	    le16_to_cpu(config.queues) != 1 ||
	    le16_to_cpu(config.descriptors) < 2 ||
	    le16_to_cpu(config.descriptors) > 256 ||
	    le32_to_cpu(config.request_bytes) < 97 ||
	    le32_to_cpu(config.request_bytes) > 16777216 ||
	    le32_to_cpu(config.response_bytes) < 92 ||
	    le32_to_cpu(config.response_bytes) > 16777216)
		return -EINVAL;
	d = kzalloc(sizeof(*d), GFP_KERNEL);
	if (!d)
		return -ENOMEM;
	d->index = ida_alloc(&va_indices, GFP_KERNEL);
	if (d->index < 0) {
		ret = d->index;
		kfree(d);
		return ret;
	}
	d->vdev = v;
	v->priv = d;
	kref_init(&d->refs);
	mutex_init(&d->lock);
	init_completion(&d->done);
	atomic_set(&d->removed, 0);
	ret = va_queue_start(d, false);
	if (ret)
		goto free;
	/* Both full arenas must fit one direct chain on this queue. */
	capacity = min_t(unsigned int, le16_to_cpu(config.descriptors),
			 virtqueue_get_vring_size(d->vq)) / 2 * PAGE_SIZE;
	capacity = min(capacity, VA_MAX_BYTES);
	if (!capacity) {
		ret = -EINVAL;
		goto stop;
	}
	d->info.abi = 1;
	d->info.protocol_major = 1;
	d->info.max_request_bytes = min(capacity, le32_to_cpu(config.request_bytes));
	d->info.max_response_bytes = min(capacity, le32_to_cpu(config.response_bytes));
	d->misc.minor = MISC_DYNAMIC_MINOR;
	d->misc.name = kasprintf(GFP_KERNEL, "virtio-accel%d", d->index);
	d->misc.fops = &va_fops;
	d->misc.parent = &v->dev;
	d->misc.mode = 0600;
	if (!d->misc.name) {
		ret = -ENOMEM;
		goto stop;
	}
	ret = misc_register(&d->misc);
	if (!ret)
		return 0;
stop:
	va_stop(d);
free:
	v->priv = NULL;
	kref_put(&d->refs, va_free);
	return ret;
}

static void va_remove(struct virtio_device *v)
{
	struct va_device *d = v->priv;

	/* Wake an exchange before waiting for its mutex. misc_deregister also
	 * synchronizes an in-progress open before the registration ref is dropped.
	 */
	atomic_set(&d->removed, 1);
	complete(&d->done);
	misc_deregister(&d->misc);
	mutex_lock(&d->lock);
	va_stop(d);
	mutex_unlock(&d->lock);
	v->priv = NULL;
	kref_put(&d->refs, va_free);
}

#ifdef CONFIG_PM_SLEEP
static int va_freeze(struct virtio_device *v)
{
	/* No migration/suspend ownership contract in this experimental adapter. */
	return -EBUSY;
}
#endif

static const struct virtio_device_id va_ids[] = {
	{ VA_EXPERIMENTAL_ID, VIRTIO_DEV_ANY_ID }, { 0 },
};

static struct virtio_driver va_driver = {
	.driver.name = "virtio_accel",
	.driver.owner = THIS_MODULE,
	.id_table = va_ids,
	.probe = va_probe,
	.remove = va_remove,
#ifdef CONFIG_PM_SLEEP
	.freeze = va_freeze,
#endif
};
module_virtio_driver(va_driver);
MODULE_DEVICE_TABLE(virtio, va_ids);
MODULE_DESCRIPTION("Experimental virtio-accel protocol 1.0 guest adapter");
MODULE_LICENSE("Dual MIT/GPL");
