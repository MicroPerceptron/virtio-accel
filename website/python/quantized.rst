Quantized projections
=====================

These helpers compose with ``jit`` and the existing execution/cache API.
INT4, INT8 and palette weights are captured constants; changing them requires
retracing and recompilation. NVFP4 byte tensors can change on every invocation.
Backend admission remains explicit. FP16 MATMUL follows TOSA's FP16 accuracy
contract; the NumPy reference uses FP32 accumulation and is not a bitwise oracle
for arbitrary native FP16 dot products.

.. automodule:: virtio_accel.quantized

A packed INT4 projection
------------------------

.. doctest::

   >>> import numpy as np
   >>> import virtio_accel as va
   >>> from virtio_accel import quantized as q
   >>> scales = np.ones((1, 1), dtype=np.float16)
   >>> f = va.jit(lambda x: q.int4_matmul(x, bytes([0x21, 0x43]), scales, group_size=4), backend="reference")
   >>> f(np.ones((1, 4), dtype=np.float16)).to_numpy().tolist()
   [[10.0]]
