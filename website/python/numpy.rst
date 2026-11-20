NumPy-style operations
======================

Shapes are static, have rank at most six, and use positive dimensions.
Arrays are immutable. Unsupported operations are rejected rather than forwarded
to NumPy. Device admission also depends on the selected backend and dtype.

.. automodule:: virtio_accel.numpy
   :exclude-members: float16, float32, bfloat16, float8_e4m3fn, float8_e5m2, int8, int32

Dtypes
------

.. autodata:: virtio_accel.numpy.float16
   :no-value:

   NumPy binary16 dtype; a supported frontend storage and arithmetic type.

.. autodata:: virtio_accel.numpy.bfloat16
   :no-value:

   ``ml_dtypes.bfloat16``; BF16 MATMUL produces FP32 accumulator output.

.. autodata:: virtio_accel.numpy.float32
   :no-value:

   NumPy binary32 dtype; backend support must still be checked.

.. autodata:: virtio_accel.numpy.float8_e4m3fn
   :no-value:

   ``ml_dtypes.float8_e4m3fn``; one-byte E4M3FN storage, casts and movement.
   Widen explicitly before arithmetic.

.. autodata:: virtio_accel.numpy.float8_e5m2
   :no-value:

   ``ml_dtypes.float8_e5m2``; one-byte E5M2 storage, casts and movement.

.. autodata:: virtio_accel.numpy.int8
   :no-value:

   Signed INT8 storage and MATMUL operands; also a raw NVFP4 byte carrier.

.. autodata:: virtio_accel.numpy.int32
   :no-value:

   Signed INT32 storage and integer MATMUL output. No implicit float promotion.
