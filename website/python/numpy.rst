NumPy-style operations
======================

Shapes are static, have rank at most six, and use positive dimensions.
Arrays are immutable. Unsupported operations are rejected rather than forwarded
to NumPy. Device admission also depends on the selected backend and dtype.

.. automodule:: virtio_accel.numpy
   :exclude-members: float16, float32, bfloat16

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
