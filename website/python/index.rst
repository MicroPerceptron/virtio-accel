Python API reference
====================

This reference is generated from the public Python modules in the same checkout
as the website. Signatures and docstrings update with the code. It describes
version |release| on ``main``; unreleased changes may precede the PyPI package.
The Python SDK version is independent of the Rust crate and protocol versions.

Start with the `Python guide <../index.html>`_ for installation, device selection,
backend prerequisites and tested examples. Calls to the NumPy-style namespace
are CPU reference operations unless traced and explicitly compiled for a device.

.. toctree::
   :maxdepth: 2

   runtime
   numpy

A reference calculation
-----------------------

This example runs in documentation CI without accelerator hardware:

.. doctest::

   >>> import numpy as np
   >>> import virtio_accel as va
   >>> import virtio_accel.numpy as vnp
   >>> @va.jit(backend="reference")
   ... def relu(x):
   ...     return vnp.maximum(x, 0)
   >>> relu(np.array([-2, 0, 3], dtype=np.float32)).to_numpy().tolist()
   [0.0, 0.0, 3.0]
