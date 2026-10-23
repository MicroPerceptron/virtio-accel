import importlib.util
import json
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import virtio_accel as va
import virtio_accel.numpy as vnp


def projection(x, weight, bias):
    return vnp.mean(vnp.maximum(x @ weight + bias, 0), axis=-1)


def samples():
    rng = np.random.default_rng(42)
    return tuple(rng.normal(size=s).astype(np.float32) for s in [(4, 8), (8, 3), (3,)])


class FrontendTests(unittest.TestCase):
    def assertArrayClose(self, actual, expected):
        self.assertEqual(actual.dtype, np.dtype("float32"))
        self.assertEqual(actual.shape, np.shape(expected))
        np.testing.assert_allclose(actual.to_numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_projection_eager_and_traced(self):
        x, w, b = samples()
        expected = np.maximum(x @ w + b, 0).mean(axis=-1)
        self.assertArrayClose(projection(*map(vnp.asarray, (x, w, b))), expected)
        compiled = va.jit(projection, backend="reference")
        self.assertArrayClose(compiled(x, w, b), expected)
        # Values are runtime inputs, not captured from the first call.
        self.assertArrayClose(compiled(x * 2, w, b), np.maximum(x * 2 @ w + b, 0).mean(axis=-1))

    def test_shared_rust_projection_fixture(self):
        fixture = json.loads((Path(__file__).parent / "data" / "projection.json").read_text())
        self.assertEqual(va.jit(projection).lower(*samples()).to_dict(), fixture)

    def test_cache_signature_binding_and_eviction(self):
        calls = []
        @va.jit(cache_size=1)
        def f(x, *, y):
            calls.append(1)
            return x + y
        a = np.ones((2,), np.float32)
        first = f.lower(a, y=a)
        self.assertIs(first, f.lower(y=a * 2, x=a * 3))
        self.assertEqual(len(calls), 1)
        f.lower(np.ones(3, np.float32), y=np.ones(3, np.float32))
        self.assertIsNot(first, f.lower(a, y=a))
        f.clear_cache()
        f.lower(a, y=a)
        self.assertEqual(len(calls), 4)

    def test_reference_explicit_and_hardware_not_claimed(self):
        f = va.jit(lambda x: x + 1)
        with self.assertRaisesRegex(RuntimeError, "device execution"):
            f(np.ones(2, np.float32))
        with self.assertRaises(NotImplementedError):
            va.jit(lambda x: x, backend="coreml")
        self.assertIn("host", repr(vnp.ones(2)))

    def test_scalar_and_broadcasting(self):
        for left, right in [((2, 3), (3,)), ((2, 1, 3), (4, 1)), ((), (2, 3)), ((), ())]:
            with self.subTest(left=left, right=right):
                a = np.full(left, 2, dtype=np.float32)
                b = np.full(right, 3, dtype=np.float32)
                f = va.jit(lambda a, b: (a + b, a - b, a * b), backend="reference")
                for actual, expected in zip(f(a, b), (a + b, a - b, a * b)):
                    self.assertArrayClose(actual, expected)
        with self.assertRaises(ValueError):
            vnp.add(vnp.ones((2, 3)), vnp.ones((4,)))

    def test_reductions(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        for axis in [None, 0, -1, (0, 2), ()]:
            for keep in [False, True]:
                with self.subTest(axis=axis, keep=keep):
                    f = va.jit(lambda a: (vnp.sum(a, axis=axis, keepdims=keep),
                                         vnp.mean(a, axis=axis, keepdims=keep)), backend="reference")
                    total, mean = f(x)
                    self.assertArrayClose(total, x.sum(axis=axis, keepdims=keep))
                    self.assertArrayClose(mean, x.mean(axis=axis, keepdims=keep))
        scalar = np.array(2, np.float32)
        self.assertArrayClose(vnp.sum(scalar), scalar)

    def test_bad_reductions(self):
        x = vnp.ones((2, 3))
        for axis in [2, -3, (0, 0), (0, -2)]:
            with self.assertRaises(ValueError):
                vnp.sum(x, axis=axis)
        with self.assertRaises(TypeError):
            vnp.sum(x, dtype=np.float64)
        with self.assertRaises(NotImplementedError):
            vnp.sum(x, out=x)
        with self.assertRaises(TypeError):
            vnp.sum(x, keepdims=x)

    def test_transpose_reshape_and_noncontiguous_host_input(self):
        x = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::-2]
        f = va.jit(lambda a: vnp.reshape(a.T, (-1, 2)), backend="reference")
        self.assertArrayClose(f(x), x.T.reshape(-1, 2))
        self.assertArrayClose(vnp.transpose(vnp.asarray(x), (-1, -2)), x.T)
        for shape in [(-1, -1), (0,), (5,), (2.5,)]:
            with self.assertRaises((ValueError, TypeError)):
                vnp.reshape(x, shape)
        for axes in [(0, 0), (2, 0), (0,), (-3, 0)]:
            with self.assertRaises(ValueError):
                vnp.transpose(x, axes)

    def test_unary_and_extrema(self):
        x = np.array([0.25, 1, 2, np.nan], np.float32)
        for name in ["exp", "log", "tanh", "sin", "cos", "abs", "negative"]:
            with self.subTest(name=name):
                fn = getattr(vnp, name)
                f = va.jit(lambda a: fn(a), backend="reference")
                self.assertArrayClose(f(x), getattr(np, name)(x))
        for name in ["maximum", "minimum"]:
            f = va.jit(lambda a: getattr(vnp, name)(a, 1), backend="reference")
            self.assertArrayClose(f(x), getattr(np, name)(x, 1))

    def test_rejects_dtypes_without_implicit_narrowing(self):
        for dtype in [np.float64, np.int32, np.bool_, np.complex64]:
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    vnp.asarray(np.ones(2, dtype=dtype))
                with self.assertRaises(TypeError):
                    vnp.zeros(2, dtype=dtype)
        self.assertArrayClose(vnp.asarray(np.ones(2), dtype=vnp.float32), np.ones(2, np.float32))
        self.assertEqual(vnp.asarray([1, 2]).dtype, np.dtype("float32"))

    def test_immutability_and_conversion(self):
        host = np.ones(2, np.float32)
        value = vnp.asarray(host)
        host[:] = 3
        self.assertArrayClose(value, np.ones(2, np.float32))
        exported = value.to_numpy()
        exported[:] = 7
        self.assertArrayClose(value, np.ones(2, np.float32))
        with self.assertRaises(TypeError):
            np.asarray(value)
        with self.assertRaises(TypeError):
            value += 1
        with self.assertRaises(TypeError):
            value[0] = 1

    def test_control_flow_and_unsupported_numpy_fail(self):
        def branch(x):
            return x if x else -x
        with self.assertRaisesRegex(TypeError, "control flow"):
            va.jit(branch).lower(np.ones(2, np.float32))
        with self.assertRaises(TypeError):
            va.jit(lambda x: np.sin(x)).lower(np.ones(2, np.float32))
        with self.assertRaises(NotImplementedError):
            va.jit(lambda x: x == x).lower(np.ones(2, np.float32))
        with self.assertRaises(NotImplementedError):
            vnp.matmul(vnp.ones(2), vnp.ones(2))
        with self.assertRaises(ValueError):
            vnp.matmul(vnp.ones((2, 3)), vnp.ones((4, 2)))

    def test_captured_constants_snapshot_and_clear_cache(self):
        bias = np.ones(2, np.float32)
        f = va.jit(lambda x: x + vnp.asarray(bias), backend="reference")
        x = np.zeros(2, np.float32)
        f(x)
        bias[:] = 5
        self.assertArrayClose(f(x), np.ones(2, np.float32))
        f.clear_cache()
        self.assertArrayClose(f(x), bias)

    def test_constants_and_passthrough_outputs(self):
        f = va.jit(lambda x: (x, x, vnp.zeros((2,))), backend="reference")
        x = np.ones(2, np.float32)
        a, b, c = f(x)
        self.assertArrayClose(a, x)
        self.assertArrayClose(b, x)
        self.assertArrayClose(c, np.zeros(2, np.float32))
        self.assertEqual(len(set(f.lower(x).to_dict()["outputs"])), 3)

    def test_reference_requires_matching_specialization(self):
        lowered = va.jit(lambda x: x + 1).lower(np.ones(2, np.float32))
        with self.assertRaises(ValueError):
            lowered.reference(np.ones(3, np.float32))
        document = lowered.to_dict()
        document["nodes"].clear()
        self.assertTrue(lowered.to_dict()["nodes"])

    def test_trace_limits_and_invalid_shapes(self):
        for shape in [(0,), (-1,), (1,) * 7, (2**30,)]:
            with self.assertRaises(ValueError):
                vnp.ones(shape)
        def long_trace(x):
            for _ in range(4100):
                x = -x
            return x
        with self.assertRaisesRegex(ValueError, "node limit"):
            va.jit(long_trace).lower(np.ones(1, np.float32))

    def test_size_limits_precede_host_copy_and_eager_output_allocation(self):
        huge_view = np.broadcast_to(np.float32(1), (2**28,))
        with self.assertRaisesRegex(ValueError, "256 MiB"):
            vnp.asarray(huge_view)
        # Inputs fit easily; the outer product must be refused before allocation.
        with self.assertRaisesRegex(ValueError, "256 MiB"):
            vnp.matmul(vnp.ones((16384, 1)), vnp.ones((1, 16384)))

    def test_concurrent_cache_use(self):
        f = va.jit(lambda x: x * 2, backend="reference")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(f, [np.full(2, i, np.float32) for i in range(12)]))
        for i, value in enumerate(results):
            self.assertArrayClose(value, np.full(2, i * 2, np.float32))


@unittest.skipUnless(importlib.util.find_spec("virtio_accel._native"), "native extension not built")
class NativeTests(unittest.TestCase):
    def test_projection_export(self):
        lowered = va.jit(projection).lower(*samples())
        first = lowered.to_tosa()
        self.assertIsInstance(first, bytes)
        self.assertEqual(first[4:8], b"TOSA")
        self.assertEqual(first, lowered.to_tosa())

    def test_all_namespace_paths_export(self):
        x = np.ones((2, 3), np.float32)
        functions = [lambda a: a, lambda a: (a, a), lambda a: vnp.zeros((2,)),
                     lambda a: vnp.sum(a), lambda a: vnp.mean(a, axis=0),
                     lambda a: a.T, lambda a: vnp.reshape(a, (6,))]
        for fn in functions:
            self.assertTrue(va.jit(fn).lower(x).to_tosa())
        for name in ["exp", "log", "tanh", "sin", "cos", "abs", "negative"]:
            fn = lambda a: getattr(vnp, name)(a)
            self.assertTrue(va.jit(fn).lower(x).to_tosa())

    def test_native_rejects_invalid_graph(self):
        from virtio_accel._native import build_tosa
        doc = va.jit(lambda x: x + 1).lower(np.ones(2, np.float32)).to_dict()
        doc["nodes"][-1]["shape"] = [3]
        with self.assertRaisesRegex(ValueError, "TOSA validation"):
            build_tosa(json.dumps(doc))


if __name__ == "__main__":
    unittest.main()
