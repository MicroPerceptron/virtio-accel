# Workload examples

Each script implements a reusable numerical function, generates deterministic demo
inputs, evaluates the traced graph on the explicit CPU reference, and checks an
independent NumPy result. `--export model.tosa` additionally writes a graph validated
by the Rust TOSA implementation. Use `--device npu` or `--device gpu` to execute through an available native provider.

| Example | Workload | Output |
| --- | --- | --- |
| `mlp_classifier.py` | Two dense layers, bias and ReLU; replace synthetic weights with a trained checkpoint | Class logits; host argmax reports predictions |
| `nearest_centroid.py` | Pairwise squared distances without a 3D differences temporary | FP32 distance matrix; host argmin assigns clusters |
| `fir_filter.py` | FIR convolution over a batch of sampled-signal windows | Filtered samples, checked against float64 convolution |

After installing the package from the repository, run from the repository root:

```sh
python python/examples/mlp_classifier.py --dtype float16
python python/examples/nearest_centroid.py --dtype float16
python python/examples/fir_filter.py --dtype float32 --export filter.tosa
```

Every example accepts `--dtype float16` and `--dtype float32`. FP16 reduces input and
constant storage; it does not imply a speedup in the NumPy reference. The classifier
uses FP32 dot-product accumulation with FP16 storage rounding at operation boundaries.
Distances explicitly promote FP16 inputs to FP32 to reduce cancellation. FIR uses host
window construction and explicit contiguous copies at the frontend boundary.

Without building the native extension, reference-only runs are also possible:

```sh
PYTHONPATH=python python -m examples.mlp_classifier --dtype float16
PYTHONPATH=python python -m examples.nearest_centroid --dtype float32
PYTHONPATH=python python -m examples.fir_filter --dtype float16
```

`examples` is a source-tree package and is not installed as a generic top-level package
by the wheel. Use the script form above with an installed wheel. Source-only runs cannot
export TOSA unless the native extension has been built into the source package (for
example with `maturin develop --manifest-path python/Cargo.toml` in a virtual environment).

Native examples (installed wheel and available device required):

```sh
python python/examples/mlp_classifier.py --dtype float16 --device npu
python python/examples/nearest_centroid.py --dtype float32 --device npu
python python/examples/fir_filter.py --dtype float16 --device gpu
```

Each native run prints its selected provider and CPU-participation metadata. Core ML
permits CPU participation. Its current tier has no CAST, so FP16 nearest-centroid
fails admission; use FP32 inputs explicitly or a provider that admits CAST. No example
automatically changes dtype or falls back after an execution failure.
