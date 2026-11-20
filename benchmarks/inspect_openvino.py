#!/usr/bin/env python3
"""Inspect production-lowered INT4 IR exported by the OpenVINO unit test.

Requires OpenVINO and NumPy. This is a compiler/correctness diagnostic, not a
timing harness. The exact IR, execution graph and JSON report are retained.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import openvino as ov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", required=True, choices=["CPU", "GPU", "NPU"])
    args = parser.parse_args()
    root = args.directory
    core = ov.Core()
    stem = "int4" if args.device == "GPU" else "int4-expanded"
    model = core.read_model(root / f"{stem}.xml")
    properties = {"EXECUTION_MODE_HINT": "ACCURACY", "PERF_COUNT": True}
    if args.device in ("CPU", "GPU"):
        properties["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = 0
    compiled = core.compile_model(model, args.device, properties)
    request = compiled.create_infer_request()
    # Same all-nibble-pattern fixture as export_compressed_projection_for_native_inspection.
    packed = np.arange(128 * 256 // 2, dtype=np.int64).astype(np.uint8)
    q = np.empty(128 * 256, dtype=np.int8)
    q[::2] = ((packed & 15).astype(np.int8) << 4) >> 4
    q[1::2] = packed.astype(np.int8) >> 4
    weights = q.reshape(128, 256).astype(np.float64) * 0.125
    rng = np.random.default_rng(218)
    for _ in range(12):
        x = rng.uniform(-1, 1, (1, 1, 256)).astype(np.float16)
        expected = (x.astype(np.float64) @ weights.T).astype(np.float16)
        actual = request.infer([x])[0]
        np.testing.assert_array_equal(actual, expected)
    path = root / f"execution-{args.device}.xml"
    ov.serialize(compiled.get_runtime_model(), path)
    layers = [
        {"type": layer.attrib["type"], **layer.find("data").attrib}
        for layer in ET.parse(path).findall("./layers/layer")
        if layer.find("data") is not None
    ]
    if args.device == "GPU":
        assert not any(l["type"] == "DynamicQuantize" for l in layers), layers
        assert any(l.get("weights_precision") == "i4" for l in layers), layers
        assert any(l.get("weights_precision") == "i4" and l.get("runtimePrecision") == "f16" for l in layers), layers
    report = {
        "openvino": ov.__version__, "device": args.device,
        "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
        "properties": properties, "exact_random_readbacks": 12,
        "lowered_weights_blob_bytes": (root / f"{stem}.bin").stat().st_size,
        "layers": layers,
    }
    (root / f"inspection-{args.device}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
