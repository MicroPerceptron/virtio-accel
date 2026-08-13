#!/usr/bin/env python3
"""Compile the public C wire header against the frozen layout manifest."""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "include"
MANIFEST = ROOT / "conformance" / "v1.0" / "layout.json"

STRUCT_NAMES = {
    "WireConfig": "virtio_accel_config",
    "RequestHeader": "virtio_accel_request_header",
    "ResponseHeader": "virtio_accel_response_header",
    "WireDeviceInfo": "virtio_accel_device_info",
    "CreateContextRequest": "virtio_accel_create_context_request",
    "ObjectPayload": "virtio_accel_object_payload",
    "AllocateBufferRequest": "virtio_accel_allocate_buffer_request",
    "TransferBufferRequest": "virtio_accel_transfer_buffer_request",
    "LoadProgramRequest": "virtio_accel_load_program_request",
    "CreateQueueRequest": "virtio_accel_create_queue_request",
    "SubmitRequest": "virtio_accel_submit_request",
    "WireBinding": "virtio_accel_binding",
    "SubmitResponse": "virtio_accel_submit_response",
    "WireEventState": "virtio_accel_event_state",
}

FIELD_NAMES = {
    ("WireDeviceInfo", "class"): "accelerator_class",
}

SCALAR_NAMESPACES = {
    "accelerator_classes": "VIRTIO_ACCEL_CLASS_",
    "memory_domains": "VIRTIO_ACCEL_MEMORY_",
    "binding_access": "VIRTIO_ACCEL_ACCESS_",
    "opcodes": "VIRTIO_ACCEL_OP_",
    "statuses": "VIRTIO_ACCEL_STATUS_",
    "event_states": "VIRTIO_ACCEL_EVENT_",
}


class Failure(Exception):
    pass


def integer(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise Failure(f"expected an integer or integer string, found {value!r}")


def keys(mapping: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(mapping, dict):
        raise Failure(f"{name} must be an object")
    actual = set(mapping)
    if actual != expected:
        raise Failure(
            f"{name} keys changed; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return mapping


def assertion(expression: str, value: object, name: str) -> str:
    expected = integer(value)
    return (
        f'VA_STATIC_ASSERT((uint64_t)({expression}) == UINT64_C({expected}), "{name}");'
    )


def manifest_assertions(manifest: dict[str, object]) -> list[str]:
    lines: list[str] = []

    protocol = keys(manifest.get("protocol"), {"major", "minor"}, "protocol")
    lines.append(assertion("VIRTIO_ACCEL_PROTOCOL_MAJOR", protocol["major"], "protocol.major"))
    lines.append(assertion("VIRTIO_ACCEL_PROTOCOL_MINOR", protocol["minor"], "protocol.minor"))

    queue_names = {
        "command_index": "VIRTIO_ACCEL_COMMAND_QUEUE",
        "baseline_count": "VIRTIO_ACCEL_BASELINE_COMMAND_QUEUES",
        "hard_max_chain_descriptors": "VIRTIO_ACCEL_HARD_MAX_CHAIN_DESCRIPTORS",
        "min_max_request_bytes": "VIRTIO_ACCEL_MIN_MAX_REQUEST_BYTES",
        "min_max_response_bytes": "VIRTIO_ACCEL_MIN_MAX_RESPONSE_BYTES",
        "hard_max_request_bytes": "VIRTIO_ACCEL_HARD_MAX_REQUEST_BYTES",
        "hard_max_response_bytes": "VIRTIO_ACCEL_HARD_MAX_RESPONSE_BYTES",
        "hard_max_bindings": "VIRTIO_ACCEL_HARD_MAX_BINDINGS",
    }
    queue = keys(manifest.get("queue"), set(queue_names), "queue")
    for name, macro in queue_names.items():
        lines.append(assertion(macro, queue[name], f"queue.{name}"))

    features = keys(manifest.get("features"), {"baseline", "reserved"}, "features")
    lines.append(
        assertion("VIRTIO_ACCEL_BASELINE_FEATURES", features["baseline"], "features.baseline")
    )
    reserved_features = keys(
        features["reserved"],
        {
            "MULTI_QUEUE",
            "EVENT_QUEUE",
            "EXTERNAL_MEMORY",
            "TIMELINE_FENCES",
            "SECURE_CONTEXTS",
        },
        "features.reserved",
    )
    reserved_mask = 0
    for name, value in reserved_features.items():
        reserved_mask |= integer(value)
        lines.append(
            assertion(
                f"UINT64_C(1) << VIRTIO_ACCEL_F_{name}", value, f"features.reserved.{name}"
            )
        )
    lines.append(
        assertion("VIRTIO_ACCEL_RESERVED_FEATURES", reserved_mask, "features.reserved mask")
    )

    capabilities = keys(
        manifest.get("capabilities"), {"assigned", "reserved"}, "capabilities"
    )
    for group in ("assigned", "reserved"):
        values = capabilities[group]
        if not isinstance(values, dict):
            raise Failure(f"capabilities.{group} must be an object")
        for name, value in values.items():
            lines.append(
                assertion(
                    f"VIRTIO_ACCEL_CAP_{name}", value, f"capabilities.{group}.{name}"
                )
            )

    for namespace, prefix in SCALAR_NAMESPACES.items():
        values = manifest.get(namespace)
        if not isinstance(values, dict) or not values:
            raise Failure(f"{namespace} must be a nonempty object")
        for name, value in values.items():
            lines.append(assertion(f"{prefix}{name}", value, f"{namespace}.{name}"))

    buffer_usage = keys(
        manifest.get("buffer_usage"),
        {
            "TRANSFER_SOURCE",
            "TRANSFER_DESTINATION",
            "PROGRAM_INPUT",
            "PROGRAM_OUTPUT",
            "MUTABLE_STATE",
            "KNOWN_BITS",
        },
        "buffer_usage",
    )
    for name, value in buffer_usage.items():
        macro = (
            "VIRTIO_ACCEL_KNOWN_BUFFER_USAGE_BITS"
            if name == "KNOWN_BITS"
            else f"VIRTIO_ACCEL_BUFFER_{name}"
        )
        lines.append(assertion(macro, value, f"buffer_usage.{name}"))

    flag_names = {
        "REQUEST": "VIRTIO_ACCEL_KNOWN_REQUEST_FLAGS",
        "CONTEXT": "VIRTIO_ACCEL_KNOWN_CONTEXT_FLAGS",
        "PROGRAM": "VIRTIO_ACCEL_KNOWN_PROGRAM_FLAGS",
        "QUEUE": "VIRTIO_ACCEL_KNOWN_QUEUE_FLAGS",
        "SUBMIT": "VIRTIO_ACCEL_KNOWN_SUBMIT_FLAGS",
    }
    flags = keys(manifest.get("known_flags"), set(flag_names), "known_flags")
    for name, macro in flag_names.items():
        lines.append(assertion(macro, flags[name], f"known_flags.{name}"))

    structures = keys(manifest.get("structures"), set(STRUCT_NAMES), "structures")
    for manifest_name, c_name in STRUCT_NAMES.items():
        layout = keys(structures[manifest_name], {"size", "align", "fields"}, manifest_name)
        lines.append(
            assertion(f"sizeof(struct {c_name})", layout["size"], f"{manifest_name}.size")
        )
        lines.append(
            assertion(f"VA_ALIGNOF(struct {c_name})", layout["align"], f"{manifest_name}.align")
        )
        fields = layout["fields"]
        if not isinstance(fields, dict) or not fields:
            raise Failure(f"{manifest_name}.fields must be a nonempty object")
        for field, offset in fields.items():
            c_field = FIELD_NAMES.get((manifest_name, field), field)
            lines.append(
                assertion(
                    f"offsetof(struct {c_name}, {c_field})",
                    offset,
                    f"{manifest_name}.{field}",
                )
            )

    return lines


def compiler(variable: str, fallbacks: tuple[str, ...]) -> list[str]:
    configured = os.environ.get(variable)
    if configured:
        return shlex.split(configured)
    for candidate in fallbacks:
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]
    raise Failure(f"no compiler found for {variable}; tried {', '.join(fallbacks)}")


def compile_source(command: list[str], source: pathlib.Path, output: pathlib.Path, cxx: bool) -> None:
    executable = pathlib.Path(command[0]).stem.lower()
    if executable in {"cl", "clang-cl"}:
        standard = "/std:c++14" if cxx else "/std:c11"
        argv = [
            *command,
            "/nologo",
            standard,
            "/W4",
            "/WX",
            f"/I{INCLUDE}",
            "/c",
            str(source),
            f"/Fo{output}",
        ]
    else:
        standard = "-std=c++11" if cxx else "-std=c11"
        argv = [
            *command,
            standard,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(INCLUDE),
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    print("$", " ".join(shlex.quote(part) for part in argv), flush=True)
    result = subprocess.run(argv, cwd=ROOT)
    if result.returncode != 0:
        raise Failure(f"{source.suffix} header check failed with exit code {result.returncode}")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "virtio-accel-layout-1":
        raise Failure(f"unexpected layout schema {manifest.get('schema')!r}")

    checks = "\n".join(manifest_assertions(manifest))
    common = f"""#include <stddef.h>
#include <stdint.h>
#include \"virtio_accel.h\"
#include \"virtio_accel.h\"

#if defined(__cplusplus)
#define VA_STATIC_ASSERT(condition, message) static_assert(condition, message)
#define VA_ALIGNOF(type) alignof(type)
#else
#define VA_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#define VA_ALIGNOF(type) _Alignof(type)
#endif

{checks}
"""

    cc = compiler("CC", ("cc", "clang", "gcc", "cl"))
    cxx = compiler("CXX", ("c++", "clang++", "g++", "clang-cl", "cl"))
    with tempfile.TemporaryDirectory(prefix="virtio-accel-c-header-") as temporary:
        directory = pathlib.Path(temporary)
        c_source = directory / "check.c"
        cxx_source = directory / "check.cc"
        c_source.write_text(common)
        cxx_source.write_text(common)
        compile_source(cc, c_source, directory / "check-c.o", cxx=False)
        compile_source(cxx, cxx_source, directory / "check-cxx.o", cxx=True)

    print(f"validated {len(STRUCT_NAMES)} structures and all scalar namespaces in C and C++")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
