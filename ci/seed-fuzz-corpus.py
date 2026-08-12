#!/usr/bin/env python3
"""Generate deterministic fuzz seed corpora from reviewed conformance vectors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "conformance" / "v1.0" / "vectors.json"
CORPUS = ROOT / "fuzz" / "corpus"
TOSA_SELECT = ROOT / "crates" / "virtio-accel-tosa" / "tests" / "data" / "select-v1.0.0.tosa"

PROTOCOL_CONTROLS = bytes([0x00, 0x40, 1, 2, 4, 8, 16, 32])
DESCRIPTOR_MARKER = 0xA5
DESCRIPTOR_RESPONSE_BYTES_MINUS_ONE = 127

# Guest-client action opcodes; see the dispatch table in fuzz/src/guest.rs.
GUEST_DEVICE_POP = 11
GUEST_DEVICE_COMPLETE = 12
GUEST_PUMP = 13
GUEST_POLL = 14
GUEST_RESET = 15
GUEST_POOL_LIMIT = 512


def main() -> None:
    corpus = json.loads(VECTORS.read_text())
    reset_target("protocol_decode")
    reset_target("descriptor_end_to_end")
    reset_target("stateful_commands")
    reset_target("guest_client")
    reset_target("tosa_parse")

    write_seed("protocol_decode", "empty", b"")
    write_seed("protocol_decode", "short_header", PROTOCOL_CONTROLS + b"\x00\x01\x02")

    for frame in corpus["frames"]:
        frame_bytes = bytes.fromhex(frame["hex"])
        if frame["kind"] == "request":
            write_seed("protocol_decode", frame["name"], PROTOCOL_CONTROLS + frame_bytes)
            write_seed("descriptor_end_to_end", frame["name"], descriptor_seed(frame_bytes))
        else:
            write_seed("protocol_decode", frame["name"], PROTOCOL_CONTROLS + frame_bytes)

    write_seed("descriptor_end_to_end", "raw_loop", raw_loop_seed())
    write_seed("descriptor_end_to_end", "raw_truncated", raw_truncated_seed())
    write_seed("descriptor_end_to_end", "raw_used_length_exceeded", raw_used_length_exceeded_seed())

    write_seed("stateful_commands", "lifecycle", stateful_lifecycle_seed())
    write_seed("stateful_commands", "stale_after_reset", stateful_stale_after_reset_seed())

    responses = response_pool(corpus)
    write_seed("guest_client", "conforming_lifecycle", guest_lifecycle_seed(responses))
    write_seed("guest_client", "hostile_completion", guest_hostile_seed(responses))
    write_seed("guest_client", "reset_with_inflight", guest_reset_seed(responses))
    write_seed("tosa_parse", "stable_select", TOSA_SELECT.read_bytes())
    write_seed("tosa_parse", "identifier_only", b"\x08\x00\x00\x00TOSA")


def reset_target(target: str) -> None:
    directory = CORPUS / target
    directory.mkdir(parents=True, exist_ok=True)
    for entry in directory.iterdir():
        if entry.is_file():
            entry.unlink()


def write_seed(target: str, name: str, data: bytes) -> None:
    path = CORPUS / target / f"{name}.seed"
    path.write_bytes(data)


def descriptor_seed(frame: bytes) -> bytes:
    return bytes(
        [
            DESCRIPTOR_MARKER,
            DESCRIPTOR_RESPONSE_BYTES_MINUS_ONE,
            0,
            1,
            3,
            7,
            2,
            5,
            11,
        ]
    ) + frame


def raw_loop_seed() -> bytes:
    return bytes([1, 0, 0, 1, 1, 0, 0, 0, 0])


def raw_truncated_seed() -> bytes:
    return bytes([3, 19, 0, 64, 1])


def raw_used_length_exceeded_seed() -> bytes:
    return bytes(
        [
            0,
            0,
            0,
            16,
            2,
            0,
            0,
            0,
            0,
        ]
    ) + bytes(16) + bytes([1])


def action(opcode: int, selector: int = 0, argument: int = 0, entropy: int = 0) -> bytes:
    return bytes([opcode & 0xFF, selector & 0xFF]) + argument.to_bytes(2, "little") + entropy.to_bytes(
        4, "little"
    )


def stateful_lifecycle_seed() -> bytes:
    return b"".join(
        [
            action(0),
            action(1),
            action(2),
            action(3),
            action(4),
            action(7),
            action(5),
            action(8),
            action(9),
            action(10),
            action(11),
            action(12),
            action(13),
            action(14),
            action(15),
        ]
    )


def response_pool(corpus: dict) -> bytes:
    """Concatenate reviewed response frames for the guest client's payload pool."""
    frames = [
        bytes.fromhex(frame["hex"]) for frame in corpus["frames"] if frame["kind"] != "request"
    ]
    return b"".join(frames)[:GUEST_POOL_LIMIT]


def guest_action(opcode: int, selector: int = 0, argument: int = 0, entropy: int = 0) -> bytes:
    return bytes([opcode & 0xFF, selector & 0xFF]) + argument.to_bytes(
        2, "little"
    ) + entropy.to_bytes(4, "little")


def guest_seed(pool: bytes, actions: bytes) -> bytes:
    return len(pool).to_bytes(2, "little") + pool + actions


def guest_lifecycle_seed(pool: bytes) -> bytes:
    """Discover, then build a context, buffer, program, queue, and event conformingly."""
    actions = b""
    for start in (0, 1, 2, 3, 4, 5, 6):
        for step in (start, GUEST_DEVICE_POP, GUEST_DEVICE_COMPLETE, GUEST_POLL):
            actions += guest_action(step)
    return guest_seed(pool, actions)


def guest_hostile_seed(pool: bytes) -> bytes:
    """Answer a discovery request with reviewed response bytes under a corrupted header."""
    actions = b"".join(
        [
            guest_action(0),
            guest_action(GUEST_DEVICE_POP),
            guest_action(GUEST_DEVICE_COMPLETE, selector=0x90, argument=0x0060),
            guest_action(GUEST_POLL),
            guest_action(GUEST_RESET),
        ]
    )
    return guest_seed(pool, actions)


def guest_reset_seed(pool: bytes) -> bytes:
    """Reset while requests are published, popped, and completed but not yet polled."""
    actions = b"".join(
        [
            guest_action(0),
            guest_action(0),
            guest_action(GUEST_DEVICE_POP),
            guest_action(GUEST_DEVICE_COMPLETE),
            guest_action(GUEST_PUMP),
            guest_action(GUEST_RESET),
            guest_action(GUEST_POLL),
        ]
    )
    return guest_seed(pool, actions)


def stateful_stale_after_reset_seed() -> bytes:
    return b"".join(
        [
            action(0),
            action(1),
            action(15),
            action(17, selector=0x80),
            action(16),
        ]
    )


if __name__ == "__main__":
    main()
