# Fixture provenance

`select-v1.0.0.tosa` was encoded with `flatc 25.2.10` and the crate's pinned schema from
`test_native_select_2048x2048x3_i8.json` in the upstream TOSA Tools repository. Unknown fields
added after the pinned tools release were ignored during encoding; every retained field belongs to
the pinned schema.

The source JSON is copyright Arm Limited and/or its affiliates and is licensed under Apache-2.0.
The encoded fixture SHA-256 is
`5299133aad7ea696939c1f3fd264d35a449c16207b82a7b483698519909ff81a`.
