# Core ML TOSA fixtures

`identity-fp32-v1.0.0.tosa` is a deterministic TOSA 1.0 FlatBuffer generated from the checked-in
schema. It contains one static `FP32[1]` input, one `IDENTITY` operator, and one static `FP32[1]`
output. The native integration test uses it to prove that the production artifact crosses the
portable boundary as TOSA and is lowered only inside `virtio-accel-coreml`.

SHA-256: `ff7eeb742556ed5fa2d0da4fd38b4bdbd1622233a617418fd970977df9f24ba6`.
