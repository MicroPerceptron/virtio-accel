#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| virtio_accel_fuzz::fuzz_tosa_parse(data));
