#![forbid(unsafe_code)]
use virtio_accel_benchmarks::cases::Case;
fn main() {
    divan::main();
}
#[divan::bench]
fn build_nvfp4() {
    divan::black_box(virtio_accel_tosa_build::quantized::nvfp4_matmul(1, 128, 256, 0.125).unwrap());
}
#[divan::bench]
fn parse_and_analyze(b: divan::Bencher) {
    let w = Case::Nvfp4F16Decode.build();
    b.bench(|| {
        let m = virtio_accel_tosa::parse(divan::black_box(&w.artifact)).unwrap();
        divan::black_box(m.analyze_for(w.target).unwrap());
    });
}
