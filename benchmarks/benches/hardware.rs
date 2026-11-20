#![forbid(unsafe_code)]
use virtio_accel_benchmarks::{
    cases::{CASES, Case},
    harness::{Phase, selected},
};
fn main() {
    divan::main();
}
#[divan::bench(args=CASES,sample_count=30,max_time=5)]
fn load_unload(b: divan::Bencher, c: Case) {
    selected(b, c, Phase::LoadUnload);
}
#[divan::bench(args=CASES,sample_count=30,max_time=5)]
fn upload(b: divan::Bencher, c: Case) {
    selected(b, c, Phase::Upload);
}
#[divan::bench(args=CASES,sample_count=30,max_time=5)]
fn download(b: divan::Bencher, c: Case) {
    selected(b, c, Phase::Download);
}
#[divan::bench(args=CASES,sample_count=30,max_time=5)]
fn warm_execution(b: divan::Bencher, c: Case) {
    selected(b, c, Phase::Warm);
}
#[divan::bench(args=CASES,sample_count=30,max_time=5)]
fn round_trip(b: divan::Bencher, c: Case) {
    selected(b, c, Phase::RoundTrip);
}

#[divan::bench(args=CASES,sample_count=30,max_time=5)]
fn pipeline_execution(b: divan::Bencher, c: Case) {
    selected(b, c, Phase::Pipeline);
}
