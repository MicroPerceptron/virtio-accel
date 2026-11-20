//! Bounded rolling submission window. Only terminal, destroyed events free a slot.
use super::{Fixture, Sink, Source, metadata, number, release};
use divan::{Bencher, counter::ItemsCount};
use std::time::{Duration, Instant};
use virtio_accel_core::*;

#[derive(Clone, Copy)]
pub(super) struct Config {
    depth: usize,
    requests: u64,
    private_inputs: bool,
}
impl Config {
    pub(super) fn from_env(limits: DeviceLimits, input_count: usize) -> Self {
        let depth = number("VA_BENCH_IN_FLIGHT", 1);
        let requests = number("VA_BENCH_REQUESTS", depth.max(256));
        let private_inputs = match std::env::var("VA_BENCH_INPUT_MODE").as_deref() {
            Ok("shared") | Err(_) => false,
            Ok("per-slot") => true,
            _ => panic!("VA_BENCH_INPUT_MODE must be shared or per-slot"),
        };
        Self::checked(limits, input_count, depth, requests, private_inputs)
            .unwrap_or_else(|e| panic!("invalid pipeline configuration: {e}"))
    }
    fn checked(
        l: DeviceLimits,
        inputs: usize,
        depth: u64,
        requests: u64,
        private_inputs: bool,
    ) -> Result<Self, &'static str> {
        if depth == 0 || requests < depth {
            return Err("require 1 <= in-flight depth <= requests per sample");
        }
        let buffers = if private_inputs {
            depth.checked_mul(inputs as u64 + 1)
        } else {
            depth.checked_add(inputs as u64)
        }
        .ok_or("buffer count overflow")?;
        // Each slot owns one program (including private native intermediate state) and output.
        // No artificial ring ceiling: only the resources actually allocated constrain depth.
        if depth > u64::from(l.max_events_per_context)
            || depth > u64::from(l.max_programs_per_context)
            || buffers > u64::from(l.max_buffers_per_context)
            || l.max_queues_per_context == 0
        {
            return Err("requested depth exceeds advertised event/program/buffer limits");
        }
        Ok(Self {
            depth: usize::try_from(depth).map_err(|_| "depth does not fit host usize")?,
            requests,
            private_inputs,
        })
    }
}
struct Pending<E> {
    event: E,
    started: Instant,
}
struct Window<E> {
    slots: Vec<Option<Pending<E>>>,
}
impl<E> Window<E> {
    fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth).map(|_| None).collect(),
        }
    }
    /// Callbacks may panic on native failure: a failed run never yields valid measurements.
    /// The backend handles retain/quarantine their resources on unwinding, as for serial benches.
    fn run(
        &mut self,
        requests: u64,
        timeout: Duration,
        wait_us: u64,
        mut submit: impl FnMut(usize) -> E,
        mut complete: impl FnMut(&E) -> bool,
        mut destroy: impl FnMut(E),
    ) -> usize {
        assert!(requests >= self.slots.len() as u64 && !self.slots.is_empty());
        assert!(self.slots.iter().all(Option::is_none));
        let mut submitted = 0;
        let mut retired = 0;
        let mut live = 0;
        let mut peak = 0;
        // Fill before polling; never turn a requested multi-flight window into depth one.
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let started = Instant::now();
            *slot = Some(Pending {
                event: submit(i),
                started,
            });
            submitted += 1;
            live += 1;
            peak = peak.max(live);
        }
        while retired < requests {
            let mut progress = false;
            // Scan every occupied slot: a slow oldest request cannot block faster completions.
            for (i, slot) in self.slots.iter_mut().enumerate() {
                let Some(pending) = slot.as_ref() else {
                    continue;
                };
                if complete(&pending.event) {
                    let pending = slot.take().unwrap();
                    destroy(pending.event);
                    retired += 1;
                    live -= 1;
                    progress = true;
                    if submitted < requests {
                        let started = Instant::now();
                        *slot = Some(Pending {
                            event: submit(i),
                            started,
                        });
                        submitted += 1;
                        live += 1;
                        peak = peak.max(live);
                    }
                } else {
                    assert!(
                        pending.started.elapsed() < timeout,
                        "pipeline event timed out; run invalid"
                    );
                }
            }
            if !progress {
                if wait_us == 0 {
                    std::hint::spin_loop();
                } else {
                    std::thread::sleep(Duration::from_micros(wait_us));
                }
            }
        }
        assert_eq!(live, 0);
        peak
    }
}
struct Lane<A: Accelerator> {
    // Lane zero borrows the fixture's handles. Other lanes own private programs/outputs.
    program: Option<A::Program>,
    inputs: Vec<A::Buffer>,
    output: Option<A::Buffer>,
}
struct Pipeline<'a, A: Accelerator> {
    fixture: &'a Fixture<A>,
    lanes: Vec<Lane<A>>,
    readback: Vec<u8>,
}
impl<'a, A: Accelerator> Pipeline<'a, A> {
    fn new(f: &'a Fixture<A>, config: Config) -> Self {
        let model = virtio_accel_tosa::parse(&f.workload.artifact).unwrap();
        let artifact = model.artifact_ref(f.workload.target, u64::MAX).unwrap();
        let mut lanes = Vec::with_capacity(config.depth);
        lanes.push(Lane {
            program: None,
            inputs: Vec::new(),
            output: None,
        });
        for _ in 1..config.depth {
            let program = f
                .backend
                .load_program(&f.context, artifact)
                .expect("pipeline program allocation failed; requested depth is never reduced");
            let allocate = |bytes: usize, usage| {
                f.backend
                    .allocate_buffer(
                        &f.context,
                        BufferDesc::new(bytes as u64, 1, MemoryDomain::Shared, usage).unwrap(),
                    )
                    .unwrap()
                    .into_parts()
                    .0
            };
            let inputs = if config.private_inputs {
                f.workload
                    .inputs
                    .iter()
                    .map(|data| {
                        let mut buffer = allocate(
                            data.len(),
                            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
                        );
                        f.backend
                            .write_buffer(&mut buffer, 0, &Source(data))
                            .unwrap();
                        buffer
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let output = allocate(
                f.workload.expected.len(),
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            );
            lanes.push(Lane {
                program: Some(program),
                inputs,
                output: Some(output),
            });
        }
        Self {
            fixture: f,
            lanes,
            readback: vec![0; f.workload.expected.len()],
        }
    }
    fn run(&self, window: &mut Window<A::Event>, requests: u64) -> usize {
        let f = self.fixture;
        window.run(
            requests,
            f.timeout,
            f.wait_us,
            |i| {
                let lane = &self.lanes[i];
                let inputs = if lane.inputs.is_empty() {
                    &f.inputs
                } else {
                    &lane.inputs
                };
                f.submit_to(
                    lane.program.as_ref().unwrap_or(&f.program),
                    inputs,
                    lane.output.as_ref().unwrap_or(&f.output),
                )
            },
            |event| match f
                .backend
                .poll_event(event)
                .expect("pipeline poll failed; run invalid")
            {
                EventState::Pending => false,
                EventState::Complete => true,
                other => panic!("pipeline execution failed: {other:?}; run invalid"),
            },
            |event| release(f.backend.destroy_event(event)),
        )
    }
    fn validate(&mut self) {
        for (index, lane) in self.lanes.iter().enumerate() {
            self.fixture
                .backend
                .read_buffer(
                    lane.output.as_ref().unwrap_or(&self.fixture.output),
                    0,
                    &mut Sink(&mut self.readback),
                )
                .unwrap();
            self.fixture
                .workload
                .validate(&self.readback)
                .unwrap_or_else(|e| panic!("pipeline slot {index}: {e}"));
        }
    }
    fn close(self) {
        for lane in self.lanes {
            if let Some(p) = lane.program {
                release(self.fixture.backend.unload_program(p));
            }
            if let Some(b) = lane.output {
                release(self.fixture.backend.free_buffer(b));
            }
            for b in lane.inputs {
                release(self.fixture.backend.free_buffer(b));
            }
        }
    }
}
pub(super) fn measure<A: Accelerator>(
    b: Bencher,
    f: &Fixture<A>,
    config: Config,
    warmups: u64,
    mut meta: serde_json::Value,
) {
    let started = Instant::now();
    let mut p = Pipeline::new(f, config);
    let mut window = Window::new(config.depth);
    let setup_ns = started.elapsed().as_nanos().to_string();
    // Exactly one request per slot ensures every slot participates in numerical validation.
    let peak = p.run(&mut window, config.depth as u64);
    p.validate();
    for _ in 0..warmups {
        p.run(&mut window, config.requests);
    }
    meta["pipeline"] = serde_json::json!({
        "in_flight":config.depth,"requests_per_iteration":config.requests,
        "peak_live_events_validation":peak,"programs":config.depth,
        "input_mode":if config.private_inputs { "per-slot" } else { "shared" },
        "resident_input_bytes":f.workload.input_bytes() as u64 * if config.private_inputs { config.depth as u64 } else { 1 },
        "resident_output_bytes":f.workload.expected.len() as u64 * config.depth as u64,
        "additional_setup_ns":setup_ns,"counter":"completed requests per second",
        "workload":"independent repeated requests; not dependent autoregressive tokens",
        "placement":"host live events are not proof of concurrent engine execution"
    });
    metadata(meta);
    b.counter(ItemsCount::new(config.requests)).bench_local(|| {
        p.run(&mut window, config.requests);
    });
    // Probe every slot again: the rolling tail may leave some slots idle earlier than others.
    p.validate();
    p.run(&mut window, config.depth as u64);
    p.validate();
    p.close();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    #[test]
    fn refills_completed_slots_before_the_slow_oldest_and_drains_every_event() {
        let submitted = Cell::new(0usize);
        let retired = RefCell::new(Vec::new());
        let polls = Cell::new(0);
        let mut window = Window::new(3);
        for _ in 0..2 {
            let start = submitted.get();
            let peak = window.run(
                10,
                Duration::from_secs(1),
                0,
                |_| {
                    assert!(submitted.get() - retired.borrow().len() < 3);
                    let id = submitted.get();
                    submitted.set(id + 1);
                    id
                },
                |id| {
                    polls.set(polls.get() + 1);
                    *id != start || submitted.get() >= start + 8
                },
                |id| retired.borrow_mut().push(id),
            );
            assert_eq!(peak, 3);
            assert!(window.slots.iter().all(Option::is_none));
            assert_eq!(submitted.get(), start + 10);
            assert_ne!(retired.borrow()[start], start);
        }
        let mut ids = retired.into_inner();
        ids.sort_unstable();
        assert_eq!(ids, (0..20).collect::<Vec<_>>());
    }
    #[test]
    fn capacity_follows_actual_resource_counts() {
        let limits = DeviceLimits {
            max_contexts: 1,
            max_buffers_per_context: 100,
            max_programs_per_context: 80,
            max_queues_per_context: 1,
            max_events_per_context: 90,
            max_bindings_per_submission: 4,
            max_buffer_bytes: u64::MAX,
            max_artifact_bytes: u64::MAX,
        };
        assert!(Config::checked(limits, 3, 80, 256, false).is_ok());
        assert!(Config::checked(limits, 3, 81, 256, false).is_err());
        assert!(Config::checked(limits, 3, 25, 256, true).is_ok());
        assert!(Config::checked(limits, 3, 26, 256, true).is_err());
        assert!(Config::checked(limits, 3, 0, 256, false).is_err());
        assert!(Config::checked(limits, 3, 8, 7, false).is_err());
        assert!(Config::checked(limits, 3, u64::MAX, u64::MAX, true).is_err());
    }
    #[test]
    #[should_panic(expected = "pipeline event timed out")]
    fn timeout_does_not_recycle_a_live_slot() {
        Window::new(1).run(
            2,
            Duration::ZERO,
            0,
            |_| (),
            |_| false,
            |_| panic!("must not release pending event"),
        );
    }
}
