#![forbid(unsafe_code)]

use std::path::Path;
use virtio_accel_core::Accelerator;
use virtio_accel_device::ResourcePolicy;
use virtio_accel_vhost_user::{Host, serve};

const USAGE: &str = "usage: virtio-accel-vhost-user --socket PATH --backend mock|openvino|vulkan|xdna [--device NAME] [--buffer-budget BYTES] [--program-budget BYTES]";

fn run<A: Accelerator>(
    socket: &Path,
    accelerator: A,
    policy: ResourcePolicy,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "virtio-accel: buffer budget={}, program resident-charge budget={}",
        policy.max_buffer_backing_bytes(),
        policy.max_program_resident_bytes()
    );
    serve(socket, Host::new(accelerator, policy)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut socket = None;
    let mut backend = None;
    let mut device = None;
    let mut buffers = 512 << 20;
    let mut programs = 256 << 20;
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("{USAGE}");
            return Ok(());
        }
        let value = args.next().ok_or(USAGE)?;
        match arg.as_str() {
            "--socket" if socket.is_none() => socket = Some(value),
            "--backend" if backend.is_none() => backend = Some(value),
            "--device" if device.is_none() => device = Some(value),
            "--buffer-budget" => buffers = value.parse()?,
            "--program-budget" => programs = value.parse()?,
            _ => return Err(USAGE.into()),
        }
    }
    let socket = socket.ok_or(USAGE)?;
    let socket = Path::new(&socket);
    let backend = backend.ok_or(USAGE)?;
    if device.is_some() && backend != "openvino" {
        return Err("--device currently selects an OpenVINO endpoint".into());
    }
    let policy = ResourcePolicy::new(buffers, programs).ok_or("budgets must be nonzero")?;
    match backend.as_str() {
        "mock" => run(
            socket,
            virtio_accel_mock::MockAccelerator::completing_on_poll(),
            policy,
        ),
        #[cfg(feature = "openvino")]
        "openvino" => {
            let accelerator = match device {
                Some(device) => virtio_accel_openvino::OpenVinoAccelerator::with_device(&device),
                None => virtio_accel_openvino::OpenVinoAccelerator::new(),
            }
            .map_err(|e| format!("OpenVINO: {e:?}"))?;
            eprintln!("virtio-accel: OpenVINO {}", accelerator.device_name());
            run(socket, accelerator, policy)
        }
        #[cfg(feature = "vulkan")]
        "vulkan" => {
            let accelerator = virtio_accel_vulkan::VulkanAccelerator::new()
                .map_err(|e| format!("Vulkan: {e:?}"))?;
            eprintln!("virtio-accel: Vulkan {}", accelerator.device_name());
            run(socket, accelerator, policy)
        }
        #[cfg(feature = "xdna")]
        "xdna" => run(
            socket,
            virtio_accel_xdna::XdnaAccelerator::new().map_err(|e| format!("XDNA: {e:?}"))?,
            policy,
        ),
        _ => Err(format!("backend {backend:?} is unavailable in this build").into()),
    }
}
