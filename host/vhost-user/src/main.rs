#![forbid(unsafe_code)]

use std::path::Path;
use virtio_accel_core::Accelerator;
use virtio_accel_device::ResourcePolicy;
use virtio_accel_vhost_user::{Host, serve};

const USAGE: &str = "usage: virtio-accel-vhost-user --backend mock|openvino|vulkan|xdna [--list-devices | --socket PATH] [--device NAME] [--buffer-budget BYTES] [--program-budget BYTES | --allow-unknown-program-residency]";

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
    let mut list = false;
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("{USAGE}");
            return Ok(());
        }
        if arg == "--list-devices" {
            list = true;
            continue;
        }
        if arg == "--allow-unknown-program-residency" {
            programs = u64::MAX;
            continue;
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
    let backend = backend.ok_or(USAGE)?;
    let built = [
        ("mock", true),
        ("openvino", cfg!(feature = "openvino")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("xdna", cfg!(feature = "xdna")),
    ]
    .into_iter()
    .any(|(name, enabled)| name == backend && enabled);
    if !built {
        return Err(format!("backend {backend:?} is unavailable in this build").into());
    }
    if device.is_some() && !matches!(backend.as_str(), "openvino" | "vulkan") {
        return Err("--device selects an OpenVINO endpoint or an exact Vulkan device name".into());
    }
    if list {
        if socket.is_some() || device.is_some() {
            return Err("--list-devices cannot be combined with --socket or --device".into());
        }
        let names = match backend.as_str() {
            "mock" => vec!["mock".to_owned()],
            #[cfg(feature = "openvino")]
            "openvino" => virtio_accel_openvino::OpenVinoAccelerator::available_devices()
                .map_err(|e| format!("OpenVINO: {e:?}"))?,
            #[cfg(feature = "vulkan")]
            "vulkan" => virtio_accel_vulkan::VulkanAccelerator::available_devices()
                .map_err(|e| format!("Vulkan: {e:?}"))?,
            #[cfg(feature = "xdna")]
            "xdna" => {
                let accelerator = virtio_accel_xdna::XdnaAccelerator::new()
                    .map_err(|e| format!("XDNA: {e:?}"))?;
                accelerator
                    .device_info()
                    .map_err(|e| format!("XDNA: {e:?}"))?;
                vec!["XDNA default endpoint (no device selector)".to_owned()]
            }
            _ => return Err(format!("backend {backend:?} is unavailable in this build").into()),
        };
        for name in names {
            println!("{name}");
        }
        return Ok(());
    }
    let socket = socket.ok_or(USAGE)?;
    let socket = Path::new(&socket);
    if backend != "mock" && programs != u64::MAX {
        return Err("native programs currently declare unknown residency; use --allow-unknown-program-residency to admit one resident program (not a physical-memory bound), or --program-budget 18446744073709551615".into());
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
            let accelerator = match device {
                Some(device) => virtio_accel_vulkan::VulkanAccelerator::with_device(&device),
                None => virtio_accel_vulkan::VulkanAccelerator::new(),
            }
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
