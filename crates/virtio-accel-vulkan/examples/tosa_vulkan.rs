fn main() {
    match virtio_accel_vulkan::VulkanAccelerator::new() {
        Ok(_) => unreachable!("the scaffold placeholder never initializes a backend"),
        Err(error) => {
            eprintln!("virtio-accel-vulkan is scaffolded but not yet executing: {error}");
        }
    }
}
