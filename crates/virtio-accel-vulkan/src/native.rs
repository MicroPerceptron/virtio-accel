//! Native Vulkan backend over `ash`: the audited `Accelerator` implementation.
//!
//! `SAFETY.md` is the audit of record; every `unsafe` block below carries a local `SAFETY:` note.
//! Each Vulkan handle has exactly one Rust owner with a `Drop` implementation, every `VkResult`
//! is checked before an out-value is trusted, and `VK_ERROR_DEVICE_LOST` poisons the whole backend
//! instance (ADR 0006). Completion is a nonblocking `vkGetFenceStatus` read: no worker thread, no
//! callback, no foreign code ever owns Rust memory.
//!
//! Handles are deliberately neither `Send` nor `Sync` (`Rc` inside): Vulkan queues and command
//! pools are externally synchronized objects, and the contract permits thread-affine providers.

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::OnceLock;

use ash::vk;
use virtio_accel_core::{
    Accelerator, AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError,
    BindingRef, BufferDesc, BufferInfo, BufferProperties, BufferUsage, ByteSink, ByteSource,
    Capabilities, ContextDesc, DeviceIdentity, DeviceInfo, DeviceLimits, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_tosa::{CapabilityDescriptor, TosaCapabilityProvider};

use crate::lower::{Kernel, LoweringError, ProgramPlan, SlotRole, lower_tosa};
use crate::shader;
use crate::{InitError, REQUIRED_RESIDENT_BYTES};

/// Maximal TOSA artifact bytes admitted before parsing (mirrors the other TOSA backends).
const MAX_TOSA_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// Stable provider-owned error namespace for unmapped `VkResult` codes (`"VULK"`).
const VULKAN_EXTERNAL_DOMAIN: u32 = 0x5655_4c4b;

/// Bounded staging allocation for explicit transfers into and out of `MemoryDomain::Device`.
const STAGING_BYTES: u64 = 4 * 1024 * 1024;

/// How long a synchronous explicit transfer may take before the device is treated as lost.
const TRANSFER_TIMEOUT_NS: u64 = 30_000_000_000;

/// `maxMemoryAllocationCount` is assumed at the spec minimum (ADR 0005): every buffer here is one
/// dedicated `VkDeviceMemory`, so the advertised aggregate buffer count plus one transient staging
/// allocation must stay inside it.
const ASSUMED_MAX_MEMORY_ALLOCATIONS: u32 = 4096;
const MAX_CONTEXTS: u32 = 16;
const MAX_BUFFERS_PER_CONTEXT: u32 = 255;
const MAX_PROGRAMS_PER_CONTEXT: u32 = 64;
const MAX_QUEUES_PER_CONTEXT: u32 = 16;
/// Ring depth per context: one (command buffer, fence, descriptor set) triple per outstanding
/// event (ADR 0006).
const RING_DEPTH: u32 = 64;
const MAX_BINDINGS_PER_SUBMISSION: u32 = 16;

const _: () = assert!(
    MAX_CONTEXTS * MAX_BUFFERS_PER_CONTEXT < ASSUMED_MAX_MEMORY_ALLOCATIONS,
    "advertised buffer aggregate must fit the assumed allocation count with one staging slot"
);

const EXCLUSIVE_ACCESS: u64 = 1 << 63;

/// The process-wide loader handle: one `dlopen` of the platform Vulkan loader.
fn entry() -> Result<ash::Entry, InitError> {
    static ENTRY: OnceLock<Result<ash::Entry, InitError>> = OnceLock::new();
    ENTRY
        .get_or_init(|| {
            // SAFETY: loading the platform Vulkan loader runs its initializers exactly once per
            // process under this `OnceLock`; nothing else in this crate loads it.
            unsafe { ash::Entry::load() }.map_err(|_| InitError::RuntimeUnavailable)
        })
        .clone()
}

fn backend_error(result: vk::Result) -> BackendError {
    match result {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            BackendError::OutOfMemory
        }
        vk::Result::ERROR_TOO_MANY_OBJECTS => BackendError::ResourceLimit,
        vk::Result::ERROR_DEVICE_LOST => BackendError::DeviceLost,
        vk::Result::ERROR_MEMORY_MAP_FAILED => BackendError::Incompatible,
        other => BackendError::External {
            domain: VULKAN_EXTERNAL_DOMAIN,
            code: i64::from(other.as_raw()),
        },
    }
}

/// Owned `VkInstance`; destroyed after every device that was created from it.
struct Instance {
    /// Kept so the loaded library outlives the instance created from it.
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl Instance {
    fn create() -> Result<Self, InitError> {
        let entry = entry()?;
        // SAFETY: querying the loader's instance version has no preconditions.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::API_VERSION_1_3 {
            return Err(InitError::RuntimeUnavailable);
        }
        let application = vk::ApplicationInfo::default()
            .application_name(c"virtio-accel-vulkan")
            .api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: `info` and the structures it points to outlive the call; no layers or
        // extensions are requested, so the loader validates nothing beyond the API version.
        let instance = match unsafe { entry.create_instance(&info, None) } {
            Ok(instance) => instance,
            Err(vk::Result::ERROR_INCOMPATIBLE_DRIVER) => {
                return Err(InitError::RuntimeUnavailable);
            }
            Err(_) => return Err(InitError::InstanceCreationFailed),
        };
        Ok(Self {
            _entry: entry,
            instance,
        })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: this owner is dropped exactly once, after every `Shared` (and thus every
        // device) created from it: `Shared` holds the `Instance` and destroys its device first.
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// Everything probed about one physical device before it is opened.
#[derive(Clone)]
struct PhysicalDeviceRecord {
    handle: vk::PhysicalDevice,
    name: String,
    device_type: vk::PhysicalDeviceType,
    vendor_id: u32,
    device_id: u32,
    uuid: [u8; 16],
    queue_family: u32,
    limits: vk::PhysicalDeviceLimits,
    memory: vk::PhysicalDeviceMemoryProperties,
    buffer_device_address: bool,
}

impl PhysicalDeviceRecord {
    /// Probe one device; `None` when it cannot host this backend (API floor, compute queue,
    /// mandatory `synchronization2`).
    fn probe(instance: &ash::Instance, handle: vk::PhysicalDevice) -> Option<Self> {
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut vulkan11);
        // SAFETY: `handle` was enumerated from `instance`; the chained structures are live locals.
        unsafe { instance.get_physical_device_properties2(handle, &mut properties) };
        let properties = properties.properties;
        if properties.api_version < vk::API_VERSION_1_3 {
            return None;
        }

        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        // SAFETY: as above; the feature chain is fully initialized before the call.
        unsafe { instance.get_physical_device_features2(handle, &mut features) };
        if vulkan13.synchronization2 == vk::FALSE {
            return None;
        }

        // SAFETY: `handle` is a live physical device of `instance`.
        let families = unsafe { instance.get_physical_device_queue_family_properties(handle) };
        // A compute-only family keeps this backend's work off the graphics queue when the device
        // offers one; otherwise the first compute-capable family serves.
        let compute_only = families.iter().position(|family| {
            family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        });
        let any_compute = families
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE));
        let queue_family = u32::try_from(compute_only.or(any_compute)?).ok()?;

        // SAFETY: `handle` is a live physical device of `instance`.
        let memory = unsafe { instance.get_physical_device_memory_properties(handle) };
        let name = CStr::from_bytes_until_nul(bytemuck_i8_to_u8(&properties.device_name))
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from("vulkan-device"));
        Some(Self {
            handle,
            name,
            device_type: properties.device_type,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            uuid: vulkan11.device_uuid,
            queue_family,
            limits: properties.limits,
            memory,
            buffer_device_address: vulkan12.buffer_device_address == vk::TRUE,
        })
    }

    /// Preference order when no device was named: discrete, integrated, virtual, CPU, other.
    fn rank(&self) -> u8 {
        match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 3,
            _ => 4,
        }
    }

    fn class(&self) -> AcceleratorClass {
        if self.device_type == vk::PhysicalDeviceType::CPU {
            AcceleratorClass::OTHER
        } else {
            AcceleratorClass::GPU
        }
    }
}

/// View a driver-filled `c_char` name array as bytes for `CStr` parsing.
fn bytemuck_i8_to_u8(name: &[std::ffi::c_char; 256]) -> &[u8; 256] {
    // SAFETY: `c_char` and `u8` have identical size and alignment; the array is plain data.
    unsafe { &*(name as *const [std::ffi::c_char; 256]).cast::<[u8; 256]>() }
}

fn enumerate(instance: &ash::Instance) -> Result<Vec<PhysicalDeviceRecord>, InitError> {
    // SAFETY: enumeration on a live instance has no other preconditions.
    let handles = unsafe { instance.enumerate_physical_devices() }
        .map_err(|_| InitError::DeviceEnumerationFailed)?;
    Ok(handles
        .into_iter()
        .filter_map(|handle| PhysicalDeviceRecord::probe(instance, handle))
        .collect())
}

/// The memory type chosen for each advertised domain (ADR 0005 memory-domain map).
#[derive(Clone, Copy, Debug)]
struct MemoryPlan {
    /// `HOST_VISIBLE | HOST_COHERENT`, preferring system memory and a host-cached type.
    host: u32,
    /// `DEVICE_LOCAL`, preferring a type the host cannot see; absent on devices without one.
    device: Option<u32>,
    /// `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`: ReBAR or UMA, never assumed.
    shared: Option<u32>,
}

impl MemoryPlan {
    /// Choose one type per domain among those `buffer_type_mask` (the `memoryTypeBits` a
    /// storage buffer of this backend reports) permits.
    fn select(memory: &vk::PhysicalDeviceMemoryProperties, buffer_type_mask: u32) -> Option<Self> {
        let types = &memory.memory_types[..memory.memory_type_count as usize];
        let usable = |index: usize, flags: vk::MemoryPropertyFlags| {
            buffer_type_mask & (1 << index) != 0
                && !flags.intersects(
                    vk::MemoryPropertyFlags::PROTECTED | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
                )
        };
        let pick = |required: vk::MemoryPropertyFlags,
                    score: fn(vk::MemoryPropertyFlags) -> u8|
         -> Option<u32> {
            types
                .iter()
                .enumerate()
                .filter(|(index, ty)| {
                    usable(*index, ty.property_flags) && ty.property_flags.contains(required)
                })
                .max_by_key(|(_, ty)| score(ty.property_flags))
                .and_then(|(index, _)| u32::try_from(index).ok())
        };
        let host_coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        Some(Self {
            host: pick(host_coherent, |flags| {
                u8::from(!flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)) * 2
                    + u8::from(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED))
            })?,
            device: pick(vk::MemoryPropertyFlags::DEVICE_LOCAL, |flags| {
                u8::from(!flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
            }),
            shared: pick(
                vk::MemoryPropertyFlags::DEVICE_LOCAL | host_coherent,
                |flags| u8::from(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED)),
            ),
        })
    }

    fn for_domain(self, domain: MemoryDomain) -> Option<u32> {
        match domain {
            MemoryDomain::Host => Some(self.host),
            MemoryDomain::Device => self.device,
            MemoryDomain::Shared => self.shared,
        }
    }

    fn capabilities(self) -> Capabilities {
        let mut capabilities = Capabilities::HOST_VISIBLE_MEMORY;
        if self.device.is_some() {
            capabilities |= Capabilities::DEVICE_LOCAL_MEMORY;
        }
        if self.shared.is_some() {
            capabilities |= Capabilities::SHARED_MEMORY;
        }
        capabilities
    }
}

/// Live provider resource totals for accounting hooks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveResources {
    pub contexts: u64,
    pub buffers: u64,
    pub programs: u64,
    pub queues: u64,
    pub events: u64,
}

#[derive(Default)]
struct Counters {
    direct_binding_admissions: Cell<u64>,
    explicit_transfer_bytes: Cell<u64>,
    contexts: Cell<u64>,
    buffers: Cell<u64>,
    programs: Cell<u64>,
    queues: Cell<u64>,
    events: Cell<u64>,
}

fn increment(cell: &Cell<u64>, by: u64) {
    cell.set(cell.get().saturating_add(by));
}

fn decrement(cell: &Cell<u64>) {
    cell.set(cell.get().saturating_sub(1));
}

/// The opened device and everything shared by all handles of one backend instance.
///
/// Field order matters for teardown: the explicit `Drop` destroys device-level objects and the
/// device, then the `instance` field's own `Drop` destroys the instance.
struct Shared {
    device: ash::Device,
    physical: PhysicalDeviceRecord,
    queue: vk::Queue,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    memory_plan: MemoryPlan,
    info: DeviceInfo,
    /// Sticky device-loss flag: after `VK_ERROR_DEVICE_LOST` no entry point is re-entered except
    /// destruction (ADR 0006).
    poisoned: Cell<bool>,
    counters: Counters,
    /// Dropped last (see the struct documentation): destroys the instance after the device.
    _instance: Instance,
}

impl Shared {
    fn open(instance: Instance, physical: PhysicalDeviceRecord) -> Result<Rc<Self>, InitError> {
        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(physical.queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        // `synchronization2` is core in 1.3 but still an opt-in feature (ADR 0005);
        // `bufferDeviceAddress` is enabled only to measure allocation alignment honestly.
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(physical.buffer_device_address);
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        // SAFETY: `physical.handle` belongs to `instance.instance`; every pointed-to structure
        // outlives the call; the requested features were reported supported by the probe.
        let device = unsafe {
            instance
                .instance
                .create_device(physical.handle, &info, None)
        }
        .map_err(|_| InitError::DeviceCreationFailed)?;
        // SAFETY: the queue family and index 0 were requested at device creation.
        let queue = unsafe { device.get_device_queue(physical.queue_family, 0) };

        // Which memory types a storage buffer of this backend may live in is a property of the
        // buffer usage, not of the heap list alone (ANV exposes types buffers cannot use), so the
        // memory-domain map is chosen against a probe buffer's `memoryTypeBits`.
        let buffer_type_mask = match probe_buffer_type_mask(&device, &physical) {
            Ok(mask) => mask,
            Err(_) => {
                // SAFETY: the device was created above and has no other objects yet.
                unsafe { device.destroy_device(None) };
                return Err(InitError::DeviceCreationFailed);
            }
        };
        let Some(memory_plan) = MemoryPlan::select(&physical.memory, buffer_type_mask) else {
            // SAFETY: as above.
            unsafe { device.destroy_device(None) };
            return Err(InitError::DeviceUnavailable);
        };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(shader::INPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(shader::OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: the device is live and `layout_info` points at live locals.
        let set_layout = match unsafe { device.create_descriptor_set_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(_) => {
                // SAFETY: the device was created above and has no other objects yet.
                unsafe { device.destroy_device(None) };
                return Err(InitError::DeviceCreationFailed);
            }
        };
        let set_layouts = [set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: the device and set layout are live.
        let pipeline_layout =
            match unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) } {
                Ok(layout) => layout,
                Err(_) => {
                    // SAFETY: both objects were created above and nothing references them.
                    unsafe {
                        device.destroy_descriptor_set_layout(set_layout, None);
                        device.destroy_device(None);
                    }
                    return Err(InitError::DeviceCreationFailed);
                }
            };

        let info = device_info(&physical, memory_plan);
        Ok(Rc::new(Self {
            device,
            physical,
            queue,
            set_layout,
            pipeline_layout,
            memory_plan,
            info,
            poisoned: Cell::new(false),
            counters: Counters::default(),
            _instance: instance,
        }))
    }

    /// Map a failed `VkResult`, latching device loss.
    fn fail(&self, result: vk::Result) -> BackendError {
        if result == vk::Result::ERROR_DEVICE_LOST {
            self.poisoned.set(true);
        }
        backend_error(result)
    }

    fn check_live(&self) -> Result<(), BackendError> {
        if self.poisoned.get() {
            Err(BackendError::DeviceLost)
        } else {
            Ok(())
        }
    }

    /// Block until the device is idle; used only on teardown paths that must not free memory a
    /// pending submission may still touch.
    fn wait_idle(&self) {
        // SAFETY: the device is live; waiting has no other preconditions. Errors are latched.
        if let Err(result) = unsafe { self.device.device_wait_idle() } {
            self.fail(result);
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every child object holds an `Rc<Shared>`, so this runs only after all of them
        // were destroyed; the layouts and device are destroyed exactly once, then the `_instance`
        // field drops and destroys the instance.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
        }
    }
}

fn device_info(physical: &PhysicalDeviceRecord, memory_plan: MemoryPlan) -> DeviceInfo {
    let largest_heap = physical.memory.memory_heaps[..physical.memory.memory_heap_count as usize]
        .iter()
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    // A storage-buffer descriptor cannot exceed `maxStorageBufferRange` (128 MiB on lavapipe),
    // so no buffer may either: a larger allocation could never be bound directly.
    let max_buffer_bytes = u64::from(physical.limits.max_storage_buffer_range)
        .min(largest_heap)
        .max(1);
    DeviceInfo {
        identity: DeviceIdentity {
            uuid: physical.uuid,
            class: physical.class(),
            vendor_id: physical.vendor_id,
            device_id: physical.device_id,
        },
        capabilities: memory_plan.capabilities(),
        limits: DeviceLimits {
            max_contexts: MAX_CONTEXTS,
            max_buffers_per_context: MAX_BUFFERS_PER_CONTEXT,
            max_programs_per_context: MAX_PROGRAMS_PER_CONTEXT,
            max_queues_per_context: MAX_QUEUES_PER_CONTEXT,
            max_events_per_context: RING_DEPTH,
            max_bindings_per_submission: MAX_BINDINGS_PER_SUBMISSION,
            max_buffer_bytes,
            max_artifact_bytes: MAX_TOSA_ARTIFACT_BYTES,
        },
    }
}

/// One preallocated ring slot: claimed by exactly one live event at a time (ADR 0006).
struct Slot {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_set: vk::DescriptorSet,
}

/// Provider state of one context: the pools, the ring, and the synchronous transfer kit.
struct ContextInner {
    shared: Rc<Shared>,
    id: u64,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    slots: Vec<Slot>,
    /// Indices into `slots` not owned by a live event.
    free_slots: RefCell<Vec<u16>>,
    /// Command buffer and fence for blocking `write_buffer`/`read_buffer` staging copies.
    transfer_command_buffer: vk::CommandBuffer,
    transfer_fence: vk::Fence,
}

impl ContextInner {
    fn create(shared: &Rc<Shared>, id: u64) -> Result<Rc<Self>, BackendError> {
        let device = &shared.device;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(shared.physical.queue_family);
        // SAFETY: the device is live; `pool_info` is a live local.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|result| shared.fail(result))?;
        let mut partial = PartialContext {
            shared,
            command_pool,
            descriptor_pool: vk::DescriptorPool::null(),
            fences: Vec::new(),
        };

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(RING_DEPTH + 1);
        // SAFETY: the pool is live; the buffers are freed with the pool.
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }
            .map_err(|result| shared.fail(result))?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: RING_DEPTH * 2,
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(RING_DEPTH)
            .pool_sizes(&pool_sizes);
        // SAFETY: the device is live; `descriptor_pool_info` is a live local.
        partial.descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }
                .map_err(|result| shared.fail(result))?;
        let set_layouts = vec![shared.set_layout; RING_DEPTH as usize];
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(partial.descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: the pool was sized for exactly these sets; they are freed with the pool.
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&set_info) }
            .map_err(|result| shared.fail(result))?;

        for _ in 0..=RING_DEPTH {
            // SAFETY: the device is live; each fence is owned by this context and destroyed once.
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(|result| shared.fail(result))?;
            partial.fences.push(fence);
        }

        let (transfer_command_buffer, ring_command_buffers) = command_buffers
            .split_last()
            .expect("RING_DEPTH + 1 buffers");
        let transfer_fence = partial.fences.pop().expect("RING_DEPTH + 1 fences");
        let slots = ring_command_buffers
            .iter()
            .zip(&partial.fences)
            .zip(&descriptor_sets)
            .map(|((command_buffer, fence), descriptor_set)| Slot {
                command_buffer: *command_buffer,
                fence: *fence,
                descriptor_set: *descriptor_set,
            })
            .collect::<Vec<_>>();
        let free_slots = (0..RING_DEPTH as u16).rev().collect();
        let descriptor_pool = partial.descriptor_pool;
        // Ownership transfers to the context; the partial guard must not destroy anything now.
        partial.fences.clear();
        partial.descriptor_pool = vk::DescriptorPool::null();
        partial.command_pool = vk::CommandPool::null();
        increment(&shared.counters.contexts, 1);
        Ok(Rc::new(Self {
            shared: Rc::clone(shared),
            id,
            command_pool,
            descriptor_pool,
            slots,
            free_slots: RefCell::new(free_slots),
            transfer_command_buffer: *transfer_command_buffer,
            transfer_fence,
        }))
    }

    fn claim_slot(&self) -> Option<u16> {
        self.free_slots.borrow_mut().pop()
    }

    fn release_slot(&self, slot: u16) {
        self.free_slots.borrow_mut().push(slot);
    }

    /// Record, submit, and wait for one buffer-to-buffer copy on the transfer kit.
    fn blocking_copy(
        &self,
        source: vk::Buffer,
        destination: vk::Buffer,
        region: vk::BufferCopy,
        host_reads_destination: bool,
    ) -> Result<(), BackendError> {
        let shared = &self.shared;
        let device = &shared.device;
        let command_buffer = self.transfer_command_buffer;
        let fence = self.transfer_fence;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        let barriers = [barrier];
        let dependency = vk::DependencyInfo::default().memory_barriers(&barriers);
        let submit_buffers =
            [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&submit_buffers)];
        // SAFETY: the transfer command buffer and fence are used only by this synchronous method,
        // which waits for the fence before returning, so no prior use is still pending; both
        // buffers are live allocations of this context and the region was bounds-checked by the
        // caller. `begin_command_buffer` implicitly resets the buffer (pool flag).
        unsafe {
            device
                .reset_fences(&[fence])
                .map_err(|result| shared.fail(result))?;
            device
                .begin_command_buffer(command_buffer, &begin)
                .map_err(|result| shared.fail(result))?;
            device.cmd_copy_buffer(command_buffer, source, destination, &[region]);
            if host_reads_destination {
                device.cmd_pipeline_barrier2(command_buffer, &dependency);
            }
            device
                .end_command_buffer(command_buffer)
                .map_err(|result| shared.fail(result))?;
            device
                .queue_submit2(shared.queue, &submits, fence)
                .map_err(|result| shared.fail(result))?;
            match device.wait_for_fences(&[fence], true, TRANSFER_TIMEOUT_NS) {
                Ok(()) => Ok(()),
                Err(vk::Result::TIMEOUT) => {
                    // A bounded copy that never completes leaves the staging allocation in
                    // unknown device use: treat the device as lost rather than free it.
                    shared.poisoned.set(true);
                    Err(BackendError::DeviceLost)
                }
                Err(result) => Err(shared.fail(result)),
            }
        }
    }
}

/// Destroys partially created context objects if creation fails midway.
struct PartialContext<'a> {
    shared: &'a Shared,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    fences: Vec<vk::Fence>,
}

impl Drop for PartialContext<'_> {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: each handle here was created by `ContextInner::create` and not yet handed to a
        // context; null handles are skipped, and destroying a pool frees its allocations.
        unsafe {
            for fence in self.fences.drain(..) {
                device.destroy_fence(fence, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        let shared = &self.shared;
        // Every child holds an `Rc<ContextInner>`, so no event can still be pending here; the
        // idle wait is defense in depth for a poisoned or misused instance.
        if self.free_slots.borrow().len() != self.slots.len() {
            shared.wait_idle();
        }
        // SAFETY: the pools own their command buffers and descriptor sets; the fences were created
        // by this context. Each is destroyed exactly once.
        unsafe {
            for slot in &self.slots {
                shared.device.destroy_fence(slot.fence, None);
            }
            shared.device.destroy_fence(self.transfer_fence, None);
            shared
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            shared.device.destroy_command_pool(self.command_pool, None);
        }
        decrement(&shared.counters.contexts);
    }
}

/// Vulkan context handle: pools plus the bounded submission ring.
pub struct VulkanContext {
    inner: Rc<ContextInner>,
}

impl std::fmt::Debug for VulkanContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanContext")
            .field("id", &self.inner.id)
            .finish_non_exhaustive()
    }
}

/// In-flight gate shared between a buffer and the events bound to it.
///
/// Zero: idle. `1..EXCLUSIVE_ACCESS`: that many read-only bindings in flight. `EXCLUSIVE_ACCESS`:
/// one writing binding in flight. Explicit transfers require zero.
#[derive(Default)]
struct BufferState {
    in_flight: Cell<u64>,
}

/// One dedicated `VkBuffer` + `VkDeviceMemory`, persistently mapped unless device-local.
pub struct VulkanBuffer {
    context: Rc<ContextInner>,
    desc: BufferDesc,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<NonNull<u8>>,
    state: Rc<BufferState>,
}

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanBuffer")
            .field("context", &self.context.id)
            .field("desc", &self.desc)
            .field("mapped", &self.mapped.is_some())
            .finish_non_exhaustive()
    }
}

impl VulkanBuffer {
    fn in_flight(&self) -> u64 {
        self.state.in_flight.get()
    }

    /// Pointer to `offset` inside the persistent mapping, when the buffer is mapped at all.
    fn mapped_at(&self, offset: usize) -> Option<*mut u8> {
        // SAFETY: callers validated `offset` (plus their length) against `desc.bytes()`, and the
        // mapping covers the whole allocation.
        self.mapped
            .map(|pointer| unsafe { pointer.as_ptr().add(offset) })
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        let shared = &self.context.shared;
        // The contract forbids dropping an in-flight buffer; if it happens anyway, never free
        // memory a submission may still address.
        if self.in_flight() != 0 {
            shared.wait_idle();
        }
        // SAFETY: this handle owns the mapping, buffer, and memory, all created together in
        // `allocate` and released exactly once here, in the reverse order.
        unsafe {
            if self.mapped.is_some() {
                shared.device.unmap_memory(self.memory);
            }
            shared.device.destroy_buffer(self.buffer, None);
            shared.device.free_memory(self.memory, None);
        }
        decrement(&shared.counters.buffers);
    }
}

/// Usage flags of every buffer this backend creates: directly bindable as a storage buffer, a
/// transfer source and destination, and device-addressable when alignment can be measured.
fn buffer_usage(physical: &PhysicalDeviceRecord) -> vk::BufferUsageFlags {
    let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    if physical.buffer_device_address {
        usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    usage
}

/// The `memoryTypeBits` a buffer with this backend's usage reports on `device`.
fn probe_buffer_type_mask(
    device: &ash::Device,
    physical: &PhysicalDeviceRecord,
) -> Result<u32, vk::Result> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(4)
        .usage(buffer_usage(physical))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the device is live; the probe buffer is never bound and is destroyed here.
    unsafe {
        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        device.destroy_buffer(buffer, None);
        Ok(requirements.memory_type_bits)
    }
}

/// Owned raw buffer + memory pair used while an allocation is being assembled or staged.
struct RawAllocation<'a> {
    shared: &'a Shared,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<NonNull<u8>>,
    allocation_bytes: u64,
    /// The smallest power-of-two alignment every measured address satisfied.
    measured_alignment: u64,
    memory_flags: vk::MemoryPropertyFlags,
}

impl<'a> RawAllocation<'a> {
    /// Create a buffer, allocate dedicated memory of `memory_type`, bind at offset 0, and map it
    /// when `map` is set. Alignment is measured, never assumed.
    fn create(
        shared: &'a Shared,
        bytes: u64,
        memory_type: u32,
        map: bool,
    ) -> Result<Self, BackendError> {
        let device = &shared.device;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(buffer_usage(&shared.physical))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live and `buffer_info` is a live local.
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .map_err(|result| shared.fail(result))?;
        let mut raw = Self {
            shared,
            buffer,
            memory: vk::DeviceMemory::null(),
            mapped: None,
            allocation_bytes: 0,
            measured_alignment: 0,
            memory_flags: shared.physical.memory.memory_types[memory_type as usize].property_flags,
        };

        // SAFETY: `buffer` is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        if requirements.memory_type_bits & (1 << memory_type) == 0 {
            return Err(BackendError::Incompatible);
        }
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        if shared.physical.buffer_device_address {
            allocate_info = allocate_info.push_next(&mut flags);
        }
        // SAFETY: the device is live; the chained structures outlive the call.
        raw.memory = unsafe { device.allocate_memory(&allocate_info, None) }
            .map_err(|result| shared.fail(result))?;
        raw.allocation_bytes = requirements.size;
        // SAFETY: fresh buffer and memory; offset 0 satisfies every alignment requirement.
        unsafe { device.bind_buffer_memory(buffer, raw.memory, 0) }
            .map_err(|result| shared.fail(result))?;

        let mut alignment = u64::MAX;
        if map {
            // SAFETY: the memory is host-visible (chosen by the memory plan), unmapped, and
            // mapping the whole allocation is always in range.
            let pointer = unsafe {
                device.map_memory(raw.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .map_err(|result| shared.fail(result))?;
            let pointer = NonNull::new(pointer.cast::<u8>()).ok_or(BackendError::Incompatible)?;
            raw.mapped = Some(pointer);
            alignment = alignment.min(address_alignment(pointer.as_ptr() as u64));
        }
        if shared.physical.buffer_device_address {
            let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
            // SAFETY: the feature is enabled, the buffer carries the device-address usage, and
            // its memory was allocated with the device-address flag.
            let address = unsafe { device.get_buffer_device_address(&address_info) };
            alignment = alignment.min(address_alignment(address));
        } else if !map {
            // Nothing observable to measure: the binding requirement is the only guarantee.
            alignment = alignment.min(requirements.alignment.max(1));
        }
        raw.measured_alignment = alignment;
        Ok(raw)
    }

    /// Transfer ownership of the handles to a `VulkanBuffer`.
    fn into_parts(self) -> (vk::Buffer, vk::DeviceMemory, Option<NonNull<u8>>) {
        let parts = (self.buffer, self.memory, self.mapped);
        std::mem::forget(self);
        parts
    }
}

impl Drop for RawAllocation<'_> {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: the handles were created by `create` and are released exactly once; null
        // memory (allocation failed) is skipped by the loader-defined null-handle rule.
        unsafe {
            if self.mapped.is_some() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            if self.memory != vk::DeviceMemory::null() {
                device.free_memory(self.memory, None);
            }
        }
    }
}

/// The largest power of two dividing `address`, capped so a zero address does not overflow.
fn address_alignment(address: u64) -> u64 {
    1_u64 << address.trailing_zeros().min(40)
}

/// Bounded host-visible staging buffer for device-local transfers; one per explicit transfer.
struct Staging<'a> {
    raw: RawAllocation<'a>,
    bytes: u64,
}

impl<'a> Staging<'a> {
    fn new(shared: &'a Shared, bytes: u64) -> Result<Self, BackendError> {
        let raw = RawAllocation::create(shared, bytes, shared.memory_plan.host, true)?;
        Ok(Self { raw, bytes })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        let pointer = self.raw.mapped.expect("staging is mapped");
        // SAFETY: the mapping covers `bytes` bytes of host-coherent memory owned by `self`; the
        // GPU never accesses it while this borrow is live (every copy is waited for).
        unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr(), self.bytes as usize) }
    }
}

/// Program-side in-flight count: pipelines stay alive until every submission using them retired.
#[derive(Default)]
struct ProgramState {
    in_flight: Cell<u32>,
}

/// Resident compute pipeline specialized for one admitted TOSA graph.
pub struct VulkanProgram {
    context: Rc<ContextInner>,
    pipeline: vk::Pipeline,
    plan: ProgramPlan,
    workgroups: u32,
    state: Rc<ProgramState>,
}

impl std::fmt::Debug for VulkanProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanProgram")
            .field("context", &self.context.id)
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanProgram {
    fn drop(&mut self) {
        let shared = &self.context.shared;
        if self.state.in_flight.get() != 0 {
            shared.wait_idle();
        }
        // SAFETY: this handle owns the pipeline, created in `load_program`, destroyed once.
        unsafe { shared.device.destroy_pipeline(self.pipeline, None) };
        decrement(&shared.counters.programs);
    }
}

/// Vulkan execution queue handle. Every queue of a context feeds the device's one compute queue.
pub struct VulkanQueue {
    context: Rc<ContextInner>,
}

impl std::fmt::Debug for VulkanQueue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanQueue")
            .field("context", &self.context.id)
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanQueue {
    fn drop(&mut self) {
        decrement(&self.context.shared.counters.queues);
    }
}

/// In-flight guard for one buffer bound to one submission.
struct Guard {
    state: Rc<BufferState>,
    exclusive: bool,
}

impl Guard {
    fn acquire(state: &Rc<BufferState>, exclusive: bool) -> Result<Self, BackendError> {
        let current = state.in_flight.get();
        let next = if exclusive {
            if current != 0 {
                return Err(BackendError::Busy);
            }
            EXCLUSIVE_ACCESS
        } else {
            if current >= EXCLUSIVE_ACCESS - 1 {
                return Err(BackendError::Busy);
            }
            current + 1
        };
        state.in_flight.set(next);
        Ok(Self {
            state: Rc::clone(state),
            exclusive,
        })
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let current = self.state.in_flight.get();
        self.state.in_flight.set(if self.exclusive {
            debug_assert_eq!(current, EXCLUSIVE_ACCESS);
            0
        } else {
            debug_assert!((1..EXCLUSIVE_ACCESS).contains(&current));
            current - 1
        });
    }
}

/// One submission: a claimed ring slot, its fence, and the guards it holds until terminal.
pub struct VulkanEvent {
    context: Rc<ContextInner>,
    slot: u16,
    program: Rc<ProgramState>,
    guards: RefCell<[Option<Guard>; MAX_BINDINGS_PER_SUBMISSION as usize]>,
    latched: Cell<Option<EventState>>,
    /// Set once the slot was returned to the ring (by `destroy_event` or `Drop`).
    released: Cell<bool>,
}

impl std::fmt::Debug for VulkanEvent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanEvent")
            .field("context", &self.context.id)
            .field("slot", &self.slot)
            .field("latched", &self.latched.get())
            .finish_non_exhaustive()
    }
}

impl VulkanEvent {
    /// Publish the first terminal state. Guards are released strictly before the latch becomes
    /// observable so a caller seeing a terminal state can transfer buffer bytes immediately.
    fn latch(&self, state: EventState) -> EventState {
        for guard in self.guards.borrow_mut().iter_mut() {
            *guard = None;
        }
        let in_flight = self.program.in_flight.get();
        self.program.in_flight.set(in_flight.saturating_sub(1));
        self.latched.set(Some(state));
        state
    }

    /// Nonblocking status read of the slot's fence (ADR 0006).
    fn poll(&self) -> Result<EventState, BackendError> {
        if let Some(state) = self.latched.get() {
            return Ok(state);
        }
        let shared = &self.context.shared;
        let fence = self.context.slots[self.slot as usize].fence;
        // SAFETY: the fence belongs to this event's claimed slot and was submitted exactly once
        // since its last reset; `vkGetFenceStatus` is a read-only query.
        match unsafe { shared.device.get_fence_status(fence) } {
            Ok(true) => Ok(self.latch(EventState::Complete)),
            Ok(false) => Ok(EventState::Pending),
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                shared.poisoned.set(true);
                Ok(self.latch(EventState::Failed(BackendError::DeviceLost)))
            }
            Err(result) => Err(shared.fail(result)),
        }
    }

    fn release(&self) {
        if self.released.replace(true) {
            return;
        }
        self.context.release_slot(self.slot);
        decrement(&self.context.shared.counters.events);
    }
}

impl Drop for VulkanEvent {
    fn drop(&mut self) {
        if self.released.get() {
            return;
        }
        // Dropping a pending event outside `destroy_event` is a contract violation; still, never
        // return a slot whose command buffer may be executing: wait for its fence first.
        if self.latched.get().is_none() {
            let shared = &self.context.shared;
            let fence = self.context.slots[self.slot as usize].fence;
            // SAFETY: the fence is this slot's, submitted once; waiting has no preconditions.
            match unsafe { shared.device.wait_for_fences(&[fence], true, u64::MAX) } {
                Ok(()) => {
                    self.latch(EventState::Complete);
                }
                Err(result) => {
                    shared.fail(result);
                    self.latch(EventState::Failed(BackendError::DeviceLost));
                }
            }
        }
        self.release();
    }
}

/// Vulkan backend instance bound to one physical device.
pub struct VulkanAccelerator {
    shared: Rc<Shared>,
    next_id: Cell<u64>,
}

impl std::fmt::Debug for VulkanAccelerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanAccelerator")
            .field("device", &self.device_name())
            .finish_non_exhaustive()
    }
}

impl VulkanAccelerator {
    /// Open the preferred Vulkan 1.3 compute device: discrete, integrated, virtual, then CPU.
    pub fn new() -> Result<Self, InitError> {
        let instance = Instance::create()?;
        let devices = enumerate(&instance.instance)?;
        let physical = devices
            .into_iter()
            .min_by_key(PhysicalDeviceRecord::rank)
            .ok_or(InitError::DeviceUnavailable)?;
        Self::open(instance, physical)
    }

    /// Open the device whose enumerated name (`available_devices`) equals `device`.
    pub fn with_device(device: &str) -> Result<Self, InitError> {
        let instance = Instance::create()?;
        let physical = enumerate(&instance.instance)?
            .into_iter()
            .find(|record| record.name == device)
            .ok_or(InitError::DeviceUnavailable)?;
        Self::open(instance, physical)
    }

    /// Enumerate the names of every suitable device visible through the loader.
    pub fn available_devices() -> Result<Vec<String>, InitError> {
        let instance = Instance::create()?;
        Ok(enumerate(&instance.instance)?
            .into_iter()
            .map(|record| record.name)
            .collect())
    }

    fn open(instance: Instance, physical: PhysicalDeviceRecord) -> Result<Self, InitError> {
        Ok(Self {
            shared: Shared::open(instance, physical)?,
            next_id: Cell::new(0),
        })
    }

    /// The enumerated name of the device this instance executes on.
    pub fn device_name(&self) -> &str {
        &self.shared.physical.name
    }

    /// Whether this instance observed device loss and refuses further work.
    pub fn is_poisoned(&self) -> bool {
        self.shared.poisoned.get()
    }

    /// Cumulative count of buffers admitted as direct bindings.
    pub fn direct_binding_admissions(&self) -> u64 {
        self.shared.counters.direct_binding_admissions.get()
    }

    /// Cumulative bytes moved by explicit `write_buffer`/`read_buffer` transfers.
    pub fn explicit_transfer_bytes(&self) -> u64 {
        self.shared.counters.explicit_transfer_bytes.get()
    }

    /// Provider handles currently alive for this instance.
    pub fn live_resources(&self) -> LiveResources {
        let counters = &self.shared.counters;
        LiveResources {
            contexts: counters.contexts.get(),
            buffers: counters.buffers.get(),
            programs: counters.programs.get(),
            queues: counters.queues.get(),
            events: counters.events.get(),
        }
    }

    fn next_id(&self) -> Result<u64, BackendError> {
        let id = self.next_id.get();
        if id == u64::MAX {
            return Err(BackendError::ResourceLimit);
        }
        self.next_id.set(id + 1);
        Ok(id)
    }

    fn checked_range(
        buffer: &VulkanBuffer,
        offset: u64,
        bytes: u64,
    ) -> Result<(usize, usize), BackendError> {
        if bytes == 0 {
            return Err(BackendError::InvalidArgument);
        }
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= buffer.desc.bytes())
            .ok_or(BackendError::OutOfBounds)?;
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = usize::try_from(end).map_err(|_| BackendError::OutOfBounds)?;
        Ok((start, end))
    }

    fn lowering_error(error: LoweringError) -> BackendError {
        match error {
            LoweringError::Parse(_) | LoweringError::Analysis(_) => BackendError::InvalidArgument,
            LoweringError::UnsupportedTarget => BackendError::Incompatible,
            LoweringError::UnsupportedGraph
            | LoweringError::UnsupportedType(_)
            | LoweringError::UnsupportedOperator(_) => BackendError::Unsupported,
            LoweringError::ResourceLimit => BackendError::ResourceLimit,
        }
    }

    /// Write into a device-local buffer through a bounded staging allocation.
    fn staged_write(
        &self,
        buffer: &VulkanBuffer,
        start: u64,
        data: &dyn ByteSource,
        len: u64,
    ) -> Result<(), BackendError> {
        let shared = &self.shared;
        let mut staging = Staging::new(shared, len.min(STAGING_BYTES))?;
        let mut done = 0_u64;
        while done < len {
            let chunk = (len - done).min(staging.bytes);
            let chunk_len = usize::try_from(chunk).map_err(|_| BackendError::OutOfBounds)?;
            data.read_at(done, &mut staging.as_mut_slice()[..chunk_len])?;
            buffer.context.blocking_copy(
                staging.raw.buffer,
                buffer.buffer,
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: start + done,
                    size: chunk,
                },
                false,
            )?;
            increment(&shared.counters.explicit_transfer_bytes, chunk);
            done += chunk;
        }
        Ok(())
    }

    /// Read out of a device-local buffer through a bounded staging allocation.
    fn staged_read(
        &self,
        buffer: &VulkanBuffer,
        start: u64,
        data: &mut dyn ByteSink,
        len: u64,
    ) -> Result<(), BackendError> {
        let shared = &self.shared;
        let mut staging = Staging::new(shared, len.min(STAGING_BYTES))?;
        let mut done = 0_u64;
        while done < len {
            let chunk = (len - done).min(staging.bytes);
            let chunk_len = usize::try_from(chunk).map_err(|_| BackendError::OutOfBounds)?;
            buffer.context.blocking_copy(
                buffer.buffer,
                staging.raw.buffer,
                vk::BufferCopy {
                    src_offset: start + done,
                    dst_offset: 0,
                    size: chunk,
                },
                true,
            )?;
            data.write_at(done, &staging.as_mut_slice()[..chunk_len])?;
            increment(&shared.counters.explicit_transfer_bytes, chunk);
            done += chunk;
        }
        Ok(())
    }

    /// Record the dispatch for one claimed slot and submit it with the slot's fence.
    fn record_and_submit(
        &self,
        slot: &Slot,
        program: &VulkanProgram,
        descriptors: &[vk::DescriptorBufferInfo],
    ) -> Result<(), vk::Result> {
        let shared = &self.shared;
        let device = &shared.device;
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(slot.descriptor_set)
                .dst_binding(shader::INPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptors[..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(slot.descriptor_set)
                .dst_binding(shader::OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptors[1..2]),
        ];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // Make the shader's storage writes visible to host reads after the fence signals.
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        let barriers = [barrier];
        let dependency = vk::DependencyInfo::default().memory_barriers(&barriers);
        let submit_buffers =
            [vk::CommandBufferSubmitInfo::default().command_buffer(slot.command_buffer)];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&submit_buffers)];
        // SAFETY: the slot is free (no submission references its command buffer, fence, or
        // descriptor set), the descriptor infos name live buffers whose ranges were validated,
        // the pipeline is live and in-flight-counted by the caller, and the pool flag lets
        // `begin_command_buffer` reset the buffer implicitly. Host writes made before this
        // submission are visible to the device by the implicit host-write ordering guarantee.
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
            device.reset_fences(&[slot.fence])?;
            device.begin_command_buffer(slot.command_buffer, &begin)?;
            device.cmd_bind_pipeline(
                slot.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                program.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                slot.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                shared.pipeline_layout,
                0,
                &[slot.descriptor_set],
                &[],
            );
            device.cmd_dispatch(slot.command_buffer, program.workgroups, 1, 1);
            device.cmd_pipeline_barrier2(slot.command_buffer, &dependency);
            device.end_command_buffer(slot.command_buffer)?;
            device.queue_submit2(shared.queue, &submits, slot.fence)
        }
    }
}

impl TosaCapabilityProvider for VulkanAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        crate::TOSA_CAPABILITIES
    }
}

impl Accelerator for VulkanAccelerator {
    type Context = VulkanContext;
    type Buffer = VulkanBuffer;
    type Program = VulkanProgram;
    type Queue = VulkanQueue;
    type Event = VulkanEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(self.shared.info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.shared.info.validate_context_desc(desc)?;
        self.shared.check_live()?;
        if self.shared.counters.contexts.get() >= u64::from(MAX_CONTEXTS) {
            return Err(BackendError::ResourceLimit);
        }
        let inner = ContextInner::create(&self.shared, self.next_id()?)?;
        Ok(VulkanContext { inner })
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        if Rc::strong_count(&context.inner) > 1 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: context,
            });
        }
        Ok(())
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        let shared = &self.shared;
        shared.info.validate_buffer_desc(desc)?;
        shared.check_live()?;
        if shared.counters.buffers.get()
            >= u64::from(MAX_BUFFERS_PER_CONTEXT) * u64::from(MAX_CONTEXTS)
        {
            return Err(BackendError::ResourceLimit);
        }
        let memory_type = shared
            .memory_plan
            .for_domain(desc.domain)
            .ok_or(BackendError::Unsupported)?;
        let map = desc.domain != MemoryDomain::Device;
        let raw = RawAllocation::create(shared, desc.bytes(), memory_type, map)?;
        if raw.measured_alignment < desc.alignment() {
            return Err(BackendError::ResourceLimit);
        }
        let mut properties = BufferProperties::DIRECT_BINDING;
        if raw.mapped.is_some() {
            properties |= BufferProperties::HOST_VISIBLE;
        }
        if raw
            .memory_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            properties |= BufferProperties::DEVICE_LOCAL;
        }
        let info = BufferInfo::new(
            desc,
            raw.allocation_bytes,
            raw.measured_alignment,
            properties,
        )?;
        let (buffer, memory, mapped) = raw.into_parts();
        increment(&shared.counters.buffers, 1);
        Ok(AllocatedBuffer::new(
            VulkanBuffer {
                context: Rc::clone(&context.inner),
                desc,
                buffer,
                memory,
                mapped,
                state: Rc::new(BufferState::default()),
            },
            info,
        ))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        if !buffer
            .desc
            .usage
            .contains(BufferUsage::TRANSFER_DESTINATION)
        {
            return Err(BackendError::PermissionDenied);
        }
        if buffer.in_flight() != 0 {
            return Err(BackendError::Busy);
        }
        self.shared.check_live()?;
        let (start, end) = Self::checked_range(buffer, offset, data.len())?;
        let len = end - start;
        let Some(target) = buffer.mapped_at(start) else {
            return self.staged_write(buffer, start as u64, data, len as u64);
        };
        // SAFETY: `target..target + len` is inside the persistent host-coherent mapping of a
        // buffer that is exclusively borrowed and not in flight; the source is a distinct
        // borrowed region. Coherent memory needs no flush.
        let target = unsafe { std::slice::from_raw_parts_mut(target, len) };
        match data.as_contiguous() {
            Some(source) if source.len() == len => target.copy_from_slice(source),
            Some(_) => return Err(BackendError::InvalidArgument),
            None => data.read_at(0, target)?,
        }
        increment(&self.shared.counters.explicit_transfer_bytes, len as u64);
        Ok(())
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        if !buffer.desc.usage.contains(BufferUsage::TRANSFER_SOURCE) {
            return Err(BackendError::PermissionDenied);
        }
        if buffer.in_flight() != 0 {
            return Err(BackendError::Busy);
        }
        self.shared.check_live()?;
        let (start, end) = Self::checked_range(buffer, offset, data.len())?;
        let len = end - start;
        let Some(source) = buffer.mapped_at(start) else {
            return self.staged_read(buffer, start as u64, data, len as u64);
        };
        // SAFETY: the range is inside the mapping and the in-flight gate proved no submission
        // still writes this buffer; every completed submission's writes were made host-visible
        // by its command buffer's barrier before its fence signaled.
        let source = unsafe { std::slice::from_raw_parts(source.cast_const(), len) };
        match data.as_contiguous_mut() {
            Some(target) if target.len() == len => target.copy_from_slice(source),
            Some(_) => return Err(BackendError::InvalidArgument),
            None => data.write_at(0, source)?,
        }
        increment(&self.shared.counters.explicit_transfer_bytes, len as u64);
        Ok(())
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        if buffer.in_flight() != 0 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: buffer,
            });
        }
        Ok(())
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        let shared = &self.shared;
        if artifact.payload.len() > shared.info.limits.max_artifact_bytes {
            return Err(BackendError::ResourceLimit);
        }
        if artifact.resident_bytes != REQUIRED_RESIDENT_BYTES {
            return Err(BackendError::ResourceLimit);
        }
        if artifact.format != virtio_accel_tosa::ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        let target = virtio_accel_tosa::Target::from_identity(artifact.target)
            .map_err(|_| BackendError::Incompatible)?;
        shared.check_live()?;
        if shared.counters.programs.get()
            >= u64::from(MAX_PROGRAMS_PER_CONTEXT) * u64::from(MAX_CONTEXTS)
        {
            return Err(BackendError::ResourceLimit);
        }
        let mut owned = Vec::new();
        let bytes = match artifact.payload.as_contiguous() {
            Some(bytes) => bytes,
            None => {
                let len = usize::try_from(artifact.payload.len())
                    .map_err(|_| BackendError::ResourceLimit)?;
                owned
                    .try_reserve_exact(len)
                    .map_err(|_| BackendError::OutOfMemory)?;
                owned.resize(len, 0);
                artifact.payload.read_at(0, &mut owned)?;
                &owned
            }
        };
        let plan = lower_tosa(bytes, target).map_err(Self::lowering_error)?;
        let workgroups = shader::elementwise_workgroups(plan.element_count);
        if workgroups > shared.physical.limits.max_compute_work_group_count[0] {
            return Err(BackendError::ResourceLimit);
        }
        let code = match plan.kernel {
            Kernel::CopyU32 => shader::copy_u32_spirv(),
        };

        let device = &shared.device;
        let module_info = vk::ShaderModuleCreateInfo::default().code(code);
        // SAFETY: `code` is the crate-authored SPIR-V module, live for the call.
        let module = unsafe { device.create_shader_module(&module_info, None) }
            .map_err(|result| shared.fail(result))?;
        let element_count = plan.element_count.to_ne_bytes();
        let entries = [vk::SpecializationMapEntry {
            constant_id: shader::ELEMENT_COUNT_SPEC_ID,
            offset: 0,
            size: std::mem::size_of::<u32>(),
        }];
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&element_count);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&specialization);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(shared.pipeline_layout);
        // SAFETY: module and layout are live; every pointed-to structure outlives the call. On
        // failure ash returns the partially created array, which holds no live pipeline for a
        // single-entry request but is destroyed defensively.
        let created = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };
        // SAFETY: the module is no longer needed once pipeline creation returned.
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = match created {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, result)) => {
                for pipeline in pipelines {
                    if pipeline != vk::Pipeline::null() {
                        // SAFETY: a non-null pipeline handed back on failure is owned by us.
                        unsafe { device.destroy_pipeline(pipeline, None) };
                    }
                }
                return Err(shared.fail(result));
            }
        };
        increment(&shared.counters.programs, 1);
        Ok(VulkanProgram {
            context: Rc::clone(&context.inner),
            pipeline,
            plan,
            workgroups,
            state: Rc::new(ProgramState::default()),
        })
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        if program.state.in_flight.get() != 0 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: program,
            });
        }
        Ok(())
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.shared.info.validate_queue_desc(desc)?;
        self.shared.check_live()?;
        if self.shared.counters.queues.get()
            >= u64::from(MAX_QUEUES_PER_CONTEXT) * u64::from(MAX_CONTEXTS)
        {
            return Err(BackendError::ResourceLimit);
        }
        increment(&self.shared.counters.queues, 1);
        Ok(VulkanQueue {
            context: Rc::clone(&context.inner),
        })
    }

    fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        Ok(())
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let shared = &self.shared;
        let reject = SubmitFailure::Rejected;
        shared.check_live().map_err(reject)?;
        // Vulkan has no cancel primitive, so a finite deadline is refused before admission rather
        // than latched against retained resources (ADR 0006).
        if let Timeout::AfterNs(_) = timeout {
            return Err(reject(BackendError::DeadlineExpired));
        }
        if bindings.is_empty() || bindings.len() > MAX_BINDINGS_PER_SUBMISSION as usize {
            return Err(reject(BackendError::ResourceLimit));
        }
        if !Rc::ptr_eq(&queue.context, &program.context) {
            return Err(reject(BackendError::InvalidArgument));
        }
        let context = &queue.context;
        let plan = &program.plan;

        // Per-binding reasons (bounds, access, slot) are reported before the aggregate count
        // check so a host learns the most specific rejection first.
        let offset_alignment = shared.physical.limits.min_storage_buffer_offset_alignment;
        let mut descriptors =
            [vk::DescriptorBufferInfo::default(); MAX_BINDINGS_PER_SUBMISSION as usize];
        let mut seen = 0_u32;
        for binding in bindings {
            if !Rc::ptr_eq(&binding.buffer.context, context) {
                return Err(reject(BackendError::InvalidArgument));
            }
            if !binding.buffer.desc.allows_access(binding.access) {
                return Err(reject(BackendError::PermissionDenied));
            }
            let (start, _) =
                Self::checked_range(binding.buffer, binding.range.offset, binding.range.bytes())
                    .map_err(reject)?;
            let index = plan
                .slots
                .iter()
                .position(|slot| slot.slot == binding.slot)
                .ok_or(reject(BackendError::Incompatible))?;
            if seen & (1 << index) != 0 {
                return Err(reject(BackendError::InvalidArgument));
            }
            seen |= 1 << index;
            let slot_plan = &plan.slots[index];
            let expected_access = match slot_plan.role {
                SlotRole::Input => AccessMode::Read,
                SlotRole::Output => AccessMode::Write,
            };
            if binding.access != expected_access {
                return Err(reject(BackendError::Incompatible));
            }
            // The descriptor covers the range directly: exact tensor bytes, scalar- and
            // `minStorageBufferOffsetAlignment`-aligned start.
            if binding.range.bytes() != slot_plan.byte_len
                || (start as u64) % slot_plan.scalar_bytes.max(offset_alignment) != 0
            {
                return Err(reject(BackendError::Incompatible));
            }
            descriptors[index] = vk::DescriptorBufferInfo {
                buffer: binding.buffer.buffer,
                offset: start as u64,
                range: binding.range.bytes(),
            };
        }
        if bindings.len() != plan.slots.len() {
            return Err(reject(BackendError::Incompatible));
        }

        let slot_index = context
            .claim_slot()
            .ok_or(reject(BackendError::ResourceLimit))?;
        let mut guards: [Option<Guard>; MAX_BINDINGS_PER_SUBMISSION as usize] =
            [const { None }; MAX_BINDINGS_PER_SUBMISSION as usize];
        for (index, binding) in bindings.iter().enumerate() {
            let exclusive = binding.access != AccessMode::Read;
            // The same buffer bound twice (an input feeding an output slot) is rejected above by
            // the role check, so each guard is a distinct allocation.
            match Guard::acquire(&binding.buffer.state, exclusive) {
                Ok(guard) => guards[index] = Some(guard),
                Err(error) => {
                    drop(guards);
                    context.release_slot(slot_index);
                    return Err(reject(error));
                }
            }
        }

        let slot = &context.slots[slot_index as usize];
        match self.record_and_submit(slot, program, &descriptors[..plan.slots.len()]) {
            Ok(()) => {}
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                // Past the admission boundary with an ambiguous outcome: the event owns the slot
                // and latches the loss; the instance is poisoned (ADR 0006).
                shared.poisoned.set(true);
                program
                    .state
                    .in_flight
                    .set(program.state.in_flight.get() + 1);
                increment(&shared.counters.events, 1);
                let event = VulkanEvent {
                    context: Rc::clone(context),
                    slot: slot_index,
                    program: Rc::clone(&program.state),
                    guards: RefCell::new(guards),
                    latched: Cell::new(None),
                    released: Cell::new(false),
                };
                event.latch(EventState::Failed(BackendError::DeviceLost));
                return Err(SubmitFailure::Indeterminate {
                    error: BackendError::DeviceLost,
                    event,
                });
            }
            Err(result) => {
                // Recording and submission failures before the queue accepted the work leave
                // every resource untouched (Vulkan guarantees this for out-of-memory results).
                drop(guards);
                context.release_slot(slot_index);
                return Err(reject(shared.fail(result)));
            }
        }
        program
            .state
            .in_flight
            .set(program.state.in_flight.get() + 1);
        increment(
            &shared.counters.direct_binding_admissions,
            bindings.len() as u64,
        );
        increment(&shared.counters.events, 1);
        Ok(VulkanEvent {
            context: Rc::clone(context),
            slot: slot_index,
            program: Rc::clone(&program.state),
            guards: RefCell::new(guards),
            latched: Cell::new(None),
            released: Cell::new(false),
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        event.poll()
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        match event.poll() {
            Ok(EventState::Pending) => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            }),
            Ok(_) => {
                event.release();
                Ok(())
            }
            Err(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: event,
            }),
        }
    }
}
