//! Vulkan instance / device / queue plumbing.
//!
//! This is the bottom of the renderer: load libvulkan dynamically, create a
//! Vulkan 1.3 instance, pick a physical device that supports the dmabuf-import
//! extensions we need to talk to GBM, create a logical device with one
//! graphics queue, and build a long-lived command pool. Everything above
//! this module assumes those handles already exist.
//!
//! Required device extensions:
//! - `VK_KHR_external_memory_fd` — pull dmabuf FDs out of `VkDeviceMemory` /
//!   push them in.
//! - `VK_KHR_external_semaphore_fd` — pull a sync-FD out of a `VkSemaphore`,
//!   which is how we hand the GPU-completion fence to calloop so the page
//!   flip can defer until rendering's done.
//! - `VK_EXT_external_memory_dma_buf` — the "type tag" for dmabufs in the
//!   external-memory-fd machinery.
//! - `VK_EXT_image_drm_format_modifier` — let us create `VkImage`s that
//!   share a DRM format modifier with the GBM-allocated scanout BO they
//!   wrap. Required for any non-`LINEAR` modifier; we still use it on
//!   linear because the import path needs the explicit modifier list.
//!
//! Vulkan 1.3 core gives us `dynamic_rendering` and `synchronization2`, so
//! they don't need explicit extension entries.

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// Long-lived Vulkan handles. Cloning is cheap (handles are `Copy` POD)
/// but we wrap in `Arc` to make ownership across modules explicit and to
/// guarantee a single point of teardown — `Drop` calls `vkDestroyDevice` /
/// `vkDestroyInstance` exactly once.
///
/// A handful of fields (`entry`, `physical`, `queue_family`, `limits`,
/// `ext_drm_mod`) aren't read in the steady-state hot path but are kept
/// alive: `entry` keeps libvulkan loaded for the device's lifetime,
/// `physical` is wanted for any future cap-query path, and the others
/// round out the API the modules below this one will pick up as we
/// extend the renderer.
#[allow(dead_code)]
pub struct VkStack {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Cached limits for things we look up frequently (push constant
    /// size, alignment requirements).
    pub limits: vk::PhysicalDeviceLimits,
    /// `VK_KHR_external_memory_fd` device-level functions. Kept on the
    /// stack because the texture cache and the GBM swap both need
    /// `vkGetMemoryFdKHR` / `vkImportMemoryFdKHR` paths.
    pub ext_mem_fd: ash::khr::external_memory_fd::Device,
    /// `VK_KHR_external_semaphore_fd` device-level functions for the
    /// sync-FD export path.
    pub ext_sem_fd: ash::khr::external_semaphore_fd::Device,
    /// `VK_EXT_image_drm_format_modifier` device-level functions —
    /// needed by the GBM swap to query the modifier picked by the
    /// driver after image creation.
    pub ext_drm_mod: ash::ext::image_drm_format_modifier::Device,
}

/// Owned wrapper that releases vk resources in the right order on drop.
/// Cloning produces an Arc-shared view; only the last clone runs Drop.
#[derive(Clone)]
pub struct SharedStack(pub Arc<VkStack>);

impl std::ops::Deref for SharedStack {
    type Target = VkStack;
    fn deref(&self) -> &VkStack {
        &self.0
    }
}

impl VkStack {
    pub fn new() -> Result<SharedStack> {
        // Dynamically load libvulkan.so.1. The user is expected to have
        // a vulkan-icd-loader installed (Arch's `vulkan-icd-loader`,
        // Debian's `libvulkan1`); the actual ICD (mesa-vulkan-radv,
        // mesa-vulkan-intel, nvidia-utils, etc.) is whatever
        // libvulkan.so.1 enumerates at runtime.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| anyhow!("load libvulkan.so.1: {e}"))?;

        let app_name = CString::new("sol")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let instance_create = vk::InstanceCreateInfo::default()
            .application_info(&app_info);
        let instance = unsafe {
            entry
                .create_instance(&instance_create, None)
                .context("create_instance")?
        };

        let (physical, queue_family) = pick_physical(&instance)
            .context("pick physical device")?;

        let mem_props =
            unsafe { instance.get_physical_device_memory_properties(physical) };
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let limits = props.limits;
        let device_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        tracing::info!(
            device = %device_name,
            api_version = format!(
                "{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
            ),
            queue_family,
            "Vulkan physical device selected"
        );

        // Logical device with one graphics queue.
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)];

        let device_extensions = [
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::khr::external_semaphore_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::ext::image_drm_format_modifier::NAME.as_ptr(),
            // Exposes VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT
            // queries through Properties2; required by image_drm
            // import path on some drivers.
            ash::khr::external_memory::NAME.as_ptr(),
            ash::khr::external_semaphore::NAME.as_ptr(),
        ];

        // Vulkan 1.3 core toggles. Both required for the dynamic-render
        // + barrier2 paths the presenter uses.
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        // Timeline semaphores aren't strictly required (we use binary
        // sync FDs), but enabling 1.2 features rings clean asserts on
        // some validation layers.
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let device = unsafe {
            instance
                .create_device(physical, &create_info, None)
                .context("create_device")?
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("create_command_pool")?
        };

        let ext_mem_fd =
            ash::khr::external_memory_fd::Device::new(&instance, &device);
        let ext_sem_fd =
            ash::khr::external_semaphore_fd::Device::new(&instance, &device);
        let ext_drm_mod =
            ash::ext::image_drm_format_modifier::Device::new(&instance, &device);

        Ok(SharedStack(Arc::new(Self {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            command_pool,
            mem_props,
            limits,
            ext_mem_fd,
            ext_sem_fd,
            ext_drm_mod,
        })))
    }

    /// Find a memory type index for an allocation given the bits the
    /// driver advertised in `VkMemoryRequirements::memoryTypeBits` and
    /// the property flags we want (e.g. `DEVICE_LOCAL`,
    /// `HOST_VISIBLE | HOST_COHERENT`). Returns the first match.
    pub fn find_memtype(
        &self,
        type_bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for i in 0..self.mem_props.memory_type_count {
            let bit = 1u32 << i;
            if (type_bits & bit) != 0
                && self.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
            {
                return Ok(i);
            }
        }
        bail!(
            "no memory type matches bits=0x{type_bits:x} flags=0x{:x}",
            flags.as_raw()
        )
    }
}

impl Drop for VkStack {
    fn drop(&mut self) {
        unsafe {
            // No queue idle wait here — by the time the Arc that owns
            // this hits zero, callers (presenter, swap, texture cache)
            // have already destroyed everything they allocated and
            // implicitly drained the queue via vkDeviceWaitIdle in
            // their own drops. Doing it again would just be an extra
            // syscall on shutdown.
            self.device
                .destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    let physicals = unsafe {
        instance
            .enumerate_physical_devices()
            .context("enumerate_physical_devices")?
    };
    if physicals.is_empty() {
        bail!("no Vulkan physical devices on this system");
    }

    let required_exts = [
        ash::khr::external_memory_fd::NAME,
        ash::khr::external_semaphore_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::image_drm_format_modifier::NAME,
    ];

    // Two-pass: prefer DISCRETE_GPU on the first pass, fall back to
    // INTEGRATED_GPU on the second. Avoids picking llvmpipe (CPU
    // software rasterizer) when a real GPU is present.
    for prefer_discrete in [true, false] {
        for &p in &physicals {
            let props = unsafe { instance.get_physical_device_properties(p) };
            let is_discrete = matches!(
                props.device_type,
                vk::PhysicalDeviceType::DISCRETE_GPU
            );
            if prefer_discrete && !is_discrete {
                continue;
            }
            // Reject software/CPU devices outright; the comp's whole
            // point is hardware acceleration.
            if matches!(props.device_type, vk::PhysicalDeviceType::CPU) {
                continue;
            }
            // Required extensions present?
            let avail = unsafe {
                match instance.enumerate_device_extension_properties(p) {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            };
            let has_all = required_exts.iter().all(|need| {
                avail.iter().any(|e| {
                    let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                    name == *need
                })
            });
            if !has_all {
                continue;
            }
            // Need a graphics queue family.
            let families = unsafe {
                instance.get_physical_device_queue_family_properties(p)
            };
            let q = families.iter().enumerate().find(|(_, f)| {
                f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });
            let Some((idx, _)) = q else {
                continue;
            };
            return Ok((p, idx as u32));
        }
    }

    bail!(
        "no Vulkan physical device exposes the required extensions \
         (VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd, \
         VK_EXT_external_memory_dma_buf, VK_EXT_image_drm_format_modifier). \
         Install mesa-vulkan drivers (Arch: pacman -S vulkan-radeon vulkan-intel \
         vulkan-mesa-layers, or nvidia-utils for NVIDIA)."
    )
}
