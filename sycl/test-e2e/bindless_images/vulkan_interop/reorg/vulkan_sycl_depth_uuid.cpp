/*
  Vulkan/SYCL Depth Test - UUID MATCHED

  clang++ -fsycl  -o vsdu.bin vulkan_sycl_depth_uuid.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  clang++ -fsycl  -o vsdu.exe vulkan_sycl_depth_uuid.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib
  
  Fixes:
  - Finds the specific Vulkan Physical Device that matches the SYCL Device UUID.
  - Prevents accidental cross-adapter sharing (iGPU <-> dGPU) which causes DEVICE_LOST.
*/
/*
  Vulkan/SYCL Depth Test - UUID MATCHED (Fixed)
  
  Fixes:
  - Macro shadowing bug resolved.
  - Matches Vulkan Physical Device to SYCL Device via UUID.
  - Prevents accidental cross-adapter sharing (iGPU <-> dGPU).
*/

#include "test_verification.hpp"
#include <vulkan/vulkan.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
#include <vulkan/vulkan_core.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif

// FIXED MACRO: Renamed 'res' to '__vkRes' to avoid shadowing outer variables
#define VK_CHECK(f) { VkResult __vkRes = (f); if (__vkRes != VK_SUCCESS) { std::cerr << "Vulkan Error: " << __vkRes << std::endl; exit(1); } }

namespace syclexp = sycl::ext::oneapi::experimental;

struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    uint32_t queueFamilyIndex;
};

struct ImageResources {
    VkImage image;
    VkDeviceMemory memory;
    VkDeviceSize allocationSize;
    VkExtent3D extent;
};

// ---------------------------------------------------------
// UUID MATCHING LOGIC
// ---------------------------------------------------------
inline VulkanContext createUUIDMatchedContext(const sycl::device& syclDev) {
    VulkanContext ctx;

    // 1. Create Instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> instanceExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
    };

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = (uint32_t)instanceExts.size();
    createInfo.ppEnabledExtensionNames = instanceExts.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &ctx.instance));

    // 2. Get SYCL UUID
    auto syclUUID = syclDev.get_info<sycl::ext::intel::info::device::uuid>();

    // 3. Find Matching Vulkan Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());

    ctx.physicalDevice = VK_NULL_HANDLE;
    std::cout << "[Setup] Scanning " << deviceCount << " Vulkan Devices..." << std::endl;

    for (const auto& dev : devices) {
        VkPhysicalDeviceIDProperties idProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(dev, &props2);

        // Compare UUIDs (VK_UUID_SIZE is 16)
        if (std::memcmp(idProps.deviceUUID, syclUUID.data(), VK_UUID_SIZE) == 0) {
            ctx.physicalDevice = dev;
            std::cout << "[Setup] MATCH FOUND: " << props2.properties.deviceName << std::endl;
            break;
        } else {
             std::cout << "[Setup] Skipping: " << props2.properties.deviceName << " (UUID Mismatch)" << std::endl;
        }
    }

    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "[Error] No Vulkan device found matching the SYCL device UUID!" << std::endl;
        exit(1);
    }

    // 4. Create Logical Device
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &queueFamilyCount, queueFamilies.data());

    ctx.queueFamilyIndex = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            ctx.queueFamilyIndex = i;
            break;
        }
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    std::vector<const char*> deviceExts = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        #ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
        #else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
        #endif
    };

    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.queueCreateInfoCount = 1;
    devInfo.enabledExtensionCount = (uint32_t)deviceExts.size();
    devInfo.ppEnabledExtensionNames = deviceExts.data();

    VK_CHECK(vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);

    return ctx;
}

// ---------------------------------------------------------
// VULKAN HELPERS
// ---------------------------------------------------------
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

ImageResources createDepthImage(VulkanContext& ctx, uint32_t w, uint32_t h) {
    ImageResources res;
    res.extent = {w, h, 1};

    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = res.extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkExternalMemoryImageCreateInfo extMemInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extMemInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    imageInfo.pNext = &extMemInfo;

    VK_CHECK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(ctx.device, res.image, &memRequirements);

    VkExportMemoryAllocateInfo exportAllocInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportAllocInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &exportAllocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    res.allocationSize = allocInfo.allocationSize;
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, res.image, res.memory, 0));
    return res;
}

// ---------------------------------------------------------
// PLATFORM GETTERS
// ---------------------------------------------------------
#ifdef _WIN32
HANDLE getMemHandle(VulkanContext& ctx, VkDeviceMemory memory) {
    VkMemoryGetWin32HandleInfoKHR getHandleInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    getHandleInfo.memory = memory;
    getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle;
    auto func = (PFN_vkGetMemoryWin32HandleKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryWin32HandleKHR");
    func(ctx.device, &getHandleInfo, &handle);
    return handle;
}
#else
int getMemFd(VulkanContext& ctx, VkDeviceMemory memory) {
    VkMemoryGetFdInfoKHR getFdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    getFdInfo.memory = memory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd;
    auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
    func(ctx.device, &getFdInfo, &fd);
    return fd;
}
#endif

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    int width = 64;
    int height = 64;

    try {
        // 1. Initialize SYCL First (This determines which GPU we MUST use)
        sycl::queue q;
        auto dev = q.get_device();
        auto ctx = q.get_context();
        std::cout << "[SYCL] Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

        // 2. Initialize Vulkan to match SYCL
        VulkanContext vkCtx = createUUIDMatchedContext(dev);

        // 3. Create Resources
        ImageResources inImg = createDepthImage(vkCtx, width, height);
        ImageResources outImg = createDepthImage(vkCtx, width, height);

        // 4. Transitions (Undefined -> General)
        {
            VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
            VkCommandPool pool; vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
            
            VkCommandBufferAllocateInfo alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            alloc.commandPool = pool; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
            VkCommandBuffer cmd; vkAllocateCommandBuffers(vkCtx.device, &alloc, &cmd);

            VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cmd, &begin);

            VkImageMemoryBarrier bar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bar.image = inImg.image;
            bar.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
            bar.image = outImg.image;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);

            vkEndCommandBuffer(cmd);
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            vkQueueSubmit(vkCtx.queue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(vkCtx.queue);
            vkDestroyCommandPool(vkCtx.device, pool, nullptr);
        }

        // 5. Import & Run SYCL
        // Use Packed Size logic from Legacy test (width*height*4)
        size_t packedSize = width * height * sizeof(float);
        
        #ifdef _WIN32
        auto inDesc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, inImg.memory), syclexp::external_mem_handle_type::win32_nt_handle, packedSize};
        auto outDesc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, outImg.memory), syclexp::external_mem_handle_type::win32_nt_handle, packedSize};
        #else
        auto inDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, inImg.memory), syclexp::external_mem_handle_type::opaque_fd, packedSize};
        auto outDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, outImg.memory), syclexp::external_mem_handle_type::opaque_fd, packedSize};
        #endif

        auto inExtMem = syclexp::import_external_memory(inDesc, q);
        auto outExtMem = syclexp::import_external_memory(outDesc, q);

        syclexp::image_descriptor imgDesc({(size_t)width, (size_t)height}, 1, sycl::image_channel_type::fp32);
        
        auto inH = syclexp::map_external_image_memory(inExtMem, imgDesc, q);
        auto outH = syclexp::map_external_image_memory(outExtMem, imgDesc, q);

        auto inImgObj = syclexp::create_image(inH, imgDesc, q);
        auto outImgObj = syclexp::create_image(outH, imgDesc, q);

        std::cout << "[SYCL] Submitting Kernel..." << std::endl;
        q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item){
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                float val = syclexp::fetch_image<float>(inImgObj, sycl::int2(x, y));
                syclexp::write_image<float>(outImgObj, sycl::int2(x, y), val);
            });
        }).wait();

        std::cout << "SUCCESS! (No Crash)" << std::endl;

        syclexp::destroy_image_handle(inImgObj, q);
        syclexp::destroy_image_handle(outImgObj, q);
        syclexp::release_external_memory(inExtMem, q);
        syclexp::release_external_memory(outExtMem, q);

    } catch (std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}