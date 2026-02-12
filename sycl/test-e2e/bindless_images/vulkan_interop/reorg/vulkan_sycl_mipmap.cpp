// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_mipmap
// REQUIRES: vulkan

// XFAIL: linux
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21131

// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21131

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out


/*
  Vulkan/SYCL Mipmap Interop Test
 

  clang++ -fsycl  -o vsmm.bin vulkan_sycl_mipmap.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  clang++ -fsycl  -o vsmm.exe vulkan_sycl_mipmap.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  
  Features:
  - Creates a 2D Image with multiple Mip Levels.
  - Uploads distinct patterns to Level 0 and Level 1.
  - SYCL Kernel samples both levels explicitly using `sample_mipmap`.
  - Verifies the blending of levels.
  - Uses UUID Matching to ensure correct Device selection.
  
  Usage:
    ./vulkan_sycl_mipmap.bin
    ./vulkan_sycl_mipmap.bin --semaphores
*/

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "test_verification.hpp"
#include <vulkan/vulkan.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
#include <vulkan/vulkan_core.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif

#define VK_CHECK(f) { VkResult __res = (f); if(__res != VK_SUCCESS) { std::cerr << "Vulkan Error: " << __res << std::endl; exit(1); } }

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// STRUCTS & CONTEXT
// ---------------------------------------------------------
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
    uint32_t mipLevels;
};

// ---------------------------------------------------------
// UUID MATCHING (The Crash Fixer)
// ---------------------------------------------------------
inline VulkanContext createUUIDMatchedContext(const sycl::device& syclDev) {
    VulkanContext ctx;
    
    // 1. Instance
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

    // 2. UUID Match
    auto syclUUID = syclDev.get_info<sycl::ext::intel::info::device::uuid>();
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data());

    ctx.physicalDevice = VK_NULL_HANDLE;
    for (const auto& dev : devices) {
        VkPhysicalDeviceIDProperties idProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(dev, &props2);
        if (std::memcmp(idProps.deviceUUID, syclUUID.data(), VK_UUID_SIZE) == 0) {
            ctx.physicalDevice = dev;
            break;
        }
    }
    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "FATAL: No matching Vulkan device found for SYCL device!" << std::endl;
        exit(1);
    }

    // 3. Logical Device
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qCount, qProps.data());
    
    ctx.queueFamilyIndex = -1;
    for(uint32_t i=0; i<qCount; ++i) {
        if(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { ctx.queueFamilyIndex = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &prio;

    std::vector<const char*> devExts = {
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
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.queueCreateInfoCount = 1;
    devInfo.enabledExtensionCount = (uint32_t)devExts.size();
    devInfo.ppEnabledExtensionNames = devExts.data();

    VK_CHECK(vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);
    
    return ctx;
}

// ---------------------------------------------------------
// MIPMAP IMAGE CREATION
// ---------------------------------------------------------
uint32_t findMemType(VkPhysicalDevice dev, uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(dev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

ImageResources createMipImage(VulkanContext& ctx, uint32_t w, uint32_t h, uint32_t mips) {
    ImageResources res;
    res.extent = {w, h, 1};
    res.mipLevels = mips;

    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = res.extent;
    imageInfo.mipLevels = mips;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT; // Standard RGBA32F
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkExternalMemoryImageCreateInfo extMemInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extMemInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    imageInfo.pNext = &extMemInfo;

    VK_CHECK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image));

    // Dedicated Allocation Logic
    VkImageMemoryRequirementsInfo2 reqInfo = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    reqInfo.image = res.image;
    VkMemoryDedicatedRequirements dedReq = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 req2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    req2.pNext = &dedReq;
    vkGetImageMemoryRequirements2(ctx.device, &reqInfo, &req2);

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = req2.memoryRequirements.size;
    allocInfo.memoryTypeIndex = findMemType(ctx.physicalDevice, req2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkExportMemoryAllocateInfo expAlloc = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    expAlloc.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    allocInfo.pNext = &expAlloc;

    VkMemoryDedicatedAllocateInfo dedAlloc = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    if (dedReq.requiresDedicatedAllocation || dedReq.prefersDedicatedAllocation) {
        dedAlloc.image = res.image;
        dedAlloc.pNext = allocInfo.pNext;
        allocInfo.pNext = &dedAlloc;
    }

    res.allocationSize = allocInfo.allocationSize;
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, res.image, res.memory, 0));
    return res;
}

// ---------------------------------------------------------
// DATA UPLOAD (Multi-Level)
// ---------------------------------------------------------
void uploadMipData(VulkanContext& ctx, ImageResources& img) {
    // We will upload 2 levels.
    // Level 0: Pure Red (1,0,0,1)
    // Level 1: Pure Blue (0,0,1,1)
    
    // Calculate total size
    size_t level0Size = img.extent.width * img.extent.height * 4 * sizeof(float);
    size_t level1Size = (img.extent.width/2) * (img.extent.height/2) * 4 * sizeof(float);
    size_t totalSize = level0Size + level1Size;

    // Create Staging
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = totalSize; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(ctx.device, &bi, nullptr, &stagingBuffer);

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(ctx.physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMemory);
    vkBindBufferMemory(ctx.device, stagingBuffer, stagingMemory, 0);

    // Fill Data
    void* data; vkMapMemory(ctx.device, stagingMemory, 0, totalSize, 0, &data);
    float* fptr = (float*)data;
    
    // Fill Level 0 (Red)
    size_t px0 = img.extent.width * img.extent.height;
    for(size_t i=0; i<px0; ++i) {
        fptr[i*4+0] = 1.0f; fptr[i*4+1] = 0.0f; fptr[i*4+2] = 0.0f; fptr[i*4+3] = 1.0f;
    }
    
    // Fill Level 1 (Blue)
    size_t px1 = (img.extent.width/2) * (img.extent.height/2);
    float* fptr1 = fptr + (px0 * 4);
    for(size_t i=0; i<px1; ++i) {
        fptr1[i*4+0] = 0.0f; fptr1[i*4+1] = 0.0f; fptr1[i*4+2] = 1.0f; fptr1[i*4+3] = 1.0f;
    }
    vkUnmapMemory(ctx.device, stagingMemory);

    // Copy to Image
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    VkCommandPool pool; vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &pool);
    VkCommandBuffer cmd; VkCommandBufferAllocateInfo alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin);

    // Barrier: Undefined -> Transfer Dst
    VkImageMemoryBarrier bar1 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar1.image = img.image;
    bar1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 0, 1}; // 2 Levels
    bar1.srcAccessMask = 0;
    bar1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar1);

    // Copy Level 0
    VkBufferImageCopy reg0 = {};
    reg0.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; // Mip 0
    reg0.imageExtent = img.extent;
    vkCmdCopyBufferToImage(cmd, stagingBuffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg0);

    // Copy Level 1
    VkBufferImageCopy reg1 = {};
    reg1.bufferOffset = level0Size;
    reg1.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1}; // Mip 1
    reg1.imageExtent = {img.extent.width/2, img.extent.height/2, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg1);

    // Barrier: Transfer Dst -> General (Read-Only/Sampled for SYCL)
    VkImageMemoryBarrier bar2 = bar1;
    bar2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar2.newLayout = VK_IMAGE_LAYOUT_GENERAL; // Safest for Bindless
    bar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar2);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.queue);

    vkDestroyCommandPool(ctx.device, pool, nullptr);
    vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
    vkFreeMemory(ctx.device, stagingMemory, nullptr);
}

// ---------------------------------------------------------
// PLATFORM GETTERS
// ---------------------------------------------------------
#ifdef _WIN32
HANDLE getMemHandle(VulkanContext& ctx, VkDeviceMemory memory) {
    VkMemoryGetWin32HandleInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle;
    auto func = (PFN_vkGetMemoryWin32HandleKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryWin32HandleKHR");
    func(ctx.device, &info, &handle);
    return handle;
}
#else
int getMemFd(VulkanContext& ctx, VkDeviceMemory memory) {
    VkMemoryGetFdInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd;
    auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
    func(ctx.device, &info, &fd);
    return fd;
}
#endif

// ---------------------------------------------------------
// MAIN TEST
// ---------------------------------------------------------
int main(int argc, char** argv) {
    int width = 32;
    int height = 32;
    int mipLevels = 2;

    try {
        // 1. SYCL Setup
        sycl::queue q;
        std::cout << "[SYCL] Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        // 2. Vulkan Setup (Matched)
        VulkanContext vkCtx = createUUIDMatchedContext(q.get_device());
        ImageResources img = createMipImage(vkCtx, width, height, mipLevels);
        
        // 3. Upload Data (LOD0=Red, LOD1=Blue)
        uploadMipData(vkCtx, img);

        // 4. Import to SYCL
        #ifdef _WIN32
        auto desc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, img.memory), syclexp::external_mem_handle_type::win32_nt_handle, img.allocationSize};
        #else
        auto desc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, img.memory), syclexp::external_mem_handle_type::opaque_fd, img.allocationSize};
        #endif
        auto extMem = syclexp::import_external_memory(desc, q);

        // Mipmap Descriptor
        syclexp::image_descriptor imgDesc({(size_t)width, (size_t)height}, 4, sycl::image_channel_type::fp32, syclexp::image_type::mipmap, mipLevels);
        auto mappedHandle = syclexp::map_external_image_memory(extMem, imgDesc, q);

        // Sampler with Nearest Mipmap Mode to distinct levels
        syclexp::bindless_image_sampler samp(
            sycl::addressing_mode::repeat,
            sycl::coordinate_normalization_mode::normalized,
            sycl::filtering_mode::nearest, 
            sycl::filtering_mode::nearest, // Nearest Mipmap filtering prevents blending between levels
            0.0f, (float)mipLevels, 8.0f
        );

        auto imgHandle = syclexp::create_image(mappedHandle, samp, imgDesc, q);

        // Output Buffer (Store result of LOD0 sample + LOD1 sample)
        sycl::buffer<float, 1> outBuf(width * height * 4);

        q.submit([&](sycl::handler& h){
            sycl::accessor outAcc(outBuf, h, sycl::write_only);
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item){
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                // Normalized Coords (Center of pixel)
                float u = (x + 0.5f) / width;
                float v = (y + 0.5f) / height;
                
                // Sample LOD 0 (Red)
                sycl::float4 val0 = syclexp::sample_mipmap<sycl::float4>(imgHandle, sycl::float2(u,v), 0.0f);
                
                // Sample LOD 1 (Blue)
                sycl::float4 val1 = syclexp::sample_mipmap<sycl::float4>(imgHandle, sycl::float2(u,v), 1.0f);

                // Store Sum
                int idx = (y * width + x) * 4;
                outAcc[idx+0] = val0.x() + val1.x();
                outAcc[idx+1] = val0.y() + val1.y();
                outAcc[idx+2] = val0.z() + val1.z();
                outAcc[idx+3] = val0.w() + val1.w();
            });
        }).wait();

        // 5. Verify
        sycl::host_accessor res(outBuf, sycl::read_only);
        int errors = 0;
        for(int i=0; i<width*height; ++i) {
            float r = res[i*4+0]; // 1 + 0 = 1
            float g = res[i*4+1]; // 0 + 0 = 0
            float b = res[i*4+2]; // 0 + 1 = 1
            
            // Allow small epsilon
            if(std::abs(r - 1.0f) > 0.01f || std::abs(g - 0.0f) > 0.01f || std::abs(b - 1.0f) > 0.01f) {
                if(errors++ < 5) std::cout << "Mismatch at " << i << " Got(" << r << "," << g << "," << b << ")" << std::endl;
            }
        }

        if(errors == 0) std::cout << "SUCCESS! Mipmap levels sampled correctly." << std::endl;
        else std::cout << "FAILURE! " << errors << " errors found." << std::endl;

        // Cleanup
        syclexp::destroy_image_handle(imgHandle, q);
        syclexp::release_external_memory(extMem, q);
        
        vkDestroyImage(vkCtx.device, img.image, nullptr);
        vkFreeMemory(vkCtx.device, img.memory, nullptr);
        vkDestroyDevice(vkCtx.device, nullptr);
        vkDestroyInstance(vkCtx.instance, nullptr);

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}