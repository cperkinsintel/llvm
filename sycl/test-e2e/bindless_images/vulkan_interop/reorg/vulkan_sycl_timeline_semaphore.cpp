// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: vulkan

// UNSUPPORTED: linux && run-mode
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21133

// XFAIL: linux && gpu-intel-dg2
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21136

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes -DTEST_L0_SUPPORTED_VK_FORMAT %}
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out



/*
  Vulkan/SYCL Timeline Semaphore & Bindless Image Test

  clang++ -fsycl  -o vsts.bin vulkan_sycl_timeline_semaphore.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  

  clang++ -fsycl  -o vsts.exe vulkan_sycl_timeline_semaphore.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  Features:
  - Uses Timeline Semaphores (uint64_t) for fine-grained sync.
  - Runs a "Stress Loop" (100 iterations) to detect race conditions.
  - Vulkan Signals (Value N) -> SYCL Waits (Value N).
  - SYCL Signals (Value N) -> Vulkan Waits (Value N).
  - Uses Bindless Images for the actual data processing.
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
#include <random>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#define PLATFORM_SEM_HANDLE_TYPE VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
#include <vulkan/vulkan_core.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#define PLATFORM_SEM_HANDLE_TYPE VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
#endif

#define VK_CHECK(f) { VkResult __res = (f); if(__res != VK_SUCCESS) { std::cerr << "Vulkan Error: " << __res << std::endl; exit(1); } }

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
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkExtent3D extent;
};

// ---------------------------------------------------------
// UUID MATCHING
// ---------------------------------------------------------
inline VulkanContext createUUIDMatchedContext(const sycl::device& syclDev) {
    VulkanContext ctx;
    
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
        std::cerr << "FATAL: No matching Vulkan device found!" << std::endl;
        exit(1);
    }

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
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
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
    
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timelineFeatures.timelineSemaphore = VK_TRUE;
    devInfo.pNext = &timelineFeatures;

    VK_CHECK(vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);
    
    return ctx;
}

// ---------------------------------------------------------
// RESOURCE CREATION helpers
// ---------------------------------------------------------
uint32_t findMemType(VkPhysicalDevice dev, uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(dev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

ImageResources createImage(VulkanContext& ctx, uint32_t w, uint32_t h) {
    ImageResources res;
    res.extent = {w, h, 1};

    // 1. Image
    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = res.extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkExternalMemoryImageCreateInfo extMemInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extMemInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    imageInfo.pNext = &extMemInfo;

    VK_CHECK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image));

    VkMemoryRequirements req; vkGetImageMemoryRequirements(ctx.device, res.image, &req);
    VkExportMemoryAllocateInfo expAlloc = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    expAlloc.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    
    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &expAlloc;
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = findMemType(ctx.physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    res.allocationSize = allocInfo.allocationSize;
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, res.image, res.memory, 0));

    // 2. Staging Buffer
    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = w * h * 4 * sizeof(float);
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(ctx.device, &bufInfo, nullptr, &res.stagingBuffer));
    
    VkMemoryRequirements bufReq; vkGetBufferMemoryRequirements(ctx.device, res.stagingBuffer, &bufReq);
    VkMemoryAllocateInfo bufAlloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bufAlloc.allocationSize = bufReq.size;
    bufAlloc.memoryTypeIndex = findMemType(ctx.physicalDevice, bufReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(ctx.device, &bufAlloc, nullptr, &res.stagingMemory));
    VK_CHECK(vkBindBufferMemory(ctx.device, res.stagingBuffer, res.stagingMemory, 0));

    return res;
}

VkSemaphore createTimelineSemaphore(VulkanContext& ctx, uint64_t initialValue) {
    VkSemaphoreTypeCreateInfo typeInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = initialValue;

    VkExportSemaphoreCreateInfo expInfo = {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    expInfo.handleTypes = PLATFORM_SEM_HANDLE_TYPE;
    expInfo.pNext = &typeInfo;

    VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    info.pNext = &expInfo;

    VkSemaphore sem;
    VK_CHECK(vkCreateSemaphore(ctx.device, &info, nullptr, &sem));
    return sem;
}

// ---------------------------------------------------------
// PLATFORM GETTERS
// ---------------------------------------------------------
#ifdef _WIN32
HANDLE getMemHandle(VulkanContext& ctx, VkDeviceMemory mem) {
    VkMemoryGetWin32HandleInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = mem; info.handleType = PLATFORM_MEM_HANDLE_TYPE;
    HANDLE h; ((PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryWin32HandleKHR"))(ctx.device, &info, &h);
    return h;
}
HANDLE getSemHandle(VulkanContext& ctx, VkSemaphore sem) {
    VkSemaphoreGetWin32HandleInfoKHR info = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    info.semaphore = sem; info.handleType = PLATFORM_SEM_HANDLE_TYPE;
    HANDLE h; ((PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreWin32HandleKHR"))(ctx.device, &info, &h);
    return h;
}
#else
int getMemFd(VulkanContext& ctx, VkDeviceMemory mem) {
    VkMemoryGetFdInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = mem; info.handleType = PLATFORM_MEM_HANDLE_TYPE;
    int fd; ((PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR"))(ctx.device, &info, &fd);
    return fd;
}
int getSemFd(VulkanContext& ctx, VkSemaphore sem) {
    VkSemaphoreGetFdInfoKHR info = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    info.semaphore = sem; info.handleType = PLATFORM_SEM_HANDLE_TYPE;
    int fd; ((PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreFdKHR"))(ctx.device, &info, &fd);
    return fd;
}
#endif

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    int width = 32;
    int height = 32;
    size_t imgSize = width * height * 4 * sizeof(float);
    int iterations = 100; // Stress Test Loop

    try {
        VulkanContext vkCtx;
        ImageResources inImg;
        ImageResources outImg;
        VkSemaphore vkToSyclSem;
        VkSemaphore syclToVkSem;
        VkCommandPool pool;
        
        {
            sycl::queue q;
            std::cout << "[SYCL] Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

            vkCtx = createUUIDMatchedContext(q.get_device());
            inImg = createImage(vkCtx, width, height);
            outImg = createImage(vkCtx, width, height);

            // Create Timeline Semaphores
            vkToSyclSem = createTimelineSemaphore(vkCtx, 0); // Vulkan Signals -> SYCL Waits
            syclToVkSem = createTimelineSemaphore(vkCtx, 0); // SYCL Signals -> Vulkan Waits

            // Import Resources to SYCL
            #ifdef _WIN32
            auto inDesc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, inImg.memory), syclexp::external_mem_handle_type::win32_nt_handle, inImg.allocationSize};
            auto outDesc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, outImg.memory), syclexp::external_mem_handle_type::win32_nt_handle, outImg.allocationSize};
            
            auto waitSemDesc = syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>{getSemHandle(vkCtx, vkToSyclSem), syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
            auto sigSemDesc = syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>{getSemHandle(vkCtx, syclToVkSem), syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
            #else
            auto inDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, inImg.memory), syclexp::external_mem_handle_type::opaque_fd, inImg.allocationSize};
            auto outDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, outImg.memory), syclexp::external_mem_handle_type::opaque_fd, outImg.allocationSize};
            
            auto waitSemDesc = syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemFd(vkCtx, vkToSyclSem), syclexp::external_semaphore_handle_type::timeline_fd};
            auto sigSemDesc = syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemFd(vkCtx, syclToVkSem), syclexp::external_semaphore_handle_type::timeline_fd};
            #endif

            auto inExtMem = syclexp::import_external_memory(inDesc, q);
            auto outExtMem = syclexp::import_external_memory(outDesc, q);
            auto syclWaitSem = syclexp::import_external_semaphore(waitSemDesc, q);
            auto syclSigSem = syclexp::import_external_semaphore(sigSemDesc, q);

            syclexp::image_descriptor imgDesc({(size_t)width, (size_t)height}, 4, sycl::image_channel_type::fp32);
            auto inMap = syclexp::map_external_image_memory(inExtMem, imgDesc, q);
            auto outMap = syclexp::map_external_image_memory(outExtMem, imgDesc, q);
            
            auto inHandle = syclexp::create_image(inMap, imgDesc, q);
            auto outHandle = syclexp::create_image(outMap, imgDesc, q);

            // Command Pool
            VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
            vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
            VkCommandBuffer cmd; VkCommandBufferAllocateInfo alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            alloc.commandPool = pool; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
            vkAllocateCommandBuffers(vkCtx.device, &alloc, &cmd);

            std::cout << "Starting Stress Test (" << iterations << " iterations)..." << std::endl;

            for(int i = 1; i <= iterations; ++i) {
                uint64_t signalVal = i;

                // --- VULKAN: Upload Data -> Signal(i) ---
                void* data; vkMapMemory(vkCtx.device, inImg.stagingMemory, 0, imgSize, 0, &data);
                float val = (float)i; // Unique value per iteration
                for(int k=0; k<width*height*4; ++k) ((float*)data)[k] = val;
                vkUnmapMemory(vkCtx.device, inImg.stagingMemory);

                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(cmd, &begin);

                // Transition Undefined -> Transfer Dst
                VkImageMemoryBarrier bar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
                // CACHE FIX: Use GENERAL for subsequent iterations to ensure cache flush? 
                // Actually, if we use UNDEFINED, we discard. If we want safety, use GENERAL if i > 1.
                if (i > 1) bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL; 
                
                bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                bar.image = inImg.image; bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                bar.srcAccessMask = (i > 1) ? VK_ACCESS_SHADER_READ_BIT : 0; 
                bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);

                VkBufferImageCopy copy = {}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
                vkCmdCopyBufferToImage(cmd, inImg.stagingBuffer, inImg.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

                // Transition Transfer Dst -> General
                bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
                
                // Also Transition Output to General
                bar.image = outImg.image; bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
                if (i > 1) bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                bar.srcAccessMask = 0; bar.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);

                vkEndCommandBuffer(cmd);

                VkTimelineSemaphoreSubmitInfo timelineInfo = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timelineInfo.signalSemaphoreValueCount = 1; timelineInfo.pSignalSemaphoreValues = &signalVal;
                
                VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.pNext = &timelineInfo;
                si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
                si.signalSemaphoreCount = 1; si.pSignalSemaphores = &vkToSyclSem;
                
                vkQueueSubmit(vkCtx.queue, 1, &si, VK_NULL_HANDLE);

                // --- SYCL: Wait(i) -> Kernel -> Signal(i) ---
                q.ext_oneapi_wait_external_semaphore(syclWaitSem, signalVal);
                
                q.submit([&](sycl::handler& cgh){
                    cgh.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item){
                        int x = item.get_id(0);
                        int y = item.get_id(1);
                        sycl::float4 px = syclexp::fetch_image<sycl::float4>(inHandle, sycl::int2(x,y));
                        // Simple op: multiply by 2
                        px = px * 2.0f; 
                        syclexp::write_image(outHandle, sycl::int2(x,y), px);
                    });
                });

                q.ext_oneapi_signal_external_semaphore(syclSigSem, signalVal);

                // --- VULKAN: Wait(i) -> Check Results ---
                // We use host wait for simplicity in checking results immediately
                VkSemaphoreWaitInfo waitInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                waitInfo.semaphoreCount = 1; waitInfo.pSemaphores = &syclToVkSem; waitInfo.pValues = &signalVal;
                vkWaitSemaphores(vkCtx.device, &waitInfo, UINT64_MAX);

                // Readback
                vkResetCommandBuffer(cmd, 0);
                vkBeginCommandBuffer(cmd, &begin);
                
                // Flush caches before readback
                VkImageMemoryBarrier readBar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                readBar.image = outImg.image;
                readBar.oldLayout = VK_IMAGE_LAYOUT_GENERAL; readBar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                readBar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; readBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &readBar);

                VkBufferImageCopy readback = {}; readback.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; readback.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
                vkCmdCopyImageToBuffer(cmd, outImg.image, VK_IMAGE_LAYOUT_GENERAL, outImg.stagingBuffer, 1, &readback);
                vkEndCommandBuffer(cmd);
                
                VkSubmitInfo siRead = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                siRead.commandBufferCount = 1; siRead.pCommandBuffers = &cmd;
                vkQueueSubmit(vkCtx.queue, 1, &siRead, VK_NULL_HANDLE);
                vkQueueWaitIdle(vkCtx.queue);

                void* resData; vkMapMemory(vkCtx.device, outImg.stagingMemory, 0, imgSize, 0, &resData);
                float expected = val * 2.0f;
                float actual = ((float*)resData)[0]; // Check first pixel
                vkUnmapMemory(vkCtx.device, outImg.stagingMemory);

                if(std::abs(actual - expected) > 0.01f) {
                    std::cerr << "FAILURE at iteration " << i << ". Expected " << expected << " Got " << actual << std::endl;
                    return 1;
                }
            }

            std::cout << "SUCCESS! All iterations passed." << std::endl;

            // Explicit SYCL Cleanup
            syclexp::destroy_image_handle(inHandle, q); syclexp::destroy_image_handle(outHandle, q);
            syclexp::release_external_semaphore(syclWaitSem, q); syclexp::release_external_semaphore(syclSigSem, q);
            syclexp::release_external_memory(inExtMem, q); syclexp::release_external_memory(outExtMem, q);
            q.wait_and_throw();
        } 
        // ~queue (SYCL Runtime caches device here)

        // workaround CMPLRLLVM-73463:  Do not destroy Vulkan Device.
        
        // vkDestroySemaphore(vkCtx.device, vkToSyclSem, nullptr); vkDestroySemaphore(vkCtx.device, syclToVkSem, nullptr);
        // vkDestroyCommandPool(vkCtx.device, pool, nullptr);
        // vkDestroyImage(vkCtx.device, inImg.image, nullptr); vkFreeMemory(vkCtx.device, inImg.memory, nullptr);
        // vkDestroyBuffer(vkCtx.device, inImg.stagingBuffer, nullptr); vkFreeMemory(vkCtx.device, inImg.stagingMemory, nullptr);
        // vkDestroyImage(vkCtx.device, outImg.image, nullptr); vkFreeMemory(vkCtx.device, outImg.memory, nullptr);
        // vkDestroyBuffer(vkCtx.device, outImg.stagingBuffer, nullptr); vkFreeMemory(vkCtx.device, outImg.stagingMemory, nullptr);
        // vkDestroyDevice(vkCtx.device, nullptr); vkDestroyInstance(vkCtx.instance, nullptr);

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}