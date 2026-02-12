/*
  Vulkan/SYCL Depth Image Interop Test
  
  Replaces: depth_format.cpp


  clang++ -fsycl  -o vsd.bin vulkan_sycl_depth.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  clang++ -fsycl  -o vsd.exe vulkan_sycl_depth.cpp -DVK_USE_PLATFORM_WIN32_KHR -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib
  
  Features:
  - Supports Depth Formats (D32_SFLOAT, D16_UNORM)
  - Handles VK_IMAGE_ASPECT_DEPTH_BIT constraints
  - Checks Unsampled Read (Fetch) and Write
  - Semaphores support
  
  Usage:
    ./vulkan_sycl_depth.bin
    ./vulkan_sycl_depth.bin --semaphores
    ./vulkan_sycl_depth.bin --d16
    ./vulkan_sycl_depth.bin 64x64
*/
/*
  Vulkan/SYCL Depth Image Interop Test - EXTENSION FIX
  
  Hypothesis: The crash occurs because 'vulkan_setup.hpp' is missing 
  VK_KHR_dedicated_allocation in the device creation list. 
  This causes the driver to ignore the dedicated allocation request, 
  leading to metadata mismatches (DEVICE_LOST) during interop.
*/

#include "test_verification.hpp"
#include "vulkan_setup.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// LOCAL VULKAN CONTEXT OVERRIDE
// ---------------------------------------------------------
// We copy-paste createVulkanContext but add the critical missing extensions.
inline VulkanContext createRobustVulkanContext() {
    VulkanContext ctx;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &ctx.instance));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
    ctx.physicalDevice = devices[0];

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

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    // --- THE FIX: Add Dedicated Allocation Extensions ---
    std::vector<const char*> robustExtensions = PLATFORM_EXTENSIONS;
    robustExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    robustExtensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    
    // Debug print
    std::cout << "[Vulkan] Enabling Extensions:" << std::endl;
    for(auto& ext : robustExtensions) std::cout << "  - " << ext << std::endl;

    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(robustExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = robustExtensions.data();

    VK_CHECK(vkCreateDevice(ctx.physicalDevice, &deviceCreateInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);

    return ctx;
}

// ---------------------------------------------------------
// DEPTH IMAGE SETUP (Standard)
// ---------------------------------------------------------
inline ImageResources createExportableDepthImage(VulkanContext& ctx, VkExtent3D extent, VkFormat format) {
    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    // Use Transfer Only to be safe (avoids HiZ on some drivers)
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkExternalMemoryImageCreateInfo extMemInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extMemInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    imageInfo.pNext = &extMemInfo;

    ImageResources res;
    res.extent = extent;
    VK_CHECK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(ctx.device, res.image, &memRequirements);

    VkExportMemoryAllocateInfo exportAllocInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportAllocInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;

    // Use Dedicated Allocation (Now actually supported by the context!)
    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedAllocInfo.image = res.image;
    dedicatedAllocInfo.buffer = VK_NULL_HANDLE;
    dedicatedAllocInfo.pNext = &exportAllocInfo;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &dedicatedAllocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    res.allocationSize = allocInfo.allocationSize;
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, res.image, res.memory, 0));
    return res;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char** argv) {
    int width = 64;
    int height = 64;

    std::cout << "DEBUG: Extension Fix Run" << std::endl;

    // Use our Robust Context
    VulkanContext vkCtx = createRobustVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};

    ImageResources inImg = createExportableDepthImage(vkCtx, extent, VK_FORMAT_D32_SFLOAT);
    ImageResources outImg = createExportableDepthImage(vkCtx, extent, VK_FORMAT_D32_SFLOAT);

    // Initial Transitions (General)
    {
        VkCommandBuffer cmd; 
        VkCommandBufferAllocateInfo alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
        VkCommandPool pool; vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
        alloc.commandPool = pool; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(vkCtx.device, &alloc, &cmd);
        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin);

        VkImageMemoryBarrier bar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_GENERAL; 
        bar.srcAccessMask = 0;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        bar.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        bar.image = inImg.image;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        bar.image = outImg.image;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(vkCtx.queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vkCtx.queue);
        vkDestroyCommandPool(vkCtx.device, pool, nullptr);
    }

    try {
        sycl::queue q;
        auto dev = q.get_device();
        auto ctx = q.get_context();
        std::cout << "[SYCL] Queue Created" << std::endl;

        // Use standard import (allocation size)
        #ifdef _WIN32
        auto inDesc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, inImg.memory), syclexp::external_mem_handle_type::win32_nt_handle, inImg.allocationSize};
        auto outDesc = syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{getMemHandle(vkCtx, outImg.memory), syclexp::external_mem_handle_type::win32_nt_handle, outImg.allocationSize};
        #else
        auto inDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, inImg.memory), syclexp::external_mem_handle_type::opaque_fd, inImg.allocationSize};
        auto outDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, outImg.memory), syclexp::external_mem_handle_type::opaque_fd, outImg.allocationSize};
        #endif
        
        auto inExtMem = syclexp::import_external_memory(inDesc, dev, ctx);
        auto outExtMem = syclexp::import_external_memory(outDesc, dev, ctx);

        syclexp::image_descriptor imgDesc({(size_t)width, (size_t)height}, 1, sycl::image_channel_type::fp32);
        
        auto inH = syclexp::map_external_image_memory(inExtMem, imgDesc, dev, ctx);
        auto outH = syclexp::map_external_image_memory(outExtMem, imgDesc, dev, ctx);

        auto inImgObj = syclexp::create_image(inH, imgDesc, dev, ctx);
        auto outImgObj = syclexp::create_image(outH, imgDesc, dev, ctx);

        std::cout << "[SYCL] Submitting Kernel..." << std::endl;
        q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item){
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                float val = syclexp::fetch_image<float>(inImgObj, sycl::int2(x, y));
                syclexp::write_image<float>(outImgObj, sycl::int2(x, y), val);
            });
        }).wait();
        std::cout << "[SYCL] Kernel Finished Successfully" << std::endl;

        syclexp::destroy_image_handle(inImgObj, dev, ctx);
        syclexp::destroy_image_handle(outImgObj, dev, ctx);
        syclexp::release_external_memory(inExtMem, dev, ctx);
        syclexp::release_external_memory(outExtMem, dev, ctx);

    } catch (std::exception& e) {
        std::cerr << "\n[FATAL ERROR] SYCL Exception caught:\n" << e.what() << std::endl;
        return 1;
    }

    vkDestroyImage(vkCtx.device, inImg.image, nullptr);
    vkFreeMemory(vkCtx.device, inImg.memory, nullptr);
    vkDestroyImage(vkCtx.device, outImg.image, nullptr);
    vkFreeMemory(vkCtx.device, outImg.memory, nullptr);
    vkDestroyDevice(vkCtx.device, nullptr);
    vkDestroyInstance(vkCtx.instance, nullptr);

    std::cout << "EXTENSION FIX RUN COMPLETE" << std::endl;
    return 0;
}