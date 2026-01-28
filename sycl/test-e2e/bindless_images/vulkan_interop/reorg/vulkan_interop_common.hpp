/*
    vulkan_interop_common.hpp

    This is pure Vulkan, no SYCL. 


*/

#pragma once

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <fstream>
#include <stdexcept>

// --- Macros & Utilities ---

#define CHECK_VK(result, msg) \
    if (result != VK_SUCCESS) { \
        std::cerr << "Vulkan error: " << msg << " (code: " << result << ")" << std::endl; \
        exit(1); \
    }

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
    VkExtent3D extent;
    VkFormat format;
};

// --- Helper Functions ---

inline uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

inline VulkanContext createVulkanContext() {
    VulkanContext ctx = {};

    // 1. Instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Interop Test";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    CHECK_VK(vkCreateInstance(&instanceInfo, nullptr, &ctx.instance), "Instance creation");

    // 2. Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
    ctx.physicalDevice = devices[0];

    // 3. Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &queueFamilyCount, queueFamilies.data());

    ctx.queueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queueFamilyIndex = i;
            break;
        }
    }

    // 4. Logical Device with Extensions
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    const char* extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,    // Added for future Semaphore work
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME // Added for future Semaphore work
    };

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 4;
    deviceInfo.ppEnabledExtensionNames = extensions;

    CHECK_VK(vkCreateDevice(ctx.physicalDevice, &deviceInfo, nullptr, &ctx.device), "Device creation");
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);

    return ctx;
}

inline ImageResources createExportableImage(VulkanContext& ctx, VkExtent3D extent, VkFormat format, VkImageType type) {
    ImageResources res = {};
    res.extent = extent;
    res.format = format;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = type;
    imageInfo.format = format;
    imageInfo.extent = extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // We enable ALL usage bits here to support both Sampled and Storage tests
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | 
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    CHECK_VK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image), "Image creation");

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(ctx.device, res.image, &memReq);

    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocInfo.pNext = &exportAllocInfo;

    CHECK_VK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory), "Memory allocation");
    CHECK_VK(vkBindImageMemory(ctx.device, res.image, res.memory, 0), "Memory bind");

    return res;
}

inline VkSemaphore createExportableSemaphore(VulkanContext& ctx) {
    VkExportSemaphoreCreateInfo exportInfo = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    semInfo.pNext = &exportInfo;

    VkSemaphore sem;
    CHECK_VK(vkCreateSemaphore(ctx.device, &semInfo, nullptr, &sem), "Semaphore creation");
    return sem;
}

// Uploads gradient data and transitions layout to GENERAL
// Returns true if the internal readback check passed
// Uploads gradient data, performs a round-trip check (Image -> VerifyBuffer), 
// and transitions layout to GENERAL.
// Returns true only if the Vulkan-side data verification passes.
inline bool uploadAndVerify(VulkanContext& ctx, ImageResources& imgRes, VkSemaphore signalSem = VK_NULL_HANDLE) {
    size_t pixelCount = imgRes.extent.width * imgRes.extent.height * imgRes.extent.depth;
    size_t dataSize = pixelCount * 4 * sizeof(float); // Assuming RGBA32F

    // 1. Create Staging Buffer (Src) and Verify Buffer (Dst)
    VkBuffer stagingBuffer, verifyBuffer;
    VkDeviceMemory stagingMem, verifyMem;
    
    // Helper lambda for buffer creation to keep this clean
    auto createBuf = [&](VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem) {
        VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size = dataSize;
        bi.usage = usage;
        vkCreateBuffer(ctx.device, &bi, nullptr, &buf);
        
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(ctx.device, buf, &req);
        VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(ctx.physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(ctx.device, &ai, nullptr, &mem);
        vkBindBufferMemory(ctx.device, buf, mem, 0);
    };

    createBuf(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingMem);
    createBuf(VK_BUFFER_USAGE_TRANSFER_DST_BIT, verifyBuffer, verifyMem);

    // 2. Fill Staging Buffer with Gradient
    void* data;
    vkMapMemory(ctx.device, stagingMem, 0, dataSize, 0, &data);
    float* floatData = (float*)data;
    for (size_t i = 0; i < pixelCount; i++) {
        floatData[i * 4 + 0] = (float)i / (float)(pixelCount - 1); // R
        floatData[i * 4 + 1] = 0.0f; // G
        floatData[i * 4 + 2] = 0.0f; // B
        floatData[i * 4 + 3] = 1.0f; // A
    }
    vkUnmapMemory(ctx.device, stagingMem);

    // 3. Command Recording
    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    VkCommandPool commandPool;
    vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAlloc.commandPool = commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);

    // A. Undefined -> Transfer Dst
    VkImageMemoryBarrier bar1 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    bar1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar1.image = imgRes.image;
    bar1.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    bar1.srcAccessMask = 0;
    bar1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&bar1);

    // B. Copy Staging -> Image
    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = imgRes.extent;
    vkCmdCopyBufferToImage(cmd, stagingBuffer, imgRes.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // C. Transfer Dst -> Transfer Src (For Verification Readback)
    VkImageMemoryBarrier bar2 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    bar2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar2.image = imgRes.image;
    bar2.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    bar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&bar2);

    // D. Copy Image -> Verify Buffer
    vkCmdCopyImageToBuffer(cmd, imgRes.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, verifyBuffer, 1, &region);

    // E. Transfer Src -> General (Final State for Interop)
    VkImageMemoryBarrier bar3 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    bar3.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar3.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    bar3.image = imgRes.image;
    bar3.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    bar3.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar3.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    
    // Bottom of pipe ensures transition completes before command buffer retires
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,0,nullptr,0,nullptr,1,&bar3);

    vkEndCommandBuffer(cmd);

    // 4. Submit
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (signalSem != VK_NULL_HANDLE) {
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &signalSem;
    }

    vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE);
    // We MUST WaitIdle to read the verification buffer, even if using semaphores for interop.
    vkQueueWaitIdle(ctx.queue);

    // 5. Verify Readback
    bool passed = true;
    void* verifyPtr;
    vkMapMemory(ctx.device, verifyMem, 0, dataSize, 0, &verifyPtr);
    float* verifyFloats = (float*)verifyPtr;
    
    for (size_t i = 0; i < pixelCount; i++) {
        float expected = (float)i / (float)(pixelCount - 1);
        float actual = verifyFloats[i * 4]; 
        if (std::abs(actual - expected) > 0.01f) {
            passed = false;
            std::cerr << "VULKAN UPLOAD CHECK FAILED at index " << i 
                      << ": Expected " << expected << ", Got " << actual << std::endl;
            break; 
        }
    }
    vkUnmapMemory(ctx.device, verifyMem);

    if (passed) {
        std::cout << "✓ Vulkan Data Verified (Internal Round-Trip Passed)" << std::endl;
    }

    // Cleanup
    vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
    vkDestroyBuffer(ctx.device, verifyBuffer, nullptr);
    vkFreeMemory(ctx.device, verifyMem, nullptr);
    vkDestroyCommandPool(ctx.device, commandPool, nullptr);

    return passed;
}

inline void cleanupVulkan(VulkanContext& ctx, ImageResources& res) {
    vkDestroyImage(ctx.device, res.image, nullptr);
    vkFreeMemory(ctx.device, res.memory, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
}

inline int getMemFd(VulkanContext& ctx, VkDeviceMemory mem) {
    auto func = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
    if (!func) return -1;
    
    VkMemoryGetFdInfoKHR info = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
    info.memory = mem;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    
    int fd = -1;
    func(ctx.device, &info, &fd);
    return fd;
}

inline int getSemaphoreFd(VulkanContext& ctx, VkSemaphore sem) {
    auto func = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreFdKHR");
    if (!func) return -1;
    
    VkSemaphoreGetFdInfoKHR info = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
    info.semaphore = sem;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    
    int fd = -1;
    func(ctx.device, &info, &fd);
    return fd;
}