#pragma once

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <type_traits>

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

#define VK_CHECK(f) \
{ \
    VkResult __vk_res = (f); \
    if (__vk_res != VK_SUCCESS) { \
        std::cerr << "Vulkan Error at line " << __LINE__ << ": " << __vk_res << std::endl; \
        throw std::runtime_error("Vulkan Error"); \
    } \
}

inline uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

inline VulkanContext createVulkanContext() {
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
    
    const char* deviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
    };
    deviceCreateInfo.enabledExtensionCount = 4;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;

    VK_CHECK(vkCreateDevice(ctx.physicalDevice, &deviceCreateInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);

    return ctx;
}

inline ImageResources createExportableImage(VulkanContext& ctx, VkExtent3D extent, VkFormat format, VkImageType type, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL) {
    ImageResources res;
    res.extent = extent;

    VkExternalMemoryImageCreateInfo extImageCreateInfo{};
    extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &extImageCreateInfo;
    imageInfo.imageType = type;
    imageInfo.extent = extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling; 
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VK_CHECK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(ctx.device, res.image, &memRequirements);
    res.allocationSize = memRequirements.size;

    VkExportMemoryAllocateInfo exportAllocInfo{};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportAllocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, res.image, res.memory, 0));

    return res;
}

inline VkSemaphore createExportableSemaphore(VulkanContext& ctx) {
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &exportInfo;

    VkSemaphore semaphore;
    VK_CHECK(vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &semaphore));
    return semaphore;
}

inline int getMemFd(VulkanContext& ctx, VkDeviceMemory memory) {
    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = memory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    auto func = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
    if (!func) throw std::runtime_error("Failed to load vkGetMemoryFdKHR");
    VK_CHECK(func(ctx.device, &getFdInfo, &fd));
    return fd;
}

inline int getSemaphoreFd(VulkanContext& ctx, VkSemaphore semaphore) {
    VkSemaphoreGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    getFdInfo.semaphore = semaphore;
    getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    auto func = (PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreFdKHR");
    if (!func) throw std::runtime_error("Failed to load vkGetSemaphoreFdKHR");
    VK_CHECK(func(ctx.device, &getFdInfo, &fd));
    return fd;
}

// -----------------------------------------------------------
//  GENERIC DATA GENERATION & VERIFICATION
// -----------------------------------------------------------

// Helper to generate a test value for a given index and channel
template <typename T>
T generateTestValue(size_t index, int channel, size_t rangeMax) {
    if constexpr (std::is_floating_point_v<T>) {
        // Floating point: 0.0 to 1.0 gradient
        float val = (float)index / (float)(rangeMax > 1 ? rangeMax - 1 : 1);
        return static_cast<T>(val + (float)channel * 0.1f);
    } else {
        // Integer: Sequential numbers wrapping around
        // e.g. (index + channel)
        return static_cast<T>((index + channel * 10) % 127); 
    }
}

// Helper to compare with tolerance
template <typename T>
bool checkValue(T actual, T expected) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(actual - expected) < 0.01f;
    } else {
        return actual == expected;
    }
}

// Templated Upload And Verify
template <typename T>
bool uploadAndVerify(VulkanContext& ctx, ImageResources& imgRes, VkSemaphore signalSemaphore = VK_NULL_HANDLE, int channels = 4) {
    size_t texWidth = imgRes.extent.width;
    size_t texHeight = imgRes.extent.height;
    size_t texDepth = imgRes.extent.depth;
    size_t totalPixels = texWidth * texHeight * texDepth;
    VkDeviceSize imageSize = totalPixels * channels * sizeof(T);

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VK_CHECK(vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &stagingBuffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &stagingBufferMemory));
    VK_CHECK(vkBindBufferMemory(ctx.device, stagingBuffer, stagingBufferMemory, 0));

    // GENERATE DATA
    void* data;
    vkMapMemory(ctx.device, stagingBufferMemory, 0, imageSize, 0, &data);
    T* pixelData = (T*)data;
    
    for (size_t i = 0; i < totalPixels; i++) {
        for(int c=0; c<channels; ++c) {
            pixelData[i * channels + c] = generateTestValue<T>(i, c, totalPixels);
        }
    }
    vkUnmapMemory(ctx.device, stagingBufferMemory);

    // COPY TO IMAGE
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    VkCommandPool commandPool;
    VK_CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmdAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = imgRes.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = imgRes.extent;

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, imgRes.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier barrier2 = barrier;
    barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    if (signalSemaphore != VK_NULL_HANDLE) {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &signalSemaphore;
    }

    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(ctx.queue);

    // COPY BACK (Round Trip Verify)
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkImageMemoryBarrier barrier3 = barrier2;
    barrier3.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier3.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier3.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier3.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr, 1, &barrier3);
    
    vkCmdCopyImageToBuffer(commandBuffer, imgRes.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &region);
    
    VkImageMemoryBarrier barrier4 = barrier3;
    barrier4.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier4.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier4.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier4.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,0,nullptr,0,nullptr, 1, &barrier4);

    vkEndCommandBuffer(commandBuffer);
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(ctx.queue);

    vkMapMemory(ctx.device, stagingBufferMemory, 0, imageSize, 0, &data);
    T* checkData = (T*)data;
    
    bool valid = true;
    for (size_t i = 0; i < totalPixels * channels; i++) {
        size_t pixelIdx = i / channels;
        int channelIdx = i % channels;
        T expected = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);
        
        if (!checkValue(checkData[i], expected)) {
            valid = false;
            // Uncomment for debugging
            // std::cout << "RoundTrip Mismatch: " << (float)checkData[i] << " != " << (float)expected << std::endl;
            break;
        }
    }
    vkUnmapMemory(ctx.device, stagingBufferMemory);

    if(valid) std::cout << "✓ Vulkan Data Verified (Internal Round-Trip Passed)" << std::endl;
    else std::cerr << "X Vulkan Data Verification Failed!" << std::endl;

    vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
    vkFreeMemory(ctx.device, stagingBufferMemory, nullptr);
    vkDestroyCommandPool(ctx.device, commandPool, nullptr);

    return valid;
}

inline void cleanupVulkan(VulkanContext& ctx, ImageResources& res) {
    vkDestroyImage(ctx.device, res.image, nullptr);
    vkFreeMemory(ctx.device, res.memory, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
}