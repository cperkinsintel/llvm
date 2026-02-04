/*

clang++ -o vos.bin vulkan_only_semaphore.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
./vos.bin

*/

#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <vulkan/vulkan.h>

#define VK_CHECK(x) \
    do { \
        VkResult err = x; \
        if (err) { \
            std::cerr << "Vulkan Error: " << err << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// VULKAN SETUP
// -----------------------------------------------------------------------------
struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physDevice;
    VkDevice device;
    VkQueue queue;
    uint32_t queueFamilyIndex;
};

VulkanContext init_vulkan() {
    VulkanContext ctx = {};

    // 1. Instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &ctx.instance));

    // 2. Physical Device
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr);
    if (count == 0) { std::cerr << "No GPU found" << std::endl; exit(1); }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data());
    ctx.physDevice = devices[0];

    // 3. Queue Family
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &count, queues.data());
    
    ctx.queueFamilyIndex = -1;
    for (uint32_t i = 0; i < count; i++) {
        if (queues[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            ctx.queueFamilyIndex = i;
            break;
        }
    }
    assert(ctx.queueFamilyIndex != (uint32_t)-1);

    // 4. Device & EXTENSIONS
    // CRITICAL: We must explicit enable these for getFd to work!
    const char* extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
    };

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.enabledExtensionCount = 4; // Enable the 4 extensions above
    devInfo.ppEnabledExtensionNames = extensions;
    
    VK_CHECK(vkCreateDevice(ctx.physDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);
    
    std::cout << "[Vulkan] Device Created with External Extensions." << std::endl;
    return ctx;
}

// -----------------------------------------------------------------------------
// BUFFER TEST
// -----------------------------------------------------------------------------
void test_export_buffer(const VulkanContext& ctx) {
    std::cout << "[Buffer] Creating Exportable Buffer..." << std::endl;
    
    VkBuffer buffer;
    VkDeviceMemory memory;

    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = 1024;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    VkExternalMemoryBufferCreateInfo extBufInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    bufInfo.pNext = &extBufInfo;

    VK_CHECK(vkCreateBuffer(ctx.device, &bufInfo, nullptr, &buffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    
    // Simple memory type search
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (memReqs.memoryTypeBits & (1 << i)) {
            allocInfo.memoryTypeIndex = i;
            break;
        }
    }

    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    allocInfo.pNext = &exportInfo;

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, buffer, memory, 0));

    // TEST EXPORT
    auto fpGetFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
    if (!fpGetFd) {
        std::cerr << "[Buffer] FAIL: Could not load vkGetMemoryFdKHR!" << std::endl;
        exit(1);
    }

    int fd = -1;
    VkMemoryGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    
    VK_CHECK(fpGetFd(ctx.device, &fdInfo, &fd));
    
    if (fd != -1) std::cout << "[Buffer] SUCCESS: Got FD " << fd << std::endl;
    else std::cout << "[Buffer] FAIL: FD is -1" << std::endl;
}

// -----------------------------------------------------------------------------
// SEMAPHORE TEST
// -----------------------------------------------------------------------------
void test_export_semaphore(const VulkanContext& ctx) {
    std::cout << "[Semaphore] Creating Exportable Semaphore..." << std::endl;

    VkSemaphore sem;
    VkExportSemaphoreCreateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semInfo.pNext = &exportInfo;

    VK_CHECK(vkCreateSemaphore(ctx.device, &semInfo, nullptr, &sem));

    // TEST EXPORT
    auto fpGetFd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreFdKHR");
    if (!fpGetFd) {
        std::cerr << "[Semaphore] FAIL: Could not load vkGetSemaphoreFdKHR!" << std::endl;
        exit(1);
    }

    int fd = -1;
    VkSemaphoreGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    fdInfo.semaphore = sem;
    fdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    
    VK_CHECK(fpGetFd(ctx.device, &fdInfo, &fd));

    if (fd != -1) std::cout << "[Semaphore] SUCCESS: Got FD " << fd << std::endl;
    else std::cout << "[Semaphore] FAIL: FD is -1" << std::endl;
}

int main() {
    VulkanContext ctx = init_vulkan();
    test_export_buffer(ctx);
    test_export_semaphore(ctx);
    return 0;
}