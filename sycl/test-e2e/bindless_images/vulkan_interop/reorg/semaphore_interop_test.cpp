/*

clang++ -fsycl -o s_i_t.bin semaphore_interop_test.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib

*/



#include <iostream>
#include <vector>
#include <cassert>
#include <unistd.h> // for close()

#include <vulkan/vulkan.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>

// Namespace alias
namespace syclexp = sycl::ext::oneapi::experimental;

// -----------------------------------------------------------------------------
// VULKAN BOILERPLATE (Minimal)
// -----------------------------------------------------------------------------
#define VK_CHECK(x) \
    do { \
        VkResult err = x; \
        if (err) { \
            std::cerr << "Vulkan Error: " << err << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

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

    // 4. Logical Device WITH EXTENSIONS
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    // FIX: Explicitly enable the External FD extensions
    const char* extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
    };

    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    
    // Pass the extensions here
    devInfo.enabledExtensionCount = 4;
    devInfo.ppEnabledExtensionNames = extensions;
    
    VK_CHECK(vkCreateDevice(ctx.physDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);
    
    return ctx;
}

// -----------------------------------------------------------------------------
// HELPER: Create Exportable Resources
// -----------------------------------------------------------------------------
struct InteropBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    int fd;
    void* mappedPtr;
    VkDeviceSize allocationSize; 
};

InteropBuffer create_buffer(const VulkanContext& ctx, size_t size) {
    InteropBuffer res = {};

    // 1. Buffer
    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkExternalMemoryBufferCreateInfo extBufInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    bufInfo.pNext = &extBufInfo;

    VK_CHECK(vkCreateBuffer(ctx.device, &bufInfo, nullptr, &res.buffer));

    // 2. Memory
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, res.buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size; // This is the aligned size (e.g. 65536)
    
    // Save it!
    res.allocationSize = allocInfo.allocationSize; 
    
    
    // Find memory type (Host Visible for checking)
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags  & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT )) {
            allocInfo.memoryTypeIndex = i;
            break;
        }
    }

    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    allocInfo.pNext = &exportInfo;

    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, res.buffer, res.memory, 0));

    // 3. Export FD
    auto fpGetFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = res.memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VK_CHECK(fpGetFd(ctx.device, &fdInfo, &res.fd));

    // 4. Map
    VK_CHECK(vkMapMemory(ctx.device, res.memory, 0, size, 0, &res.mappedPtr));

    return res;
}

struct InteropSemaphore {
    VkSemaphore sem;
    int fd;
};

InteropSemaphore create_semaphore(const VulkanContext& ctx) {
    InteropSemaphore res = {};

    VkExportSemaphoreCreateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semInfo.pNext = &exportInfo;

    VK_CHECK(vkCreateSemaphore(ctx.device, &semInfo, nullptr, &res.sem));

    auto fpGetFd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetSemaphoreFdKHR");
    VkSemaphoreGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    fdInfo.semaphore = res.sem;
    fdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VK_CHECK(fpGetFd(ctx.device, &fdInfo, &res.fd));

    return res;
}

// -----------------------------------------------------------------------------
// MAIN TEST
// -----------------------------------------------------------------------------
int main() {
    std::cout << "[Test] Starting Semaphore Interop Test..." << std::endl;
    
    // 1. Setup Vulkan
    VulkanContext vkCtx = init_vulkan();
    const size_t bufSize = 1024 * sizeof(int); // 4KB
    const int VAL_VULKAN = 11111111;
    const int VAL_SYCL   = 22222222;

    InteropBuffer vkBuf = create_buffer(vkCtx, bufSize);
    InteropSemaphore vkSem = create_semaphore(vkCtx);

    // Initialize Host Data to 0
    std::memset(vkBuf.mappedPtr, 0, bufSize);

    // 2. Submit Vulkan Work: Fill with VAL_VULKAN -> Signal Sem
    VkCommandPool pool;
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool = pool;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(vkCtx.device, &cmdInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &beginInfo);
    vkCmdFillBuffer(cmd, vkBuf.buffer, 0, bufSize, VAL_VULKAN);
    
    // Barrier to make write visible
    VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.buffer = vkBuf.buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &vkSem.sem;

    std::cout << "[Test] Submitting Vulkan Work..." << std::endl;
    vkQueueSubmit(vkCtx.queue, 1, &submitInfo, VK_NULL_HANDLE);

    // 3. SYCL Work: Wait Sem -> Overwrite with VAL_SYCL
    try {
        std::cout << "[Test] Starting SYCL Work..." << std::endl;
        sycl::queue q; 
        auto dev = q.get_device();
        auto ctx = q.get_context();

        // FIX 1: Use 'syclexp::resource_fd' instead of 'int'
        // This matches the template instantiation in your library.
        syclexp::external_semaphore_descriptor<syclexp::resource_fd> semDesc{
            vkSem.fd, 
            syclexp::external_semaphore_handle_type::opaque_fd
        };
        auto syclSem = syclexp::import_external_semaphore(semDesc, dev, ctx);

        
        // Matching the working code's constructor signature.
        syclexp::external_mem_descriptor<syclexp::resource_fd> memDesc{
            vkBuf.fd, 
            syclexp::external_mem_handle_type::opaque_fd,
            vkBuf.allocationSize // <--- USE THIS instead of bufSize
        };
        auto syclMem = syclexp::import_external_memory(memDesc, dev, ctx);

        // Map as 1D Image (Linear)
        syclexp::image_descriptor imgDesc(
            sycl::range<1>{1024}, 
            1, 
            sycl::image_channel_type::unsigned_int32,
            syclexp::image_type::standard, 
            1, 
            1
        );
        
        // Use (dev, ctx) for mapping
        auto imgMemHandle = syclexp::map_external_image_memory(syclMem, imgDesc, dev, ctx);
        
        // Use create_image for the handle
        auto imgHandle = syclexp::create_image(imgMemHandle, imgDesc, dev, ctx);

        q.submit([&](sycl::handler& cgh) {
            cgh.ext_oneapi_wait_external_semaphore(syclSem);
            
            cgh.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
                 sycl::uint4 color{static_cast<unsigned int>(VAL_SYCL), 0, 0, 0};
                 syclexp::write_image(imgHandle, int(idx[0]), color);
            });
        }).wait();

        std::cout << "[Test] SYCL Work Complete." << std::endl;
        
        // ceanup
        syclexp::destroy_image_handle(imgHandle, dev, ctx);
        syclexp::release_external_memory(syclMem, dev, ctx);
        syclexp::release_external_semaphore(syclSem, dev, ctx);

    } catch (sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    // 4. Verify
    vkQueueWaitIdle(vkCtx.queue);
    
    // Invalidate mapped memory if non-coherent (assumed coherent here for simplicity)
    int* ptr = static_cast<int*>(vkBuf.mappedPtr);
    int errors = 0;
    for (size_t i = 0; i < 1024; ++i) {
        if (ptr[i] != VAL_SYCL) {
            std::cerr << "Mismatch at [" << i << "]: Expected " << VAL_SYCL 
                      << ", Got " << ptr[i] << std::endl;
            errors++;
            if (errors > 5) break;
        }
    }

    if (errors == 0) {
        std::cout << "[PASS] Semaphores synchronized correctly." << std::endl;
    } else {
        std::cout << "[FAIL] Data mismatch. Sync failed or Kernel failed." << std::endl;
    }

    // Cleanup omitted for brevity (OS cleans up FDs)
    return (errors == 0) ? 0 : 1;
}