// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// REQUIRES: target-spir
// REQUIRES: vulkan

// XFAIL: windows && run-mode
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21125

// clang-format off

// UNSUPPORTED: arch-intel_gpu_pvc
// UNSUPPORTED-INTENDED: Our PVC runners don't have the userspace Vulkan driver installed

// clang-format on

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out



/*
  SYCL -> Vulkan Buffer Export Test
  
  clang++ -fsycl  -o vseb.bin vulkan_sycl_export_buffer.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  clang++ -fsycl  -o vseb.exe vulkan_sycl_export_buffer.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib


  Features:
  - SYCL allocates exportable memory.
  - SYCL exports handle (FD/Win32).
  - Vulkan queries handle properties to find correct memory index.
  - Vulkan imports handle into VkDeviceMemory.
  - Vulkan maps memory and verifies data.
*/

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "test_verification.hpp"
#include <vulkan/vulkan.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>


#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
#include <vulkan/vulkan_core.h>
#define PLATFORM_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif

#define VK_CHECK(f) { VkResult __vkRes = (f); if (__vkRes != VK_SUCCESS) { std::cerr << "Vulkan Error: " << __vkRes << std::endl; exit(1); } }

namespace syclexp = sycl::ext::oneapi::experimental;

struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    uint32_t queueFamilyIndex;
};

// ---------------------------------------------------------
// UUID MATCHING LOGIC
// ---------------------------------------------------------
inline VulkanContext createUUIDMatchedContext(const sycl::device& syclDev) {
    VulkanContext ctx;

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> instanceExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME
    };

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = (uint32_t)instanceExts.size();
    createInfo.ppEnabledExtensionNames = instanceExts.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &ctx.instance));

    auto syclUUID = syclDev.get_info<sycl::ext::intel::info::device::uuid>();

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

        if (std::memcmp(idProps.deviceUUID, syclUUID.data(), VK_UUID_SIZE) == 0) {
            ctx.physicalDevice = dev;
            std::cout << "[Setup] MATCH FOUND: " << props2.properties.deviceName << std::endl;
            break;
        }
    }

    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "[Error] No Vulkan device found matching the SYCL device UUID!" << std::endl;
        exit(1);
    }

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
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        #ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
        #else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
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
// VULKAN IMPORT HELPERS
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

// Query the handle directly for its compatible memory type
uint32_t getMemoryTypeFromHandle(VulkanContext& ctx, void* handle) {
#ifdef _WIN32
    auto func = (PFN_vkGetMemoryWin32HandlePropertiesKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryWin32HandlePropertiesKHR");
    if (!func) throw std::runtime_error("Could not load vkGetMemoryWin32HandlePropertiesKHR");

    VkMemoryWin32HandlePropertiesKHR props = {VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    VkResult res = func(ctx.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, handle, &props);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to get Win32 handle properties");

    std::cout << "[Debug] Handle MemoryTypeBits: " << std::hex << props.memoryTypeBits << std::dec << std::endl;

    if (props.memoryTypeBits == 0) {
        throw std::runtime_error("Handle reports NO compatible memory types! (Is the handle valid?)");
    }

    // RELAXED CONSTRAINT:
    // We try to find DEVICE_LOCAL first for performance, but if that fails, 
    // we accept ANY memory type the handle supports.
    try {
        return findMemoryType(ctx.physicalDevice, props.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    } catch (...) {
        std::cout << "[Debug] Device Local not found for handle. Falling back to any compatible type." << std::endl;
        return findMemoryType(ctx.physicalDevice, props.memoryTypeBits, 0);
    }

#else
    // Linux equivalent (FD)
    auto func = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdPropertiesKHR");
    if (!func) throw std::runtime_error("Could not load vkGetMemoryFdPropertiesKHR");
    
    int fd = (int)(intptr_t)handle; 
    VkMemoryFdPropertiesKHR props = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    VkResult res = func(ctx.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, fd, &props);
    if (res != VK_SUCCESS) throw std::runtime_error("Failed to get FD properties");

    if (props.memoryTypeBits == 0) {
        throw std::runtime_error("Handle reports NO compatible memory types!");
    }

    try {
        return findMemoryType(ctx.physicalDevice, props.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    } catch (...) {
        return findMemoryType(ctx.physicalDevice, props.memoryTypeBits, 0);
    }
#endif
}

// CHANGE: Pass the buffer handle so we can link the memory to it dedicated-ly
VkDeviceMemory importMemoryFromHandle(VulkanContext& ctx, VkDeviceSize size, uint32_t memTypeIndex, void* handle, VkBuffer buffer) {
    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    // 1. The Import Struct
#ifdef _WIN32
    VkImportMemoryWin32HandleInfoKHR importInfo = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    importInfo.handle = handle;
    allocInfo.pNext = &importInfo;
#else
    VkImportMemoryFdInfoKHR importInfo = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    importInfo.fd = (int)(intptr_t)handle;
    allocInfo.pNext = &importInfo;
#endif

    // 2. The Dedicated Allocation Struct (CRITICAL FIX)
    // We chain this to the import struct (or vice versa, order doesn't matter for pNext)
    VkMemoryDedicatedAllocateInfo dedicatedInfo = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedInfo.image = VK_NULL_HANDLE;
    dedicatedInfo.buffer = buffer; // Tell driver: "This memory is for THIS buffer"
    
    // Chain it: alloc -> import -> dedicated
    dedicatedInfo.pNext = allocInfo.pNext; 
    allocInfo.pNext = &dedicatedInfo;

    VkDeviceMemory mem;
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &mem));
    return mem;
}

// ---------------------------------------------------------
// MAIN TEST
// ---------------------------------------------------------
int main(int argc, char** argv) {
    size_t alignment = 65536; 
    size_t bufferSize = 65536; 
    size_t numElements = bufferSize / sizeof(uint32_t); // 16384 elements

    try {
        // 1. SYCL Setup
        sycl::queue q;
        auto dev = q.get_device();
        auto ctx = q.get_context();
        std::cout << "[SYCL] Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

        if (!dev.has(sycl::aspect::ext_oneapi_exportable_device_mem)) {
            std::cerr << "Device does not support exportable memory!" << std::endl;
            return 1;
        }

        // 2. Vulkan Setup (Matched)
        VulkanContext vkCtx = createUUIDMatchedContext(dev);

        // 3. SYCL Allocation
        #ifdef _WIN32
        auto handleType = syclexp::external_mem_handle_type::win32_nt_handle;
        #else
        auto handleType = syclexp::external_mem_handle_type::opaque_fd;
        #endif
        
        void* syclPtr = syclexp::alloc_exportable_device_mem(alignment, bufferSize, handleType, dev, ctx);

        // 4. Fill Data in SYCL
        std::vector<uint32_t> initData(numElements);
        std::iota(initData.begin(), initData.end(), 0); 
        q.memcpy(syclPtr, initData.data(), bufferSize).wait();
        std::cout << "[SYCL] Memory allocated and filled." << std::endl;

        // 5. Export Handle
        #ifdef _WIN32
        void* nativeHandle = syclexp::export_device_mem_handle<syclexp::external_mem_handle_type::win32_nt_handle>(syclPtr, dev, ctx);
        #else
        int fd = syclexp::export_device_mem_handle<syclexp::external_mem_handle_type::opaque_fd>(syclPtr, dev, ctx);
        void* nativeHandle = (void*)(intptr_t)fd; // store in void* for generic passing
        #endif
        std::cout << "[SYCL] Handle exported: " << nativeHandle << std::endl;

        // 6. Vulkan Import (CORRECTED FLOW)
        
        // Step A: Create the buffer container
        VkBuffer buffer;
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VkExternalMemoryBufferCreateInfo extBufInfo = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        extBufInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
        bufferInfo.pNext = &extBufInfo;

        VK_CHECK(vkCreateBuffer(vkCtx.device, &bufferInfo, nullptr, &buffer));

        // Step B: Ask the HANDLE which memory type it needs
        // (This replaces the old logic of guessing based on buffer requirements)
        uint32_t memTypeIndex = getMemoryTypeFromHandle(vkCtx, nativeHandle);
        std::cout << "[Vulkan] Handle is compatible with Memory Type Index: " << memTypeIndex << std::endl;

        // Step C: Allocate (Import) and Bind
        VkDeviceMemory importedMem = importMemoryFromHandle(vkCtx, bufferSize, memTypeIndex, nativeHandle, buffer);
        VK_CHECK(vkBindBufferMemory(vkCtx.device, buffer, importedMem, 0));
        std::cout << "[Vulkan] Handle imported and bound." << std::endl;

        // 7. Verification (Copy Vulkan Buffer -> Host Staging)
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkBufferCreateInfo stageInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stageInfo.size = bufferSize;
        stageInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(vkCtx.device, &stageInfo, nullptr, &stagingBuffer);

        VkMemoryRequirements stageReq;
        vkGetBufferMemoryRequirements(vkCtx.device, stagingBuffer, &stageReq);
        VkMemoryAllocateInfo stageAlloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        stageAlloc.allocationSize = stageReq.size;
        stageAlloc.memoryTypeIndex = findMemoryType(vkCtx.physicalDevice, stageReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(vkCtx.device, &stageAlloc, nullptr, &stagingMemory);
        vkBindBufferMemory(vkCtx.device, stagingBuffer, stagingMemory, 0);

        // Copy (Imported -> Staging)
        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
        VkCommandPool pool; vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
        VkCommandBuffer cmd; VkCommandBufferAllocateInfo alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = pool; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(vkCtx.device, &alloc, &cmd);

        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin);
        VkBufferCopy copyRegion = {0, 0, bufferSize};
        vkCmdCopyBuffer(cmd, buffer, stagingBuffer, 1, &copyRegion);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
        vkQueueSubmit(vkCtx.queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(vkCtx.queue);

        // Map and Check
        void* mappedData;
        vkMapMemory(vkCtx.device, stagingMemory, 0, bufferSize, 0, &mappedData);
        uint32_t* result = (uint32_t*)mappedData;
        
        int errors = 0;
        for(size_t i=0; i<numElements; ++i) {
            if(result[i] != i) {
                if(errors++ < 10) std::cerr << "Mismatch at " << i << ": Expected " << i << " Got " << result[i] << std::endl;
            }
        }

        vkUnmapMemory(vkCtx.device, stagingMemory);

        if(errors == 0) std::cout << "SUCCESS! Data verified." << std::endl;
        else std::cout << "FAILURE! " << errors << " errors found." << std::endl;

        // Cleanup
        syclexp::free_exportable_memory(syclPtr, dev, ctx);

        // TODO: restore the resource freeing below.
	    // workaround CMPLRLLVM-73463:  Do not destroy Vulkan Device.
       
        // vkDestroyBuffer(vkCtx.device, stagingBuffer, nullptr);
        // vkFreeMemory(vkCtx.device, stagingMemory, nullptr);
        // vkDestroyCommandPool(vkCtx.device, pool, nullptr);
        // vkDestroyBuffer(vkCtx.device, buffer, nullptr);
        // vkFreeMemory(vkCtx.device, importedMem, nullptr);
        // vkDestroyDevice(vkCtx.device, nullptr);
        // vkDestroyInstance(vkCtx.instance, nullptr);
	

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}