/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 2D Sampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader.comp -o vulkan_shader.spv

  clang++ -fsycl -std=c++17 -o vsu_test.bin vulkan_sycl_unsamp_rgba32_sfloat_2d_test.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vs_test.bin vulkan_sycl_unsamp_rgba32_sfloat_2d_test.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vsu_test.bin 




 */
/*
  Vulkan/SYCL Test: Unsampled (Storage) Image Interop
  
  1. Creates a Vulkan Image (USAGE_STORAGE_BIT).
  2. Uploads data via Staging Buffer.
  3. Transitions to VK_IMAGE_LAYOUT_GENERAL.
  4. Exports Opaque FD.
  5. Imports into SYCL as 'unsampled_image_handle'.
  6. Reads data using 'fetch_image' (Data Port).
*/

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <fstream>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>

#define CHECK_VK(result, msg) \
    if (result != VK_SUCCESS) { \
        std::cerr << "Vulkan error at " << __LINE__ << ": " << msg << " (code: " << result << ")" << std::endl; \
        return 1; \
    }

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

int main() {
    std::cout << "Starting Vulkan/SYCL Unsampled (Storage) Image Test..." << std::endl;

    // Constants
    const uint32_t IMAGE_WIDTH = 4;
    const uint32_t IMAGE_HEIGHT = 4;
    const VkFormat IMAGE_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

    // --- VULKAN SETUP ---
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Unsampled Test";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    CHECK_VK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "Failed to create instance");

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice physicalDevice = devices[0];

    // Find Compute Queue
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t computeQueueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Extensions for Interop
    const char* deviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = 2;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;

    VkDevice device;
    CHECK_VK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device), "Failed to create device");
    std::cout << "✓ Created logical device" << std::endl;

    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

    // --- RESOURCE CREATION ---

    // 1. Create Target Image (Storage + Sampled + Transfer)
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = IMAGE_FORMAT;
    imageInfo.extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    CHECK_VK(vkCreateImage(device, &imageInfo, nullptr, &image), "Failed to create image");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocInfo.pNext = &exportAllocInfo;

    VkDeviceMemory imageMemory;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory), "Failed to allocate image memory");
    CHECK_VK(vkBindImageMemory(device, image, imageMemory, 0), "Failed to bind image memory");

    // 2. Create Staging Buffer (Upload Source)
    VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferInfo.size = IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float);
    stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer;
    CHECK_VK(vkCreateBuffer(device, &stagingBufferInfo, nullptr, &stagingBuffer), "Failed to create staging buffer");

    VkMemoryRequirements stagingMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemReq);
    
    VkMemoryAllocateInfo stagingAllocInfo = {};
    stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAllocInfo.allocationSize = stagingMemReq.size;
    stagingAllocInfo.memoryTypeIndex = findMemoryType(physicalDevice, stagingMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory stagingMemory;
    CHECK_VK(vkAllocateMemory(device, &stagingAllocInfo, nullptr, &stagingMemory), "Failed to allocate staging memory");
    CHECK_VK(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0), "Failed to bind staging buffer");

    // Fill Staging Buffer
    void* data;
    vkMapMemory(device, stagingMemory, 0, stagingBufferInfo.size, 0, &data);
    float* floatData = static_cast<float*>(data);
    for (uint32_t y = 0; y < IMAGE_HEIGHT; y++) {
        for (uint32_t x = 0; x < IMAGE_WIDTH; x++) {
            float value = static_cast<float>(x + y * IMAGE_WIDTH) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
            uint32_t idx = (y * IMAGE_WIDTH + x) * 4;
            floatData[idx + 0] = value; // R
            floatData[idx + 1] = 0.0f;  // G
            floatData[idx + 2] = 0.0f;  // B
            floatData[idx + 3] = 1.0f;  // A
        }
    }
    vkUnmapMemory(device, stagingMemory);

    // 3. Create Verify Buffer (Diagnostic Readback)
    VkBufferCreateInfo verifyBufferInfo = {};
    verifyBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    verifyBufferInfo.size = stagingBufferInfo.size;
    verifyBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    VkBuffer verifyBuffer;
    CHECK_VK(vkCreateBuffer(device, &verifyBufferInfo, nullptr, &verifyBuffer), "Failed to create verify buffer");
    
    VkMemoryRequirements verifyMemReq;
    vkGetBufferMemoryRequirements(device, verifyBuffer, &verifyMemReq);
    
    VkMemoryAllocateInfo verifyAllocInfo = {};
    verifyAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    verifyAllocInfo.allocationSize = verifyMemReq.size;
    verifyAllocInfo.memoryTypeIndex = findMemoryType(physicalDevice, verifyMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkDeviceMemory verifyMemory;
    CHECK_VK(vkAllocateMemory(device, &verifyAllocInfo, nullptr, &verifyMemory), "Failed to allocate verify memory");
    CHECK_VK(vkBindBufferMemory(device, verifyBuffer, verifyMemory, 0), "Failed to bind verify buffer");

    // --- COMMAND RECORDING ---
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    VkCommandPool commandPool;
    CHECK_VK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "Failed to create pool");

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer), "Failed to allocate cmd buffer");

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // 1. Transition UNDEFINED -> TRANSFER_DST (For Upload)
    VkImageMemoryBarrier uploadBarrier = {};
    uploadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    uploadBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    uploadBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    uploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uploadBarrier.image = image;
    uploadBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    uploadBarrier.srcAccessMask = 0;
    uploadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &uploadBarrier);

    // 2. Copy Staging -> Image
    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { IMAGE_WIDTH, IMAGE_HEIGHT, 1 };
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // 3. Transition TRANSFER_DST -> TRANSFER_SRC (For Readback)
    VkImageMemoryBarrier verifyBarrier = {};
    verifyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    verifyBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    verifyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    verifyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    verifyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    verifyBarrier.image = image;
    verifyBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    verifyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    verifyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &verifyBarrier);

    // 4. Copy Image -> Verify Buffer
    vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, verifyBuffer, 1, &region);

    // 5. Transition TRANSFER_SRC -> GENERAL (For SYCL Unsampled)
    VkImageMemoryBarrier syclBarrier = {};
    syclBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    syclBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    syclBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    syclBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    syclBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    syclBarrier.image = image;
    syclBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    syclBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    syclBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    // Note: BOTTOM_OF_PIPE ensures this transition is done before the buffer retires
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &syclBarrier);

    vkEndCommandBuffer(commandBuffer);

    // --- SUBMIT ---
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE), "Failed to submit");
    CHECK_VK(vkQueueWaitIdle(computeQueue), "Failed to wait for queue");

    std::cout << "✓ Vulkan Setup & Upload Complete" << std::endl;

    // --- DIAGNOSTIC VERIFICATION ---
    bool vulkanPassed = true;
    void* verifyData;
    vkMapMemory(device, verifyMemory, 0, verifyBufferInfo.size, 0, &verifyData);
    float* verifyFloats = static_cast<float*>(verifyData);
    
    std::cout << "Checking Vulkan Readback..." << std::endl;
    for (uint32_t i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
        float expected = static_cast<float>(i) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
        float actual = verifyFloats[i * 4]; // Check R channel
        if (std::fabs(actual - expected) > 0.01f) {
            vulkanPassed = false;
            std::cout << "Vulkan Mismatch [" << i << "] " << actual << " != " << expected << std::endl;
        }
    }
    vkUnmapMemory(device, verifyMemory);
    
    if (vulkanPassed) std::cout << "✓ Vulkan Readback Passed" << std::endl;
    else std::cout << "✗ Vulkan Readback FAILED" << std::endl;

    // --- SYCL INTEROP ---
    namespace syclexp = sycl::ext::oneapi::experimental;
    bool syclPassed = true;

    // 1. Get FD
    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) return 1;

    VkMemoryGetFdInfoKHR fdInfo = {};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = imageMemory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    CHECK_VK(vkGetMemoryFdKHR(device, &fdInfo, &fd), "Failed to get FD");
    std::cout << "✓ Got FD: " << fd << std::endl;

    try {
        sycl::queue q;
        std::cout << "✓ SYCL Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        // 2. Import Memory
        size_t imgSize = IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float);
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            fd, syclexp::external_mem_handle_type::opaque_fd, imgSize
        };
        syclexp::external_mem extMem = syclexp::import_external_memory(extMemDesc, q.get_device(), q.get_context());

        // 3. Map to Image Handle
        syclexp::image_descriptor imgDesc(
            sycl::range<2>(IMAGE_WIDTH, IMAGE_HEIGHT),
            4, sycl::image_channel_type::fp32
        );
        syclexp::image_mem_handle deviceMemHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());

        // 4. Create Unsampled Handle
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(deviceMemHandle, imgDesc, q.get_device(), q.get_context());
        
        // 5. Run Kernel
        sycl::buffer<float, 1> checkBuf(IMAGE_WIDTH * IMAGE_HEIGHT);
        q.submit([&](sycl::handler& h) {
            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            h.parallel_for(sycl::range<2>(IMAGE_WIDTH, IMAGE_HEIGHT), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                sycl::int2 coords(x, y);
                sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(unsampledHandle, coords);
                outAcc[y * IMAGE_WIDTH + x] = pixel.x();
            });
        }).wait();
        std::cout << "✓ SYCL Kernel Executed" << std::endl;

        // 6. Cleanup SYCL
        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());

        // 7. Verify SYCL Data
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        for (uint32_t i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
            float expected = static_cast<float>(i) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
            if (std::fabs(hostAcc[i] - expected) > 0.01f) {
                syclPassed = false;
                std::cout << "SYCL Mismatch [" << i << "] " << hostAcc[i] << " != " << expected << std::endl;
            }
        }
        if (syclPassed) std::cout << "✓ SYCL Verification Passed" << std::endl;
        else std::cout << "✗ SYCL Verification FAILED" << std::endl;

    } catch (sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    // --- CLEANUP ---
    vkDestroyBuffer(device, verifyBuffer, nullptr);
    vkFreeMemory(device, verifyMemory, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    vkDestroyImage(device, image, nullptr);
    vkFreeMemory(device, imageMemory, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return (vulkanPassed && syclPassed) ? 0 : 1;
}