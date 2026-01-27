/*
 * Minimal Vulkan Test: VK_FORMAT_R32_SFLOAT 2D Sampled Image
 * 
 * Compilation (Linux, clang++):
 * clang++ -std=c++17 -o v_test.bin vulkan_r32_sfloat_2d_test.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
 * 
 * Prerequisites:
 * - Vulkan SDK installed
 * - Vulkan loader library available
 * 
 * Run with:
 * ./v_test.bin

 ./v_test.bin 
Starting Vulkan R32_SFLOAT 2D Sampled Image Test...
✓ Created Vulkan instance
✓ Using device: Intel(R) Graphics (BMG G21)
✓ Created logical device
✓ Created 2D image (4x4, VK_FORMAT_R32_SFLOAT)
✓ Allocated and bound image memory
✓ Filled staging buffer with test data
✓ Uploaded test data to image
✓ Created image view
✓ Created sampler
✓ Created output buffer
✓ Updated descriptor sets
✓ Created compute pipeline
✓ Executed compute shader (sampling image)

=== Verification ===
Sampled values from image:
[0,0] = 0 (expected: 0) ✓
[1,0] = 0 (expected: 0.0666667) ✗
[2,0] = 0 (expected: 0.133333) ✗
[3,0] = 0 (expected: 0.2) ✗
[0,1] = 0 (expected: 0.266667) ✗
[1,1] = 0 (expected: 0.333333) ✗
[2,1] = 0 (expected: 0.4) ✗
[3,1] = 0 (expected: 0.466667) ✗
[0,2] = 0 (expected: 0.533333) ✗
[1,2] = 0 (expected: 0.6) ✗
[2,2] = 0 (expected: 0.666667) ✗
[3,2] = 0 (expected: 0.733333) ✗
[0,3] = 0 (expected: 0.8) ✗
[1,3] = 0 (expected: 0.866667) ✗
[2,3] = 0 (expected: 0.933333) ✗
[3,3] = 0 (expected: 1) ✗

=== Test Result ===
✗ TEST FAILED: Some sampled values do not match!

 */

 /*
 * Minimal Vulkan Test: VK_FORMAT_R32_SFLOAT 2D Sampled Image
 * 
 * Compilation (Linux, clang++):
 * clang++ -std=c++17 -o vulkan_r32_sfloat_2d_test vulkan_r32_sfloat_2d_test.cpp -lvulkan
 * 
 * Prerequisites:
 * - Vulkan SDK installed
 * - Vulkan loader library available
 * 
 * Run with:
 * ./vulkan_r32_sfloat_2d_test
 */

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define CHECK_VK(result, msg) \
    if (result != VK_SUCCESS) { \
        std::cerr << "Vulkan error at " << __LINE__ << ": " << msg << " (code: " << result << ")" << std::endl; \
        return 1; \
    }

// Simple compute shader that samples from an image and writes to a buffer
// SPIR-V compiled from:
// #version 450
// layout(binding = 0) uniform sampler2D inputImage;
// layout(binding = 1, std430) buffer OutputBuffer { float values[]; };
// layout(local_size_x = 1, local_size_y = 1) in;
// void main() {
//     vec2 uv = vec2(gl_GlobalInvocationID.xy) / vec2(4.0, 4.0);
//     values[gl_GlobalInvocationID.y * 4 + gl_GlobalInvocationID.x] = texture(inputImage, uv).r;
// }
const uint32_t SAMPLE_SHADER_SPIRV[] = {
    0x07230203, 0x00010000, 0x0008000a, 0x00000039, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00060010, 0x00000004,
    0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000009, 0x00007675, 0x00080005, 0x0000000d,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00050005, 0x00000018,
    0x70747561, 0x6d497475, 0x00656761, 0x00060005, 0x0000001f, 0x7074754f, 0x75427475, 0x72656666,
    0x00000000, 0x00060006, 0x0000001f, 0x00000000, 0x756c6176, 0x00007365, 0x00000000, 0x00030005,
    0x00000021, 0x00000000, 0x00040047, 0x0000000d, 0x0000000b, 0x0000001c, 0x00040047, 0x00000018,
    0x00000022, 0x00000000, 0x00040047, 0x00000018, 0x00000021, 0x00000000, 0x00050048, 0x0000001f,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000001f, 0x00000003, 0x00040047, 0x00000020,
    0x00000022, 0x00000000, 0x00040047, 0x00000020, 0x00000021, 0x00000001, 0x00040047, 0x00000038,
    0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040020, 0x00000008,
    0x00000007, 0x00000007, 0x00040015, 0x0000000a, 0x00000020, 0x00000000, 0x00040017, 0x0000000b,
    0x0000000a, 0x00000003, 0x00040020, 0x0000000c, 0x00000001, 0x0000000b, 0x0004003b, 0x0000000c,
    0x0000000d, 0x00000001, 0x00040017, 0x0000000e, 0x0000000a, 0x00000002, 0x00090019, 0x00000015,
    0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x0003001b,
    0x00000016, 0x00000015, 0x00040020, 0x00000017, 0x00000000, 0x00000016, 0x0004003b, 0x00000017,
    0x00000018, 0x00000000, 0x00040017, 0x0000001a, 0x00000006, 0x00000004, 0x0004002b, 0x00000006,
    0x0000001c, 0x40800000, 0x0005002c, 0x00000007, 0x0000001d, 0x0000001c, 0x0000001c, 0x0003001d,
    0x0000001e, 0x00000006, 0x0003001e, 0x0000001f, 0x0000001e, 0x00040020, 0x00000020, 0x00000002,
    0x0000001f, 0x0004003b, 0x00000020, 0x00000021, 0x00000002, 0x00040015, 0x00000022, 0x00000020,
    0x00000001, 0x0004002b, 0x00000022, 0x00000023, 0x00000000, 0x00040020, 0x00000025, 0x00000001,
    0x0000000a, 0x0004002b, 0x0000000a, 0x00000028, 0x00000001, 0x0004002b, 0x0000000a, 0x0000002d,
    0x00000000, 0x0004002b, 0x0000000a, 0x00000031, 0x00000004, 0x00040020, 0x00000035, 0x00000002,
    0x00000006, 0x0004002b, 0x0000000a, 0x00000037, 0x00000001, 0x0006002c, 0x0000000b, 0x00000038,
    0x00000037, 0x00000037, 0x00000037, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003d, 0x0000000b,
    0x0000000f, 0x0000000d, 0x0007004f, 0x0000000e, 0x00000010, 0x0000000f, 0x0000000f, 0x00000000,
    0x00000001, 0x00040070, 0x00000007, 0x00000011, 0x00000010, 0x00050088, 0x00000007, 0x00000014,
    0x00000011, 0x0000001d, 0x0003003e, 0x00000009, 0x00000014, 0x0004003d, 0x00000016, 0x00000019,
    0x00000018, 0x0004003d, 0x00000007, 0x0000001b, 0x00000009, 0x00050057, 0x0000001a, 0x00000024,
    0x00000019, 0x0000001b, 0x00050041, 0x00000025, 0x00000026, 0x0000000d, 0x00000028, 0x0004003d,
    0x0000000a, 0x00000027, 0x00000026, 0x00050041, 0x00000025, 0x00000029, 0x0000000d, 0x0000002d,
    0x0004003d, 0x0000000a, 0x0000002a, 0x00000029, 0x00050084, 0x0000000a, 0x0000002c, 0x00000027,
    0x00000031, 0x00050080, 0x0000000a, 0x0000002e, 0x0000002c, 0x0000002a, 0x00050051, 0x00000006,
    0x00000034, 0x00000024, 0x00000000, 0x00060041, 0x00000035, 0x00000036, 0x00000021, 0x00000023,
    0x0000002e, 0x0003003e, 0x00000036, 0x00000034, 0x000100fd, 0x00010038
};

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
    std::cout << "Starting Vulkan R32_SFLOAT 2D Sampled Image Test..." << std::endl;

    // Constants for test
    const uint32_t IMAGE_WIDTH = 4;
    const uint32_t IMAGE_HEIGHT = 4;
    const VkFormat IMAGE_FORMAT = VK_FORMAT_R32_SFLOAT;

    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Image Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    CHECK_VK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "Failed to create instance");
    std::cout << "✓ Created Vulkan instance" << std::endl;

    // Get physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "No Vulkan devices found!" << std::endl;
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice physicalDevice = devices[0];

    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    std::cout << "✓ Using device: " << deviceProps.deviceName << std::endl;

    // Find queue family with compute support
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

    if (computeQueueFamily == UINT32_MAX) {
        std::cerr << "No compute queue family found!" << std::endl;
        return 1;
    }

    // Create logical device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device;
    CHECK_VK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device), "Failed to create device");
    std::cout << "✓ Created logical device" << std::endl;

    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

    // Create image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = IMAGE_FORMAT;
    imageInfo.extent.width = IMAGE_WIDTH;
    imageInfo.extent.height = IMAGE_HEIGHT;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    CHECK_VK(vkCreateImage(device, &imageInfo, nullptr, &image), "Failed to create image");
    std::cout << "✓ Created 2D image (" << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << ", VK_FORMAT_R32_SFLOAT)" << std::endl;

    // Allocate image memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceMemory imageMemory;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory), "Failed to allocate image memory");
    CHECK_VK(vkBindImageMemory(device, image, imageMemory, 0), "Failed to bind image memory");
    std::cout << "✓ Allocated and bound image memory" << std::endl;

    // Create staging buffer for uploading data
    VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferInfo.size = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);
    stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer;
    CHECK_VK(vkCreateBuffer(device, &stagingBufferInfo, nullptr, &stagingBuffer), "Failed to create staging buffer");

    VkMemoryRequirements stagingMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemReq);

    VkMemoryAllocateInfo stagingAllocInfo = {};
    stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAllocInfo.allocationSize = stagingMemReq.size;
    stagingAllocInfo.memoryTypeIndex = findMemoryType(physicalDevice, stagingMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory stagingMemory;
    CHECK_VK(vkAllocateMemory(device, &stagingAllocInfo, nullptr, &stagingMemory), "Failed to allocate staging memory");
    CHECK_VK(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0), "Failed to bind staging buffer");

    // Fill staging buffer with test data (simple gradient pattern)
    void* data;
    vkMapMemory(device, stagingMemory, 0, stagingBufferInfo.size, 0, &data);
    float* floatData = static_cast<float*>(data);
    for (uint32_t y = 0; y < IMAGE_HEIGHT; y++) {
        for (uint32_t x = 0; x < IMAGE_WIDTH; x++) {
            floatData[y * IMAGE_WIDTH + x] = static_cast<float>(x + y * IMAGE_WIDTH) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
        }
    }
    vkUnmapMemory(device, stagingMemory);
    std::cout << "✓ Filled staging buffer with test data" << std::endl;

    // Create command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool commandPool;
    CHECK_VK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "Failed to create command pool");

    // Create command buffer
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer), "Failed to allocate command buffer");

    // Copy data to image
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Transition image to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy buffer to image
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition image to SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE), "Failed to submit copy command");
    CHECK_VK(vkQueueWaitIdle(computeQueue), "Failed to wait for queue");
    std::cout << "✓ Uploaded test data to image" << std::endl;

    // DIAGNOSTIC 1: Copy image back to a buffer to verify upload worked
    std::cout << "\n=== Diagnostic: Verifying Upload ===" << std::endl;
    
    VkBufferCreateInfo verifyBufferInfo = {};
    verifyBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    verifyBufferInfo.size = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);
    verifyBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    verifyBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer verifyBuffer;
    CHECK_VK(vkCreateBuffer(device, &verifyBufferInfo, nullptr, &verifyBuffer), "Failed to create verify buffer");

    VkMemoryRequirements verifyMemReq;
    vkGetBufferMemoryRequirements(device, verifyBuffer, &verifyMemReq);

    VkMemoryAllocateInfo verifyAllocInfo = {};
    verifyAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    verifyAllocInfo.allocationSize = verifyMemReq.size;
    verifyAllocInfo.memoryTypeIndex = findMemoryType(physicalDevice, verifyMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory verifyMemory;
    CHECK_VK(vkAllocateMemory(device, &verifyAllocInfo, nullptr, &verifyMemory), "Failed to allocate verify memory");
    CHECK_VK(vkBindBufferMemory(device, verifyBuffer, verifyMemory, 0), "Failed to bind verify buffer");

    // Copy image to verify buffer
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Transition to TRANSFER_SRC
    VkImageMemoryBarrier transferBarrier = {};
    transferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    transferBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    transferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    transferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transferBarrier.image = image;
    transferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    transferBarrier.subresourceRange.baseMipLevel = 0;
    transferBarrier.subresourceRange.levelCount = 1;
    transferBarrier.subresourceRange.baseArrayLayer = 0;
    transferBarrier.subresourceRange.layerCount = 1;
    transferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    transferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &transferBarrier);

    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};

    vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, verifyBuffer, 1, &copyRegion);

    // Transition back to SHADER_READ_ONLY
    transferBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    transferBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    transferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &transferBarrier);

    vkEndCommandBuffer(commandBuffer);

    CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE), "Failed to submit verify command");
    CHECK_VK(vkQueueWaitIdle(computeQueue), "Failed to wait for queue");

    // Read back verification data
    void* verifyData;
    vkMapMemory(device, verifyMemory, 0, verifyBufferInfo.size, 0, &verifyData);
    float* verifyFloats = static_cast<float*>(verifyData);

    std::cout << "Direct readback from image (bypass sampling):" << std::endl;
    bool uploadWorked = true;
    const float tolerance = 0.01f;
    for (uint32_t y = 0; y < IMAGE_HEIGHT; y++) {
        for (uint32_t x = 0; x < IMAGE_WIDTH; x++) {
            uint32_t idx = y * IMAGE_WIDTH + x;
            float expected = static_cast<float>(idx) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
            float actual = verifyFloats[idx];
            bool match = std::fabs(actual - expected) < tolerance;
            
            std::cout << "[" << x << "," << y << "] = " << actual 
                     << " (expected: " << expected << ") "
                     << (match ? "✓" : "✗") << std::endl;
            
            if (!match) uploadWorked = false;
        }
    }

    vkUnmapMemory(device, verifyMemory);
    
    if (uploadWorked) {
        std::cout << "✓ Upload verification PASSED - data is in the image correctly" << std::endl;
    } else {
        std::cout << "✗ Upload verification FAILED - data didn't make it to the image!" << std::endl;
    }
    std::cout << std::endl;

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = IMAGE_FORMAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    CHECK_VK(vkCreateImageView(device, &viewInfo, nullptr, &imageView), "Failed to create image view");
    std::cout << "✓ Created image view" << std::endl;

    // Create sampler
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkSampler sampler;
    CHECK_VK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler), "Failed to create sampler");
    std::cout << "✓ Created sampler" << std::endl;

    // Create output buffer to read sampled values
    VkBufferCreateInfo outputBufferInfo = {};
    outputBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outputBufferInfo.size = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);
    outputBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    outputBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer outputBuffer;
    CHECK_VK(vkCreateBuffer(device, &outputBufferInfo, nullptr, &outputBuffer), "Failed to create output buffer");

    VkMemoryRequirements outputMemReq;
    vkGetBufferMemoryRequirements(device, outputBuffer, &outputMemReq);

    VkMemoryAllocateInfo outputAllocInfo = {};
    outputAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    outputAllocInfo.allocationSize = outputMemReq.size;
    outputAllocInfo.memoryTypeIndex = findMemoryType(physicalDevice, outputMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory outputMemory;
    CHECK_VK(vkAllocateMemory(device, &outputAllocInfo, nullptr, &outputMemory), "Failed to allocate output memory");
    CHECK_VK(vkBindBufferMemory(device, outputBuffer, outputMemory, 0), "Failed to bind output buffer");
    std::cout << "✓ Created output buffer" << std::endl;

    // Create descriptor set layout
    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bufferLayoutBinding = {};
    bufferLayoutBinding.binding = 1;
    bufferLayoutBinding.descriptorCount = 1;
    bufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {samplerLayoutBinding, bufferLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout), "Failed to create descriptor set layout");

    // Create descriptor pool
    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 2;
    poolCreateInfo.pPoolSizes = poolSizes;
    poolCreateInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    CHECK_VK(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool), "Failed to create descriptor pool");

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo descAllocInfo = {};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descriptorPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    CHECK_VK(vkAllocateDescriptorSets(device, &descAllocInfo, &descriptorSet), "Failed to allocate descriptor set");

    // Update descriptor set
    VkDescriptorImageInfo imageDescInfo = {};
    imageDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageDescInfo.imageView = imageView;
    imageDescInfo.sampler = sampler;

    VkDescriptorBufferInfo bufferDescInfo = {};
    bufferDescInfo.buffer = outputBuffer;
    bufferDescInfo.offset = 0;
    bufferDescInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrites[2] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &imageDescInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &bufferDescInfo;

    vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);
    std::cout << "✓ Updated descriptor sets" << std::endl;

    // Create compute shader module
    VkShaderModuleCreateInfo shaderModuleInfo = {};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = sizeof(SAMPLE_SHADER_SPIRV);
    shaderModuleInfo.pCode = SAMPLE_SHADER_SPIRV;

    VkShaderModule computeShaderModule;
    CHECK_VK(vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &computeShaderModule), "Failed to create shader module");

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    CHECK_VK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "Failed to create pipeline layout");

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = computeShaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline), "Failed to create compute pipeline");
    std::cout << "✓ Created compute pipeline" << std::endl;

    // Execute compute shader to sample from image
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // DIAGNOSTIC 2: Add explicit memory barrier to ensure image is ready
    VkImageMemoryBarrier preComputeBarrier = {};
    preComputeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    preComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    preComputeBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    preComputeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preComputeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preComputeBarrier.image = image;
    preComputeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    preComputeBarrier.subresourceRange.baseMipLevel = 0;
    preComputeBarrier.subresourceRange.levelCount = 1;
    preComputeBarrier.subresourceRange.baseArrayLayer = 0;
    preComputeBarrier.subresourceRange.layerCount = 1;
    preComputeBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    preComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, 
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &preComputeBarrier);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, IMAGE_WIDTH, IMAGE_HEIGHT, 1);

    // Add barrier to ensure compute writes are complete
    VkBufferMemoryBarrier bufferBarrier = {};
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = outputBuffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);

    vkEndCommandBuffer(commandBuffer);

    CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE), "Failed to submit compute command");
    CHECK_VK(vkQueueWaitIdle(computeQueue), "Failed to wait for queue");
    std::cout << "✓ Executed compute shader (sampling image)" << std::endl;

    // Read back and verify results
    void* outputData;
    vkMapMemory(device, outputMemory, 0, outputBufferInfo.size, 0, &outputData);
    float* outputFloats = static_cast<float*>(outputData);

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Sampled values from image:" << std::endl;
    bool testPassed = true;
    const float tolerance2 = 0.01f;

    for (uint32_t y = 0; y < IMAGE_HEIGHT; y++) {
        for (uint32_t x = 0; x < IMAGE_WIDTH; x++) {
            uint32_t idx = y * IMAGE_WIDTH + x;
            float expected = static_cast<float>(idx) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
            float actual = outputFloats[idx];
            bool match = std::fabs(actual - expected) < tolerance2;
            
            std::cout << "[" << x << "," << y << "] = " << actual 
                     << " (expected: " << expected << ") "
                     << (match ? "✓" : "✗") << std::endl;
            
            if (!match) testPassed = false;
        }
    }

    vkUnmapMemory(device, outputMemory);

    std::cout << "\n=== Test Result ===" << std::endl;
    if (testPassed) {
        std::cout << "✓ TEST PASSED: All sampled values match expected values!" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED: Some sampled values do not match!" << std::endl;
    }

    // Cleanup
    vkDestroyBuffer(device, verifyBuffer, nullptr);
    vkFreeMemory(device, verifyMemory, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, outputMemory, nullptr);
    vkDestroySampler(device, sampler, nullptr);
    vkDestroyImageView(device, imageView, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    vkDestroyImage(device, image, nullptr);
    vkFreeMemory(device, imageMemory, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return testPassed ? 0 : 1;
}
