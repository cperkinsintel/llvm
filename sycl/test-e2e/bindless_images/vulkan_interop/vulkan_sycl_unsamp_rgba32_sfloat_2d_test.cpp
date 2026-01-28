/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 2D Sampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader.comp -o vulkan_shader.spv

  clang++ -fsycl -std=c++17 -o vsu_test.bin vulkan_sycl_unsamp_rgba32_sfloat_2d_test.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vs_test.bin vulkan_sycl_unsamp_rgba32_sfloat_2d_test.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vsu_test.bin 




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

// Compute shader using imageLoad (storage image) with r32f format
// Compute shader using texture() with sampler for RGBA32F
// SPIR-V compiled from:
// #version 450
// layout(binding = 0) uniform sampler2D inputImage;
// layout(binding = 1, std430) buffer OutputBuffer { float values[]; };
// layout(local_size_x = 1, local_size_y = 1) in;
// void main() {
//     vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) / vec2(4.0, 4.0);
//     values[gl_GlobalInvocationID.y * 4 + gl_GlobalInvocationID.x] = texture(inputImage, uv, 0.0).r;   // <-- 0.0 for textureLod
// }


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

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

int main() {
    std::cout << "Starting Vulkan and SYCL UNsampled VK_FORMAT_R32G32B32A32_SFLOAT 2D Image Test..." << std::endl;


    // Load the SPIR-V binary from disk
    // Make sure "vulkan_shader.spv" is in the same directory where you run the binary
    std::vector<char> shaderCode;
    try {
        shaderCode = readFile("vulkan_shader.spv");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Constants for test
    const uint32_t IMAGE_WIDTH = 4;
    const uint32_t IMAGE_HEIGHT = 4;
    const VkFormat IMAGE_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan and SYCL Unsampled Image Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    
    // Enable validation layers for better error messages
    const char* validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    
    bool validationAvailable = false;
    for (const auto& layerProps : availableLayers) {
        if (strcmp(layerProps.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            validationAvailable = true;
            break;
        }
    }
    
    if (validationAvailable) {
        instanceCreateInfo.enabledLayerCount = 1;
        instanceCreateInfo.ppEnabledLayerNames = validationLayers;
        std::cout << "✓ Validation layers enabled" << std::endl;
    } else {
        std::cout << "⚠ Validation layers not available" << std::endl;
    }

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

    // Define the extensions we need for Interop
    const char* deviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    
    // --> ADD THIS: Enable the extensions
    deviceCreateInfo.enabledExtensionCount = 2;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;

    VkDevice device;
    CHECK_VK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device), "Failed to create device");
    std::cout << "✓ Created logical device with External Memory extensions" << std::endl;

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
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    CHECK_VK(vkCreateImage(device, &imageInfo, nullptr, &image), "Failed to create image");
    std::cout << "✓ Created 2D image (" << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << ", VK_FORMAT_R32G32B32A32_SFLOAT)" << std::endl;

    // Allocate image memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);


    // Enable the extension capability (requires VK_KHR_external_memory_fd)
    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    // On Linux/Intel, use OPAQUE_FD. On Windows, use OPAQUE_WIN32.
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    // Chain it to existing allocation info
    allocInfo.pNext = &exportAllocInfo;




    VkDeviceMemory imageMemory;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory), "Failed to allocate image memory");
    CHECK_VK(vkBindImageMemory(device, image, imageMemory, 0), "Failed to bind image memory");
    std::cout << "✓ Allocated and bound image memory" << std::endl;

    // Create staging buffer for uploading data
    VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferInfo.size = IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float); // 4 channels
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

    // Fill staging buffer with test data (simple gradient pattern in all 4 channels)
    void* data;
    vkMapMemory(device, stagingMemory, 0, stagingBufferInfo.size, 0, &data);
    float* floatData = static_cast<float*>(data);
    for (uint32_t y = 0; y < IMAGE_HEIGHT; y++) {
        for (uint32_t x = 0; x < IMAGE_WIDTH; x++) {
            float value = static_cast<float>(x + y * IMAGE_WIDTH) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
            uint32_t idx = (y * IMAGE_WIDTH + x) * 4;
            floatData[idx + 0] = value;  // R
            floatData[idx + 1] = 0.0f;   // G  
            floatData[idx + 2] = 0.0f;   // B
            floatData[idx + 3] = 1.0f;   // A
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
    verifyBufferInfo.size = IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float); // 4 channels
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

    // DEFINE A FRESH BARRIER TO TRANSITION TO GENERAL
    VkImageMemoryBarrier handoffBarrier = {};
    handoffBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    handoffBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    handoffBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; // <--- Critical for Unsampled Access
    handoffBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    handoffBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    handoffBarrier.image = image;
    handoffBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    handoffBarrier.subresourceRange.baseMipLevel = 0;
    handoffBarrier.subresourceRange.levelCount = 1;
    handoffBarrier.subresourceRange.baseArrayLayer = 0;
    handoffBarrier.subresourceRange.layerCount = 1;
    
    handoffBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    handoffBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, 
        VK_PIPELINE_STAGE_TRANSFER_BIT, 
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &handoffBarrier);

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
            uint32_t idx = (y * IMAGE_WIDTH + x) * 4;  // 4 channels
            float expected = static_cast<float>(x + y * IMAGE_WIDTH) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
            float actual = verifyFloats[idx];  // Check R channel
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
    
    // Cleanup verification resources immediately to avoid hitting driver limits
    vkDestroyBuffer(device, verifyBuffer, nullptr);
    vkFreeMemory(device, verifyMemory, nullptr);
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
    shaderModuleInfo.codeSize = shaderCode.size();
    shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
    //shaderModuleInfo.codeSize = sizeof(SAMPLE_SHADER_SPIRV);
    //shaderModuleInfo.pCode = SAMPLE_SHADER_SPIRV;

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

    // ---------------------------------------------------------
    // PHASE 2: SYCL BINDLESS INTEROP  - U
    // ---------------------------------------------------------
    namespace syclexp = sycl::ext::oneapi::experimental;

    // 1. GET THE FILE DESCRIPTOR (Same as before)
    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) return 1;

    VkMemoryGetFdInfoKHR fdInfo = {};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = imageMemory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    CHECK_VK(vkGetMemoryFdKHR(device, &fdInfo, &fd), "Failed to get file descriptor");
    std::cout << "✓ Got Opaque FD: " << fd << std::endl;

    // ... inside main, after getting the FD ...

    // 2. SYCL SETUP
    try {
        sycl::queue q;
        std::cout << "✓ SYCL running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        // 3. IMPORT EXTERNAL MEMORY
        size_t imgSize = IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float);
        
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            fd, 
            syclexp::external_mem_handle_type::opaque_fd, 
            imgSize
        };

        // Step A: Import the raw FD
        syclexp::external_mem extMem = syclexp::import_external_memory(
            extMemDesc, q.get_device(), q.get_context());
            
        std::cout << "✓ Imported FD as External Memory" << std::endl;

        // Step B: Define the Image Descriptor EARLY
        // We need this to tell the mapper how to interpret the raw bytes
        syclexp::image_descriptor imgDesc(
            sycl::range<2>(IMAGE_WIDTH, IMAGE_HEIGHT),
            4, // num_channels
            sycl::image_channel_type::fp32
        );

        // Step C: Map External Memory to an Image Handle
        // This was the missing link!
        syclexp::image_mem_handle deviceMemHandle = syclexp::map_external_image_memory(
            extMem, 
            imgDesc, 
            q.get_device(), 
            q.get_context()
        );
        
        std::cout << "✓ Mapped External Memory to Image Handle" << std::endl;

        // 4. CREATE UNSAMPLED HANDLE
        // Notice: No sampler argument here
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(
            deviceMemHandle, 
            imgDesc, 
            q.get_device(), 
            q.get_context()
        );
        std::cout << "✓ Created Unsampled Handle" << std::endl;

        // 5. RUN KERNEL
        sycl::buffer<float, 1> checkBuf(IMAGE_WIDTH * IMAGE_HEIGHT);

        q.submit([&](sycl::handler& h) {
        sycl::accessor outAcc(checkBuf, h, sycl::write_only);
        
        h.parallel_for(sycl::range<2>(IMAGE_WIDTH, IMAGE_HEIGHT), [=](sycl::item<2> item) {
            int x = item.get_id(0);
            int y = item.get_id(1);

            // Fetch uses Integer coordinates
            sycl::int2 coords(x, y);
            
            // fetch_image<Return Type>(handle, coords)
            sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(unsampledHandle, coords);

            outAcc[y * IMAGE_WIDTH + x] = pixel.x();
        });
    }).wait();
        
        std::cout << "✓ SYCL Unsampled Kernel Executed" << std::endl;

        // 6. CLEANUP
        // Destroy the view
        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        
        // Release the import (this tears down the mapping too)
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        
        // ... Verification Logic (Same as before) ...
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        
        std::cout << "\n=== SYCL Verification ===" << std::endl;
        bool syclPassed = true;
        for (uint32_t y = 0; y < IMAGE_HEIGHT; y++) {
            for (uint32_t x = 0; x < IMAGE_WIDTH; x++) {
                uint32_t idx = y * IMAGE_WIDTH + x;
                float expected = static_cast<float>(idx) / (IMAGE_WIDTH * IMAGE_HEIGHT - 1);
                float actual = hostAcc[idx];
                
                bool match = std::fabs(actual - expected) < 0.01f;
                if (!match) {
                    syclPassed = false;
                    std::cout << "SYCL Mismatch [" << x << "," << y << "] " << actual << " != " << expected << std::endl;
                }
            }
        }
        
        if (syclPassed) std::cout << "✓ SYCL PASSED: Data matches!" << std::endl;
        else std::cout << "✗ SYCL FAILED" << std::endl;

    } catch (sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }




    // Cleanup
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

