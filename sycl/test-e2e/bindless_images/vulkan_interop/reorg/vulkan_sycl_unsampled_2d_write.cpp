/*
   $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vsu_2d_w_test.bin vulkan_sycl_unsampled_2d_write.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
    ./vsu_2d_w_test.bin 

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, int32, uint8). Default is float
    WxH            Set custom Width x Height (e.g. 8x4)
    
    ./vsu_2d_w_test.bin --semaphores --channels 2 --linear 8x4

*/
#include "vulkan_interop_common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>

// ---------------------------------------------------------
// TYPE MAPPING HELPERS
// ---------------------------------------------------------
template <typename T> VkFormat getVulkanFormat(int channels);

template <> VkFormat getVulkanFormat<float>(int channels) {
    switch(channels) {
        case 1: return VK_FORMAT_R32_SFLOAT;
        case 2: return VK_FORMAT_R32G32_SFLOAT;
        case 4: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: throw std::runtime_error("Unsupported channels for float");
    }
}

template <> VkFormat getVulkanFormat<int32_t>(int channels) {
    switch(channels) {
        case 1: return VK_FORMAT_R32_SINT;
        case 2: return VK_FORMAT_R32G32_SINT;
        case 4: return VK_FORMAT_R32G32B32A32_SINT;
        default: throw std::runtime_error("Unsupported channels for int32");
    }
}

template <> VkFormat getVulkanFormat<uint8_t>(int channels) {
    switch(channels) {
        case 1: return VK_FORMAT_R8_UINT;
        case 2: return VK_FORMAT_R8G8_UINT;
        case 4: return VK_FORMAT_R8G8B8A8_UINT;
        default: throw std::runtime_error("Unsupported channels for uint8");
    }
}

template <typename T> sycl::image_channel_type getSyclChannelType();
template <> sycl::image_channel_type getSyclChannelType<float>() { return sycl::image_channel_type::fp32; }
template <> sycl::image_channel_type getSyclChannelType<int32_t>() { return sycl::image_channel_type::signed_int32; }
template <> sycl::image_channel_type getSyclChannelType<uint8_t>() { return sycl::image_channel_type::unsigned_int8; }

// ---------------------------------------------------------
// KERNEL GENERATOR HELPER
// (Must match generateTestValue in common header)
// ---------------------------------------------------------
template <typename T>
T getKernelValue(size_t index, int channel, size_t rangeMax) {
    if constexpr (std::is_floating_point_v<T>) {
        float val = (float)index / (float)(rangeMax > 1 ? rangeMax - 1 : 1);
        return static_cast<T>(val + (float)channel * 0.1f);
    } else {
        // Integer pattern: (index + channel*10) % 127
        return static_cast<T>((index + channel * 10) % 127);
    }
}

// ---------------------------------------------------------
// TEMPLATED TEST RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(int width, int height, int channels, bool useLinear, bool useSemaphores) {
    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
    VkFormat vkFormat = getVulkanFormat<T>(channels);

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
    ImageResources imgRes = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling);

    // Initial Transition to GENERAL
    {
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
        VkCommandPool pool;
        vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
        
        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo cmdAlloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdAlloc.commandPool = pool;
        cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAlloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(vkCtx.device, &cmdAlloc, &cmd);
        
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &beginInfo);
        
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image = imgRes.image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; 
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,0,nullptr,0,nullptr,1,&barrier);
        vkEndCommandBuffer(cmd);
        
        VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vkQueueSubmit(vkCtx.queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(vkCtx.queue);
        vkDestroyCommandPool(vkCtx.device, pool, nullptr);
    }

    // 2. Export Handles
    int memFd = getMemFd(vkCtx, imgRes.memory);
    int semFd = -1;
    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) {
        vkSem = createExportableSemaphore(vkCtx);
        semFd = getSemaphoreFd(vkCtx, vkSem);
    }

    // 3. SYCL Interop
    namespace syclexp = sycl::ext::oneapi::experimental;
    
    try {
        sycl::queue q;
        
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            memFd, syclexp::external_mem_handle_type::opaque_fd, imgRes.allocationSize
        };
        syclexp::external_mem extMem = syclexp::import_external_memory(extMemDesc, q.get_device(), q.get_context());

        syclexp::external_semaphore extSem;
        if (useSemaphores) {
             syclexp::external_semaphore_descriptor<syclexp::resource_fd> extSemDesc{
                semFd, syclexp::external_semaphore_handle_type::opaque_fd
            };
            extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(), q.get_context());
        }

        syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), channels, getSyclChannelType<T>());
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // Step A: Kernel
        sycl::event kernelEvent = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                size_t index = y * width + x;
                size_t totalPixels = width * height;
                
                if (channels == 1) {
                    T val = getKernelValue<T>(index, 0, totalPixels);
                    syclexp::write_image(unsampledHandle, sycl::int2(x, y), val);
                } 
                else if (channels == 2) {
                    using Vec2 = sycl::vec<T, 2>;
                    Vec2 px(getKernelValue<T>(index, 0, totalPixels), 
                            getKernelValue<T>(index, 1, totalPixels));
                    syclexp::write_image(unsampledHandle, sycl::int2(x, y), px);
                } 
                else { // 4
                    using Vec4 = sycl::vec<T, 4>;
                    Vec4 px(getKernelValue<T>(index, 0, totalPixels), 
                            getKernelValue<T>(index, 1, totalPixels),
                            getKernelValue<T>(index, 2, totalPixels),
                            getKernelValue<T>(index, 3, totalPixels));
                    syclexp::write_image(unsampledHandle, sycl::int2(x, y), px);
                }
            });
        });

        // Step B: Signal
        if (useSemaphores) {
            q.submit([&](sycl::handler& h) {
                h.depends_on(kernelEvent);
                h.ext_oneapi_signal_external_semaphore(extSem);
            });
        }

        q.wait();
        std::cout << "SYCL Write Kernel Executed." << std::endl;

        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) {
            syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context());
        }

    } catch (std::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    // 4. Vulkan Verification
    vkDeviceWaitIdle(vkCtx.device);

    VkBuffer verifyBuffer;
    VkDeviceMemory verifyMem;
    size_t dataSize = width * height * channels * sizeof(T);
    
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = dataSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(vkCtx.device, &bi, nullptr, &verifyBuffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vkCtx.device, verifyBuffer, &req);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(vkCtx.physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(vkCtx.device, &ai, nullptr, &verifyMem);
    vkBindBufferMemory(vkCtx.device, verifyBuffer, verifyMem, 0);

    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    VkCommandPool pool;
    vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAlloc.commandPool = pool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(vkCtx.device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);
    
    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = extent;
    vkCmdCopyImageToBuffer(cmd, imgRes.image, VK_IMAGE_LAYOUT_GENERAL, verifyBuffer, 1, &region);
    
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    
    std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_TRANSFER_BIT };
    if (useSemaphores) {
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &vkSem;
        submit.pWaitDstStageMask = waitStages.data();
    }

    vkQueueSubmit(vkCtx.queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkCtx.queue);

    // Verify Data
    bool passed = true;
    void* verifyPtr;
    vkMapMemory(vkCtx.device, verifyMem, 0, dataSize, 0, &verifyPtr);
    T* verifyData = (T*)verifyPtr;
    
    size_t totalPixels = width * height;
    int errorCount = 0;
    
    for(size_t i=0; i < totalPixels * channels; ++i) {
        size_t pixelIdx = i / channels;
        int channelIdx = i % channels;
        
        // Use the SHARED generator logic to verify (same as kernel logic)
        T expected = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);
        T actual = verifyData[i];

        if(!checkValue(actual, expected)) {
            passed = false;
            if (errorCount < 5) {
                std::cout << "Mismatch at " << i << " Got: " << (double)actual << " Exp: " << (double)expected << std::endl;
            }
            errorCount++;
        }
    }
    vkUnmapMemory(vkCtx.device, verifyMem);
    
    if(passed) std::cout << "SUCCESS!" << std::endl;
    else std::cout << "FAILURE! (" << errorCount << " errors)" << std::endl;

    vkDestroyCommandPool(vkCtx.device, pool, nullptr);
    vkDestroyBuffer(vkCtx.device, verifyBuffer, nullptr);
    vkFreeMemory(vkCtx.device, verifyMem, nullptr);
    if(useSemaphores) vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
    cleanupVulkan(vkCtx, imgRes);

    return 0;
}

// ---------------------------------------------------------
// MAIN DISPATCHER
// ---------------------------------------------------------
int main(int argc, char** argv) {
    int width = 4;
    int height = 4;
    int channels = 4;
    bool useLinear = false;
    bool useSemaphores = false;
    std::string type = "float";

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg == "--channels" && i+1 < argc) channels = std::stoi(argv[++i]);
        else if(arg == "--type" && i+1 < argc) type = argv[++i];
        else if(arg.find("x") != std::string::npos) {
            size_t xPos = arg.find("x");
            try {
                width = std::stoi(arg.substr(0, xPos));
                height = std::stoi(arg.substr(xPos+1));
            } catch (...) { }
        }
    }

    if (channels != 1 && channels != 2 && channels != 4) {
        std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
        return 1;
    }

    std::cout << "Running UNSAMPLED 2D WRITE Test | Type: " << type
              << " | Size: " << width << "x" << height 
              << " | Channels: " << channels
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

    if (type == "float") return runTest<float>(width, height, channels, useLinear, useSemaphores);
    if (type == "int32") return runTest<int32_t>(width, height, channels, useLinear, useSemaphores);
    if (type == "uint8") return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores);

    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
}