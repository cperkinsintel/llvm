/*
   $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vsu_2d_w_test.bin vulkan_sycl_unsampled_2d_write.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vsu_2d_w_test.bin vulkan_sycl_unsampled_2d_write.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vsu_2d_w_test.bin 

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --WxH          Set custom Width x Height (e.g. 8x4)

    ./vsu_2d_w_test.bin --semaphores --linear 8x4

*/
#include "vulkan_interop_common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>

int main(int argc, char** argv) {
    // Defaults
    int width = 4;
    int height = 4;
    bool useLinear = false;
    bool useSemaphores = false;

    // Argument Parsing
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg.find("x") != std::string::npos) {
            size_t xPos = arg.find("x");
            try {
                width = std::stoi(arg.substr(0, xPos));
                height = std::stoi(arg.substr(xPos+1));
            } catch (...) {
                std::cerr << "Invalid size format. Use WxH (e.g. 8x4)" << std::endl;
                return 1;
            }
        }
    }

    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

    std::cout << "Running UNSAMPLED WRITE Test | Size: " << width << "x" << height 
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
    
    // Create Image (Empty)
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D, tiling);

    // Initial Transition to GENERAL (Manual, because we aren't uploading data)
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

        syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), 4, sycl::image_channel_type::fp32);
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // Step A: Kernel
        sycl::event kernelEvent = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                // Gradient: (x + y*w) / total
                float val = (float)(x + y * width) / (float)(width * height - 1);
                sycl::float4 pixel(val, 0.0f, 0.0f, 1.0f);
                
                syclexp::write_image(unsampledHandle, sycl::int2(x, y), pixel);
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
    size_t dataSize = width * height * 4 * sizeof(float);
    
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
    
    // Copy GENERAL -> Buffer
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
    float* verifyFloats = (float*)verifyPtr;
    
    size_t totalPixels = width * height;
    int errorCount = 0;
    
    for(size_t i=0; i < totalPixels; ++i) {
        float expected = (float)i / (float)(totalPixels - 1);
        float actual = verifyFloats[i * 4];
        if(std::abs(actual - expected) > 0.01f) {
            passed = false;
            if (errorCount < 5) {
                std::cout << "Mismatch at " << i << " (" << i%width << "," << i/width << ")"
                          << " Got: " << actual << " Exp: " << expected << std::endl;
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