/*
   $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vsu_2d_w_test.bin vulkan_sycl_unsampled_2d_write.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vsu_2d_w_test.bin vulkan_sycl_unsampled_2d_write.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vsu_2d_w_test.bin 

*/

#include "vulkan_interop_common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>

int main(int argc, char** argv) {
    bool useSemaphores = false;
    if (argc > 1 && std::string(argv[1]) == "--semaphores") {
        useSemaphores = true;
    }

    std::cout << "Running UNSAMPLED WRITE Test | Semaphores: " 
              << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {4, 4, 1}; 
    
    // Create Image (OPTIMAL Tiling)
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL);

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

        syclexp::image_descriptor imgDesc(sycl::range<2>(extent.width, extent.height), 4, sycl::image_channel_type::fp32);
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // FIX: Chain submissions. Kernel First -> Then Signal.
        
        // Step A: The Kernel (returns an event)
        sycl::event kernelEvent = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(extent.width, extent.height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                float val = (float)(x + y * extent.width) / (float)(extent.width * extent.height - 1);
                sycl::float4 pixel(val, 0.0f, 0.0f, 1.0f);
                syclexp::write_image(unsampledHandle, sycl::int2(x, y), pixel);
            });
        });

        // Step B: The Signal (Depends on Kernel)
        if (useSemaphores) {
            q.submit([&](sycl::handler& h) {
                h.depends_on(kernelEvent);
                h.ext_oneapi_signal_external_semaphore(extSem);
            });
        }

        q.wait(); // Wait for submission to hit the GPU

        std::cout << "SYCL Write Kernel Executed." << std::endl;

        // Cleanup SYCL objects
        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) {
            syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context());
        }

    } catch (std::exception& e) {
        std::cerr << "SYCL Error: " << e.what() << std::endl;
        return 1;
    }

    // 4. Vulkan Verification
    // Use heavy barriers just to be safe, but rely on Semaphores if enabled.
    vkDeviceWaitIdle(vkCtx.device);

    VkBuffer verifyBuffer;
    VkDeviceMemory verifyMem;
    size_t dataSize = extent.width * extent.height * 4 * sizeof(float);
    
    // Buffer Setup
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

    // Command Buffer
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

    // FIX: Pass valid begin info
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);
    
    // Copy GENERAL -> Buffer
    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = extent;
    vkCmdCopyImageToBuffer(cmd, imgRes.image, VK_IMAGE_LAYOUT_GENERAL, verifyBuffer, 1, &region);
    
    vkEndCommandBuffer(cmd);

    // Submit
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    
    // Wait for Semaphore (The critical part)
    std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_TRANSFER_BIT };
    if (useSemaphores) {
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &vkSem;
        submit.pWaitDstStageMask = waitStages.data();
    }

    vkQueueSubmit(vkCtx.queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkCtx.queue);

    // Check Data
    bool passed = true;
    void* verifyPtr;
    vkMapMemory(vkCtx.device, verifyMem, 0, dataSize, 0, &verifyPtr);
    float* verifyFloats = (float*)verifyPtr;
    
    size_t totalPixels = extent.width * extent.height;
    for(size_t i=0; i < totalPixels; ++i) {
        float expected = (float)i / (float)(totalPixels - 1);
        float actual = verifyFloats[i * 4];
        if(std::abs(actual - expected) > 0.01f) {
            passed = false;
            std::cout << "Mismatch at " << i << " Got: " << actual << " Exp: " << expected << std::endl;
            break;
        }
    }
    vkUnmapMemory(vkCtx.device, verifyMem);
    
    if(passed) std::cout << "SUCCESS!" << std::endl;
    else std::cout << "FAILURE!" << std::endl;

    // Cleanup
    vkDestroyCommandPool(vkCtx.device, pool, nullptr);
    vkDestroyBuffer(vkCtx.device, verifyBuffer, nullptr);
    vkFreeMemory(vkCtx.device, verifyMem, nullptr);
    if(useSemaphores) vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
    cleanupVulkan(vkCtx, imgRes);

    return 0;
}