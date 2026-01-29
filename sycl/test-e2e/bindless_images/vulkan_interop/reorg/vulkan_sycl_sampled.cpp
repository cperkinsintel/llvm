/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 2D Sampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vss_test.bin vulkan_sycl_sampled.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vss_test.bin vulkan_sycl_sampled.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vss_test.bin 
    ./vss_test.bin --semaphores



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

    std::cout << "Running SAMPLED (Texture) Test | Semaphores: " 
              << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {4, 4, 1};
    // Note: Common header sets USAGE_SAMPLED_BIT automatically
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D);

    // 2. Prepare Semaphore (if enabled)
    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) {
        vkSem = createExportableSemaphore(vkCtx);
    }

    // 3. Upload Data (Signals semaphore if provided)
    // Common helper transitions image to VK_IMAGE_LAYOUT_GENERAL, which is valid for sampling too.
    uploadAndVerify(vkCtx, imgRes, vkSem);

    // 4. Export Handles
    int memFd = getMemFd(vkCtx, imgRes.memory);
    int semFd = -1;
    if (useSemaphores) {
        semFd = getSemaphoreFd(vkCtx, vkSem);
        std::cout << "Got Semaphore FD: " << semFd << std::endl;
    }

    // 5. SYCL Interop
    namespace syclexp = sycl::ext::oneapi::experimental;
    
    try {
        sycl::queue q;
        
        // Import Memory
        size_t size = extent.width * extent.height * 4 * sizeof(float);
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            memFd, syclexp::external_mem_handle_type::opaque_fd, size
        };
        syclexp::external_mem extMem = syclexp::import_external_memory(extMemDesc, q.get_device(), q.get_context());

        // Import Semaphore (If enabled)
        syclexp::external_semaphore extSem;
        if (useSemaphores) {
             syclexp::external_semaphore_descriptor<syclexp::resource_fd> extSemDesc{
                semFd, syclexp::external_semaphore_handle_type::opaque_fd
            };
            extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(), q.get_context());
        }

        // Map Memory
        syclexp::image_descriptor imgDesc(
            sycl::range<2>(extent.width, extent.height), 4, sycl::image_channel_type::fp32
        );
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());

        // --- SAMPLED SPECIFIC SETUP ---
        
        // Define Sampler
        syclexp::bindless_image_sampler sampler(
            sycl::addressing_mode::clamp_to_edge,
            sycl::coordinate_normalization_mode::unnormalized,
            sycl::filtering_mode::linear
        );

        // Create Sampled Handle
        syclexp::sampled_image_handle sampledHandle = syclexp::create_image(
            devHandle, sampler, imgDesc, q.get_device(), q.get_context()
        );

        // --- KERNEL ---
        sycl::buffer<float, 1> checkBuf(extent.width * extent.height);
        
        // Step A: Handle Semaphore Wait (Separate Submission)
        sycl::event dependencyEvent;
        if (useSemaphores) {
            dependencyEvent = q.submit([&](sycl::handler& h) {
                h.ext_oneapi_wait_external_semaphore(extSem);
            });
        }

        // Step B: Submit Kernel
        q.submit([&](sycl::handler& h) {
            if (useSemaphores) {
                h.depends_on(dependencyEvent);
            }

            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            
            h.parallel_for(sycl::range<2>(extent.width, extent.height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                // Sampled uses Float coordinates + 0.5 offset for pixel center
                sycl::float2 coords(x + 0.5f, y + 0.5f);
                
                sycl::float4 px = syclexp::sample_image<sycl::float4>(sampledHandle, coords);
                outAcc[y * extent.width + x] = px.x();
            });
        }).wait();

        std::cout << "SYCL Kernel Executed." << std::endl;
        
        // Verify
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        for(int i=0; i<16; ++i) {
            float expected = (float)i / 15.0f;
            if(std::abs(hostAcc[i] - expected) > 0.01f) passed = false;
        }

        if(passed) std::cout << "SUCCESS!" << std::endl;
        else std::cout << "FAILURE!" << std::endl;

        // Cleanup
        syclexp::destroy_image_handle(sampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) {
            syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context());
            vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
        }

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    cleanupVulkan(vkCtx, imgRes);
    return 0;
}

