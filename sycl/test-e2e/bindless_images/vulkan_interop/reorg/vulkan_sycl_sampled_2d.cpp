/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 2D Sampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vss_2d_test.bin vulkan_sycl_sampled_2d.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vss_2d_test.bin vulkan_sycl_sampled_2d.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vss_2d_test.bin 
    ./vss_2d_test.bin --semaphores

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --WxH          Set custom Width x Height (e.g. 8x4)

    ./vss_2d_test.bin --semaphores --linear 8x4



 */
#include "vulkan_interop_common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>

int main(int argc, char** argv) {
    // defaults
    int width = 4;
    int height = 4;
    bool useLinear = false;
    bool useSemaphores = false;

    // Simple arg parsing
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg.find("x") != std::string::npos) {
            // Parse WxH (e.g. 8x4)
            size_t xPos = arg.find("x");
            width = std::stoi(arg.substr(0, xPos));
            height = std::stoi(arg.substr(xPos+1));
        }
    }

    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

    std::cout << "Running 2D Sweep: " << width << "x" << height 
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
    
    // Note: IMAGE_TYPE_2D
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D, tiling);

    // 2. Prepare Semaphore
    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) {
        vkSem = createExportableSemaphore(vkCtx);
    }

    // 3. Upload Data
    if (!uploadAndVerify(vkCtx, imgRes, vkSem)) {
        std::cerr << "Vulkan Upload Failed!" << std::endl;
        return 1;
    }

    // 4. Export Handles
    int memFd = getMemFd(vkCtx, imgRes.memory);
    int semFd = -1;
    if (useSemaphores) {
        semFd = getSemaphoreFd(vkCtx, vkSem);
    }

    // 5. SYCL Interop
    namespace syclexp = sycl::ext::oneapi::experimental;
    
    try {
        sycl::queue q;
        
        // Import Memory
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            memFd, syclexp::external_mem_handle_type::opaque_fd, imgRes.allocationSize
        };
        syclexp::external_mem extMem = syclexp::import_external_memory(extMemDesc, q.get_device(), q.get_context());

        // Import Semaphore
        syclexp::external_semaphore extSem;
        if (useSemaphores) {
             syclexp::external_semaphore_descriptor<syclexp::resource_fd> extSemDesc{
                semFd, syclexp::external_semaphore_handle_type::opaque_fd
            };
            extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(), q.get_context());
        }

        // Map Image
        syclexp::image_descriptor imgDesc(
            sycl::range<2>(width, height), 
            4, 
            sycl::image_channel_type::fp32
        );
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());

        // Sampler (Nearest to avoid interpolation noise)
        syclexp::bindless_image_sampler sampler(
            sycl::addressing_mode::clamp_to_edge,
            sycl::coordinate_normalization_mode::unnormalized,
            sycl::filtering_mode::nearest
        );

        syclexp::sampled_image_handle sampledHandle = syclexp::create_image(
            devHandle, sampler, imgDesc, q.get_device(), q.get_context()
        );

        // Kernel
        size_t totalPixels = width * height;
        sycl::buffer<float, 1> checkBuf(totalPixels);
        
        sycl::event dependencyEvent;
        if (useSemaphores) {
            dependencyEvent = q.submit([&](sycl::handler& h) {
                h.ext_oneapi_wait_external_semaphore(extSem);
            });
        }

        q.submit([&](sycl::handler& h) {
            if (useSemaphores) h.depends_on(dependencyEvent);
            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                // Add 0.5f to hit pixel center
                sycl::float2 coords(x + 0.5f, y + 0.5f);
                sycl::float4 px = syclexp::sample_image<sycl::float4>(sampledHandle, coords);
                
                outAcc[y * width + x] = px.x();
            });
        }).wait();

        // Verify
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        int errorCount = 0;

        for(size_t i=0; i < totalPixels; ++i) {
            float expected = (float)i / (float)(totalPixels - 1);
            if(std::abs(hostAcc[i] - expected) > 0.01f) {
                passed = false;
                if (errorCount < 5) { // Print first 5 errors
                     std::cout << "Mismatch at idx " << i << " (" << i%width << "," << i/width << ")"
                               << " Got: " << hostAcc[i] << " Exp: " << expected << std::endl;
                }
                errorCount++;
            }
        }

        if(passed) std::cout << "SUCCESS" << std::endl;
        else std::cout << "FAILURE (" << errorCount << " errors)" << std::endl;

        // Cleanup (Minimal for sweep speed)
        syclexp::destroy_image_handle(sampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) {
            syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context());
            vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
        }

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    cleanupVulkan(vkCtx, imgRes);
    return 0;
}

