/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 1D UnSampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_1d.comp -o vulkan_shader_1d.spv

  clang++ -fsycl -std=c++17 -o vsu_1d_test.bin vulkan_sycl_unsampled_1d_read.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  

    ./vsu_1d_test.bin 
    ./vsu_1d_test.bin --semaphores

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --Wx          Set custom Width .  Put "x" after 

    ./vsu_1d_test.bin --semaphores --linear 64x



 */

#include "vulkan_interop_common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>

int main(int argc, char** argv) {
    // Defaults
    int width = 16; // Standard 1D size
    bool useLinear = false;
    bool useSemaphores = false;

    // Argument Parsing
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg.find("x") != std::string::npos) {
            // Even though it's 1D, we allow WxH format but ignore H
            // or just parse a single integer. Let's support simple int.
            try {
                width = std::stoi(arg);
            } catch (...) {
                // Try parsing WxH and take W
                size_t xPos = arg.find("x");
                if (xPos != std::string::npos) {
                     width = std::stoi(arg.substr(0, xPos));
                }
            }
        }
    }

    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

    std::cout << "Running UNSAMPLED 1D Read Test | Width: " << width
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, 1, 1};
    
    // Note: VK_IMAGE_TYPE_1D
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_1D, tiling);

    // 2. Prepare Semaphore
    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) {
        vkSem = createExportableSemaphore(vkCtx);
    }

    // 3. Upload Data (Uploads Gradient based on Index)
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

        // Map Image (Range<1>)
        syclexp::image_descriptor imgDesc(sycl::range<1>(width), 4, sycl::image_channel_type::fp32);
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // Kernel
        sycl::buffer<float, 1> checkBuf(width);
        
        sycl::event dependencyEvent;
        if (useSemaphores) {
            dependencyEvent = q.submit([&](sycl::handler& h) {
                h.ext_oneapi_wait_external_semaphore(extSem);
            });
        }

        q.submit([&](sycl::handler& h) {
            if (useSemaphores) h.depends_on(dependencyEvent);
            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            
            // Parallel For 1D
            h.parallel_for(sycl::range<1>(width), [=](sycl::item<1> item) {
                int x = item.get_id(0);
                
                // Fetch using int coordinate
                sycl::float4 px = syclexp::fetch_image<sycl::float4>(unsampledHandle, x);
                outAcc[x] = px.x();
            });
        }).wait();

        std::cout << "SYCL Kernel Executed." << std::endl;
        
        // Verify
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        int errorCount = 0;

        for(int i=0; i < width; ++i) {
            float expected = (float)i / (float)(width * 1 * 1 - 1); // Logic matches uploadAndVerify 1D flattening
            if(std::abs(hostAcc[i] - expected) > 0.01f) {
                passed = false;
                if (errorCount < 5) {
                     std::cout << "Mismatch at idx " << i << " Got: " << hostAcc[i] << " Exp: " << expected << std::endl;
                }
                errorCount++;
            }
        }

        if(passed) std::cout << "SUCCESS!" << std::endl;
        else std::cout << "FAILURE! (" << errorCount << " errors)" << std::endl;

        // Cleanup
        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) {
            syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context());
            vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
        }

    } catch (std::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    cleanupVulkan(vkCtx, imgRes);
    return 0;
}