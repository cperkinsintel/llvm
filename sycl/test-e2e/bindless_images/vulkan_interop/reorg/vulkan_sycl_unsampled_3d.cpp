/*
  Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 3D UnSampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_3d.comp -o vulkan_shader_3d.spv

  clang++ -fsycl -std=c++17 -o vsu_3d_test.bin vulkan_sycl_unsampled_3d.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vsu_3d_test.bin vulkan_sycl_unsampled_3d.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vsu_3d_test.bin 
    ./vsu_3d_test.bin --semaphores

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

    std::cout << "Running UNSAMPLED 3D Test (Asymmetric) | Semaphores: " 
              << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan (Asymmetric: 4x3x2)
    VulkanContext vkCtx = createVulkanContext();
    // Width=4, Height=3, Depth=2
    VkExtent3D extent = {4, 3, 2}; 
    
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_3D);

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
        size_t size = imgRes.allocationSize;
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            memFd, syclexp::external_mem_handle_type::opaque_fd, size
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
        // Note: We use (Width, Height, Depth) order for the descriptor
        syclexp::image_descriptor imgDesc(
            sycl::range<3>(extent.width, extent.height, extent.depth), 
            4, 
            sycl::image_channel_type::fp32
        );
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());

        // Create Unsampled Handle
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // Kernel
        size_t totalPixels = extent.width * extent.height * extent.depth;
        sycl::buffer<float, 1> checkBuf(totalPixels);
        
        // A. Wait Semaphore
        sycl::event dependencyEvent;
        if (useSemaphores) {
            dependencyEvent = q.submit([&](sycl::handler& h) {
                h.ext_oneapi_wait_external_semaphore(extSem);
            });
        }

        // B. Run Kernel
        q.submit([&](sycl::handler& h) {
            if (useSemaphores) {
                h.depends_on(dependencyEvent);
            }

            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            
            // NOTE: We pass dimensions as (Width, Height, Depth)
            h.parallel_for(sycl::range<3>(extent.width, extent.height, extent.depth), [=](sycl::item<3> item) {
                // In SYCL, item[0] is typically the FIRST dimension passed to range.
                // If range(w, h, d), then id(0)=x, id(1)=y, id(2)=z
                int x = item.get_id(0);
                int y = item.get_id(1);
                int z = item.get_id(2);
                
                sycl::float4 px = syclexp::fetch_image<sycl::float4>(unsampledHandle, sycl::int3(x, y, z));
                
                // Flatten: Z * (W*H) + Y * W + X
                size_t linearIdx = z * (extent.width * extent.height) + y * extent.width + x;
                outAcc[linearIdx] = px.x();
            });
        }).wait();

        std::cout << "SYCL 3D Kernel Executed." << std::endl;
        
        // Verify
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        int errorCount = 0;
        
        for(size_t i=0; i < totalPixels; ++i) {
            float expected = (float)i / (float)(totalPixels - 1);
            if(std::abs(hostAcc[i] - expected) > 0.01f) {
                passed = false;
                if (errorCount < 10) {
                     std::cout << "Mismatch at idx " << i << " (Coords: " 
                               << i % extent.width << "," 
                               << (i / extent.width) % extent.height << ","
                               << i / (extent.width * extent.height) << ")"
                               << " Got: " << hostAcc[i] << " Exp: " << expected << std::endl;
                }
                errorCount++;
            }
        }

        if(passed) std::cout << "SUCCESS!" << std::endl;
        else std::cout << "FAILURE! Total Errors: " << errorCount << std::endl;

        // Cleanup
        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
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