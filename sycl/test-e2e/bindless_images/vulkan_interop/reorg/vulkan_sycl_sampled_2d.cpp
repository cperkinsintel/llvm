/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 2D Sampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vss_2d_test.bin vulkan_sycl_sampled_2d.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
    ./vss_2d_test.bin 
    ./vss_2d_test.bin --semaphores

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    WxH            Set custom Width x Height (e.g. 8x4)

    ./vss_2d_test.bin --semaphores --linear --channels 2 8x4

    NOTE: presently --linear is not working with 2D.

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
    int channels = 4;
    bool useLinear = false;
    bool useSemaphores = false;

    // Argument Parsing
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg == "--channels" && i+1 < argc) {
            channels = std::stoi(argv[++i]);
        }
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

    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
    VkFormat vkFormat = getFloatFormat(channels);

    std::cout << "Running SAMPLED 2D Test | Size: " << width << "x" << height 
              << " | Channels: " << channels
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
    ImageResources imgRes = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling);

    // 2. Semaphores
    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) vkSem = createExportableSemaphore(vkCtx);

    // 3. Upload Data
    if (!uploadAndVerify(vkCtx, imgRes, vkSem, channels)) {
        std::cerr << "Vulkan Upload Failed!" << std::endl;
        return 1;
    }

    // 4. Export Handles
    int memFd = getMemFd(vkCtx, imgRes.memory);
    int semFd = -1;
    if (useSemaphores) semFd = getSemaphoreFd(vkCtx, vkSem);

    // 5. SYCL Interop
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

        // Map Image
        syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), channels, sycl::image_channel_type::fp32);
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());

        // Create Sampler (Nearest)
        syclexp::bindless_image_sampler sampler(
            sycl::addressing_mode::clamp_to_edge,
            sycl::coordinate_normalization_mode::unnormalized,
            sycl::filtering_mode::nearest
        );

        syclexp::sampled_image_handle sampledHandle = syclexp::create_image(devHandle, sampler, imgDesc, q.get_device(), q.get_context());

        // Output Buffer
        size_t totalValues = width * height * channels;
        sycl::buffer<float, 1> checkBuf(totalValues);
        
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
                
                // Sample 2D (Float Coords, Center)
                sycl::float2 coords(x + 0.5f, y + 0.5f);
                
                // Sampler ALWAYS returns float4, regardless of image format
                sycl::float4 px = syclexp::sample_image<sycl::float4>(sampledHandle, coords);
                
                // Manually unpack into buffer based on expected channels
                size_t baseIdx = (y * width + x) * channels;
                
                outAcc[baseIdx + 0] = px.x();
                if (channels >= 2) outAcc[baseIdx + 1] = px.y();
                if (channels >= 4) {
                    outAcc[baseIdx + 2] = px.z();
                    outAcc[baseIdx + 3] = px.w();
                }
            });
        }).wait();

        std::cout << "SYCL Kernel Executed." << std::endl;
        
        // Verify
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        int errorCount = 0;
        size_t totalPixels = width * height;

        for(size_t i=0; i < totalValues; ++i) {
            size_t pixelIdx = i / channels;
            int channelIdx = i % channels;
            
            float baseVal = (float)pixelIdx / (float)(totalPixels > 1 ? totalPixels - 1 : 1);
            float expected = baseVal + (float)channelIdx * 0.1f;

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

        syclexp::destroy_image_handle(sampledHandle, q.get_device(), q.get_context());
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
