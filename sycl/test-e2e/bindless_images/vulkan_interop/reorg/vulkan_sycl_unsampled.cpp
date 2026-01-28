/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 2D Sampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader.comp -o vulkan_shader.spv

  clang++ -fsycl -std=c++17 -o vsu_test.bin vulkan_sycl_unsampled.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
  export VULTURE_SDK=/iusers/cperkins/sycl_workspace/1.4.328.1/x86_64/
  clang++ -fsycl -std=c++17 -o vsu_test.bin vulkan_sycl_unsampled.cpp -lvulkan -I$VULTURE_SDK/include -L$VULTURE_SDK/lib

    ./vsu_test.bin 




 */
/*
  Vulkan/SYCL Test: Unsampled (Storage) Image Interop
  
  1. Creates a Vulkan Image (USAGE_STORAGE_BIT).
  2. Uploads data via Staging Buffer.
  3. Transitions to VK_IMAGE_LAYOUT_GENERAL.
  4. Exports Opaque FD.
  5. Imports into SYCL as 'unsampled_image_handle'.
  6. Reads data using 'fetch_image' (Data Port).
*/

#include "vulkan_interop_common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>



int main(int argc, char** argv) {
    std::cout << "Running Unsampled Test..." << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {4, 4, 1};
    ImageResources imgRes = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D);

    // 2. Upload Data (Transitions to GENERAL)
    uploadAndVerify(vkCtx, imgRes);

    // 3. Export FD
    int fd = getMemFd(vkCtx, imgRes.memory);
    std::cout << "✓ Got FD: " << fd << std::endl;

    // 4. SYCL Interop
    namespace syclexp = sycl::ext::oneapi::experimental;
    
    try {
        sycl::queue q;
        std::cout << "✓ SYCL Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        // Import
        size_t size = extent.width * extent.height * 4 * sizeof(float);
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
            fd, syclexp::external_mem_handle_type::opaque_fd, size
        };
        syclexp::external_mem extMem = syclexp::import_external_memory(extMemDesc, q.get_device(), q.get_context());

        // Map
        syclexp::image_descriptor imgDesc(
            sycl::range<2>(extent.width, extent.height),
            4, sycl::image_channel_type::fp32
        );
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());

        // Create Unsampled Handle
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // Kernel
        sycl::buffer<float, 1> checkBuf(extent.width * extent.height);
        q.submit([&](sycl::handler& h) {
            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            h.parallel_for(sycl::range<2>(extent.width, extent.height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                sycl::float4 px = syclexp::fetch_image<sycl::float4>(unsampledHandle, sycl::int2(x, y));
                outAcc[y * extent.width + x] = px.x();
            });
        }).wait();

        std::cout << "✓ SYCL Kernel Executed." << std::endl;
        
        // Verify
        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        for(int i=0; i<16; ++i) {
            float expected = (float)i / 15.0f;
            if(std::abs(hostAcc[i] - expected) > 0.01f) passed = false;
        }

        if(passed) std::cout << "✓ SUCCESS!" << std::endl;
        else std::cout << "✗ FAILURE!" << std::endl;

        // Cleanup SYCL
        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    cleanupVulkan(vkCtx, imgRes);
    return 0;
}