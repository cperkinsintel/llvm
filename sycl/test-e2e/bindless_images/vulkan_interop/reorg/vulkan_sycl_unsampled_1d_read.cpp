/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_R32G32B32A32_SFLOAT 1D UnSampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_1d.comp -o vulkan_shader_1d.spv

  clang++ -fsycl -std=c++17 -o vsu_1d_test.bin vulkan_sycl_unsampled_1d_read.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  

    ./vsu_1d_test.bin 
    ./vsu_1d_test.bin --semaphores

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, int32, uint8). Default is float
    Wx             Set custom Width .  Put "x" after 

    ./vsu_1d_test.bin --semaphores --linear --channels 2 64x



 */
#include "test_verification.hpp"
#include "vulkan_setup.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>

// ---------------------------------------------------------
// SYCL TYPE MAPPING HELPERS
// ---------------------------------------------------------

template <typename T>
sycl::image_channel_type getSyclChannelType();

template <> sycl::image_channel_type getSyclChannelType<float>() { return sycl::image_channel_type::fp32; }
template <> sycl::image_channel_type getSyclChannelType<int32_t>() { return sycl::image_channel_type::signed_int32; }
template <> sycl::image_channel_type getSyclChannelType<uint8_t>() { return sycl::image_channel_type::unsigned_int8; }

// ---------------------------------------------------------
// TEST RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(int width, int channels, bool useLinear, bool useSemaphores) {
    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
    VkFormat vkFormat = getVulkanFormat<T>(channels);
    std::cout << "VK Format: " << getFormatString(vkFormat) << std::endl;

    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, 1, 1};
    ImageResources imgRes = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_1D, tiling);

    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) vkSem = createExportableSemaphore(vkCtx);

    if (!uploadAndVerify<T>(vkCtx, imgRes, vkSem, channels)) {
        std::cerr << "Vulkan Upload Failed!" << std::endl; return 1;
    }

    int memFd = getMemFd(vkCtx, imgRes.memory);
    int semFd = -1;
    if (useSemaphores) semFd = getSemaphoreFd(vkCtx, vkSem);

    namespace syclexp = sycl::ext::oneapi::experimental;
    try {
        sycl::queue q;
        syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{memFd, syclexp::external_mem_handle_type::opaque_fd, imgRes.allocationSize};
        syclexp::external_mem extMem = syclexp::import_external_memory(extMemDesc, q.get_device(), q.get_context());
        
        syclexp::external_semaphore extSem;
        if (useSemaphores) {
             syclexp::external_semaphore_descriptor<syclexp::resource_fd> extSemDesc{semFd, syclexp::external_semaphore_handle_type::opaque_fd};
            extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(), q.get_context());
        }

        syclexp::image_descriptor imgDesc(sycl::range<1>(width), channels, getSyclChannelType<T>());
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        size_t totalValues = width * channels;
        sycl::buffer<T, 1> checkBuf(totalValues);
        
        sycl::event dependencyEvent;
        if (useSemaphores) dependencyEvent = q.submit([&](sycl::handler& h) { h.ext_oneapi_wait_external_semaphore(extSem); });

        q.submit([&](sycl::handler& h) {
            if (useSemaphores) h.depends_on(dependencyEvent);
            sycl::accessor outAcc(checkBuf, h, sycl::write_only);
            h.parallel_for(sycl::range<1>(width), [=](sycl::item<1> item) {
                int x = item.get_id(0);
                if (channels == 1) {
                    outAcc[x] = syclexp::fetch_image<T>(unsampledHandle, x);
                } else if (channels == 2) {
                    using Vec2 = sycl::vec<T, 2>;
                    Vec2 px = syclexp::fetch_image<Vec2>(unsampledHandle, x);
                    outAcc[x * 2 + 0] = px.x(); outAcc[x * 2 + 1] = px.y();
                } else {
                    using Vec4 = sycl::vec<T, 4>;
                    Vec4 px = syclexp::fetch_image<Vec4>(unsampledHandle, x);
                    outAcc[x * 4 + 0] = px.x(); outAcc[x * 4 + 1] = px.y(); outAcc[x * 4 + 2] = px.z(); outAcc[x * 4 + 3] = px.w();
                }
            });
        }).wait();

        sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
        bool passed = true;
        int errorCount = 0;
        for(size_t i=0; i < totalValues; ++i) {
            T expected = generateTestValue<T>(i / channels, i % channels, width);
            if(!checkValue(hostAcc[i], expected)) {
                passed = false;
                if (errorCount++ < 5) std::cout << "Mismatch at " << i << " Got: " << (double)hostAcc[i] << " Exp: " << (double)expected << std::endl;
            }
        }
        if(passed) std::cout << "SUCCESS!" << std::endl;
        else std::cout << "FAILURE! (" << errorCount << " errors)" << std::endl;

        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) { syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context()); vkDestroySemaphore(vkCtx.device, vkSem, nullptr); }
    } catch (std::exception& e) { std::cerr << "SYCL Exception: " << e.what() << std::endl; cleanupVulkan(vkCtx, imgRes); return 1; }
    cleanupVulkan(vkCtx, imgRes);
    return 0;
}

int main(int argc, char** argv) {
    int width = 16, channels = 4;
    bool useLinear = false, useSemaphores = false;
    std::string type = "float";
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg == "--channels" && i+1 < argc) channels = std::stoi(argv[++i]);
        else if(arg == "--type" && i+1 < argc) type = argv[++i];
        else if(arg.find("x") != std::string::npos) { try { width = std::stoi(arg); } catch(...) {} }
        else { try { width = std::stoi(arg); } catch(...) {} }
    }
    std::cout << "Running UNSAMPLED 1D Read Test | Type: " << type << " | Width: " << width << " | Channels: " << channels << std::endl;
    if (type == "float") return runTest<float>(width, channels, useLinear, useSemaphores);
    if (type == "int32") return runTest<int32_t>(width, channels, useLinear, useSemaphores);
    if (type == "uint8") return runTest<uint8_t>(width, channels, useLinear, useSemaphores);
    return 1;
}