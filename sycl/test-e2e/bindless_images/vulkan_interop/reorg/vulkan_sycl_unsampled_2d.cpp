/*
  Minimal Vulkan/SYCL Test: VK_FORMAT_XXXX_SFLOAT 2D UnSampled Image

  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv

  clang++ -fsycl -std=c++17 -o vsu_2d_test.bin vulkan_sycl_unsampled_2d.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  
    ./vsu_2d_test.bin 
    ./vsu_2d_test.bin --semaphores

    FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). Default is float
    WxH            Set custom Width x Height (e.g. 8x4)


    ./vsu_2d_test.bin --semaphores --channels 2 --linear 8x4


    NOTE: presently --linear is not working with 2D.
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

#include "test_verification.hpp"
#include "vulkan_setup.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>
#include <map>
#include <optional>

// ---------------------------------------------------------
// SYCL TYPE MAPPING HELPERS
// ---------------------------------------------------------

template <typename T>
sycl::image_channel_type getSyclChannelType();

template <> inline sycl::image_channel_type getSyclChannelType<float>() { return sycl::image_channel_type::fp32; }

template <> inline sycl::image_channel_type getSyclChannelType<int32_t>() { return sycl::image_channel_type::signed_int32; }
template <> inline sycl::image_channel_type getSyclChannelType<uint32_t>() { return sycl::image_channel_type::unsigned_int32; }

template <> inline sycl::image_channel_type getSyclChannelType<int16_t>() {  return sycl::image_channel_type::signed_int16; }
template <> inline sycl::image_channel_type getSyclChannelType<uint16_t>() { return sycl::image_channel_type::unsigned_int16; }

template <> inline sycl::image_channel_type getSyclChannelType<uint8_t>() { return sycl::image_channel_type::unsigned_int8; }
template <> inline sycl::image_channel_type getSyclChannelType<int8_t>() { return sycl::image_channel_type::signed_int8; }

sycl::image_channel_order getSyclChannelOrder(int channels) {
    switch (channels) {
        case 1: return sycl::image_channel_order::r;
        case 2: return sycl::image_channel_order::rg;
        case 4: return sycl::image_channel_order::rgba;
        default: throw std::runtime_error("Unsupported channel count for SYCL Order");
    }
}



// half
template <> inline VkFormat getVulkanFormat<sycl::half>(int channels) {
    switch(channels) {
        case 1: return VK_FORMAT_R16_SFLOAT;
        case 2: return VK_FORMAT_R16G16_SFLOAT;
        case 4: return VK_FORMAT_R16G16B16A16_SFLOAT;
        default: throw std::runtime_error("Unsupported channels for half");
    }
}
template <> inline sycl::image_channel_type getSyclChannelType<sycl::half>() { return sycl::image_channel_type::fp16; }

// ---------------------------------------------------------
//  TEST RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(int width, int height, int channels, bool useLinear, bool useSemaphores, 
            VkFormat fmtOverride = VK_FORMAT_UNDEFINED, 
            std::optional<sycl::image_channel_type> syclOverride = std::nullopt)  {

    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

    VkFormat vkFormat = (fmtOverride != VK_FORMAT_UNDEFINED) 
                      ? fmtOverride 
                      : getVulkanFormat<T>(channels);
    std::cout << "VK Format: " << getFormatString(vkFormat) << std::endl;

    // 1. Setup Vulkan
    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
    ImageResources imgRes = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling);

    // 2. Semaphores
    VkSemaphore vkSem = VK_NULL_HANDLE;
    if (useSemaphores) vkSem = createExportableSemaphore(vkCtx);

    // 3. Upload Data (Explicit Template Call)
    if (!uploadAndVerify<T>(vkCtx, imgRes, vkSem, channels)) {
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

        size_t pitchA = 0; // 0 means "compute automatically" (Tight)
        if (useLinear) {
            pitchA = getRowPitch(vkCtx, imgA.image);
            // Note: If A and B are same dims/format, pitch is likely same
        }
        sycl::image_channel_order order = getSyclChannelOrder(channels);

        sycl::image_channel_type syclType = syclOverride.has_value()  ? syclOverride.value() : getSyclChannelType<T>();
        syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), channels, syclType);
        syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(extMem, imgDesc, q.get_device(), q.get_context());
        syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(devHandle, imgDesc, q.get_device(), q.get_context());

        // Output Buffer
        size_t totalValues = width * height * channels;
        sycl::buffer<T, 1> checkBuf(totalValues);
        
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

                // unorm is special snowflake
                bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
                if(isUnorm){
                    using Vec4 = sycl::vec<float, 4>;
                    // Fetch normalized floats (0.0..1.0)
                    Vec4 px = syclexp::fetch_image<Vec4>(unsampledHandle, sycl::int2(x, y));
                    
                    // Convert back to bytes (0..255) with rounding
                    Vec4 scaled = px * 255.0f; // Scale up

                    size_t baseIdx = (y * width + x) * channels;
                    outAcc[baseIdx + 0] = static_cast<T>(sycl::round(scaled.x()));
                    if(channels > 1) outAcc[baseIdx + 1] = static_cast<T>(sycl::round(scaled.y()));
                    if(channels > 2) outAcc[baseIdx + 2] = static_cast<T>(sycl::round(scaled.z()));
                    if(channels > 3) outAcc[baseIdx + 3] = static_cast<T>(sycl::round(scaled.w()));
                    
                    return; // Early exit. special snowflake gets to leave early.
                }


                
                if (channels == 1) {
                    T px = syclexp::fetch_image<T>(unsampledHandle, sycl::int2(x, y));
                    outAcc[y * width + x] = px;
                } else if (channels == 2) {
                    using Vec2 = sycl::vec<T, 2>;
                    Vec2 px = syclexp::fetch_image<Vec2>(unsampledHandle, sycl::int2(x, y));
                    outAcc[(y * width + x) * 2 + 0] = px.x();
                    outAcc[(y * width + x) * 2 + 1] = px.y();
                } else { // 4
                    using Vec4 = sycl::vec<T, 4>;
                    Vec4 px = syclexp::fetch_image<Vec4>(unsampledHandle, sycl::int2(x, y));
                    outAcc[(y * width + x) * 4 + 0] = px.x();
                    outAcc[(y * width + x) * 4 + 1] = px.y();
                    outAcc[(y * width + x) * 4 + 2] = px.z();
                    outAcc[(y * width + x) * 4 + 3] = px.w();
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
            
            T expected = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);
            
            if(!checkValue(hostAcc[i], expected)) {
                passed = false;
                if (errorCount < 5) {
                     std::cout << "Mismatch at idx " << i << " Got: " << (double)hostAcc[i] << " Exp: " << (double)expected << std::endl;
                }
                errorCount++;
            }
        }

        if(passed) std::cout << "SUCCESS!" << std::endl;
        else std::cout << "FAILURE! (" << errorCount << " errors)" << std::endl;

        syclexp::destroy_image_handle(unsampledHandle, q.get_device(), q.get_context());
        syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
        if (useSemaphores) {
            syclexp::release_external_semaphore(extSem, q.get_device(), q.get_context());
            vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
        }

    } catch (std::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        cleanupVulkan(vkCtx, imgRes);
        return 1;
    }

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
    std::string type = "float"; // Default

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

    std::cout << "Running UNSAMPLED 2D Read Test | Type: " << type 
              << " | Size: " << width << "x" << height 
              << " | Channels: " << channels
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

    if (type == "float")  return runTest<float>(width, height, channels, useLinear, useSemaphores);
    if (type == "half")   return runTest<sycl::half>(width, height, channels, useLinear, useSemaphores);
    
    if (type == "int32")  return runTest<int32_t>(width, height, channels, useLinear, useSemaphores);
    if (type == "uint32") return runTest<uint32_t>(width, height, channels, useLinear, useSemaphores);
    
    if (type == "int16")  return runTest<int16_t>(width, height, channels, useLinear, useSemaphores);
    if (type == "uint16") return runTest<uint16_t>(width, height, channels, useLinear, useSemaphores);
    
    if (type == "uint8")  return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores);
    if (type == "int8")  return runTest<int8_t>(width, height, channels, useLinear, useSemaphores);

    if (type == "unorm8") {
        // unorm8 is one of those scaled floats. 0-1.0  
        return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores, 
                              getUnorm8Format(channels), 
                              sycl::image_channel_type::unorm_int8); 
    }

    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
}