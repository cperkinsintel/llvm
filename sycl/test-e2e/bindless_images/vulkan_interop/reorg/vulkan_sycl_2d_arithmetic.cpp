/*
  BOSS BATTLE: Vulkan/SYCL 2D Arithmetic (A + B = C)
  
  $VULKAN_SDK/bin/glslangValidator -V vulkan_shader_2d.comp -o vulkan_shader_2d.spv
  clang++ -fsycl -std=c++17 -o vs_2d_arith.bin vulkan_sycl_2d_arithmetic.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib

  FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). Default is float
    WxH            Set custom Width x Height (e.g. 8x4)


    --sampled      A flag for THIS test only. Choose between sampled or unsampled.  
  
  ./vs_2d_arith.bin --type float --semaphores
  ./vs_2d_arith.bin --type unorm8 --sampled --semaphores
*/
#include "test_verification.hpp"
#include "vulkan_setup.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <string>
#include <optional>
#include <vector>
#include <algorithm> 

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// SYCL TYPE MAPPING
// ---------------------------------------------------------
template <typename T> sycl::image_channel_type getSyclChannelType();
template <> inline sycl::image_channel_type getSyclChannelType<float>() { return sycl::image_channel_type::fp32; }
template <> inline sycl::image_channel_type getSyclChannelType<int32_t>() { return sycl::image_channel_type::signed_int32; }
template <> inline sycl::image_channel_type getSyclChannelType<uint32_t>() { return sycl::image_channel_type::unsigned_int32; }
template <> inline sycl::image_channel_type getSyclChannelType<int16_t>() {  return sycl::image_channel_type::signed_int16; }
template <> inline sycl::image_channel_type getSyclChannelType<uint16_t>() { return sycl::image_channel_type::unsigned_int16; }
template <> inline sycl::image_channel_type getSyclChannelType<uint8_t>() { return sycl::image_channel_type::unsigned_int8; }
template <> inline sycl::image_channel_type getSyclChannelType<int8_t>() { return sycl::image_channel_type::signed_int8; }

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
// GENERATORS
// ---------------------------------------------------------
template <typename T>
T generateValueA(size_t x, size_t y, int channel) {
    float val = (float)(x + y) / 100.0f; 
    if constexpr (std::is_floating_point_v<T>) return static_cast<T>(val + channel * 0.1f);
    else return static_cast<T>((x + y + channel * 10) % 64);
}

template <typename T>
T generateValueB(size_t x, size_t y, int channel) {
    float val = (float)(x * 2 + y) / 100.0f; 
    if constexpr (std::is_floating_point_v<T>) return static_cast<T>(val + channel * 0.2f);
    else return static_cast<T>((x * 2 + y + channel * 5) % 64);
}

// ---------------------------------------------------------
// THE BOSS TEST
// ---------------------------------------------------------
template <typename T>
int runTest(int width, int height, int channels, bool useLinear, bool useSemaphores, bool useSampled,
            VkFormat fmtOverride = VK_FORMAT_UNDEFINED, 
            std::optional<sycl::image_channel_type> syclOverride = std::nullopt)  {
    
    VkImageTiling tiling = useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
    VkFormat vkFormat = (fmtOverride != VK_FORMAT_UNDEFINED) ? fmtOverride : getVulkanFormat<T>(channels);
    
    std::cout << "  VK Format: " << getFormatString(vkFormat) << std::endl;
    std::cout << "  Mode: " << (useSampled ? "SAMPLED Input" : "UNSAMPLED Input") << std::endl;

    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
    
    // 1. Create Images & Semaphores
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if(useSampled) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

    ImageResources imgA = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);
    ImageResources imgB = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);
    ImageResources imgOut = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);

    VkSemaphore semA = VK_NULL_HANDLE;
    VkSemaphore semB = VK_NULL_HANDLE;
    if (useSemaphores) {
        semA = createExportableSemaphore(vkCtx);
        semB = createExportableSemaphore(vkCtx);
    }

    // 2. Upload Data
    uploadImage(vkCtx, imgA,  channels, semA, [&](size_t i, int c) {
        return generateValueA<T>(i % width, i / width, c);
    });
    uploadImage(vkCtx, imgB,  channels, semB, [&](size_t i, int c) {
        return generateValueB<T>(i % width, i / width, c);
    });

    // 3. Import into SYCL
    sycl::queue q;
    auto extMemA = syclexp::import_external_memory(syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, imgA.memory), syclexp::external_mem_handle_type::opaque_fd, imgA.allocationSize}, q.get_device(), q.get_context());
    auto extMemB = syclexp::import_external_memory(syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, imgB.memory), syclexp::external_mem_handle_type::opaque_fd, imgB.allocationSize}, q.get_device(), q.get_context());
    auto extMemOut = syclexp::import_external_memory(syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, imgOut.memory), syclexp::external_mem_handle_type::opaque_fd, imgOut.allocationSize}, q.get_device(), q.get_context());

    syclexp::external_semaphore extSemA, extSemB, extSemOut;
    VkSemaphore semOutVk = VK_NULL_HANDLE;
    if (useSemaphores) {
        extSemA = syclexp::import_external_semaphore(syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemaphoreFd(vkCtx, semA), syclexp::external_semaphore_handle_type::opaque_fd}, q.get_device(), q.get_context());
        extSemB = syclexp::import_external_semaphore(syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemaphoreFd(vkCtx, semB), syclexp::external_semaphore_handle_type::opaque_fd}, q.get_device(), q.get_context());
        
        semOutVk = createExportableSemaphore(vkCtx);
        extSemOut = syclexp::import_external_semaphore(syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemaphoreFd(vkCtx, semOutVk), syclexp::external_semaphore_handle_type::opaque_fd}, q.get_device(), q.get_context());
    }

    sycl::image_channel_type syclType = syclOverride.has_value() ? syclOverride.value() : getSyclChannelType<T>();
    syclexp::image_descriptor imgDesc({(size_t)width, (size_t)height}, channels, syclType);

    auto imgMemA = syclexp::map_external_image_memory(extMemA, imgDesc, q.get_device(), q.get_context());
    auto imgMemB = syclexp::map_external_image_memory(extMemB, imgDesc, q.get_device(), q.get_context());
    auto imgMemOut = syclexp::map_external_image_memory(extMemOut, imgDesc, q.get_device(), q.get_context());
    
    auto handleOut = syclexp::create_image(imgMemOut, imgDesc, q.get_device(), q.get_context());

    // 4. Kernel Submission (Scoped lifetime)
    sycl::event kernelEvent;

    // Signal helper
    auto queueSignal = [&](sycl::event kEvent) {
        if (useSemaphores) {
            q.submit([&](sycl::handler& h) {
                h.depends_on(kEvent);
                h.ext_oneapi_signal_external_semaphore(extSemOut);
            });
        }
    };

    if (useSampled) {
        // --- PATH A: SAMPLED ---
        syclexp::bindless_image_sampler sampler(
            sycl::addressing_mode::clamp_to_edge,
            sycl::coordinate_normalization_mode::unnormalized, 
            sycl::filtering_mode::nearest 
        );

        auto handleA = syclexp::create_image(imgMemA, sampler, imgDesc, q.get_device(), q.get_context());
        auto handleB = syclexp::create_image(imgMemB, sampler, imgDesc, q.get_device(), q.get_context());

        kernelEvent = q.submit([&](sycl::handler& h) {
            if (useSemaphores) {
                h.ext_oneapi_wait_external_semaphore(extSemA);
                h.ext_oneapi_wait_external_semaphore(extSemB);
            }
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                float u = (float)x + 0.5f; float v = (float)y + 0.5f;
                bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
                using Vec4 = sycl::vec<float, 4>;
                
                Vec4 valA = syclexp::sample_image<Vec4>(handleA, sycl::float2(u, v));
                Vec4 valB = syclexp::sample_image<Vec4>(handleB, sycl::float2(u, v));
                Vec4 sum = valA + valB;

                if (isUnorm) {
                   sum.x() = sycl::clamp(sum.x(), 0.0f, 1.0f); sum.y() = sycl::clamp(sum.y(), 0.0f, 1.0f);
                   sum.z() = sycl::clamp(sum.z(), 0.0f, 1.0f); sum.w() = sycl::clamp(sum.w(), 0.0f, 1.0f);
                   if(channels==1) syclexp::write_image(handleOut, sycl::int2(x,y), sum.x());
                   else if(channels==2) syclexp::write_image(handleOut, sycl::int2(x,y), sycl::float2(sum.x(), sum.y()));
                   else syclexp::write_image(handleOut, sycl::int2(x,y), sum);
                } else {
                   if(channels==1) syclexp::write_image(handleOut, sycl::int2(x,y), static_cast<T>(sum.x()));
                   else if(channels==2) syclexp::write_image(handleOut, sycl::int2(x,y), sycl::vec<T,2>(static_cast<T>(sum.x()), static_cast<T>(sum.y())));
                   else syclexp::write_image(handleOut, sycl::int2(x,y), sycl::vec<T,4>(static_cast<T>(sum.x()), static_cast<T>(sum.y()), static_cast<T>(sum.z()), static_cast<T>(sum.w())));
                }
            });
        });

        // Queue Signal, Wait, THEN Destroy handles
        queueSignal(kernelEvent);
        q.wait();

        syclexp::destroy_image_handle(handleA, q.get_device(), q.get_context());
        syclexp::destroy_image_handle(handleB, q.get_device(), q.get_context());

    } else {
        // --- PATH B: UNSAMPLED ---
        auto handleA = syclexp::create_image(imgMemA, imgDesc, q.get_device(), q.get_context());
        auto handleB = syclexp::create_image(imgMemB, imgDesc, q.get_device(), q.get_context());

        kernelEvent = q.submit([&](sycl::handler& h) {
            if (useSemaphores) {
                h.ext_oneapi_wait_external_semaphore(extSemA);
                h.ext_oneapi_wait_external_semaphore(extSemB);
            }
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
                using Vec4 = sycl::vec<float, 4>;
                Vec4 valA(0,0,0,0); Vec4 valB(0,0,0,0);

                if (isUnorm) {
                    valA = syclexp::fetch_image<Vec4>(handleA, sycl::int2(x, y));
                    valB = syclexp::fetch_image<Vec4>(handleB, sycl::int2(x, y));
                } else {
                    auto fetchT = [&](auto& hdl) {
                        Vec4 v(0,0,0,0);
                        if(channels==1) v.x() = (float)syclexp::fetch_image<T>(hdl, sycl::int2(x,y));
                        else {
                           auto raw = syclexp::fetch_image<sycl::vec<T,4>>(hdl, sycl::int2(x,y));
                           v.x()=(float)raw.x(); v.y()=(float)raw.y(); v.z()=(float)raw.z(); v.w()=(float)raw.w();
                        }
                        return v;
                    };
                    valA = fetchT(handleA);
                    valB = fetchT(handleB);
                }

                Vec4 sum = valA + valB;

                if (isUnorm) {
                   sum.x() = sycl::clamp(sum.x(), 0.0f, 1.0f); sum.y() = sycl::clamp(sum.y(), 0.0f, 1.0f);
                   sum.z() = sycl::clamp(sum.z(), 0.0f, 1.0f); sum.w() = sycl::clamp(sum.w(), 0.0f, 1.0f);
                   if(channels==1) syclexp::write_image(handleOut, sycl::int2(x,y), sum.x());
                   else if(channels==2) syclexp::write_image(handleOut, sycl::int2(x,y), sycl::float2(sum.x(), sum.y()));
                   else syclexp::write_image(handleOut, sycl::int2(x,y), sum);
                } else {
                   if(channels==1) syclexp::write_image(handleOut, sycl::int2(x,y), static_cast<T>(sum.x()));
                   else if(channels==2) syclexp::write_image(handleOut, sycl::int2(x,y), sycl::vec<T,2>(static_cast<T>(sum.x()), static_cast<T>(sum.y())));
                   else syclexp::write_image(handleOut, sycl::int2(x,y), sycl::vec<T,4>(static_cast<T>(sum.x()), static_cast<T>(sum.y()), static_cast<T>(sum.z()), static_cast<T>(sum.w())));
                }
            });
        });

        // Queue Signal, Wait, THEN Destroy handles
        queueSignal(kernelEvent);
        q.wait();

        syclexp::destroy_image_handle(handleA, q.get_device(), q.get_context());
        syclexp::destroy_image_handle(handleB, q.get_device(), q.get_context());
    }

    // 5. Cleanup Shared Resources (Handle First, Then Mem)
    syclexp::destroy_image_handle(handleOut, q.get_device(), q.get_context());
    
    // !!! FIX: Free Image Memory Handles created by Map !!!
    syclexp::free_image_mem(imgMemA, q.get_device(), q.get_context());
    syclexp::free_image_mem(imgMemB, q.get_device(), q.get_context());
    syclexp::free_image_mem(imgMemOut, q.get_device(), q.get_context());

    syclexp::release_external_memory(extMemA, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemB, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemOut, q.get_device(), q.get_context());

    if(useSemaphores) {
        syclexp::release_external_semaphore(extSemA, q.get_device(), q.get_context());
        syclexp::release_external_semaphore(extSemB, q.get_device(), q.get_context());
        syclexp::release_external_semaphore(extSemOut, q.get_device(), q.get_context());
    }

    // 6. Verify (Vulkan)
    bool passed = verifyImage(vkCtx, imgOut, channels, semOutVk, [&](size_t i, int c) {
        size_t x = i % width; size_t y = i / width;
        T a = generateValueA<T>(x, y, c);
        T b = generateValueB<T>(x, y, c);
        
        if (syclOverride.has_value() && syclOverride.value() == sycl::image_channel_type::unorm_int8) {
             float fa = (float)a / 255.0f; 
             float fb = (float)b / 255.0f;
             float sum = std::min(fa + fb, 1.0f);
             return static_cast<T>(sum * 255.0f + 0.5f);
        } else {
             return static_cast<T>(a + b);
        }
    });

    if(passed) std::cout << "SUCCESS!" << std::endl;
    else std::cout << "FAILURE!" << std::endl;

    // Cleanup Vulkan
    if(useSemaphores) { 
        vkDestroySemaphore(vkCtx.device, semA, nullptr); 
        vkDestroySemaphore(vkCtx.device, semB, nullptr); 
        vkDestroySemaphore(vkCtx.device, semOutVk, nullptr); 
    }

    cleanupImageResources(vkCtx, imgA);
    cleanupImageResources(vkCtx, imgB);
    cleanupVulkan(vkCtx, imgOut);
    
    return passed ? 0 : 1;
}

int main(int argc, char** argv) {
    int width = 4;
    int height = 4;
    int channels = 4;
    bool useLinear = false;
    bool useSemaphores = false;
    bool useSampled = false;
    std::string type = "float";

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg == "--channels" && i+1 < argc) channels = std::stoi(argv[++i]);
        else if(arg == "--type" && i+1 < argc) type = argv[++i];
        else if(arg == "--sampled") useSampled = true;
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

    std::cout << "Running 2D ARITHMETIC Test (C = A + B) | Type: " << type 
              << " | Size: " << width << "x" << height 
              << " | Channels: " << channels
              << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
              << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;


    // Dispatcher
    if (type == "float")  return runTest<float>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "half")   return runTest<sycl::half>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "int32")  return runTest<int32_t>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "uint32") return runTest<uint32_t>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "int16")  return runTest<int16_t>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "uint16") return runTest<uint16_t>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "uint8")  return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores, useSampled);
    if (type == "int8")   return runTest<int8_t>(width, height, channels, useLinear, useSemaphores, useSampled);
    
    if (type == "unorm8") {
        return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores, useSampled, 
                                getUnorm8Format(channels), sycl::image_channel_type::unorm_int8);
    }

    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
}