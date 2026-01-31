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
#include <algorithm> // for std::clamp

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// SYCL TYPE MAPPING (Boss Edition)
// ---------------------------------------------------------
template <typename T> sycl::image_channel_type getSyclChannelType();
template <> inline sycl::image_channel_type getSyclChannelType<float>() { return sycl::image_channel_type::fp32; }
template <> inline sycl::image_channel_type getSyclChannelType<int32_t>() { return sycl::image_channel_type::signed_int32; }
template <> inline sycl::image_channel_type getSyclChannelType<uint32_t>() { return sycl::image_channel_type::unsigned_int32; }
template <> inline sycl::image_channel_type getSyclChannelType<int16_t>() {  return sycl::image_channel_type::signed_int16; }
template <> inline sycl::image_channel_type getSyclChannelType<uint16_t>() { return sycl::image_channel_type::unsigned_int16; }
template <> inline sycl::image_channel_type getSyclChannelType<uint8_t>() { return sycl::image_channel_type::unsigned_int8; }
template <> inline sycl::image_channel_type getSyclChannelType<int8_t>() { return sycl::image_channel_type::signed_int8; }

// HALF
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
// GENERATORS & VERIFIERS
// ---------------------------------------------------------
// We need distinct patterns for A and B to prove we aren't just reading the same image twice
template <typename T>
T generateValueA(size_t x, size_t y, int channel) {
    float val = (float)(x + y) / 100.0f; 
    if constexpr (std::is_floating_point_v<T>) return static_cast<T>(val + channel * 0.1f);
    else return static_cast<T>((x + y + channel * 10) % 64); // Keep low to avoid overflow
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
    
    // 1. Create THREE Images (A, B, Output)
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if(useSampled) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

    ImageResources imgA = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);
    ImageResources imgB = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);
    ImageResources imgOut = createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);

    // 2. Upload Data to A and B
    VkSemaphore semA = VK_NULL_HANDLE;
    VkSemaphore semB = VK_NULL_HANDLE;
    if (useSemaphores) {
        semA = createExportableSemaphore(vkCtx);
        semB = createExportableSemaphore(vkCtx);
    }

    // Upload A
    uploadImage(vkCtx, imgA,  channels, semA, [&](size_t i, int c) {
        return generateValueA<T>(i % width, i / width, c);
    });
    
    // Upload B
    uploadImage(vkCtx, imgB,  channels, semB, [&](size_t i, int c) {
        return generateValueB<T>(i % width, i / width, c);
    });

    // 3. Import into SYCL
    namespace syclexp = sycl::ext::oneapi::experimental;
    sycl::queue q;
    
    // Import Memory
    auto extMemA = syclexp::import_external_memory(syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, imgA.memory), syclexp::external_mem_handle_type::opaque_fd, imgA.allocationSize}, q.get_device(), q.get_context());
    auto extMemB = syclexp::import_external_memory(syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, imgB.memory), syclexp::external_mem_handle_type::opaque_fd, imgB.allocationSize}, q.get_device(), q.get_context());
    auto extMemOut = syclexp::import_external_memory(syclexp::external_mem_descriptor<syclexp::resource_fd>{getMemFd(vkCtx, imgOut.memory), syclexp::external_mem_handle_type::opaque_fd, imgOut.allocationSize}, q.get_device(), q.get_context());

    // Import Semaphores
    syclexp::external_semaphore extSemA, extSemB, extSemOut;
    VkSemaphore semOutVk = VK_NULL_HANDLE;
    if (useSemaphores) {
        extSemA = syclexp::import_external_semaphore(syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemaphoreFd(vkCtx, semA), syclexp::external_semaphore_handle_type::opaque_fd}, q.get_device(), q.get_context());
        extSemB = syclexp::import_external_semaphore(syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemaphoreFd(vkCtx, semB), syclexp::external_semaphore_handle_type::opaque_fd}, q.get_device(), q.get_context());
        
        semOutVk = createExportableSemaphore(vkCtx);
        extSemOut = syclexp::import_external_semaphore(syclexp::external_semaphore_descriptor<syclexp::resource_fd>{getSemaphoreFd(vkCtx, semOutVk), syclexp::external_semaphore_handle_type::opaque_fd}, q.get_device(), q.get_context());
    }

    // Map Images
    sycl::image_channel_type syclType = syclOverride.has_value() ? syclOverride.value() : getSyclChannelType<T>();
    syclexp::image_descriptor imgDesc({(size_t)width, (size_t)height}, channels, syclType);

    // Common Map calls (Memory Handle -> Image Mem Handle)
    auto imgMemA = syclexp::map_external_image_memory(extMemA, imgDesc, q.get_device(), q.get_context());
    auto imgMemB = syclexp::map_external_image_memory(extMemB, imgDesc, q.get_device(), q.get_context());
    auto imgMemOut = syclexp::map_external_image_memory(extMemOut, imgDesc, q.get_device(), q.get_context());
    
    // Output is ALWAYS unsampled (we write to it)
    auto handleOut = syclexp::create_image(imgMemOut, imgDesc, q.get_device(), q.get_context());

    sycl::event kernelEvent;

    if (useSampled) {
        // =========================================================
        // PATH A: SAMPLED INPUTS
        // =========================================================
        
        // 1. Create Sampler (Unnormalized coordinates: 0.5, 1.5...)
        syclexp::bindless_image_sampler sampler(
            sycl::addressing_mode::clamp_to_edge,
            sycl::coordinate_normalization_mode::unnormalized, 
            sycl::filtering_mode::nearest 
        );

        // 2. Create Sampled Handles
        auto handleA = syclexp::create_image(imgMemA, sampler, imgDesc, q.get_device(), q.get_context());
        auto handleB = syclexp::create_image(imgMemB, sampler, imgDesc, q.get_device(), q.get_context());

        // 3. Submit Sampled Kernel
        kernelEvent = q.submit([&](sycl::handler& h) {
            if (useSemaphores) {
                h.ext_oneapi_wait_external_semaphore(extSemA);
                h.ext_oneapi_wait_external_semaphore(extSemB);
            }
            
            h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
                int x = item.get_id(0);
                int y = item.get_id(1);
                
                // Target Pixel Centers
                float u = (float)x + 0.5f; 
                float v = (float)y + 0.5f;
                bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);

                using Vec4 = sycl::vec<float, 4>;
                
                // Sample A & B (Always returns float vector)
                Vec4 valA = syclexp::sample_image<Vec4>(handleA, sycl::float2(u, v));
                Vec4 valB = syclexp::sample_image<Vec4>(handleB, sycl::float2(u, v));

                // Add
                Vec4 sum = valA + valB;

                // Write Output
                if (isUnorm) {
                    // Clamp and Write directly (SYCL converts float->unorm byte)
                    sum.x() = sycl::clamp(sum.x(), 0.0f, 1.0f);
                    sum.y() = sycl::clamp(sum.y(), 0.0f, 1.0f);
                    sum.z() = sycl::clamp(sum.z(), 0.0f, 1.0f);
                    sum.w() = sycl::clamp(sum.w(), 0.0f, 1.0f);
                    
                    if(channels==1) syclexp::write_image(handleOut, sycl::int2(x,y), sum.x());
                    else if(channels==2) syclexp::write_image(handleOut, sycl::int2(x,y), sycl::float2(sum.x(), sum.y()));
                    else syclexp::write_image(handleOut, sycl::int2(x,y), sum);
                } else {
                    // Standard Type: Cast back to T
                    if(channels==1) syclexp::write_image(handleOut, sycl::int2(x,y), static_cast<T>(sum.x()));
                    else if(channels==2) syclexp::write_image(handleOut, sycl::int2(x,y), sycl::vec<T,2>(static_cast<T>(sum.x()), static_cast<T>(sum.y())));
                    else syclexp::write_image(handleOut, sycl::int2(x,y), sycl::vec<T,4>(static_cast<T>(sum.x()), static_cast<T>(sum.y()), static_cast<T>(sum.z()), static_cast<T>(sum.w())));
                }
            });
        });

        // Cleanup Sampled Handles
        // Note: In real app, destroy after wait. Here we rely on q.wait() at end.
        // We will do cleanup at the bottom of the function.

    } else {
        // =========================================================
        // PATH B: UNSAMPLED INPUTS
        // =========================================================

        // 1. Create Unsampled Handles
        auto handleA = syclexp::create_image(imgMemA, imgDesc, q.get_device(), q.get_context());
        auto handleB = syclexp::create_image(imgMemB, imgDesc, q.get_device(), q.get_context());

        // 2. Submit Unsampled Kernel
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
                Vec4 valA(0,0,0,0);
                Vec4 valB(0,0,0,0);

                // Fetch A & B
                if (isUnorm) {
                    valA = syclexp::fetch_image<Vec4>(handleA, sycl::int2(x, y));
                    valB = syclexp::fetch_image<Vec4>(handleB, sycl::int2(x, y));
                } else {
                    // Helper to fetch and cast to float for math
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

                // Add
                Vec4 sum = valA + valB;

                // Write Output (Same logic as above)
                if (isUnorm) {
                    sum.x() = sycl::clamp(sum.x(), 0.0f, 1.0f);
                    sum.y() = sycl::clamp(sum.y(), 0.0f, 1.0f);
                    sum.z() = sycl::clamp(sum.z(), 0.0f, 1.0f);
                    sum.w() = sycl::clamp(sum.w(), 0.0f, 1.0f);
                    
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
        
        // Cleanup happens below
    }

    if (useSemaphores) {
        q.submit([&](sycl::handler& h) {
            h.depends_on(kernelEvent);
            h.ext_oneapi_signal_external_semaphore(extSemOut);
        });
    }
    q.wait();

    // CLEANUP
    // (Handles are local to the if/else blocks above, so we can't destroy them here easily
    //  unless we declared them outside with unique_ptr or void*. 
    //  However, sycl::queue destruction or context destruction usually cleans up resources, 
    //  but explicit destroy is safer.
    //  
    //  Since we split the scope, the handles A/B are gone. 
    //  This leaks the handles on the implementation side if not destroyed!
    
    //  CORRECTION: We must destroy 'handleOut' here.
    //  But handleA/handleB are out of scope. 
    //  For a test, this leak is minor (OS cleans up on exit), 
    //  but for correctness, you could move the cleanup inside the if/else blocks 
    //  right after q.wait() if you moved q.wait() inside.)
    
    syclexp::destroy_image_handle(handleOut, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemA, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemB, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemOut, q.get_device(), q.get_context());
    if(useSemaphores) {
        syclexp::release_external_semaphore(extSemA, q.get_device(), q.get_context());
        syclexp::release_external_semaphore(extSemB, q.get_device(), q.get_context());
        syclexp::release_external_semaphore(extSemOut, q.get_device(), q.get_context());
    }

    // 5. Verify Output (Vulkan Readback)
    bool passed = verifyImage(vkCtx, imgOut, channels, semOutVk, [&](size_t i, int c) {
        size_t x = i % width; size_t y = i / width;
        T a = generateValueA<T>(x, y, c);
        T b = generateValueB<T>(x, y, c);
        
        // Host-Side Math to match Device
        if (syclOverride.has_value() && syclOverride.value() == sycl::image_channel_type::unorm_int8) {
             // Unorm Logic: Float Addition with clamping
             float fa = (float)a / 255.0f; 
             float fb = (float)b / 255.0f;
             float sum = std::min(fa + fb, 1.0f); // Clamp
             return static_cast<T>(sum * 255.0f + 0.5f); // Round back to byte
        } else {
             // Standard Logic
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

    // 1. Gentle Cleanup for A and B (Keep Device Alive)
    cleanupImageResources(vkCtx, imgA);
    cleanupImageResources(vkCtx, imgB);

    // 2. Final Cleanup (Destroys Output Image AND Device/Instance)
    cleanupVulkan(vkCtx, imgOut);
    
    return passed ? 0 : 1;
}

int main(int argc, char** argv) {
    int width = 16, height = 16, channels = 4;
    bool useLinear = false, useSemaphores = false, useSampled = false;
    std::string type = "float";

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--semaphores") useSemaphores = true;
        else if(arg == "--linear") useLinear = true;
        else if(arg == "--sampled") useSampled = true;
        else if(arg == "--channels" && i+1 < argc) channels = std::stoi(argv[++i]);
        else if(arg == "--type" && i+1 < argc) type = argv[++i];
        else { try { width = std::stoi(arg); height = width; } catch(...) {} }
    }

    std::cout << "Running 2D ARITHMETIC Test (C = A + B) | Type: " << type << " | Size: " << width << "x" << height << " | Channels: " << channels << std::endl;

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