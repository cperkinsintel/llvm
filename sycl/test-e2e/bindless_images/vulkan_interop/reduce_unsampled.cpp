// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes -DTEST_L0_SUPPORTED_VK_FORMAT %}
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT
/*

TEST_L0_SUPPORTED - spirv - ( which is L0 and OCL)
3D float: OK
3D half2: Kernel fail
3D half4: kerneal fail
2D float: OK
2D half2: kernel fail
2D half4: OK
2D half3: kernel fail copying staging memory to images

"not spirv"  - ( which is CUDA and HIP)
3D uint4: kernel fail
3D uint2: kernel fail
3d uint:  kernel fail copying staging memory to images
3D float4: kernel fail copying staging memory to images - BUT other operations do not fail.  Doesn't get correct results (obviously)
3D float2: some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
3D float:  some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
2D uint4:  some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
2D uint2:  some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
2D uint:   some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
2D float4: some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
2D float2: some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)
2D float:  some problem submitting image layout transition - BUT other operations do not fail.  Doesn't get correct results (obviously)



clang++ -fsycl  -DVERBOSE_PRINT=1 -DTEST_SEMAPHORE_IMPORT=1 -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan  -o ru_default_semaphore.bin reduce_prob.cpp
clang++ -fsycl  -DVERBOSE_PRINT=1                           -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan  -o ru_default.bin reduce_prob.cpp

clang++ -fsycl  -DVERBOSE_PRINT=1 -DTEST_SEMAPHORE_IMPORT=1 -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan -DCP_RES=1  -o ru_cp_res_semaphore.bin reduce_unsampled.cpp
clang++ -fsycl  -DVERBOSE_PRINT=1                           -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan -DCP_RES=1  -o ru_cp_res.bin reduce_unsampled.cpp


NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 ./ru_cp_res.bin

// ignore AOT for now. No discernable difference yet.
clang++ -fsycl -fsycl-targets=spir64_gen -Xs '-device bmg' -DVERBOSE_PRINT=1 -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan -DTEST_L0_SUPPORTED_VK_FORMAT  -o rp_aot.bin reduce_prob.cpp


*/

#include "../../CommonUtils/vulkan_common.hpp"
#include "../helpers/common.hpp"
#include <sycl/properties/queue_properties.hpp>

#include <random>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Helpers and utilities
namespace util {
struct handles_t {
  syclexp::external_mem input_external_mem_1, input_external_mem_2,
      output_external_mem;
  syclexp::image_mem_handle input_mem_handle_1, input_mem_handle_2,
      output_mem_handle;
  syclexp::external_semaphore sycl_wait_external_semaphore,
      sycl_done_external_semaphore;
  syclexp::unsampled_image_handle input_1, input_2, output;
};

template <typename InteropMemHandleT, typename InteropSemHandleT>
handles_t create_test_handles(
    sycl::context &ctxt, sycl::device &dev,
    InteropMemHandleT img_in_interop_handle_1,
    InteropMemHandleT img_in_interop_handle_2,
    InteropMemHandleT img_out_interop_handle,
    [[maybe_unused]] InteropSemHandleT sycl_wait_semaphore_handle,
    [[maybe_unused]] InteropSemHandleT sycl_done_semaphore_handle,
    const size_t img_size,
    sycl::ext::oneapi::experimental::image_descriptor &desc) {
  // Extension: map the external memory descriptors

#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      input_ext_mem_desc_1{img_in_interop_handle_1,
                           syclexp::external_mem_handle_type::win32_nt_handle,
                           img_size};
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      input_ext_mem_desc_2{img_in_interop_handle_2,
                           syclexp::external_mem_handle_type::win32_nt_handle,
                           img_size};
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      output_ext_mem_desc{img_out_interop_handle,
                          syclexp::external_mem_handle_type::win32_nt_handle,
                          img_size};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> input_ext_mem_desc_1{
      img_in_interop_handle_1, syclexp::external_mem_handle_type::opaque_fd,
      img_size};
  syclexp::external_mem_descriptor<syclexp::resource_fd> input_ext_mem_desc_2{
      img_in_interop_handle_2, syclexp::external_mem_handle_type::opaque_fd,
      img_size};
  syclexp::external_mem_descriptor<syclexp::resource_fd> output_ext_mem_desc{
      img_out_interop_handle, syclexp::external_mem_handle_type::opaque_fd,
      img_size};
#endif

  // Extension: create interop memory handles
  syclexp::external_mem input_external_mem_1 =
      syclexp::import_external_memory(input_ext_mem_desc_1, dev, ctxt);
  syclexp::external_mem input_external_mem_2 =
      syclexp::import_external_memory(input_ext_mem_desc_2, dev, ctxt);
  syclexp::external_mem output_external_mem =
      syclexp::import_external_memory(output_ext_mem_desc, dev, ctxt);

  // Extension: map image memory handles
  syclexp::image_mem_handle input_mapped_mem_handle_1 =
      syclexp::map_external_image_memory(input_external_mem_1, desc, dev, ctxt);
  syclexp::image_mem_handle input_mapped_mem_handle_2 =
      syclexp::map_external_image_memory(input_external_mem_2, desc, dev, ctxt);
  syclexp::image_mem_handle output_mapped_mem_handle =
      syclexp::map_external_image_memory(output_external_mem, desc, dev, ctxt);

  // Extension: create the image and return the handle
  syclexp::unsampled_image_handle input_1 =
      syclexp::create_image(input_mapped_mem_handle_1, desc, dev, ctxt);
  syclexp::unsampled_image_handle input_2 =
      syclexp::create_image(input_mapped_mem_handle_2, desc, dev, ctxt);
  syclexp::unsampled_image_handle output =
      syclexp::create_image(output_mapped_mem_handle, desc, dev, ctxt);

#ifdef TEST_SEMAPHORE_IMPORT
  // Extension: import semaphores
#ifdef _WIN32
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      sycl_wait_external_semaphore_desc{
          sycl_wait_semaphore_handle,
          syclexp::external_semaphore_handle_type::win32_nt_handle};
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      sycl_done_external_semaphore_desc{
          sycl_done_semaphore_handle,
          syclexp::external_semaphore_handle_type::win32_nt_handle};
#else
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      sycl_wait_external_semaphore_desc{
          sycl_wait_semaphore_handle,
          syclexp::external_semaphore_handle_type::opaque_fd};
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      sycl_done_external_semaphore_desc{
          sycl_done_semaphore_handle,
          syclexp::external_semaphore_handle_type::opaque_fd};
#endif

  syclexp::external_semaphore sycl_wait_external_semaphore =
      syclexp::import_external_semaphore(sycl_wait_external_semaphore_desc, dev,
                                         ctxt);
  syclexp::external_semaphore sycl_done_external_semaphore =
      syclexp::import_external_semaphore(sycl_done_external_semaphore_desc, dev,
                                         ctxt);
#else  // #ifdef TEST_SEMAPHORE_IMPORT
  syclexp::external_semaphore sycl_wait_external_semaphore{};
  syclexp::external_semaphore sycl_done_external_semaphore{};
#endif // #ifdef TEST_SEMAPHORE_IMPORT

  return {input_external_mem_1,
          input_external_mem_2,
          output_external_mem,
          input_mapped_mem_handle_1,
          input_mapped_mem_handle_2,
          output_mapped_mem_handle,
          sycl_wait_external_semaphore,
          sycl_done_external_semaphore,
          input_1,
          input_2,
          output};
}

void cleanup_test(sycl::context &ctxt, sycl::device &dev, handles_t handles) {
#ifdef TEST_SEMAPHORE_IMPORT
  syclexp::release_external_semaphore(handles.sycl_wait_external_semaphore, dev,
                                      ctxt);
  syclexp::release_external_semaphore(handles.sycl_done_external_semaphore, dev,
                                      ctxt);
#endif
  syclexp::destroy_image_handle(handles.input_1, dev, ctxt);
  syclexp::destroy_image_handle(handles.input_2, dev, ctxt);
  syclexp::destroy_image_handle(handles.output, dev, ctxt);
  syclexp::unmap_external_image_memory(
      handles.input_mem_handle_1, syclexp::image_type::standard, dev, ctxt);
  syclexp::unmap_external_image_memory(
      handles.input_mem_handle_2, syclexp::image_type::standard, dev, ctxt);
  syclexp::unmap_external_image_memory(
      handles.output_mem_handle, syclexp::image_type::standard, dev, ctxt);
  syclexp::release_external_memory(handles.input_external_mem_1, dev, ctxt);
  syclexp::release_external_memory(handles.input_external_mem_2, dev, ctxt);
  syclexp::release_external_memory(handles.output_external_mem, dev, ctxt);
}

template <typename InteropMemHandleT, typename InteropSemHandleT, int NDims,
          typename DType, sycl::image_channel_type CType, int NChannels,
          typename KernelName>
void run_ndim_test(sycl::range<NDims> global_size,
                   sycl::range<NDims> local_size,
                   InteropMemHandleT img_in_interop_handle_1,
                   InteropMemHandleT img_in_interop_handle_2,
                   InteropMemHandleT img_out_interop_handle,
                   InteropSemHandleT sycl_wait_semaphore_handle,
                   InteropSemHandleT sycl_done_semaphore_handle
#ifdef CP_RES
                   , size_t optimal_image_size
#endif                  
                  ) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::device dev;
  sycl::queue q{dev, {sycl::property::queue::in_order{}}};
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  syclexp::image_descriptor desc(global_size, NChannels, CType);

#ifdef CP_RES
  const size_t img_size = optimal_image_size;
#else
  const size_t img_size = global_size.size() * sizeof(DType) * NChannels;
#endif

  auto handles = create_test_handles(
      ctxt, dev, img_in_interop_handle_1, img_in_interop_handle_2,
      img_out_interop_handle, sycl_wait_semaphore_handle,
      sycl_done_semaphore_handle, img_size, desc);

#ifdef TEST_SEMAPHORE_IMPORT
  // Extension: wait for imported semaphore
  q.ext_oneapi_wait_external_semaphore(handles.sycl_wait_external_semaphore);
#endif

  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(
          sycl::nd_range<NDims>{global_size, local_size},
          [=](sycl::nd_item<NDims> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);

            if constexpr (NDims == 2) {
              if constexpr (NChannels > 1) {
                VecType px1 = syclexp::fetch_image<VecType>(
                    handles.input_1, sycl::int2(dim0, dim1));
                VecType px2 = syclexp::fetch_image<VecType>(
                    handles.input_2, sycl::int2(dim0, dim1));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<VecType>(
                    handles.output, sycl::int2(dim0, dim1), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_image<DType>(handles.input_1,
                                                        sycl::int2(dim0, dim1));
                DType px2 = syclexp::fetch_image<DType>(handles.input_2,
                                                        sycl::int2(dim0, dim1));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<DType>(handles.output,
                                            sycl::int2(dim0, dim1), DType(sum));
              }
            } else {
              size_t dim2 = it.get_global_id(2);

              if constexpr (NChannels > 1) {
                VecType px1 = syclexp::fetch_image<VecType>(
                    handles.input_1, sycl::int3(dim0, dim1, dim2));
                VecType px2 = syclexp::fetch_image<VecType>(
                    handles.input_2, sycl::int3(dim0, dim1, dim2));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<VecType>(
                    handles.output, sycl::int3(dim0, dim1, dim2), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_image<DType>(
                    handles.input_1, sycl::int3(dim0, dim1, dim2));
                DType px2 = syclexp::fetch_image<DType>(
                    handles.input_2, sycl::int3(dim0, dim1, dim2));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<DType>(
                    handles.output, sycl::int3(dim0, dim1, dim2), DType(sum));
              }
            }
          });
    });

#ifdef TEST_SEMAPHORE_IMPORT
    // Extension: signal imported semaphore
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_signal_external_semaphore(
          handles.sycl_done_external_semaphore);
    });
#endif

    // Wait for kernel completion before destroying external objects
    q.wait_and_throw();

    // Cleanup
    cleanup_test(ctxt, dev, handles);
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }
}
} // namespace util


// FUN STUFF
struct vulkan_image_test_resources_cp {
  VkImage vkImage;
  VkDeviceMemory imageMemory;
  size_t optimalImageSizeBytes;
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;

// Don't pass in imageSizeBytes.
vulkan_image_test_resources_cp(VkImageType imgType, VkFormat format,
                              VkExtent3D ext, size_t elementSizeBytes, int NChannels) {

    // Staging Buffer (Uses Linear Size)
    // This calculation IS correct for the staging buffer
    const size_t linearSizeBytes = ext.width * ext.height * ext.depth * NChannels * elementSizeBytes;

    stagingBuffer = vkutil::createBuffer(linearSizeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkMemoryRequirements bufferRequirements;
    vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &bufferRequirements);

    auto stagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory = vkutil::allocateDeviceMemory(
        /*linearSizeBytes*/ bufferRequirements.size, stagingMemoryTypeIndex, nullptr /*image*/, false /*exportable*/);
    VK_CHECK_CALL( vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0) );


    // Device Image (Uses Optimal Size)
        // VK_IMAGE_TILING_OPTIMAL
    vkImage = vkutil::createImage(imgType, format, ext,                                      
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  VK_IMAGE_USAGE_TRANSFER_DST_BIT,  // usage
        1,                                        // mipLevels
        false);                                   // linearTiling should be false  OPTIMAL image
                                                  // Optimal Tiling is the default, and that is 
                                                  // what unsampled_images.cpp does too.  Confusing

    // 
    VkMemoryRequirements memRequirements;
    auto imageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
        vkImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
    optimalImageSizeBytes = memRequirements.size;

    // Use memRequirements.size, NOT the user's linear size
    imageMemory = vkutil::allocateDeviceMemory(
        memRequirements.size, imageMemoryTypeIndex, vkImage);

    VK_CHECK_CALL( vkBindImageMemory(vk_device, vkImage, imageMemory, 0) );
}
  ~vulkan_image_test_resources_cp() {
    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkDestroyImage(vk_device, vkImage, nullptr);
    vkFreeMemory(vk_device, stagingMemory, nullptr);
    vkFreeMemory(vk_device, imageMemory, nullptr);
  }
};


// RUN TEST

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> local_size,
              unsigned int seed = 0) {
  uint32_t width = static_cast<uint32_t>(dims[0]);
  uint32_t height = 1;
  uint32_t depth = 1;

  size_t num_elems = dims[0];
  VkImageType imgType = VK_IMAGE_TYPE_1D;

  if (NDims > 1) {
    num_elems *= dims[1];
    height = static_cast<uint32_t>(dims[1]);
    imgType = VK_IMAGE_TYPE_2D;
  }
  if (NDims > 2) {
    num_elems *= dims[2];
    depth = static_cast<uint32_t>(dims[2]);
    imgType = VK_IMAGE_TYPE_3D;
  }

  VkFormat format = vkutil::to_vulkan_format(COrder, CType);
  const size_t imageSizeBytes = num_elems * NChannels * sizeof(DType);

#ifdef CP_RES
  vulkan_image_test_resources_cp inVkImgRes1(imgType, format, {width, height, depth}, sizeof(DType), NChannels);
  vulkan_image_test_resources_cp inVkImgRes2(imgType, format, {width, height, depth}, sizeof(DType), NChannels);
  vulkan_image_test_resources_cp outVkImgRes(imgType, format, {width, height, depth}, sizeof(DType), NChannels);
#else
  vkutil::vulkan_image_test_resources_t inVkImgRes1(imgType, format, {width, height, depth}, imageSizeBytes);
  vkutil::vulkan_image_test_resources_t inVkImgRes2(imgType, format, {width, height, depth}, imageSizeBytes);
  vkutil::vulkan_image_test_resources_t outVkImgRes(imgType, format, {width, height, depth}, imageSizeBytes);
#endif

  printString("Populating staging buffer\n");
  // Populate staging memory
  std::vector<DType> input_vector_0(num_elems * NChannels,
                                    static_cast<DType>(0));
  std::srand(seed);
  bindless_helpers::fill_rand(input_vector_0);

  DType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inVkImgRes1.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < (num_elems * NChannels); ++i) {
    inputStagingData[i] = input_vector_0[i];
  }
  vkUnmapMemory(vk_device, inVkImgRes1.stagingMemory);

  std::vector<DType> input_vector_1(num_elems * NChannels,
                                    static_cast<DType>(0));
  std::srand(seed);
  bindless_helpers::fill_rand(input_vector_1);

  VK_CHECK_CALL(vkMapMemory(vk_device, inVkImgRes2.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < (num_elems * NChannels); ++i) {
    inputStagingData[i] = input_vector_1[i];
  }
  vkUnmapMemory(vk_device, inVkImgRes2.stagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput1 =
        vkutil::createImageMemoryBarrier(inVkImgRes1.vkImage, 1 /*mipLevels*/);
    VkImageMemoryBarrier barrierInput2 =
        vkutil::createImageMemoryBarrier(inVkImgRes2.vkImage, 1 /*mipLevels*/);

    VkImageMemoryBarrier barrierOutput =
        vkutil::createImageMemoryBarrier(outVkImgRes.vkImage, 1 /*mipLevels*/);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput1);

    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput2);

    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierOutput);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

#ifdef TEST_SEMAPHORE_IMPORT
  // Create semaphore to later import in SYCL
  printString("Creating semaphores\n");
  VkSemaphore syclWaitSemaphore;
  {
    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#ifdef _WIN32
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    VK_CHECK_CALL(
        vkCreateSemaphore(vk_device, &sci, nullptr, &syclWaitSemaphore));
  }

  VkSemaphore syclDoneSemaphore;
  {
    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#ifdef _WIN32
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    VK_CHECK_CALL(
        vkCreateSemaphore(vk_device, &sci, nullptr, &syclDoneSemaphore));
  }
#endif // #ifdef TEST_SEMAPHORE_IMPORT

  printString("Copying staging memory to images\n");
  // Copy staging to main image memory
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, depth};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inVkImgRes1.stagingBuffer,
                           inVkImgRes1.vkImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inVkImgRes2.stagingBuffer,
                           inVkImgRes2.vkImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];

#ifdef TEST_SEMAPHORE_IMPORT
    submission.signalSemaphoreCount = 1;
    submission.pSignalSemaphores = &syclWaitSemaphore;
#endif
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
#ifndef TEST_SEMAPHORE_IMPORT
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
#endif
  }

  printString("Getting memory interop handles\n");

  // Pass memory to SYCL for modification
  auto global_size = dims;
#ifdef _WIN32
  auto input_mem_handle_1 =
      vkutil::getMemoryWin32Handle(inVkImgRes1.imageMemory);
  auto input_mem_handle_2 =
      vkutil::getMemoryWin32Handle(inVkImgRes2.imageMemory);
  auto output_mem_handle =
      vkutil::getMemoryWin32Handle(outVkImgRes.imageMemory);
#else
  auto input_mem_handle_1 = vkutil::getMemoryOpaqueFD(inVkImgRes1.imageMemory);
  auto input_mem_handle_2 = vkutil::getMemoryOpaqueFD(inVkImgRes2.imageMemory);
  auto output_mem_handle = vkutil::getMemoryOpaqueFD(outVkImgRes.imageMemory);
#endif

  printString("Getting semaphore interop handles\n");

#ifdef TEST_SEMAPHORE_IMPORT
  // Pass semaphores to SYCL for synchronization
#ifdef _WIN32
  auto sycl_wait_semaphore_handle =
      vkutil::getSemaphoreWin32Handle(syclWaitSemaphore);
  auto sycl_done_semaphore_handle =
      vkutil::getSemaphoreWin32Handle(syclDoneSemaphore);
#else
  auto sycl_wait_semaphore_handle =
      vkutil::getSemaphoreOpaqueFD(syclWaitSemaphore);
  auto sycl_done_semaphore_handle =
      vkutil::getSemaphoreOpaqueFD(syclDoneSemaphore);
#endif
#else  // #ifdef TEST_SEMAPHORE_IMPORT
  void *sycl_wait_semaphore_handle = nullptr;
  void *sycl_done_semaphore_handle = nullptr;
#endif // #ifdef TEST_SEMAPHORE_IMPORT

  // CP
  printString("Calling into SYCL with interop memory and semaphore handles\n");

  util::run_ndim_test<decltype(input_mem_handle_1),
                      decltype(sycl_wait_semaphore_handle), NDims, DType, CType,
                      NChannels, KernelName>(
      global_size, local_size, input_mem_handle_1, input_mem_handle_2,
      output_mem_handle, sycl_wait_semaphore_handle,
      sycl_done_semaphore_handle
#ifdef CP_RES
      , inVkImgRes1.optimalImageSizeBytes
#endif     
    );

  

  printString("Copying image memory to staging memory\n");
  // Copy main image memory to staging
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, depth};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[1], &cbbi));
    vkCmdCopyImageToBuffer(vk_transferCmdBuffers[1], outVkImgRes.vkImage,
                           VK_IMAGE_LAYOUT_GENERAL, outVkImgRes.stagingBuffer,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[1]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[1];

#ifdef TEST_SEMAPHORE_IMPORT
    submission.waitSemaphoreCount = 1;
    submission.pWaitSemaphores = &syclDoneSemaphore;
#endif
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  printString("Validating\n");
  // Validate that SYCL made changes to the memory
  bool validated = true;
  DType *outputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, outVkImgRes.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&outputStagingData));

  for (int i = 0; i < (num_elems * NChannels); ++i) {
    DType expected = input_vector_0[i] + input_vector_1[i];
    // Use helper function to determine if data is accepted
    // For integers, exact results are expected
    // For floats, accepted error variance is passed
    if (!util::is_equal(outputStagingData[i], expected)) {
      std::cerr << "Result mismatch! actual[" << i
                << "] == " << outputStagingData[i]
                << " : expected == " << expected << "\n";
      validated = false;
    }
    if (!validated)
      break;
  }
  vkUnmapMemory(vk_device, outVkImgRes.stagingMemory);

  if (validated) {
    printString("  Results are correct!\n");
  }

#ifdef TEST_SEMAPHORE_IMPORT
  // Cleanup
  vkDestroySemaphore(vk_device, syclWaitSemaphore, nullptr);
  vkDestroySemaphore(vk_device, syclDoneSemaphore, nullptr);
#endif

  // CP
  return validated; // true; //validated;
}

bool run_all() {
  unsigned int seed = 0;
  bool valid = true;

// BEGIN L0 SUPPORTED
  std::cout << " BEGIN L0 SUPPORTED TESTS " << std::endl;

  // OK
  printString("Running 3D float\n");
  valid &= run_test<3, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class fp32_3d_c1>(
      {1024, 1024, 16}, {16, 16, 1}, seed);

  // kernel fail
  printString("Running 3D half2\n");
  valid &= run_test<3, sycl::half, 2, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rg, class fp16_3d_c2>(
      {1920, 1080, 8}, {16, 8, 2}, seed);

  // kernel fail
  // printString("Running 3D half4\n");
  // valid &= run_test<3, sycl::half, 4, sycl::image_channel_type::fp16,
  //                   sycl::image_channel_order::rgba, class fp16_3d_c4>(
  //     {2048, 2048, 4}, {16, 16, 1}, seed);

  // OK
  printString("Running 2D float\n");
  valid &= run_test<2, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class fp32_2d_c1>(
      {1024, 1024}, {16, 16}, seed);

  // kernel fail
  // printString("Running 2D half2\n");
  // valid &= run_test<2, sycl::half, 2, sycl::image_channel_type::fp16,
  //                   sycl::image_channel_order::rg, class fp16_2d_c2>(
  //     {1920, 1080}, {16, 8}, seed);

  // OK
  printString("Running 2D half4\n");
  valid &= run_test<2, sycl::half, 4, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rgba, class fp16_2d_c4>(
      {2048, 2048}, {16, 16}, seed);

  // 3-channels
  // // kernel fail copying staging memory to images
  // printString("Running 2D half3\n");
  // valid &= run_test<2, sycl::half, 3, sycl::image_channel_type::fp16,
  //                   sycl::image_channel_order::rgb, class fp16_2d_c3>(
  //     {2048, 2048}, {2, 2}, seed);


// END L0 SUPPORTED
  std::cout << " END L0 SUPPORTED TESTS " << std::endl;

  // //kernel fail -- RERUN --- THIS TEST MISTAKENLY USED SIGNED INSTEAD OF UNSIGNED
  // printString("Running 3D uint4\n");
  // valid &= run_test<3, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
  //                   sycl::image_channel_order::rgba, class uint4_3d>(
  //     {272, 144, 4}, {16, 16, 4}, seed);

  // //kernel fail
  // printString("Running 3D uint2\n");
  // valid &= run_test<3, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
  //                   sycl::image_channel_order::rg, class uint2_3d>(
  //     {272, 144, 4}, {16, 16, 4}, seed);

  //kernel fail copying staging memory to images
  // printString("Running 3D uint\n");
  // valid &= run_test<3, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
  //                   sycl::image_channel_order::r, class uint1_3d>(
  //     {272, 144, 4}, {16, 16, 4}, seed);


  printString("Running 3D float4\n");
  valid &= run_test<3, float, 4, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rgba, class float4_3d>(
      {16, 16, 16}, {16, 16, 4}, seed);

  printString("Running 3D float2\n");
  valid &= run_test<3, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float2_3d>(
      {128, 128, 16}, {16, 16, 4}, seed);

  printString("Running 3D float\n");
  valid &= run_test<3, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class float1_3d>(
      {1024, 1024, 16}, {16, 16, 4}, seed);

  printString("Running 2D uint4\n");
  valid &= run_test<2, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rgba, class uint4_2d>(
      {1024, 1024}, {2, 2}, seed);

  printString("Running 2D uint2\n");
  valid &= run_test<2, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rg, class uint2_2d>(
      {128, 128}, {2, 2}, seed);

  printString("Running 2D uint\n");
  valid &= run_test<2, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::r, class uint1_2d>({512, 512},
                                                                  {2, 2}, seed);

  printString("Running 2D float4\n");
  valid &= run_test<2, float, 4, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rgba, class float4_2d>(
      {128, 64}, {2, 2}, seed);

  printString("Running 2D float2\n");
  valid &= run_test<2, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float2_2d>(
      {1024, 512}, {2, 2}, seed);

  printString("Running 2D float\n");
  valid &= run_test<2, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class float1_2d>(
      {32, 32}, {2, 2}, seed);

  return valid;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  sycl::device dev;

  if (vkutil::setupDevice(dev) != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  auto run_ok = run_all();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  if (run_ok) {
    std::cout << "All tests passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
