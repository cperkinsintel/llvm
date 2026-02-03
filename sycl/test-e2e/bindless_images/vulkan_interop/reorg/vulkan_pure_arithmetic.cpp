/*
  PURE VULKAN TEST: 2D Arithmetic (A + B = C) with Semaphores


  $VULKAN_SDK/bin/glslangValidator -V vulkan_arithmetic.comp -o vulkan_arithmetic.spv
  
  Compile:
  clang++ -std=c++17 -o vk_pure_arith.bin vulkan_pure_arithmetic.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib

  Run:
  ./vk_pure_arith.bin 16x16
*/



#include "test_verification.hpp" 
#include "vulkan_setup.hpp"      
#include <vector>
#include <fstream>

// --- LOCAL HELPERS ---

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
    return imageView;
}

VkCommandBuffer beginSingleTimeCommands(const VulkanContext& ctx, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open shader file!");
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module!");
    return shaderModule;
}

// --- MAIN ---

int main(int argc, char** argv) {
    int width = 16;
    int height = 16;
    if(argc > 1) {
        std::string arg = argv[1];
        size_t xPos = arg.find("x");
        if(xPos != std::string::npos) {
            width = std::stoi(arg.substr(0, xPos));
            height = std::stoi(arg.substr(xPos+1));
        }
    }

    std::cout << "Running PURE VULKAN Arithmetic Test (" << width << "x" << height << ")" << std::endl;

    VulkanContext vkCtx = createVulkanContext();
    VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};

    // FIXED: Renamed to cmdPoolInfo to avoid collision
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(vkCtx.device, &cmdPoolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    // 1. Create Resources
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    
    ImageResources imgA = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage);
    ImageResources imgB = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage);
    ImageResources imgOut = createExportableImage(vkCtx, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage);

    // Manually create views
    VkImageView viewA = createImageView(vkCtx.device, imgA.image, VK_FORMAT_R32G32B32A32_SFLOAT);
    VkImageView viewB = createImageView(vkCtx.device, imgB.image, VK_FORMAT_R32G32B32A32_SFLOAT);
    VkImageView viewOut = createImageView(vkCtx.device, imgOut.image, VK_FORMAT_R32G32B32A32_SFLOAT);

    VkSemaphore semA = createExportableSemaphore(vkCtx);
    VkSemaphore semB = createExportableSemaphore(vkCtx);
    VkSemaphore semOut = createExportableSemaphore(vkCtx);

    // 2. Upload Data
    uploadImage(vkCtx, imgA, 4, semA, [&](size_t i, int c) { return (float)(i + c); });
    uploadImage(vkCtx, imgB, 4, semB, [&](size_t i, int c) { return (float)(i * 2 + c); });

    // 3. Pipeline Setup
    VkDescriptorSetLayoutBinding bindings[3] = {};
    for(int i=0; i<3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    
    VkDescriptorSetLayout descLayout;
    vkCreateDescriptorSetLayout(vkCtx.device, &layoutInfo, nullptr, &descLayout);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descLayout;
    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(vkCtx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkShaderModule compShaderModule = createShaderModule(vkCtx.device, "vulkan_arithmetic.spv");
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = compShaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;
    
    VkPipeline computePipeline;
    vkCreateComputePipelines(vkCtx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3};
    
    // FIXED: Renamed to descPoolInfo to avoid collision
    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;
    descPoolInfo.maxSets = 1;
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(vkCtx.device, &descPoolInfo, nullptr, &descPool);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descLayout;
    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(vkCtx.device, &allocInfo, &descSet);

    // Bind Views
    VkDescriptorImageInfo imgInfos[3];
    imgInfos[0] = {VK_NULL_HANDLE, viewA, VK_IMAGE_LAYOUT_GENERAL};
    imgInfos[1] = {VK_NULL_HANDLE, viewB, VK_IMAGE_LAYOUT_GENERAL};
    imgInfos[2] = {VK_NULL_HANDLE, viewOut, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet descriptorWrites[3] = {};
    for(int i=0; i<3; ++i) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &imgInfos[i];
    }
    vkUpdateDescriptorSets(vkCtx.device, 3, descriptorWrites, 0, nullptr);

    // 4. Execution
    VkCommandBuffer cmd = beginSingleTimeCommands(vkCtx, commandPool);

    auto transition = [&](VkImage img) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = img;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0; 
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    };
    transition(imgA.image);
    transition(imgB.image);
    transition(imgOut.image);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    vkEndCommandBuffer(cmd);

    // 5. Submit (The Test)
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    // WAIT for A and B to finish uploading
    VkSemaphore waitSemaphores[] = {semA, semB};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    // SIGNAL Out when compute is done
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semOut;

    if (vkQueueSubmit(vkCtx.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    // 6. Verify
    bool passed = verifyImage(vkCtx, imgOut, 4, semOut, [&](size_t i, int c) {
        float a = (float)(i + c);
        float b = (float)(i * 2 + c);
        return a + b;
    });

    if(passed) std::cout << "SUCCESS!" << std::endl;
    else std::cout << "FAILURE!" << std::endl;

    // Cleanup
    vkDestroyShaderModule(vkCtx.device, compShaderModule, nullptr);
    vkDestroyPipeline(vkCtx.device, computePipeline, nullptr);
    vkDestroyPipelineLayout(vkCtx.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(vkCtx.device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(vkCtx.device, descLayout, nullptr);
    
    vkDestroySemaphore(vkCtx.device, semA, nullptr);
    vkDestroySemaphore(vkCtx.device, semB, nullptr);
    vkDestroySemaphore(vkCtx.device, semOut, nullptr);
    
    vkDestroyImageView(vkCtx.device, viewA, nullptr);
    vkDestroyImageView(vkCtx.device, viewB, nullptr);
    vkDestroyImageView(vkCtx.device, viewOut, nullptr);

    vkDestroyCommandPool(vkCtx.device, commandPool, nullptr);

    cleanupImageResources(vkCtx, imgA);
    cleanupImageResources(vkCtx, imgB);
    cleanupVulkan(vkCtx, imgOut);
    
    return passed ? 0 : 1;
}