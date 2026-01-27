#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

// --- THE SHADER (Hardcoded, Valid Logical SPIR-V: NO-OP) ---
// #version 450
// void main() {}
// OpMemoryModel Logical GLSL450
const uint32_t shader_code[] = {
    0x07230203, 0x00010000, 0x000d000a, 0x00000006, 0x00000000,
    0x00020011, 0x00000001, // OpCapability Shader
    0x0003000e, 0x00000000, 0x00000001, // OpMemoryModel Logical GLSL450
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, // OpEntryPoint GLCompute %4 "main"
    0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001, // OpExecutionMode LocalSize 1 1 1
    0x00020013, 0x00000002, // OpTypeVoid
    0x00030021, 0x00000003, 0x00000002, // OpTypeFunction %2
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, // OpFunction %2 None %3
    0x000200f8, 0x00000005, // OpLabel %5
    0x000100fd, // OpReturn
    0x00010038  // OpFunctionEnd
};

// --- MINIMAL VK UTIL ---
// Inlined to ensure no header mismatches
VkInstance inst;
VkPhysicalDevice phys;
VkDevice dev;
VkQueue q;
uint32_t qFam;
VkCommandPool cmdPool;

#define VK_CHECK(x) { VkResult r = x; if (r != VK_SUCCESS) { std::cerr << "Fail at line " << __LINE__ << " code " << r << "\n"; exit(1); } }

void setup() {
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_2; // Request 1.2 for BMG safety
    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &inst));

    uint32_t count = 1;
    vkEnumeratePhysicalDevices(inst, &count, &phys);
    
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, qProps.data());
    for(uint32_t i=0; i<count; i++) {
        if(qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qFam = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qInfo.queueFamilyIndex = qFam;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &prio;
    
    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    VK_CHECK(vkCreateDevice(phys, &devInfo, nullptr, &dev));
    
    vkGetDeviceQueue(dev, qFam, 0, &q);
    
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = qFam;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(dev, &poolInfo, nullptr, &cmdPool));
}

int main() {
    setup();
    std::cout << "Vulkan Initialized.\n";

    // 1. Pipeline Layout (Empty)
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    // 2. Shader Module
    VkShaderModuleCreateInfo shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = sizeof(shader_code);
    shaderInfo.pCode = shader_code;
    VkShaderModule shaderMod;
    VK_CHECK(vkCreateShaderModule(dev, &shaderInfo, nullptr, &shaderMod));

    // 3. Pipeline
    VkComputePipelineCreateInfo pipeInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderMod, "main", nullptr};
    pipeInfo.layout = pipelineLayout;
    VkPipeline pipe;
    
    // THIS IS THE MOMENT OF TRUTH
    VkResult res = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe);
    if (res != VK_SUCCESS) {
        std::cerr << "Pipeline Creation Failed: " << res << "\n";
        return 1;
    }

    std::cout << "Pipeline Created Successfully!\n";

    // 4. Run (Optional, to prove it executes)
    VkCommandBufferAllocateInfo cmdAlloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cmdAlloc, &cmd));

    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    vkBeginCommandBuffer(cmd, &begin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdDispatch(cmd, 1, 1, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(q, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(q));

    std::cout << "TEST PASSED.\n";
    return 0;
}