#include <cassert>

#ifdef _WIN32

#include <Windows.h>
#include <direct.h>
#include <malloc.h>
#include <shlwapi.h>

#endif


#include <iostream>
#include <map>
#include <string>



#include "win_proxy_loader.hpp"



#ifdef _WIN32

// ------------------------------------

static constexpr const char *DirSep = "\\";
using OSModuleHandle = intptr_t;
/// Module handle for the executable module - it is assumed there is always
/// single one at most.
static constexpr OSModuleHandle ExeModuleHandle = -1;


// cribbed from sycl/source/detail/os_util.cpp
std::string getDirName(const char *Path) {
  std::string Tmp(Path);
  // Remove trailing directory separators
  Tmp.erase(Tmp.find_last_not_of("/\\") + 1, std::string::npos);

  size_t pos = Tmp.find_last_of("/\\");
  if (pos != std::string::npos)
    return Tmp.substr(0, pos);

  // If no directory separator is present return initial path like dirname does
  return Tmp;
}

// cribbed from sycl/source/detail/os_util.cpp
OSModuleHandle getOSModuleHandle(const void *VirtAddr) {
  HMODULE PhModule;
  DWORD Flag = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  auto LpModuleAddr = reinterpret_cast<LPCSTR>(VirtAddr);
  if (!GetModuleHandleExA(Flag, LpModuleAddr, &PhModule)) {
    // Expect the caller to check for zero and take
    // necessary action
    return 0;
  }
  if (PhModule == GetModuleHandleA(nullptr))
    return ExeModuleHandle;
  return reinterpret_cast<OSModuleHandle>(PhModule);
}

// cribbed from sycl/source/detail/os_util.cpp
/// Returns an absolute path where the object was found.
std::string getCurrentDSODir() {
  char Path[MAX_PATH];
  Path[0] = '\0';
  Path[sizeof(Path) - 1] = '\0';
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODir));
  DWORD Ret = GetModuleFileNameA(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle),
      reinterpret_cast<LPSTR>(&Path), sizeof(Path));
  assert(Ret < sizeof(Path) && "Path is longer than PATH_MAX?");
  assert(Ret > 0 && "GetModuleFileNameA failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpecA(reinterpret_cast<LPSTR>(&Path));
  assert(RetCode && "PathRemoveFileSpecA failed");
  (void)RetCode;

  return Path;
}

  

// these are cribbed from sycl/detail/pi.hpp
#define __SYCL_OPENCL_PLUGIN_NAME "pi_opencl.dll"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "pi_level_zero.dll"
#define __SYCL_CUDA_PLUGIN_NAME "pi_cuda.dll"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "pi_esimd_emulator.dll"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dll"

// ------------------------------------


static std::map<std::string, void*> dllMap;

/// load the five libraries and store them in a map.
void preloadLibraries(){
  // this path duplicates sycl/detail/pi.cpp:initializePlugins
  const std::string LibSYCLDir =  getCurrentDSODir() + DirSep;

  std::string ocl_path = LibSYCLDir + __SYCL_OPENCL_PLUGIN_NAME;
  dllMap.emplace(ocl_path, LoadLibraryA(ocl_path.c_str()));
  
  std::string l0_path = LibSYCLDir + __SYCL_LEVEL_ZERO_PLUGIN_NAME;
  dllMap.emplace(l0_path, LoadLibraryA(l0_path.c_str()));
  
  std::string cuda_path = LibSYCLDir + __SYCL_CUDA_PLUGIN_NAME;
  dllMap.emplace(cuda_path, LoadLibraryA(cuda_path.c_str()));
  
  std::string esimd_path = LibSYCLDir + __SYCL_ESIMD_EMULATOR_PLUGIN_NAME;
  dllMap.emplace(esimd_path, LoadLibraryA(esimd_path.c_str()));
  
  std::string hip_path = LibSYCLDir + __SYCL_HIP_PLUGIN_NAME;
  dllMap.emplace(hip_path, LoadLibraryA(hip_path.c_str()));
}


/// windows_pi.cpp:loadOsLibrary() calls this to get the DLL we loaded earlier.
__declspec(dllexport) void* getPreloadedPlugin(const std::string &PluginPath) {

  auto match  = dllMap.find(PluginPath); // result might be nullptr, which is perfectly valid.
  if(match == dllMap.end()){
    // but asking for something we don't know about at all, is an issue.
    std::cout << "unknown plugin: " << PluginPath << std::endl;
    assert(false && "getPreloadedPlugin was given an unknown plugin path.");
    return nullptr;
  }
  return match->second;
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, // handle to DLL module
                    DWORD fdwReason,    // reason for calling function
                    LPVOID lpReserved)  // reserved
{
  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    preloadLibraries();
    
    break;
  case DLL_PROCESS_DETACH:
  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:    
    break;
  }
  return TRUE;
}

#endif //WIN32

