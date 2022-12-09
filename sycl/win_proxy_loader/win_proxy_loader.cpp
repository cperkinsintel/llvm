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
#endif //WIN32
  

// these are cribbed from sycl/detail/pi.hpp
#define __SYCL_OPENCL_PLUGIN_NAME "pi_opencl.dll"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "pi_level_zero.dll"
#define __SYCL_CUDA_PLUGIN_NAME "pi_cuda.dll"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "pi_esimd_emulator.dll"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dll" 

void preloadLibraries(){
  // this path duplicates sycl/detail/pi.cpp:initializePlugins
  const std::string LibSYCLDir =  getCurrentDSODir() + DirSep;

}


static void* oclPtr = nullptr;
static void* l0Ptr  = nullptr;

static std::map<std::string, void*> dllMap;

__declspec(dllexport) void* preserve_lib(const std::string &PluginPath) {
  std::cout << "preserve_lib: " << PluginPath <<  std::endl;
  if(PluginPath.find("opencl.dll") != std::string::npos){
    return oclPtr;
  }
  if(PluginPath.find("level_zero.dll") != std::string::npos){
    return l0Ptr;
  }
  return nullptr;

}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, // handle to DLL module
                    DWORD fdwReason,    // reason for calling function
                    LPVOID lpReserved)  // reserved
{
  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    std::cout << "win_proxy_loader process_attach" << std::endl;
    oclPtr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\build\\bin\\pi_opencl.dll");
    ////oclPtr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\junk-drawer\\dll_unload\\xmain\\deploy\\win_prod\\bin\\pi_opencl.dll");
    l0Ptr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\build\\bin\\pi_level_zero.dll");
    break;
  case DLL_PROCESS_DETACH:
    std::cout << "win_proxy_loader  process_detach" << std::endl;
    
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}


