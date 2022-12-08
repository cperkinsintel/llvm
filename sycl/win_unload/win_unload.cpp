#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

#include "win_unload.hpp"

// 100% this only works if the Manually loaded DLL are loaded before. 
#define LOAD_BEFORE 1


// working.
// need namespace, etc.

static void* oclPtr = nullptr;
static void* l0Ptr  = nullptr;

__declspec(dllexport) void* preserve_lib(const std::string &PluginPath) {
  std::cout << "preserve_lib: " << PluginPath <<  std::endl;
#ifdef LOAD_BEFORE
  if(PluginPath.find("opencl.dll") != std::string::npos){
    return oclPtr;
  }
  if(PluginPath.find("level_zero.dll") != std::string::npos){
    return l0Ptr;
  }
  return nullptr;
#else
  void*  Result = (void *)LoadLibraryA(PluginPath.c_str());
  return Result;
#endif
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, // handle to DLL module
                    DWORD fdwReason,    // reason for calling function
                    LPVOID lpReserved)  // reserved
{
  //TCHAR dllFilePath[512 + 1] = { 0 };
  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    //GetModuleFileNameA(hinstDLL, dllFilePath, 512);
    //printf(">> Module   load: %s\n", dllFilePath);

    std::cout << "win_unload process_attach" << std::endl;
#ifdef LOAD_BEFORE 
    oclPtr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\build\\bin\\pi_opencl.dll");
    ////oclPtr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\junk-drawer\\dll_unload\\xmain\\deploy\\win_prod\\bin\\pi_opencl.dll");
    l0Ptr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\build\\bin\\pi_level_zero.dll");
#endif   
    break;
  case DLL_PROCESS_DETACH:
    std::cout << "win_unload  process_detach" << std::endl;
    
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}


