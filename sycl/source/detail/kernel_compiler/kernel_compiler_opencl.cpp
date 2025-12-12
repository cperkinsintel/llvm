//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/ur.hpp> // getOsLibraryFuncAddress
#include <sycl/exception.hpp> // make_error_code

#include "kernel_compiler_opencl.hpp"

#include "../split_string.hpp"
#include "ocloc_api.h"
#include <detail/jit_compiler.hpp>

#include <cstring>    // strlen
#include <functional> // for std::function
#include <numeric>    // for std::accumulate
#include <regex>
#include <sstream>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// forward declaration
std::string InvokeOclocQuery(const std::vector<uint32_t> &IPVersionVec,
                             const char *identifier);

// ensures the OclocLibrary has the right version, etc.
void checkOclocLibrary(void *OclocLibrary) {
  void *OclocVersionHandle =
      sycl::detail::ur::getOsLibraryFuncAddress(OclocLibrary, "oclocVersion");
  // The initial versions of ocloc library did not have the oclocVersion()
  // function. Those versions had the same API as the first version of ocloc
  // library having that oclocVersion() function.
  int LoadedVersion = ocloc_version_t::OCLOC_VERSION_1_0;
  if (OclocVersionHandle) {
    decltype(::oclocVersion) *OclocVersionFunc =
        reinterpret_cast<decltype(::oclocVersion) *>(OclocVersionHandle);
    LoadedVersion = OclocVersionFunc();
  }
  // The loaded library with version (A.B) is compatible with expected API/ABI
  // version (X.Y) used here if A == B and B >= Y.
  int LoadedVersionMajor = LoadedVersion >> 16;
  int LoadedVersionMinor = LoadedVersion & 0xffff;
  int CurrentVersionMajor = ocloc_version_t::OCLOC_VERSION_CURRENT >> 16;
  int CurrentVersionMinor = ocloc_version_t::OCLOC_VERSION_CURRENT & 0xffff;
  if (LoadedVersionMajor != CurrentVersionMajor ||
      LoadedVersionMinor < CurrentVersionMinor) {
    throw sycl::exception(
        make_error_code(errc::build),
        std::string("Found incompatible version of ocloc library: (") +
            std::to_string(LoadedVersionMajor) + "." +
            std::to_string(LoadedVersionMinor) +
            "). The supported versions are (" +
            std::to_string(CurrentVersionMajor) +
            ".N), where (N >= " + std::to_string(CurrentVersionMinor) + ").");
  }
}

static std::unique_ptr<void, std::function<void(void *)>>
    OclocLibrary(nullptr, [](void *StoredPtr) {
      if (!StoredPtr)
        return;
      std::ignore = sycl::detail::ur::unloadOsLibrary(StoredPtr);
    });

void loadOclocLibrary(const std::vector<uint32_t> &IPVersionVec) {
#ifdef __SYCL_RT_OS_WINDOWS
  // first the environment, if not compatible will move on to absolute path.
  static const std::vector<std::string_view> OclocPaths = {
      "ocloc64.dll",
      "C:\\Program Files (x86)\\Intel\\oneAPI\\ocloc\\latest\\ocloc64.dll"};
#else
  static const std::vector<std::string_view> OclocPaths = {"libocloc.so"};
#endif

  // attemptLoad() sets OclocLibrary value by side effect.
  auto attemptLoad = [&](std::string_view OclocPath_sv) {
    std::string OclocPath(OclocPath_sv);
    try {
      // Load then perform checks. Each check throws.
      void *tempPtr = sycl::detail::ur::loadOsLibrary(OclocPath);
      OclocLibrary.reset(tempPtr);

      if (tempPtr == nullptr)
        throw sycl::exception(make_error_code(errc::build),
                              "Unable to load ocloc from " + OclocPath);

      checkOclocLibrary(tempPtr);

      InvokeOclocQuery(IPVersionVec, "CL_DEVICE_OPENCL_C_ALL_VERSIONS");
    } catch (const sycl::exception &) {
      OclocLibrary.reset(nullptr);
      return false;
    }
    return true;
  };
  for (const std::string_view &result : OclocPaths) {
    if (attemptLoad(result))
      return; // exit on successful attempt
  }
  // If we haven't exited yet, then throw to indicate failure.
  throw sycl::exception(make_error_code(errc::build), "Unable to load ocloc");
}

bool OpenCLC_Compilation_Available(const std::vector<uint32_t> &IPVersionVec) {
  // Already loaded?
  if (OclocLibrary != nullptr)
    return true;

  try {
    // loads and checks version
    loadOclocLibrary(IPVersionVec);
    return true;
  } catch (...) {
    return false;
  }
}

using voidPtr = void *;

void SetupLibrary(voidPtr &oclocInvokeHandle, voidPtr &oclocFreeOutputHandle,
                  std::error_code the_errc,
                  const std::vector<uint32_t> &IPVersionVec) {
  if (OclocLibrary == nullptr)
    loadOclocLibrary(IPVersionVec);

  if (!oclocInvokeHandle) {
    oclocInvokeHandle = sycl::detail::ur::getOsLibraryFuncAddress(
        OclocLibrary.get(), "oclocInvoke");
    if (!oclocInvokeHandle)
      throw sycl::exception(the_errc, "Cannot load oclocInvoke() function");

    oclocFreeOutputHandle = sycl::detail::ur::getOsLibraryFuncAddress(
        OclocLibrary.get(), "oclocFreeOutput");
    if (!oclocFreeOutputHandle)
      throw sycl::exception(the_errc, "Cannot load oclocFreeOutput() function");
  }
}

std::string IPVersionsToString(const std::vector<uint32_t> IPVersionVec) {
  std::stringstream ss;
  ss.imbue(std::locale::classic());
  bool amFirst = true;
  for (uint32_t ipVersion : IPVersionVec) {
    // if any device is not intelGPU, bail.
    if (ipVersion < 0x02000000)
      return "";

    if (!amFirst)
      ss << ",";
    amFirst = false;
    ss << ipVersion;
  }
  return ss.str();
}

std::string InvokeOclocQuery(const std::vector<uint32_t> &IPVersionVec,
                             const char *identifier) {

  std::string QueryLog = "";

  // handles into ocloc shared lib
  static void *oclocInvokeHandle = nullptr;
  static void *oclocFreeOutputHandle = nullptr;
  std::error_code the_errc = make_error_code(errc::runtime);

  SetupLibrary(oclocInvokeHandle, oclocFreeOutputHandle, the_errc,
               IPVersionVec);

  uint32_t NumOutputs = 0;
  uint8_t **Outputs = nullptr;
  uint64_t *OutputLengths = nullptr;
  char **OutputNames = nullptr;

  std::vector<const char *> Args = {"ocloc", "query"};
  std::string IPVersionsStr = IPVersionsToString(IPVersionVec);
  if (!IPVersionsStr.empty()) {
    Args.push_back("-device");
    Args.push_back(IPVersionsStr.c_str());
  }
  Args.push_back(identifier);

  decltype(::oclocInvoke) *OclocInvokeFunc =
      reinterpret_cast<decltype(::oclocInvoke) *>(oclocInvokeHandle);

  int InvokeError = OclocInvokeFunc(
      Args.size(), Args.data(), 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
      nullptr, &NumOutputs, &Outputs, &OutputLengths, &OutputNames);

  // Gather the results.
  for (uint32_t i = 0; i < NumOutputs; i++) {
    if (!strcmp(OutputNames[i], "stdout.log")) {
      if (OutputLengths[i] > 0) {
        const char *LogText = reinterpret_cast<const char *>(Outputs[i]);
        QueryLog.append(LogText, OutputLengths[i]);
      }
    }
  }

  // Try to free memory before reporting possible error.
  decltype(::oclocFreeOutput) *OclocFreeOutputFunc =
      reinterpret_cast<decltype(::oclocFreeOutput) *>(oclocFreeOutputHandle);
  int MemFreeError =
      OclocFreeOutputFunc(&NumOutputs, &Outputs, &OutputLengths, &OutputNames);

  if (InvokeError)
    throw sycl::exception(the_errc,
                          "ocloc reported errors: {\n" + QueryLog + "\n}");

  if (MemFreeError)
    throw sycl::exception(the_errc, "ocloc cannot safely free resources");

  return QueryLog;
}

spirv_vec_t
OpenCLC_to_SPIRV(const std::string &Source,
                 const std::vector<uint32_t> &IPVersionVec,
                 const std::vector<sycl::detail::string_view> &UserArgs,
                 std::string *LogPtr) {
  // Convert UserArgs to std::string
  std::vector<std::string> StringUserArgs;
  StringUserArgs.reserve(UserArgs.size());
  for (const auto &Arg : UserArgs) {
    StringUserArgs.emplace_back(Arg.data(), std::string_view(Arg).size());
  }

  // Use sycl-jit to compile OpenCL C to SPIR-V
  auto &JIT = sycl::detail::jit_compiler::get_instance();
  if (!JIT.isAvailable()) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "JIT compiler is not available for OpenCL C compilation");
  }

  // We perform compilation.
  std::string CompilationID = "opencl_compilation";

  // Note: IncludePairs is empty as we don't support headers via this path yet
  // (ocloc didn't clearly either via this specific API).
  std::vector<std::pair<std::string, std::string>> IncludePairs;

  auto Result =
      JIT.compileOpenCLC(CompilationID, Source, IncludePairs, StringUserArgs,
                         LogPtr, ::jit_compiler::BinaryFormat::SPIRV);

  sycl_device_binaries Binaries = Result.first;

  if (Binaries->NumDeviceBinaries == 0) {
    throw sycl::exception(make_error_code(errc::build),
                          "JIT compiler returned no binaries");
  }

  // Extract the first binary (assuming SPIR-V)
  const auto &Binary = Binaries->DeviceBinaries[0];
  const uint8_t *Start = Binary.BinaryStart;
  size_t Size = Binary.BinaryEnd - Binary.BinaryStart;

  spirv_vec_t SpirV(Start, Start + Size);

  // Clean up binaries managed by JIT
  JIT.destroyDeviceBinaries(Binaries);

  return SpirV;
}

bool OpenCLC_Feature_Available(const std::string &Feature, uint32_t IPVersion) {
  static std::string FeatureLog = "";
  if (FeatureLog.empty()) {
    try {
      FeatureLog = InvokeOclocQuery({IPVersion}, "CL_DEVICE_OPENCL_C_FEATURES");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // Allright, we have FeatureLog, so let's find that feature!
  return (FeatureLog.find(Feature) != std::string::npos);
}

bool OpenCLC_Supports_Version(
    const ext::oneapi::experimental::cl_version &Version, uint32_t IPVersion) {
  static std::string VersionLog = "";
  if (VersionLog.empty()) {
    try {
      VersionLog =
          InvokeOclocQuery({IPVersion}, "CL_DEVICE_OPENCL_C_ALL_VERSIONS");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // Have VersionLog, will search.
  // "OpenCL C":1.0.0 "OpenCL C":1.1.0 "OpenCL C":1.2.0 "OpenCL C":3.0.0
  std::stringstream ss;
  ss << Version.major << "." << Version.minor << "." << Version.patch;
  return VersionLog.find(ss.str()) != std::string::npos;
}

bool OpenCLC_Supports_Extension(
    const std::string &Name, ext::oneapi::experimental::cl_version *VersionPtr,
    uint32_t IPVersion) {
  std::error_code rt_errc = make_error_code(errc::runtime);
  static std::string ExtensionByVersionLog = "";
  if (ExtensionByVersionLog.empty()) {
    try {
      ExtensionByVersionLog =
          InvokeOclocQuery({IPVersion}, "CL_DEVICE_EXTENSIONS_WITH_VERSION");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // ExtensionByVersionLog is ready. Time to find Name, and update VersionPtr.
  // cl_khr_byte_addressable_store:1.0.0 cl_khr_device_uuid:1.0.0 ...
  size_t where = ExtensionByVersionLog.find(Name);
  if (where == std::string::npos) {
    return false;
  } // not there

  size_t colon = ExtensionByVersionLog.find(':', where);
  if (colon == std::string::npos) {
    throw sycl::exception(
        rt_errc,
        "trouble parsing query returned from CL_DEVICE_EXTENSIONS_WITH_VERSION "
        "- extension not followed by colon (:)");
  }

  // Note that VersionPtr is an optional parameter in
  // ext_oneapi_supports_cl_extension().
  if (!VersionPtr)
    return true;

  colon++; // move it forward

  size_t space = ExtensionByVersionLog.find(' ', colon); // could be npos

  size_t count = (space == std::string::npos) ? space : (space - colon);

  std::string versionStr = ExtensionByVersionLog.substr(colon, count);
  std::vector<std::string> versionVec =
      sycl::detail::split_string(versionStr, '.');
  if (versionVec.size() != 3) {
    throw sycl::exception(
        rt_errc,
        "trouble parsing query returned from  "
        "CL_DEVICE_EXTENSIONS_WITH_VERSION - version string unexpected: " +
            versionStr);
  }

  VersionPtr->major = std::stoi(versionVec[0]);
  VersionPtr->minor = std::stoi(versionVec[1]);
  VersionPtr->patch = std::stoi(versionVec[2]);
  return true;
}

std::string OpenCLC_Profile(uint32_t IPVersion) {
  try {
    std::string result = InvokeOclocQuery({IPVersion}, "CL_DEVICE_PROFILE");
    // NOTE: result has \n\n amended. Clean it up.
    // TODO: remove this once the ocloc query is fixed.
    result.erase(std::remove_if(result.begin(), result.end(),
                                [](char c) {
                                  return !std::isprint(c) || std::isspace(c);
                                }),
                 result.end());

    return result;
  } catch (sycl::exception &) {
    return "";
  }
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

