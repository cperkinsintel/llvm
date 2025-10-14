
set(SYCL_JIT_RESOURCE_MODE "GLOB" CACHE STRING 
    "Resource generation mode: GLOB, GLOB_AND_MANIFEST, or MANIFEST")
set_property(CACHE SYCL_JIT_RESOURCE_MODE PROPERTY STRINGS GLOB GLOB_AND_MANIFEST MANIFEST)

set(SYCL_JIT_RESOURCE_MODE "MANIFEST")

set(SYCL_JIT_RESOURCE_BLACKLIST "" CACHE FILEPATH "Path to a blacklist file for sycl-jit resources.")
set(SYCL_JIT_RESOURCE_BLACKLIST "/iusers/cperkins/sycl_workspace/glob_blacklist.txt")


set(SYCL_JIT_RESOURCE_CPP "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp")
set(SYCL_JIT_RESOURCE_OBJ "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp.o")
#set(GLOB_MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/glob_manifest.txt)
set(GLOB_MANIFEST "/iusers/cperkins/sycl_workspace/rel_glob_manifest.txt")
#set(GLOB_MANIFEST "/iusers/cperkins/sycl_workspace/filtered_manifest.txt")
set(DEPS_FILE ${CMAKE_CURRENT_BINARY_DIR}/probe_dependencies.d)
set(FILTERED_MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/filtered_manifest.txt)
set(FILTER_PY_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/utils/filter_manifest.py)


if (WIN32)
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "c:/sycl-jit-toolchain/")
else()
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "/sycl-jit-toolchain/")
endif()


# this target can be run on demand. It outputs the dependencies 
# required to build some kernel probe.cpp file.
# $ ninja generate-deps
add_custom_target(generate-deps
    # This is the command that will be executed
    COMMAND ${clang_exe}
        -fsycl-device-only
        --target=x86_64-unknown-linux-gnu
        -I ${CMAKE_CURRENT_SOURCE_DIR}/include
        -O2
        -Wno-c23-extensions
        -std=c++17
        -fPIC
        -fvisibility=hidden
        -M
        -MF ${DEPS_FILE}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/probe.cpp
    
    # This ensures the clang executable is built before we try to use it
    DEPENDS clang
    
    COMMENT "Generating dependencies from probe.cpp into probe_dependencies.d"
    VERBATIM
)

add_custom_target(filter-manifest
    COMMAND ${Python3_EXECUTABLE} ${FILTER_PY_SCRIPT}
        ${GLOB_MANIFEST}
        ${DEPS_FILE}
        ${FILTERED_MANIFEST}
    # This target depends on the other two finishing first
    # DEPENDS generate-glob-manifest generate-deps
    COMMENT "Filtering manifests into ${FILTERED_MANIFEST}"
    VERBATIM
)

set(SYCL_JIT_RESOURCE_DEPS
  sycl-headers                # include/sycl
  opencl-resource-headers     # lib/clang/N/include
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py)

if ("libclc" IN_LIST LLVM_ENABLE_PROJECTS)
  # Somehow just "libclc" doesn't build "remangled-*" (and maybe whatever else).
  list(APPEND SYCL_JIT_RESOURCE_DEPS libclc libspirv-builtins) # lib/clc/*.bc
endif()

if ("libdevice" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND SYCL_JIT_RESOURCE_DEPS libsycldevice) # lib/*.bc
endif()

set(GENERATE_PY ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py)

set(GENERATE_PY_BASE_COMMAND ${Python3_EXECUTABLE} ${GENERATE_PY}
    --toolchain-dir ${CMAKE_BINARY_DIR}
    --output ${SYCL_JIT_RESOURCE_CPP}
    --prefix ${SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT}
)
set(GENERATE_PY_DEPENDS ${SYCL_JIT_RESOURCE_DEPS})

# Conditionally add arguments and dependencies based on the mode
if(SYCL_JIT_RESOURCE_MODE STREQUAL "MANIFEST")
    message(STATUS "sycl-jit: Using MANIFEST-based resource generation.")
    #set(MANIFEST_FILE "/iusers/cperkins/sycl_workspace/glob_manifest.txt")
    list(APPEND GENERATE_PY_BASE_COMMAND --manifest-input ${GLOB_MANIFEST})
    list(APPEND GENERATE_PY_DEPENDS ${GLOB_MANIFEST})
elseif(SYCL_JIT_RESOURCE_MODE STREQUAL "GLOB_AND_MANIFEST")
    message(STATUS "sycl-jit: Using GLOB_AND_MANIFEST resource generation.")
    list(APPEND GENERATE_PY_BASE_COMMAND --manifest-output ${CMAKE_CURRENT_BINARY_DIR}/glob_manifest.txt)
else()
    message(STATUS "sycl-jit: Using GLOB-only resource generation.")
endif()

if(EXISTS "${SYCL_JIT_RESOURCE_BLACKLIST}")
    message(STATUS "sycl-jit: Using resource blacklist file: ${SYCL_JIT_RESOURCE_BLACKLIST}")
    list(APPEND GENERATE_PY_BASE_COMMAND --blacklist ${SYCL_JIT_RESOURCE_BLACKLIST})
    # Make the build depend on the blacklist file, so it reruns if the file changes
    list(APPEND GENERATE_PY_DEPENDS ${SYCL_JIT_RESOURCE_BLACKLIST})
endif()

add_custom_command(
    OUTPUT ${SYCL_JIT_RESOURCE_CPP}
    COMMAND ${GENERATE_PY_BASE_COMMAND}
    DEPENDS ${GENERATE_PY_DEPENDS}
    COMMENT "Generating resource.cpp..."
    VERBATIM
)

# We use C23/C++26's `#embed` to implement this resource creation, and "current"
# CMAKE_CXX_COMPILER might not have support for it. As such, use freshly built
# `clang++` instead.
if (WIN32)
  set(clang_exe ${CMAKE_BINARY_DIR}/bin/clang-cl.exe)
  set(SYCL_JIT_RESOURCE_CXX_FLAGS /O2 /std:c++17 /W0)

  # Determine if we should use static (/MT) or dynamic (/MD) runtime
  set(USE_STATIC_RUNTIME 0)
  if(CMAKE_MSVC_RUNTIME_LIBRARY AND NOT CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
    set(USE_STATIC_RUNTIME 1)
  endif()

  list(APPEND SYCL_JIT_RESOURCE_CXX_FLAGS /M$<IF:${USE_STATIC_RUNTIME},T,D>$<$<CONFIG:Debug>:d>)
else()
  get_host_tool_path( clang CLANG clang_exe clang_target )
  set(SYCL_JIT_RESOURCE_CXX_FLAGS -O2 -Wno-c23-extensions -std=c++17 -fPIC -fvisibility=hidden)

  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    # Hack based on the information at
    # https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_SYSROOT.html
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE SYCL_JIT_OSX_SYSROOT)
    list(APPEND SYCL_JIT_RESOURCE_CXX_FLAGS -isysroot ${SYCL_JIT_OSX_SYSROOT})
  endif()
endif()

add_custom_command(
  OUTPUT ${SYCL_JIT_RESOURCE_OBJ}
  COMMAND
  # ${clang_exe}'s default target is not necessarily ${LLVM_HOST_TRIPLE}: when
  # cross compiling, it will be whatever the host tools were configured with,
  # and when building a cross compiler, it will be
  # ${LLVM_DEFAULT_TARGET_TRIPLE}. Rather than special casing these, just always
  # specify --target=${LLVM_HOST_TRIPLE}.
  ${clang_exe} --target=${LLVM_HOST_TRIPLE} ${SYCL_JIT_RESOURCE_CPP} -I ${CMAKE_CURRENT_SOURCE_DIR}/include -c -o ${SYCL_JIT_RESOURCE_OBJ} ${SYCL_JIT_RESOURCE_CXX_FLAGS}
  DEPENDS
  ${SYCL_JIT_RESOURCE_CPP}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Resource.h
)

add_llvm_library(sycl-jit
   lib/translation/JITContext.cpp
   lib/translation/SPIRVLLVMTranslation.cpp
   lib/translation/Translation.cpp
   lib/materializer/MaterializerPipeline.cpp
   lib/materializer/Materializer.cpp
   lib/rtc/DeviceCompilation.cpp
   lib/rtc/ESIMD.cpp
   lib/rtc/RTC.cpp
   lib/helper/ConfigHelper.cpp
   lib/helper/ErrorHelper.cpp

   ${SYCL_JIT_RESOURCE_OBJ}

   SHARED

   DEPENDS
   intrinsics_gen

   LINK_COMPONENTS
   BitReader
   BitWriter
   Core
   Support
   Option
   Analysis
   IPO
   TransformUtils
   Passes
   IRReader
   Linker
   ScalarOpts
   InstCombine
   Target
   TargetParser
   MC
   SYCLLowerIR
   SYCLPostLink
   SPIRVLib
   ${LLVM_TARGETS_TO_BUILD}

   LINK_LIBS
   clangBasic
   clangDriver
   clangFrontend
   clangCodeGen
   clangTooling
   clangSerialization
)

if(WIN32)
  target_link_libraries(sycl-jit PRIVATE Shlwapi)
endif()

target_compile_options(sycl-jit PRIVATE ${SYCL_JIT_WARNING_FLAGS})

# Mark LLVM and SPIR-V headers as system headers to ignore warnigns in them.
# This classification remains intact even if the same paths are added as normal
# include paths in GCC and Clang.
target_include_directories(sycl-jit
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_SPIRV_INCLUDE_DIRS}
  ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/tools/clang/include
)
target_include_directories(sycl-jit
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

find_package(Threads REQUIRED)

target_link_libraries(sycl-jit
  PRIVATE
  SYCLJITPasses
  ${CMAKE_THREAD_LIBS_INIT}
)

add_dependencies(sycl-jit sycl-headers)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(sycl-jit PRIVATE JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(sycl-jit PRIVATE JIT_SUPPORT_AMDGCN)
endif()

if(NOT MSVC AND NOT APPLE)
  # Manage symbol visibility through the linker to make sure no LLVM symbols
  # are exported and confuse the drivers.
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
  target_link_libraries(
    sycl-jit PRIVATE "-Wl,--version-script=${linker_script}")
  set_target_properties(sycl-jit PROPERTIES LINK_DEPENDS ${linker_script})
endif()
