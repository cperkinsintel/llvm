//==--- sycl_selfcontained.cpp --- kernel_compiler extension tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out | FileCheck %s --check-prefixes=CHECK,CHECK-SYSTEM
// RUN: %{l0_leak_check} %{run} %t.out %S/../../.. | FileCheck %s --check-prefixes=CHECK,CHECK-SELFCNTD --implicit-check-not /usr/include --implicit-check-not ucrt

// CHECK-SYSTEM: Using system headers
// CHECK-SELFCNTD: Running self-contained
// CHECK: COMPUTATION OK

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <string>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

static constexpr auto SYCLSource = R"""(
#include <sycl/sycl.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}
)""";

syclexp::build_options get_selfcontained_options(
    const std::string &DPCPPRoot, const std::string &ClangMajor,
    const std::string &LibCXXRoot, const std::string &StubLibCRoot) {
  std::vector<std::string> opts = {"-nostdinc",
                                   "-isystem",
                                   LibCXXRoot + "/include",
                                   "-isystem",
                                   DPCPPRoot + "/lib/clang/" + ClangMajor +
                                       "/include",
                                   "-isystem",
                                   StubLibCRoot + "/include/no_triple",
                                   "-isystem",
                                   StubLibCRoot + "/include",
                                   "-include",
                                   StubLibCRoot + "/libc.h",
                                   "-H"};
  return syclexp::build_options{opts};
}

int main(int argc, char **argv) {
  syclexp::build_options options;
  if (argc == 2) {
    std::cout << "Running self-contained\n";
    std::string RepoRoot = argv[1];
    options = get_selfcontained_options(RepoRoot + "/build", "21",
                                        RepoRoot + "/libcxx",
                                        RepoRoot + "/sycl-jit/stub-libc");
  } else {
    std::cout << "Using system headers\n";
  }

  try {
    sycl::queue q;

    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclexp::create_kernel_bundle_from_source(
            q.get_context(), syclexp::source_language::sycl, SYCLSource);

    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclexp::build(kb_src, syclexp::properties{options});

    sycl::kernel iota = kb_exe.ext_oneapi_get_kernel("iota");

    float *ptr = sycl::malloc_shared<float>(NUM, q);
    q.submit([&](sycl::handler &cgh) {
       cgh.set_args(3.14f, ptr);

       sycl::nd_range ndr{{NUM}, {WGSIZE}};
       cgh.parallel_for(ndr, iota);
     }).wait();

    constexpr float eps = 0.001;
    for (int i = 0; i < NUM; i++) {
      const float truth = 3.14f + static_cast<float>(i);
      if (std::abs(ptr[i] - truth) > eps) {
        std::cout << "Result: " << ptr[i] << " expected " << i << "\n";
        sycl::free(ptr, q);
        return 1;
      }
    }

    std::cout << "COMPUTATION OK\n";
    sycl::free(ptr, q);
  } catch (sycl::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
  return 0;
}
