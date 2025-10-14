// probe.cpp
// This file contains a collection of minimal SYCL kernels that
// reference various SYCL features, to be used to test that the
// appropriate headers are included when compiling kernels that
// use those features.
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <sycl/ext/intel/math.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// Basic functionality
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void probe_basic(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}

// device math library
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void probe_math(float *ptr) { ptr[0] = sycl::sin(ptr[0]); }

// group algorithms
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void probe_group_algo(int *ptr) {
  auto group = syclext::this_work_item::get_work_group<1>();
  // sycl::leader is a simple group function that will pull in the right headers
  if (group.leader()) {
    ptr[0] = 1;
  }
}

// atomics
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void probe_atomic(int *ptr) {
  auto ref = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                              sycl::memory_scope::device>(ptr[0]);
  ref.fetch_add(1);
}

// pull in ESIMD headers
[[sycl::device]] void probe_esimd_func() {
  using namespace sycl::ext::intel::esimd;
  simd<float, 16> data = 1.0f;
}

// ext_intel_devicelib_imf
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    sycl::ext::oneapi::experimental::single_task_kernel)
void imf_kernel(float *ptr) {
  // cl_intel_devicelib_imf
  ptr[0] = sycl::ext::intel::math::sqrt(ptr[0] * 2);

  // cl_intel_devicelib_imf_bf16
  ptr[1] = sycl::ext::intel::math::float2bfloat16(ptr[1] * 0.5f);
}
