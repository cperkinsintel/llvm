// RUN: not %clangxx -fsycl %s -o %t 2>&1 | FileCheck --check-prefix=CHECK-NO-PREVIEW %s
// RUN: %clangxx -fsycl -fpreview-major-release %s -o %t
// REQUIRES: preview-major-release-lib && linux

// Checks that the preview-major-release marker is present only when the
// -fpreview-major-release option is used. This implies two things:
//  1. The driver links against the right library, i.e. sycl-preview.
//  2. The sycl-preview library has the __SYCL_PREVIEW_MAJOR_RELEASE__ macro
//     defined.

#include <sycl/sycl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

extern void PreviewMajorReleaseMarker();

} // namespace detail
} // namespace _V1
} // namespace sycl

int main() {
  sycl::detail::PreviewMajorReleaseMarker();
  return 0;
}

// CHECK-NO-PREVIEW: undefined reference to `sycl::_V1::detail::PreviewMajorReleaseMarker()'
