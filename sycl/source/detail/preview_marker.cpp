//==----- preview_marker.cpp --- Preview library marker symbol -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/export.hpp>

#ifdef __SYCL_PREVIEW_MAJOR_RELEASE__
namespace sycl {
inline namespace _V1 {
namespace detail {

// Exported marker function to help verify that the preview library correctly
// defines the __SYCL_PREVIEW_MAJOR_RELEASE__ macro and is linked with when the
// -fpreview-major-release option is used.
__SYCL_EXPORT void PreviewMajorReleaseMarker() {}

} // namespace detail
} // namespace _V1
} // namespace sycl
#endif // __SYCL_PREVIEW_MAJOR_RELEASE__
