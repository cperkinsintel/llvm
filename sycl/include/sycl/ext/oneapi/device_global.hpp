//==--------------- device_global.hpp -
//SYCL_ONEAPI_EXT_DEVICE_GLOBAL--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

/*
template<typename T>
class device_global {
T *usmptr;
public:
T& get() noexcept { return *usmptr; }
// other member functions
};
*/

// PROPERTIES (pending SYCL_EXT_ONEAPI_PROPERTY_LIST )

// DEVICE GLOBAL CLASS

template <typename T, typename PropertyListT = sycl::property_list>
// enable_if  trivial_destructor && trivial default constructor ?
class device_global {
  using element_type = std::remove_extent_t<T>;

  static_assert(std::is_trivially_default_constructible_v<T>,
                "Type T must be trivially default constructable (until C++20 "
                "consteval is supported and enabled)");

  static_assert(std::is_trivially_destructible_v<T>,
                "Type T must be trivially destructible.");

  T val; // must be first in struct
public:
  // Only default construction is allowed.  The underlying memory allocations
  // of type T on devices will be zero-initialized before any non-initialization
  // accesses occur.
  // Also ensures zero initialization
  device_global() : val() {}

  // delete other constructors and assignment operators
  device_global(const device_global &) = delete;
  device_global(const device_global &&) = delete;
  device_global &operator=(const device_global &) = delete;
  device_global &operator=(const device_global &&) = delete;

  // access::decorated doesn't exist yet, and multi_ptr does not expose
  // decoration at this time. template <access::decorated IsDecorated>
  // multi_ptr<T, access::address_space::global_space, IsDecorated>
  // get_multi_ptr() noexcept; template <access::decorated IsDecorated>
  // multi_ptr<const T, access::address_space::global_space, IsDecorated>
  // get_multi_ptr() const noexcept;

  multi_ptr<T, access::address_space::global_space> get_multi_ptr() noexcept {
    return sycl::multi_ptr(val);
  }
  multi_ptr<const T, access::address_space::global_space>
  get_multi_ptr() const noexcept {
    return sycl::multi_ptr(val);
  }

  // Access the underlying data
  operator T &() noexcept { return val; }
  operator const T &() const noexcept { return val; }

  T &get() noexcept { return val; }
  const T &get() const noexcept { return val; }

  // Enable assignments from underlying type
  device_global &operator=(const T &data) noexcept {
    val = data;
    return *this;
  };

  // Available if the operator[] is valid for objects of type T
  // using subscript_return_t =
  // std::remove_reference_t<decltype(std::declval<T>()[std::ptrdiff_t{}])>;
  // subscript_return_t& operator[]( std::ptrdiff_t idx ) noexcept;
  // const subscript_return_t& operator[]( std::ptrdiff_t idx ) const noexcept;

  // template<typename U = T>
  // typename std::enable_if_t<std::is_integral<T>::value, U>
  // operator[](size_t idx) noexcept { return 3; }

  // template<class U = T, typename
  // std::enable_if_t<std::is_integral_v<std::decay_t<decltype(U{}[0])>>, int> =
  // false>
  // std::remove_reference_t<decltype(std::declval<T>()[std::ptrdiff_t{}])>
  // operator[](size_t idx) noexcept { return 3; }

  // operator[]
  // Available if the operator[] is valid for objects of type T
  template <typename U = T>
  typename std::enable_if_t<
      std::is_integral_v<std::decay_t<decltype(U{}[0])>>,
      std::remove_reference_t<decltype(std::declval<U>()[std::ptrdiff_t{}])> &>
  operator[](std::ptrdiff_t idx) noexcept {
    return val[idx];
  }

  template <typename U = T>
  typename std::enable_if_t<std::is_integral_v<std::decay_t<decltype(U{}[0])>>,
                            const std::remove_reference_t<decltype(
                                std::declval<U>()[std::ptrdiff_t{}])> &>
  operator[](std::ptrdiff_t idx) const noexcept {
    return val[idx];
  }

  // operator->
  // Available if the operator-> is valid for objects of type T
  // the device_global class assertions already enforce trivially default
  // constructible. CP what is an example of this? must be trivially default
  // constructible CP typename
  // std::enable_if_t<std::is_integral_v<std::decay_t<decltype(*U{})>>, U&>
  template <typename U = T>
  typename std::enable_if_t<std::is_pointer<U>::value, U &>
  operator->() noexcept {
    return val;
  }

  template <typename U = T>
  typename std::enable_if_t<std::is_pointer<U>::value, const U &>
  operator->() const noexcept {
    return val;
  }

  // Note that there is no need for "device_global" to define member functions
  // for operators like "++", comparison, etc.  Instead, the type "T" need only
  // define these operators as non-member functions.  Because there is an
  // implicit conversion from "device_global" to "T&", the operations can be
  // applied to objects of type "device_global<T>".

  // PROPERTIES
  // (pending SYCL_EXT_ONEAPI_PROPERTY_LIST compile time properties)
};

} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)