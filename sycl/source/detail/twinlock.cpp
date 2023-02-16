//==-------------------- twinlock.cpp --- Twin Spin lock ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <detail/twinlock.hpp>
//#include <sycl/detail/defines.hpp>

//#include <atomic>
//#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif


namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

#ifdef _WIN32

  void TwinLock::lock(){
    if(!HMutex)
      HMutex = CreateMutex(NULL, false, NULL);

    lockStatus = WaitForSingleObject(HMutex, INFINITY);
  }

  void TwinLock::unlock(){
    if(!HMutex){
      HMutex = CreateMutex(NULL, false, NULL);
      return;
    }

    ReleaseMutex(HMutex);
  }

  bool TwinLock::try_lock(){
    if(!HMutex)
      HMutex = CreateMutex(NULL, false, NULL);

    lockStatus = WaitForSingleObject(HMutex, 0);
    return(lockStatus != WAIT_TIMEOUT);
  }

#else
  
  void TwinLock::lock() {
    while (MLock.test_and_set(std::memory_order_acquire))
      std::this_thread::yield();
  }
  void TwinLock::unlock() {
    MLock.clear(std::memory_order_release);
  }

  bool TwinLock::try_lock(){
    return !MLock.test_and_set(std::memory_order_acquire);
  }
#endif

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
