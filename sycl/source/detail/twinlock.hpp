//==-------------------- twinlock.hpp --- Twin Spin lock ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>

#include <atomic>
#include <mutex>
#include <thread>


namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
/// SpinLock is a synchronization primitive, that uses atomic variable and
/// causes thread trying acquire lock wait in loop while repeatedly check if
/// the lock is available.
///
/// One important feature of this implementation is that std::atomic<bool> can
/// be zero-initialized. This allows SpinLock to have trivial constructor and
/// destructor, which makes it possible to use it in global context (unlike
/// std::mutex, that doesn't provide such guarantees).
  class TwinLock : public std::mutex {

#ifdef _WIN32
  void*  HMutex = nullptr;
  long  lockStatus = 0;
#else
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
#endif
  
public:
  void lock();
  void unlock();

  bool try_lock();

#ifdef _WIN32
  // if we have the lock, we can get its status.
  // for a successful lock acquisition it will likely be  WAIT_OBJECT_0 ( 0x000L )
  // but could also be WAIT_ABANDONED ( 0x080L ) if the owning thread was terminated.
  // both are successful lock acquisitions, but WAIT_ABANDONED means the other thread didn't
  // complete its action
  // for WAIT_TIMEOUT ( 0x102L ) the lock was not acquired. 
  // WAIT_FAILED ( 0xFFFF... ) would inicate an error, and 'acquisition' is meaningless
  long getLockStatus(){
    return lockStatus;
  }
#endif
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
