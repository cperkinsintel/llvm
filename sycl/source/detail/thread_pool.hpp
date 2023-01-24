//===-- thread_pool.hpp - Simple thread pool --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <sycl/detail/defines.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

class ThreadPool {
  std::vector<std::thread> MLaunchedThreads;

  size_t MThreadCount;
  std::queue<std::function<void()>> MJobQueue;
  std::mutex MJobQueueMutex;
  std::condition_variable MDoSmthOrStop;
  std::atomic_bool MStop;
  std::atomic_uint MJobsInPool;

  void worker() {
    std::cout << "ThreadPool::worker() MJobsInPool/MThreadCount: " << MJobsInPool << "/" << MThreadCount << std::endl;
    GlobalHandler::instance().registerSchedulerUsage(/*ModifyCounter*/ false);
    std::unique_lock<std::mutex> Lock(MJobQueueMutex);
    while (true) {
      MDoSmthOrStop.wait(
          Lock, [this]() { return !MJobQueue.empty() || MStop.load(); });

      if (MStop.load())
        break;

      std::function<void()> Job = std::move(MJobQueue.front());
      MJobQueue.pop();
      Lock.unlock();

      std::cout << "start Job()" << std::endl;
      Job();
      std::cout << "end Job()" << std::endl;

      Lock.lock();

      MJobsInPool--;
    }
  }

  void start() {
    std::cout << "ThreadPool::start() MJobsInPool/MThreadCount: " << MJobsInPool << "/" << MThreadCount << std::endl;
    MLaunchedThreads.reserve(MThreadCount);

    MStop.store(false);
    MJobsInPool.store(0);

    for (size_t Idx = 0; Idx < MThreadCount; ++Idx)
      MLaunchedThreads.emplace_back([this] { worker(); });
  }

public:
  void drain() {
    std::cout << "ThreadPool::drain() MJobsInPool/MThreadCount: " << MJobsInPool << "/" << MThreadCount << std::endl;
    while (MJobsInPool != 0)
      std::this_thread::yield();
  }

  ThreadPool(unsigned int ThreadCount = 1) : MThreadCount(ThreadCount) {
    start();
  }

  ~ThreadPool() { finishAndWait(); }

  void finishAndWait() {
    std::cout << "ThreadPool::finishAndWait() MJobsInPool/MThreadCount: " << MJobsInPool << "/" << MThreadCount << std::endl;
    MStop.store(true);

    MDoSmthOrStop.notify_all();

    for (std::thread &Thread : MLaunchedThreads)
      if (Thread.joinable())
        Thread.join();
  }

  template <typename T> void submitWork(T &&Func) {
    {
      std::cout << "ThreadPool::submit() MJobsInPool/MThreadCount: " << MJobsInPool << "/" << MThreadCount << std::endl;
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MJobQueue.emplace([F = std::move(Func)]() { F(); });
      MJobsInPool++;
    }
   // MJobsInPool++;
    MDoSmthOrStop.notify_one();
  }

  void submitWork(std::function<void()> &&Func) {
    {
      std::cout << "ThreadPool::submit std::function() MJobsInPool/MThreadCount: " << MJobsInPool << "/" << MThreadCount << std::endl;
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MJobQueue.emplace(Func);
      MJobsInPool++;
    }
    //MJobsInPool++;
    MDoSmthOrStop.notify_one();
  }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
