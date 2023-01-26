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
    GlobalHandler::instance().registerSchedulerUsage(/*ModifyCounter*/ false);
    std::unique_lock<std::mutex> Lock(MJobQueueMutex);
    while (true) {
      MDoSmthOrStop.wait(
          Lock, [this]() { return !MJobQueue.empty() || MStop.load(); });

      //we have the lock.  
      if (MStop.load()){
	Lock.unlock(); //<--probably not necessary
        break;
      }

      MJobsInPool--;
      std::function<void()> Job = std::move(MJobQueue.front());
      MJobQueue.pop();
      Lock.unlock(); // allow MJobQueue or MJobsInPool to be updated while the Job runs.
      
      std::cout << "start Job() id: " << std::this_thread::get_id() << std::endl;
      Job();
      std::cout << "end Job()" << std::endl;

      
      //Lock.lock(); // lock and update MJobsInPool
      //MJobsInPool--;
      
      //Lock.unlock(); // free lock again, allowing updates to MJobQueue or MJobsInPool
      MDoSmthOrStop.notify_one();
    }
  }

  void start() {
    MLaunchedThreads.reserve(MThreadCount);

    MStop.store(false);
    MJobsInPool.store(0);

    for (size_t Idx = 0; Idx < MThreadCount; ++Idx)
      MLaunchedThreads.emplace_back([this] { worker(); });
  }

public:
  void drain() {
    std::cout << "ThreadPool::drain() MJobsInPool: " << MJobsInPool << " MThreadCount: (" << MThreadCount << ")" 
              << "id: " << std::this_thread::get_id() << std::endl;

    while (MJobsInPool > 0)
      std::this_thread::yield();
    //finishAndWait();

    std::cout << "~drain() completed" << std::endl;
  }

  ThreadPool(unsigned int ThreadCount = 1) : MThreadCount(ThreadCount) {
    start();
  }

  ~ThreadPool() { finishAndWait(); }

  void finishAndWait() {
    std::cout << "finishAndWait() id: " << std::this_thread::get_id() << std::endl;
    MStop.store(true); // <-- prevents Job() from running. ? 
    MDoSmthOrStop.notify_all();

    std::cout << "finishAndWait, MLaunchedThreads.size(): " << MLaunchedThreads.size() << std::endl;

    for (std::thread &Thread : MLaunchedThreads){
      std::cout << "Thread: " << Thread.get_id() << std::endl;
      if (Thread.joinable())
        Thread.join();
    }
  }

  template <typename T> void submit(T &&Func) {
    {
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MJobQueue.emplace([F = std::move(Func)]() { F(); });
      MJobsInPool++;
    }
    //MJobsInPool++;
    MDoSmthOrStop.notify_one();
  }

  void submit(std::function<void()> &&Func) {
    {
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
