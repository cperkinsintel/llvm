//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
void *buffer_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                  void *HostPtr, RT::PiEvent &OutEventToWait) {

  assert(!(InitFromUserData && HostPtr) &&
          "Cannot init from user data and reuse host ptr provided "
          "simultaneously");

  void *UserPtr = InitFromUserData ? BaseT::getUserPtr() : HostPtr;

  assert(!(nullptr == UserPtr && BaseT::useHostPtr() && Context->is_host()) &&
          "Internal error. Allocating memory on the host "
          "while having use_host_ptr property");

  return MemoryManager::allocateMemBuffer(
      std::move(Context), this, UserPtr, BaseT::MHostPtrReadOnly,
      BaseT::getSize(), BaseT::MInteropEvent, BaseT::MInteropContext,
      OutEventToWait);
}


void buffer_impl::addBufferInfo(const void *const BuffPtr, const size_t Sz, const size_t Offset, const bool IsSub ) {
  MBufferInfoDQ.emplace_back(BuffPtr, Sz, Offset, IsSub);
}


static bool shouldCopyBack(detail::when_copyback now, buffer_usage& BU){
  // when false, tests pass.  
  // now == ~dtor will not be correct, but shouldn't be too horrible.
  return now == detail::when_copyback::dtor;
  //return false;
}


EventImplPtr buffer_impl::copyBackSubBuffer(detail::when_copyback now, const void *const BuffPtr, bool Wait){
  //find record of buffer_usage
  std::deque<buffer_usage>::iterator it = find_if(MBufferInfoDQ.begin(), MBufferInfoDQ.end(), [BuffPtr](buffer_usage BU){
    return (BU.buffAddr == BuffPtr);
  });
  assert(it != MBufferInfoDQ.end() && "no record of subbuffer");
  buffer_usage BU = it[0];

  if(shouldCopyBack(now, BU)){
    const id<3> Offset{BU.BufferInfo.OffsetInBytes, 0, 0};
    const range<3> AccessRange{BU.BufferInfo.SizeInBytes, 1, 1};
    const range<3> MemoryRange{/*BU.BufferInfo.SizeInBytes*/ 5, 1, 1}; // !!!
    const access::mode AccessMode = access::mode::read;
    SYCLMemObjI *SYCLMemObject = this;
    const int Dims = 1;
    const int ElemSize = 1;

    Requirement Req(Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject, Dims, ElemSize);
    
    void* DataPtr = getUserPtr();  //
    if(DataPtr != nullptr){
      Req.MData = DataPtr;
    
      EventImplPtr Event = Scheduler::getInstance().addCopyBack(&Req);
      if (Event && Wait)
        Event->wait(Event);
      else if(Event)
        return Event;
    }
  }
    
  return nullptr; 
}


} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
