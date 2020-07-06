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

static bool isAReadMode(access::mode Mode){
  if(Mode == access::mode::write || Mode == access::mode::discard_write)
    return false;
  else
    return true;
}

static bool isAWriteMode(access::mode Mode){
  if(Mode == access::mode::read || Mode == access::mode::discard_write || Mode == access::mode::discard_read_write)
    return false;
  else
    return true;
}

static bool needDtorCopyBack(buffer_usage& BU){
  using hentry = std::pair<bool, access::mode>;

  bool updateOnDtor = false;
  
  find_if(BU.MHistory.begin(), BU.MHistory.end(), [&updateOnDtor](hentry HEntry){
    // returns at first consequential entry. Set updateOnDtor by side effect

    //writing on device - set bool, end search.
    if(!HEntry.first  &&  isAWriteMode(HEntry.second)){
      updateOnDtor = true; // 
      return true;
    }
    //blocking host read (was updated via map op), do not set bool, end search
    if(HEntry.first && isAReadMode(HEntry.second))
      return true;

    //continue
    return false;
  });

  return updateOnDtor; 
}
/*
// when is the appropriate time for some sub/buffer to update the host memory?
// return value of 'immediate' means when scheduling the command group (addCCG), usually via a Map operation
// 'dtor' means during buffer destructor, by calling addCopyBack.
// and 'never' is a possibility as well.
static detail::when_copyback whenCopyBack(buffer_usage& BU){
  using hentry = std::pair<sycl::device, access::mode>;

  hentry HEntry = BU.MHistory.front();

  //if the last operation was a blocking host read, then this copied back immediately.
  if(HEntry.first.is_host() && isAReadMode(HEntry.second))
    return when_copyback::immediate;

  //if we find something that wrote to a device, 
  std::deque<hentry>::iterator It = find_if(BU.MHistory.begin(), BU.MHistory.end(), [](hentry HEntry){
      return ((!HEntry.first.is_host())  &&  isAWriteMode(HEntry.second));
  });
  if(It != BU.MHistory.end()){ return when_copyback::dtor; }
  else{ return when_copyback::never; }
}

// given a now of immediate or dtor, returns a boolean.
// give a now of 'undetermined' returns true if this copies data back at either stage
// takes set_write_back into consideration.
static bool shouldCopyBack(detail::when_copyback now, buffer_usage& BU){
  assert(now != when_copyback::never);
  
  detail::when_copyback when = whenCopyBack(BU);

  //check immediate case first. It is unaffected by set_write_back (addCG does not check WB, so neither do we)
  if(now == when_copyback::immediate && when == when_copyback::immediate)
    return true;

  if(BU.MWriteBackSet == settable_bool::set_false)
    return false;

  if(now == when_copyback::undetermined)
    return when != when_copyback::never; 
  else
    return (now == when);
}
*/

bool buffer_impl::hasSubBuffers(){
  return MBufferInfoDQ.size() > 1;
}

void buffer_impl::recordAccessorUsage(const void *const BuffPtr, access::mode Mode,  handler &CGH){
  std::deque<buffer_usage>::iterator it = find_if(MBufferInfoDQ.begin(), MBufferInfoDQ.end(), [BuffPtr](buffer_usage& BU){
    return (BU.buffAddr == BuffPtr);
  });
  assert(it != MBufferInfoDQ.end() && "no record of (sub)buffer");
  buffer_usage &BU = it[0];

  BU.MHistory.emplace_front(false, Mode);
}

void buffer_impl::recordAccessorUsage(const void *const BuffPtr, access::mode Mode){
  std::deque<buffer_usage>::iterator it = find_if(MBufferInfoDQ.begin(), MBufferInfoDQ.end(), [BuffPtr](buffer_usage& BU){
    return (BU.buffAddr == BuffPtr);
  });
  assert(it != MBufferInfoDQ.end() && "no record of (sub)buffer");
  buffer_usage &BU = it[0];

  BU.MHistory.emplace_front(true, Mode );
}

EventImplPtr buffer_impl::copyBackSubBuffer(detail::when_copyback now, const void *const BuffPtr, bool Wait){
  //find record of buffer_usage
  std::deque<buffer_usage>::iterator it = find_if(MBufferInfoDQ.begin(), MBufferInfoDQ.end(), [BuffPtr](buffer_usage& BU){
    return (BU.buffAddr == BuffPtr);
  });
  assert(it != MBufferInfoDQ.end() && "no record of subbuffer");
  buffer_usage &BU = it[0];

  if(needDtorCopyBack(BU)){   //(shouldCopyBack(now, BU)){
    const id<3> Offset{BU.BufferInfo.OffsetInBytes, 0, 0};
    const range<3> AccessRange{BU.BufferInfo.SizeInBytes, 1, 1};
    const range<3> MemoryRange{BU.BufferInfo.SizeInBytes, 1, 1}; // seems to not be used.
    const access::mode AccessMode = access::mode::read;
    SYCLMemObjI *SYCLMemObject = this;
    const int Dims = 1;
    const int ElemSize = 1;

    Requirement Req(Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject, Dims, ElemSize, BU.BufferInfo.OffsetInBytes, true);
    
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
