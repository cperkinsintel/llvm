//==------------ sycl_mem_obj_t.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once


#include <CL/sycl/context.hpp>



#include <stack>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
    class handler;
namespace detail {

// allows us to represent whether a user has set (or not) a boolean value, and, if so, to what.
enum SettableBool { 
    set_false = -1,
    not_set,
    set_true
};

// need to track information about a sub/buffer, 
// even after its destruction, we may need to know about it.
struct buffer_info {
    const size_t SizeInBytes;
    const size_t OffsetInBytes;
    const bool IsSubBuffer;

    buffer_info(const size_t sz, const size_t offset, const bool IsSub) :
        SizeInBytes(sz), OffsetInBytes(offset), IsSubBuffer(IsSub) {}
};

// given a sub/buffer, this tracks how it was used (were there write accessors, etc.)

struct buffer_usage {
    //the address of a sub/buffer is used to uniquely identify it, but is never dereferenced.
    const void *const buffAddr;
    // basic info about the buffer (range, offset, isSub)
    buffer_info BufferInfo;
    // did the user set the writeback?
    SettableBool MWriteBackSet;
    // the list of CommandGroupHandlers used to request any write-capable accessors
    std::stack<handler*> CghWithWriteAcc;
    // the list of CommandGroupHandlers used to request any read-capable accessors
    std::stack<handler*> CghWithReadAcc;
    // did the host get a read or write accessor?
    bool HostHasReadAcc;
    bool HostHasWriteAcc;
    //ctor
    buffer_usage(const void *const BuffPtr, const size_t Sz, const size_t Offset, const bool IsSub) : 
        buffAddr(BuffPtr) , BufferInfo(Sz, Offset, IsSub), MWriteBackSet(SettableBool::not_set),
        HostHasReadAcc(false), HostHasWriteAcc(false) {}
};

} //namespaces 
}
}