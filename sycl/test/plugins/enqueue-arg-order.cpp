// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// XFAIL: *

/*
  Manual
    clang++ -fsycl -o eao.bin enqueue-arg-order.cpp
    SYCL_PI_TRACE=2 ./eao.bin

    clang++ --driver-mode=g++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -o eao.bin enqueue-arg-order.cpp
    SYCL_PI_TRACE=2 SYCL_BE=PI_CUDA ./eao.bin

    llvm-lit --param SYCL_BE=PI_CUDA -v enqueue-arg-order.cpp
*/

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

constexpr long width = 16;
constexpr long height = 5;
constexpr long total = width * height;

constexpr long depth = 3;
constexpr long total3D = total * depth;

void remind() {
  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBufferRect.html

    buffer_origin defines the (x, y, z) offset in the memory region associated
    with buffer. For a 2D rectangle region, the z value given by
    buffer_origin[2] should be 0. The offset in bytes is computed as
    buffer_origin[2] × buffer_slice_pitch + buffer_origin[1] × buffer_row_pitch
    + buffer_origin[0].

    region defines the (width in bytes, height in rows, depth in slices) of the
    2D or 3D rectangle being read or written. For a 2D rectangle copy, the depth
    value given by region[2] should be 1. The values in region cannot be 0.


    buffer_row_pitch is the length of each row in bytes to be used for the
    memory region associated with buffer. If buffer_row_pitch is 0,
    buffer_row_pitch is computed as region[0].

    buffer_slice_pitch is the length of each 2D slice in bytes to be used for
    the memory region associated with buffer. If buffer_slice_pitch is 0,
    buffer_slice_pitch is computed as region[1] × buffer_row_pitch.
  */
  std::cout << "For BUFFERS" << std::endl;
  std::cout << "         Region SHOULD be : " << width * sizeof(float) << "/"
            << height << "/" << depth << std::endl; // 64/5/3
  std::cout << "  RowPitch SHOULD be 0 or : " << width * sizeof(float)
            << std::endl; // 0 or 64
  std::cout << "SlicePitch SHOULD be 0 or : " << width * sizeof(float) * height
            << std::endl
            << std::endl; // 0 or 320

  // NOTE: presently we see 20/16/1 for Region and 20 for row pitch.  both
  // incorrect.

  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadImage.html

    row_pitch in clEnqueueReadImage and input_row_pitch in clEnqueueWriteImage
    is the length of each row in bytes. This value must be greater than or equal
    to the element size in bytes × width. If row_pitch (or input_row_pitch) is
    set to 0, the appropriate row pitch is calculated based on the size of each
    element in bytes multiplied by width.

    slice_pitch in clEnqueueReadImage and input_slice_pitch in
    clEnqueueWriteImage is the size in bytes of the 2D slice of the 3D region of
    a 3D image or each image of a 1D or 2D image array being read or written
    respectively.
  */

  std::cout << "For IMAGES" << std::endl;
  std::cout << "           Region SHOULD be : " << width << "/" << height << "/"
            << depth << std::endl; // 16/5/3
  std::cout << "   row_pitch SHOULD be 0 or : " << width * sizeof(sycl::float4)
            << std::endl; // 0 or 256
  std::cout << " slice_pitch SHOULD be 0 or : "
            << width * sizeof(sycl::float4) * height << std::endl
            << std::endl; // 0 or 1280

  // NOTE: presently we see 5/16/1 for image Region and 80 for row pitch.  both
  // incorrect
}




void testcopyD2HBuffer() {
  std::cout << "start copyD2H-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);

  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyD2H_1D>(buffer_from_1D.get_range(), [=](id<1> index) {
        write[index] = read[index] * -1;
      });
    });
  } // ~buffer 1D

  {
    buffer<float, 2> buffer_from_2D(data_from_2D.data(), range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyD2H_2D>(buffer_from_2D.get_range(), [=](id<2> index) {
        write[index] = read[index] * -1;
      });
    });
  } // ~buffer 2D

  {
    buffer<float, 3> buffer_from_3D(data_from_3D.data(), range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(), range<3>(depth, height, width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyD2H_3D>(buffer_from_3D.get_range(), [=](id<3> index) {
        write[index] = read[index] * -1;
      });
    });
  } // ~buffer 3D
  
  std::cout << "end copyD2H-buffer" << std::endl;
}

void testcopyH2DBuffer() {
  // copy between two queues triggers a piEnqueueMemBufferMap followed by
  // copyH2D, followed by a copyD2H, followed by a piEnqueueMemUnmap 
  // Here we only care about checking copyH2D

  std::cout << "start copyH2D-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);

  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_1D>(buffer_from_1D.get_range(), [=](id<1> index) {
        write[index] = read[index] * -1;
      });
    });
    myQueue.wait();

    otherQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_1D_2nd>(buffer_from_1D.get_range(), [=](id<1> index) {
          write[index] = read[index] * 10; 
      });
    });
  } // ~buffer 1D

  {
    buffer<float, 2> buffer_from_2D(data_from_2D.data(), range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_2D>(buffer_from_2D.get_range(), [=](id<2> index) {
        write[index] = read[index] * -1;
      });
    });

    otherQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_2D_2nd>(buffer_from_2D.get_range(), [=](id<2> index) {
          write[index] = read[index] * 10; 
      });
    });
  } // ~buffer 22

  {
    buffer<float, 3> buffer_from_3D(data_from_3D.data(), range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(), range<3>(depth, height, width));
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_3D>(buffer_from_3D.get_range(), [=](id<3> index) {
        write[index] = read[index] * -1;
      });
    });

    otherQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_3D_2nd>(buffer_from_3D.get_range(), [=](id<3> index) {
          write[index] = read[index] * 10; 
      });
    });
  } // ~buffer 3D

  std::cout << "end copyH2D-buffer" << std::endl;
}

void testcopyD2DBuffer() {
  std::cout << "start copyD2D-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);
  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));
    buffer<float, 2> buffer_from_2D(data_from_2D.data(), range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));
    buffer<float, 3> buffer_from_3D(data_from_3D.data(), range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(), range<3>(depth, height, width));

    queue myQueue;
    auto e1 = myQueue.submit([&](handler &cgh) {
      auto read  = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.copy(read, write);
    });
    auto e2 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e1);
      auto read  = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.copy(read, write);
    });
    auto e3 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e2);
      auto read  = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.copy(read, write);
    });
    

  } // ~buffer
  std::cout << "end copyD2D-buffer" << std::endl;
}

void testFill_Buffer(){
  std::cout << "start testFill Buffer" << std::endl;
  std::vector<float> data_1D(width, 0);
  std::vector<float> data_2D(total, 0);
  std::vector<float> data_3D(total3D, 0);
  {
    buffer<float, 1> buffer_1D(data_1D.data(), range<1>(width));
    buffer<float, 2> buffer_2D(data_2D.data(), range<2>(height, width));
    buffer<float, 3> buffer_3D(data_3D.data(), range<3>(depth, height, width));

    queue myQueue;
    auto e1 = myQueue.submit([&](handler &cgh) {
      auto acc1D = buffer_1D.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.fill(acc1D, float{1});
    });
    auto e2 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e1);  
      auto acc2D = buffer_2D.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.fill(acc2D, float{2});
    });
    auto e3 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e2); 
      auto acc3D = buffer_3D.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.fill(acc3D, float{3});
    });
  }// ~buffer
  std::cout << "end testFill Buffer" << std::endl;
}


void testCopy_1D_D2HImage() {
  // copyD2H
  std::cout << "start 1D copyD2H-Image" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<1> Img1Size(width);
  const sycl::range<1> Img2Size(width);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<1> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<1> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopy_1D>(Img1Size, [=](sycl::item<1> Item) {
        sycl::float4 Data = Img1Acc.read(int(Item[0])); // , Item[1], Item[2]});
        Img2Acc.write(int(Item[0]), Data);
      });
    });
  } // ~image
  std::cout << "end 1D copyD2H-Image" << std::endl;
}



void testCopy_2D_D2HImage() {
  // copyD2H
  std::cout << "start copyD2H-Image" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<2> Img1Size(height, width);
  const sycl::range<2> Img2Size(height, width);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<2> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<2> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopy>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
  } // ~image
  std::cout << "end copyD2H-Image" << std::endl;
}


void testCopy_3D_D2HImage() {
  // copyD2H
  std::cout << "start 3D copyD2H-Image" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<3> Img1Size(depth, height, width);
  const sycl::range<3> Img2Size(depth, height, width);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<3> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<3> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopy_3D>(Img1Size, [=](sycl::item<3> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int4{Item[0], Item[1], Item[2],0});
        Img2Acc.write(sycl::int4{Item[0], Item[1], Item[2], 0}, Data);
      });
    });
  } // ~image
  std::cout << "end 3D copyD2H-Image" << std::endl;
}


void testCopyTwiceImage() {
  // copyD2H and copyH2D
  std::cout << "start copyTwiceImage" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<2> Img1Size(height, width);
  const sycl::range<2> Img2Size(height, width);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<2> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<2> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    queue Q;
    queue otherQueue;

    // first op
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyTwice0>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });

    // second op
    otherQueue.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img2.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img1.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyTwice1>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
  } // ~image
  std::cout << "end copyTwiceImage" << std::endl;
}

int main() {
    remind();
        
    //testcopyD2HBuffer();
    testcopyH2DBuffer();
    //testcopyD2DBuffer();
    // testFill_Buffer();
    

    // testCopy_1D_D2HImage();
    // testCopy_2D_D2HImage();
    // testCopyTwiceImage();
    // testCopy_3D_D2HImage();
    
}


//CHECK: start copyD2H-buffer
//CHECK: ---> piEnqueueMemBufferRead(
//CHECK: <unknown> : 64
//CHECK: ---> piEnqueueMemBufferReadRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK-NEXT: <unknown> : 64
//CHECK: ---> piEnqueueMemBufferReadRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK: end copyD2H-buffer

//CHECK: start copyH2D-buffer
//CHECK: ---> piEnqueueMemBufferWrite(
//CHECK: <unknown> : 64
//CHECK:  ---> piEnqueueMemBufferWriteRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 0
//CHECK-NEXT: <unknown> : 64
//CHECK:  ---> piEnqueueMemBufferWriteRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK: end copyH2D-buffer



//CHECK: start copyD2D-buffer
//CHECK: ---> piEnqueueMemBufferCopy(
//CHECK: <unknown> : 64
//CHECK: ---> piEnqueueMemBufferCopyRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK-NEXT: <unknown> : 64
//CHECK-NEXT: <unknown> : 320
//CHECK: end copyD2D-buffer

//CHECK: start testFill Buffer
//CHECK: ---> piEnqueueMemBufferFill(
//CHECK: <unknown> : 4
//CHECK-NEXT: <unknown> : 0
//CHECK-NEXT: <unknown> : 64
//CHECK: end testFill Buffer


//CHECK: start 1D copyD2H-Image
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: end 1D copyD2H-Image

//CHECK: start copyD2H-Image
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: <unknown> : 256
//CHECK: end copyD2H-Image

//CHECK: start copyTwiceImage
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: <unknown> : 256
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: <unknown> : 256
//CHECK: end copyTwiceImage

//CHECK: start 3D copyD2H-Image
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/3
//CHECK: <unknown> : 256
//CHECK: <unknown> : 1280
//CHECK: end 3D copyD2H-Image