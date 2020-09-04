// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// TFAIL: *

/*
  Manual
    clang++ -fsycl -o eao.bin enqueue-arg-order.cpp
    clang++ -fsycl -g -o eao.d enqueue-arg-order.cpp
    SYCL_PI_TRACE=2 ./eao.bin

    clang++ --driver-mode=g++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -o eao.bin enqueue-arg-order.cpp
    SYCL_PI_TRACE=2 SYCL_BE=PI_CUDA ./eao.bin

    llvm-lit --param SYCL_BE=PI_CUDA -v enqueue-arg-order.cpp
*/

#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>
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

// ----------- FUNCTIONAL
template <template <int> class T>
static void printRangeId(T<3> arr) {
  std::cout << ":: " << "{" << arr[0] << ", " << arr[1] << ", " << arr[2] << "}" << std::endl;
}

void testDetailConvertToArrayOfN(){
  //ranges
  range<1> range_1D(width);
  range<2> range_2D(height, width);
  range<3> range_3D(depth, height, width);

  range<3> arr1 = sycl::detail::convertToArrayOfN<3,1>(range_1D); 
  //should be: {1,1,16}
  printRangeId(arr1);
  assert(arr1[0] == 1 && arr1[1] == 1 && arr1[2] == width && "arr1 should be {1,1,16} ");

  // range<3> arrSz = sycl::detail::convertToArrayOfN<3,1>(size_t{width}); 
  // printRangeId(arrSz);
  // assert(arrSz[0] == 1 && arrSz[1] == 1 && arrSz[2] == width && "arrSz should be {1,1,16} ");


  range<3> arr2 = sycl::detail::convertToArrayOfN<3,1>(range_2D);
  //should be: {1, 5, 16}
  printRangeId(arr2);
  assert(arr2[0] == 1 && arr2[1] == height && arr2[2] == width && "arr2 should be {1,5,16} ");

  range<3> arr3 = sycl::detail::convertToArrayOfN<3,1>(range_3D);
  //should be: {3, 5, 16}
  printRangeId(arr3);
  assert(arr3[0] == depth && arr3[1] == height && arr3[2] == width && "arr3 should be {3,5,16} ");

  range<2> smaller2 = sycl::detail::convertToArrayOfN<2,1>(range_3D);
  assert(smaller2[0] == height && smaller2[1] == width  && "smaller2 should be {5,16} ");

  range<1> smaller1 = sycl::detail::convertToArrayOfN<1,1>(range_3D);
  assert(smaller1[0] == width && "smaller1 should be {16} ");
}

// class to give access to protected function getLinearIndex
template <typename T, int Dims>
class AccTest
  : public accessor<T, Dims, access::mode::read_write, access::target::host_buffer, access::placeholder::false_t> {
  using AccessorT = accessor<T, Dims, access::mode::read_write, access::target::host_buffer, access::placeholder::false_t>;

  public:
    AccTest(AccessorT acc) : AccessorT(acc) {}
      
    size_t gLI(id<Dims> idx){
      return AccessorT::getLinearIndex(idx);
    }
};


void testGetLinearIndex(){
  constexpr int x = 4, y = 3, z = 1;
  // width=16, height=5, depth = 3. 
  // row is 16 (ie. width)
  // slice is 80 (ie width * height)
  size_t target_1D = x;
  size_t target_2D = (y * width) + x; // s.b. (3*16) + 4 => 52
  size_t target_3D = (height * width * z) + (y * width) + x; // s.b. 80 + (3*16) + 4 => 132

  std::vector<float> data_1D(width, 13);
  std::vector<float> data_2D(total, 7);
  std::vector<float> data_3D(total3D, 17);

  // test accessor protected function
  {
    buffer<float, 1> buffer_1D(data_1D.data(), range<1>(width));
    buffer<float, 2> buffer_2D(data_2D.data(), range<2>(height, width));
    buffer<float, 3> buffer_3D(data_3D.data(), range<3>(depth, height, width));
  
    auto acc_1D = buffer_1D.get_access<access::mode::read_write>();
    auto accTest_1D = AccTest<float, 1>(acc_1D);
    size_t linear_1D = accTest_1D.gLI(id<1>(x));  //s.b. 4
    std::cout << "linear_1D: " << linear_1D << "  target_1D: " << target_1D << std::endl;
    assert(linear_1D == target_1D && "linear_1D s.b. 4");

    auto acc_2D = buffer_2D.get_access<access::mode::read_write>();
    auto accTest_2D = AccTest<float, 2>(acc_2D);
    size_t linear_2D = accTest_2D.gLI(id<2>(y, x));   
    std::cout << "linear_2D: " << linear_2D << "  target_2D: " << target_2D << std::endl;
    assert(linear_2D == target_2D && "linear_2D s.b. 52");

    auto acc_3D = buffer_3D.get_access<access::mode::read_write>();
    auto accTest_3D = AccTest<float, 3>(acc_3D);
    size_t linear_3D = accTest_3D.gLI(id<3>(z, y, x));   
    std::cout << "linear_3D: " << linear_3D << "  target_3D: " << target_3D << std::endl;
    assert(linear_3D ==  target_3D &&  "linear_3D s.b. 132" );
  }

  // common.hpp variant of getLinearIndex
  size_t lin_1D = getLinearIndex(id<1>(x), range<1>(width));
  std::cout << "lin_1D: " << lin_1D << std::endl;
  assert(lin_1D == target_1D && "lin_1D s.b. 4");

  size_t lin_2D = getLinearIndex(id<2>(y, x), range<2>(height, width));
  std::cout << "lin_2D: " << lin_2D << "  target_2D: " << target_2D << std::endl;
  assert(lin_2D == target_2D && "lin_2D s.b. 52");

  size_t lin_3D = getLinearIndex(id<3>(z, y, x), range<3>(depth, height, width));
  std::cout << "lin_3D: " << lin_3D << "  target_3D: " << target_3D << std::endl;
  assert(lin_3D ==  target_3D &&  "lin_3D s.b. 132" );
}

// ----------- BUFFERS

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

// ----------- IMAGES

void testcopyD2HImage(){
   // copyD2H
  std::cout << "start copyD2H-Image" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;
  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<1> ImgSize_1D(width);
  const sycl::range<2> ImgSize_2D(height, width);
  const sycl::range<3> ImgSize_3D(depth, height, width);

  std::vector<sycl::float4> data_from_1D(ImgSize_1D.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> data_to_1D(ImgSize_1D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_2D(ImgSize_2D.size(), {7, 7, 7, 7});
  std::vector<sycl::float4> data_to_2D(ImgSize_2D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_3D(ImgSize_3D.size(), {11, 11, 11, 11});
  std::vector<sycl::float4> data_to_3D(ImgSize_3D.size(), {0, 0, 0, 0});

  {
    sycl::image<1> image_from_1D(data_from_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    sycl::image<1> image_to_1D(data_to_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyD2H_1D>(ImgSize_1D, [=](sycl::item<1> Item) {
        sycl::float4 Data = readAcc.read(int(Item[0]));
        writeAcc.write(int(Item[0]), Data);
      });
    });
  } // ~image 1D

  {
    sycl::image<2> image_from_2D(data_from_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    sycl::image<2> image_to_2D(data_to_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyD2H_2D>(ImgSize_2D, [=](sycl::item<2> Item) {
        sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
        writeAcc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
  } // ~image 2D

  {
    sycl::image<3> image_from_3D(data_from_3D.data(), ChanOrder, ChanType, ImgSize_3D);
    sycl::image<3> image_to_3D(data_to_3D.data(), ChanOrder, ChanType, ImgSize_3D);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyD2H_3D>(ImgSize_3D, [=](sycl::item<3> Item) {
        sycl::float4 Data = readAcc.read(sycl::int4{Item[0], Item[1], Item[2],0});
        writeAcc.write(sycl::int4{Item[0], Item[1], Item[2],0}, Data);
      });
    });
  } // ~image 3D
  
  std::cout << "end copyD2H-Image" << std::endl;
}

void testcopyH2DImage() {
  // copy between two queues triggers a copyH2D, followed by a copyD2H
  // Here we only care about checking copyH2D
  std::cout << "start copyH2D-image" << std::endl;

  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;
  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<1> ImgSize_1D(width);
  const sycl::range<2> ImgSize_2D(height, width);
  const sycl::range<3> ImgSize_3D(depth, height, width);

  std::vector<sycl::float4> data_from_1D(ImgSize_1D.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> data_to_1D(ImgSize_1D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_2D(ImgSize_2D.size(), {7, 7, 7, 7});
  std::vector<sycl::float4> data_to_2D(ImgSize_2D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_3D(ImgSize_3D.size(), {11, 11, 11, 11});
  std::vector<sycl::float4> data_to_3D(ImgSize_3D.size(), {0, 0, 0, 0});

  // 1D 
  {
    sycl::image<1> image_from_1D(data_from_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    sycl::image<1> image_to_1D(data_to_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    queue Q;
    queue otherQueue;
    //first op
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_1D>(ImgSize_1D, [=](sycl::item<1> Item) {
        sycl::float4 Data = readAcc.read(int(Item[0]));
        writeAcc.write(int(Item[0]), Data);
      });
    });
    //second op
    otherQueue.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_1D_2nd>(ImgSize_1D, [=](sycl::item<1> Item) {
        sycl::float4 Data = readAcc.read(int(Item[0]));
        writeAcc.write(int(Item[0]), Data);
      });
    });
  } // ~image 1D

  //2D
  {
    sycl::image<2> image_from_2D(data_from_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    sycl::image<2> image_to_2D(data_to_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    queue Q;
    queue otherQueue;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_2D>(ImgSize_2D, [=](sycl::item<2> Item) {
        sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
        writeAcc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
    otherQueue.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_2D_2nd>(ImgSize_2D, [=](sycl::item<2> Item) {
        sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
        writeAcc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
  } // ~image 2D

  //3D
  {
    sycl::image<3> image_from_3D(data_from_3D.data(), ChanOrder, ChanType, ImgSize_3D);
    sycl::image<3> image_to_3D(data_to_3D.data(), ChanOrder, ChanType, ImgSize_3D);
    queue Q;
    queue otherQueue;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_3D>(ImgSize_3D, [=](sycl::item<3> Item) {
        sycl::float4 Data = readAcc.read(sycl::int4{Item[0], Item[1], Item[2],0});
        writeAcc.write(sycl::int4{Item[0], Item[1], Item[2],0}, Data);
      });
    });
    otherQueue.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_3D_2nd>(ImgSize_3D, [=](sycl::item<3> Item) {
        sycl::float4 Data = readAcc.read(sycl::int4{Item[0], Item[1], Item[2],0});
        writeAcc.write(sycl::int4{Item[0], Item[1], Item[2],0}, Data);
      });
    });
  } // ~image 3D

  std::cout << "end copyH2D-image" << std::endl;
}

void testcopyD2DImage(){
   // copyD2D
  std::cout << "start copyD2D-Image" << std::endl;
  // COPY and FILL not working with image accessors yet.
  /*
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;
  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<1> ImgSize_1D(width);
  const sycl::range<2> ImgSize_2D(height, width);
  const sycl::range<3> ImgSize_3D(depth, height, width);

  std::vector<sycl::float4> data_from_1D(ImgSize_1D.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> data_to_1D(ImgSize_1D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_2D(ImgSize_2D.size(), {7, 7, 7, 7});
  std::vector<sycl::float4> data_to_2D(ImgSize_2D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_3D(ImgSize_3D.size(), {11, 11, 11, 11});
  std::vector<sycl::float4> data_to_3D(ImgSize_3D.size(), {0, 0, 0, 0});

  {
    sycl::image<1> image_from_1D(data_from_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    sycl::image<1> image_to_1D(data_to_1D.data(), ChanOrder, ChanType, ImgSize_1D);
    sycl::image<2> image_from_2D(data_from_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    sycl::image<2> image_to_2D(data_to_2D.data(), ChanOrder, ChanType, ImgSize_2D);
    sycl::image<3> image_from_3D(data_from_3D.data(), ChanOrder, ChanType, ImgSize_3D);
    sycl::image<3> image_to_3D(data_to_3D.data(), ChanOrder, ChanType, ImgSize_3D);

    queue Q;
    auto e1 = Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.copy(readAcc, writeAcc);
    });
    auto e2 = Q.submit([&](sycl::handler &CGH) {
      //CGH.depends_on(e1);
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.copy(readAcc, writeAcc);
    });
    auto e3 = Q.submit([&](sycl::handler &CGH) {
      //CGH.depends_on(e2);
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.copy(readAcc, writeAcc);
    });
  } // ~images
  */
  std::cout << "end copyD2D-Image" << std::endl;
}

// --------------

int main() {
    remind();

    testDetailConvertToArrayOfN();
    testGetLinearIndex();
        
    // testcopyD2HBuffer();
    // testcopyH2DBuffer();
    // testcopyD2DBuffer();
    // testFill_Buffer();
    

    // testcopyD2HImage();
    // testcopyH2DImage();
    // testcopyD2DImage();
       
}

// ----------- BUFFERS

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

// ----------- IMAGES

//CHECK: start copyD2H-Image
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK-NEXT: <unknown> : 256
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/3
//CHECK-NEXT: <unknown> : 256
//CHECK-NEXT: <unknown> : 1280
//CHECK: end copyD2H-Image

//CHECK: start copyH2D-image
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/1/1
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK-NEXT: <unknown> : 256
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK-NEXT: <unknown> : 256
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/5/3
//CHECK-NEXT: <unknown> : 256
//CHECK-NEXT: <unknown> : 1280
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/5/3
//CHECK-NEXT: <unknown> : 256
//CHECK-NEXT: <unknown> : 1280
//CHECK: end copyH2D-image

//CHECK: start copyD2D-Image
//CHECK: end copyD2D-Image

// ----------- 




