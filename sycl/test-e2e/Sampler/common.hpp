#include <iostream>
#include <limits>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

template <typename vecType, int numOfElems>
std::string vec2string(const vec<vecType, numOfElems> &vec) {
  std::string str = "";
  for (size_t i = 0; i < numOfElems - 1; ++i) {
    str += std::to_string(vec[i]) + ",";
  }
  str = "{" + str + std::to_string(vec[numOfElems - 1]) + "}";
  return str;
}

template <typename vecType, int numOfElems>
void check_pixel(const vec<vecType, numOfElems> &result,
                 const vec<vecType, numOfElems> &ref, int index) {
  // 0 ULP difference is allowed for integer type
  int precision = 0;
  // 1 ULP difference is allowed for float type
  if (std::is_floating_point<vecType>::value) {
    precision = 1;
  }

  int4 resultInt = result.template as<int4>();
  int4 refInt = ref.template as<int4>();
  int4 diff = resultInt - refInt;

  // 4.17.9 Unless otherwise specified, these functions 
  // return a vector component of -1 (i.e. all bits set) when
  // the comparison is true and 0 when the comparison is false.
  auto isLess = (diff <= precision);     // {-1, -1, -1, -1}
  auto isMore = (diff >= (-precision));  // {-1, -1, -1, -1}
  auto andBoth = (isLess && isMore);     // {-1, -1, -1, -1}
  std::cout << "isLess: " << vec2string(isLess) 
            << " isMore: " << vec2string(isMore) 
            << " andBoth: " << vec2string(andBoth)
            << std::endl;
  std::cout << "underlying type of isMore/andBoth: " << typeid(isMore).name()
            << " / " << typeid(andBoth).name() << std::endl;

  std::cout << "is_vigeninteger: " << detail::is_vigeninteger<decltype(andBoth)>() << std::endl; // 1

  auto wtf_01 = detail::rel_sign_bit_test_arg_t<decltype(andBoth)>(andBoth);
  // weirdly, typeid is same, but must cast to get it to be accepted by vec2string
  std::cout << "rel_sign_bit_test_arg_t: " << typeid(wtf_01).name() << "  " << vec2string((vec<int, 4>)wtf_01)  << std::endl;  //{-1, -1, -1, -1}

  //auto wtf_02 = detail::rel_sign_bit_test_ret_t<decltype(andBoth)>(wtf_01);
  //std::cout << "rel_sign_bit_test_ret_t: " << typeid(wtf_02).name() << "  " << vec2string((vec<int, 4>)wtf_02)  << std::endl;

  // this is getting wrong value. builtins.hpp:50 calls  convertDataToType<>() generic_type_traits.hpp:620   with 0 as the arg. 
  auto invocationVal = __sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<decltype(andBoth)>>(wtf_01);

  // << invocationVal.value    =>   error: member reference base type 'int' is not a structure or union
  // << invocationVal          =>   error: use of overloaded operator '<<' is ambiguous (with operand types 'basic_ostream<char, char_traits<char>>' and 'sycl::detail::Boolean<1>')
  std::cout << "invocationVal: " << typeid(invocationVal).name() << std::endl; // << (detail::Boolean<1>)invocationVal  << std::endl;  // i    it's value is:  0


  //auto res_1 = __sycl_std::sycl_host_All(wtf_01);  // opencl::cl_int

  //from builtins.hpp:2037  definition of all(T x)
  //detail::rel_sign_bit_test_ret_t<T>(__sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<T>>(detail::rel_sign_bit_test_arg_t<T>(x)));
  // __invoke_All comes from builtins.hpp:277
  // __SYCL_MAKE_CALL_ARG1(All, __FUNC_PREFIX_CORE)                // all



  // from builtins_helperx.hpp:80
  //  Fun: sycl_host_ALL, Call: __All, N, Ret: sycl::cl_int , Arg1: sycl::cl_int/long
  // #define __MAKE_SR_1V_AND(Fun, Call, N, Ret, Arg1)                              \
  // __SYCL_EXPORT Ret Fun __NOEXC(sycl::vec<Arg1, N> x) {                        \
  //   Ret r;                                                                     \
  //   detail::helper<N - 1>().run_1v_sr_and(                                     \
  //       r, [](Arg1 x) { return __host_std::Call(x); }, x);                     \
  //   return r;                                                                  \
  // }

  // builtins_helper.hpp:290
  // template <typename Res, typename Op, typename T1>
  // inline void run_1v_sr_and(Res &r, Op op, T1 x) {
  //   helper<N - 1>().run_1v_sr_and(r, op, x);
  //   r = (op(x.template swizzle<N>()) && r);
  // }

  // builtins_helper.hpp:385
  // template <typename Res, typename Op, typename T1>
  // inline void run_1v_sr_and(Res &r, Op op, T1 x) {
  //   r = op(x.template swizzle<0>());
  // }


  // from builtins_relational.cpp:70
  // template <typename T> s::cl_int inline __All(T x) { return d::msbIsSet(x); }

  std::cout << "msbIsSet: "  << detail::msbIsSet(andBoth[0]) << " " 
                             << detail::msbIsSet(andBoth[1]) << " " 
                             << detail::msbIsSet(andBoth[2]) << " " 
                             << detail::msbIsSet(andBoth[3]) << std::endl;   // 1 1 1 1  

  
  //__sycl_std::detail::helper<3>().run_1v_sr_and(myRes,  [](sycl::opencl::cl_int x) { return detail::msbIsSet(x); }, andBoth);

  auto operation = [](sycl::opencl::cl_int x) { return detail::msbIsSet(x); };
  sycl::opencl::cl_int myRes = operation(andBoth.template swizzle<0>() );
  std::cout << "myRes at 0: " << myRes;
  myRes = myRes && operation(andBoth.template swizzle<1>());
  std::cout << "  1: " << myRes;
  myRes = myRes && operation(andBoth.template swizzle<2>());
  std::cout << "  2: " << myRes;
  myRes = myRes && operation(andBoth.template swizzle<3>());
  std::cout << "  3: " << myRes  << std::endl;   //  =>  myRes at 0: 1  1: 1  2: 1  3: 1


  //int isCorrect = all((diff <= precision) && (diff >= (-precision))); //should return most significant bit set for each.  
  int isCorrect = all(andBoth);
  if (isCorrect) {
    std::cout << index << " -- " << vec2string(result) << std::endl;
  } else {
    std::string errMsg = "unexpected result: " + vec2string(result) +
                         " vs reference: " + vec2string(ref);
    std::cout << index << " -- " << errMsg  << std::endl;
    std::cout << "diff: " << vec2string(diff) << " precision: " << precision << std::endl;
    exit(1);
  }
}

template <typename accType, typename pixelType>
void check_pixels(accType &pixels, const std::vector<pixelType> &ref,
                  size_t &offset) {
  for (int i = offset, ref_i = 0; i < ref.size(); i++, ref_i++) {
    pixelType testPixel = pixels[i];
    check_pixel(testPixel, ref[ref_i], i);
  }
  offset += ref.size();
}