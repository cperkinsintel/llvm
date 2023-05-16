#include <iostream>
#include <limits>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

// ---------------------------------------------------------

/*
  when run with the logging we see

    read four pixels, no sampler
    bothAnd: {-1,-1,-1,-1}

    r type/val: i / 0
    s type/val: i / 1

    0 -- unexpected result: {1,2,3,4} vs reference: {1,2,3,4}


*/

// builtins_relational.cpp:72
template <typename T> opencl::cl_int inline __cpAll(T x) { return detail::msbIsSet(x); }

// builtins_helper.hpp:291
template <int N> struct helper {
  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_and(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_sr_and(r, op, x);
    r = (op(x.template swizzle<N>()) && r);
  }
};

// builtins_helper.hpp:358
template <> struct helper<0> {
  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_and(Res &r, Op op, T1 x) {
    r = op(x.template swizzle<0>());
  }
};

// builtins_helper.hpp:80
#define __CPMAKE_SR_1V_AND(Fun, Call, N, Ret, Arg1)                              \
  __SYCL_EXPORT Ret Fun __NOEXC(sycl::vec<Arg1, N> x) {                        \
    Ret r;                                                                     \
    helper<N - 1>().run_1v_sr_and(                                     \
        r, [](Arg1 x) { return /*__host_std::*/Call(x); }, x);                     \
    return r;                                                                  \
  }

// builtins_helper.hpp:176
#define CPMAKE_SR_1V_AND(Fun, Call, Ret, Arg1)                                   \
  __CPMAKE_SR_1V_AND(Fun, Call, 1, Ret, Arg1)                                    \
  __CPMAKE_SR_1V_AND(Fun, Call, 2, Ret, Arg1)                                    \
  __CPMAKE_SR_1V_AND(Fun, Call, 3, Ret, Arg1)                                    \
  __CPMAKE_SR_1V_AND(Fun, Call, 4, Ret, Arg1)                                    \
  __CPMAKE_SR_1V_AND(Fun, Call, 8, Ret, Arg1)                                    \
  __CPMAKE_SR_1V_AND(Fun, Call, 16, Ret, Arg1)

// builtins_relational.cpp:480
CPMAKE_SR_1V_AND(cpycl_host_All, __cpAll, opencl::cl_int, opencl::cl_char)
CPMAKE_SR_1V_AND(cpycl_host_All, __cpAll, opencl::cl_int, opencl::cl_short)
CPMAKE_SR_1V_AND(cpycl_host_All, __cpAll, opencl::cl_int, opencl::cl_int)
CPMAKE_SR_1V_AND(cpycl_host_All, __cpAll, opencl::cl_int, opencl::cl_long)

// builtins.hpp:40
#define __CPYCL_PPCAT_NX(A, B) A##B
#define __CPYCL_PPCAT(A, B) __CPYCL_PPCAT_NX(A, B)

// builtins.hpp:20
#ifdef __SYCL_DEVICE_ONLY__
  #define __FUNC_PREFIX_CORE __spirv_
  #define __CPYCL_EXTERN_IT1(Ret, prefix, call, Arg1)
#else
  #define __FUNC_PREFIX_CORE cpycl_host_
  #define __CPYCL_EXTERN_IT1(Ret, prefix, call, Arg)                              \
  extern Ret __CPYCL_PPCAT(prefix, call)(Arg)
#endif


// builtins.hpp:43
#define __CPYCL_MAKE_CALL_ARG1(call, prefix)                                    \
  template <typename R, typename T1>                                           \
  inline __SYCL_ALWAYS_INLINE R __cp_invoke_##call(T1 t1) __NOEXC {            \
    using Ret = sycl::detail::ConvertToOpenCLType_t<R>;                        \
    using Arg1 = sycl::detail::ConvertToOpenCLType_t<T1>;                      \
    __CPYCL_EXTERN_IT1(Ret, prefix, call, Arg1);                                \
    Arg1 arg1 = sycl::detail::convertDataToType<T1, Arg1>(t1);                 \
    Ret ret = __CPYCL_PPCAT(prefix, call)(arg1);                                \
    return sycl::detail::convertDataToType<Ret, R>(ret);                       \
  }

// builtint.hpp:277
__CPYCL_MAKE_CALL_ARG1(All, __FUNC_PREFIX_CORE) 

// --------------------------------------------

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


  // -----------------------------
  //   logging 
  auto lessThan = (diff <= precision);
  auto moreThan = (diff >= (-precision));
  auto bothAnd = lessThan && moreThan;
  std::cout << "bothAnd: " << vec2string(bothAnd) << std::endl;  // {-1 -1 -1 -1}  <== this is correct


  // our __invoke_All is incorrectly returning 0.  
  auto r = __sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<decltype(bothAnd)>>(detail::rel_sign_bit_test_arg_t<decltype(bothAnd)>(bothAnd));
  std::cout << "r type/val: " <<  typeid(r).name() << " / " << (bool)r << std::endl; // type: i / 0

  // but using the same definitions (copy/pasted above) we get 1. 
  auto s = __cp_invoke_All<detail::rel_sign_bit_test_ret_t<decltype(bothAnd)>>(detail::rel_sign_bit_test_arg_t<decltype(bothAnd)>(bothAnd));
  std::cout << "s type/val: " <<  typeid(s).name() <<  " / " << (bool)s << std::endl; // type i  / 1
  // ------------------------------




  int isCorrect = all((diff <= precision) && (diff >= (-precision)));
  if (isCorrect) {
    std::cout << index << " -- " << vec2string(result) << std::endl;
  } else {
    std::string errMsg = "unexpected result: " + vec2string(result) +
                         " vs reference: " + vec2string(ref);
    std::cout << index << " -- " << errMsg << std::endl;
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