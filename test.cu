#include <iostream>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

typedef thrust::device_vector<float> dvec;

namespace ActivationFunction {

struct CustomTuple2 {

  float val0;
  float val1;

};

struct TransformationCustomTuple2 {
  
  template <typename Tuple>
  __device__ const CustomTuple2 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d ist the delta value    
    CustomTuple2 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);

    return result;
  }

};

struct SumCustomTuple2 {
 
  __device__ const CustomTuple2 operator()(const CustomTuple2& lhs, const CustomTuple2& rhs) {
    
    CustomTuple2 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;

    return result;
  }

};

}


std::int32_t main()
{

  float a_host[4] = {4.f, 5.f, 10.f, 2.f}; 
  float b_host[4] = {4.f, 3.f, -3.f, 3.f}; 
  float delta_host[4] = {-1.f, 1.f, -1.f, 1.f};

  dvec a(a_host, a_host + 4);
  dvec b(b_host, b_host + 4);
  dvec delta(delta_host, delta_host + 4);

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), a.begin(), b.begin()));
  auto end = thrust::make_zip_iterator(thrust::make_tuple(delta.end(), a.end(), b.end()));

  ActivationFunction::CustomTuple2 init;
  init.val0 = 0.0f;
  init.val1 = 0.0f;

  ActivationFunction::CustomTuple2 res = thrust::transform_reduce(begin, end, ActivationFunction::TransformationCustomTuple2(), init, ActivationFunction::SumCustomTuple2());
  
  std::cout << res.val0 << std::endl;
  std::cout << res.val1 << std::endl;
  std::cout << "done" << std::endl;

  return 0;
}
