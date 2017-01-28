//A collection of useful functions and functors - alphebetical order

namespace utils
{

//Performs the classic axpy operation
template <typename T>
struct axpy
{

  const T alpha;

  axpy(T _alpha) : alpha(_alpha) {}

  __device__ T operator()(const T &x, T &y) const
  {
    return alpha * x + y;
  }
};

//Transformation that squares the result
template <typename T>
struct square : public thrust::unary_function<T, T>
{

  __device__ T operator()(const T &x) const
  {
    return x * x;
  }
};
}
