//A collection of useful functions and functors - alphebetical order

namespace utils
{

//Performs the classic axpy operation
template <typename T>
struct axpy
{

  const T alpha;
  const T beta;
  
  axpy(T _alpha, T _beta) : alpha(_alpha), beta(_beta) {}

  __device__ T operator()(const T &x, T &y) const
  {
    return alpha * x + beta * y;
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
