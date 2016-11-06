//A collection of useful functions and functors - alphebetical order

namespace utils {

  //Performs the classic saxpy operation
  template <typename T>
  struct axpy {

    const T learning_rate;

    axpy(T _learning_rate) : learning_rate(_learning_rate) {}

    __device__
    T operator()(const T &x, T &y) const { 
      return learning_rate * x + y;
    }

  };

  //Transformation that squares the result
  template <typename T>
  struct square : public thrust::unary_function<T,T> {

    __device__ T operator()(const T &x) const
    {
      return x*x;
    }

  };

}
