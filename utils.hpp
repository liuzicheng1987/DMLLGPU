//A collection of useful functions and functors - alphebetical order

namespace utils {

  //Performs the classic saxpy operation
  template <typename T>
  struct saxpy {

    const T LearningRate;

    saxpy(T _LearningRate) : LearningRate(_LearningRate) {}

    __device__
    T operator()(const T &x, T &y) const { 
      return LearningRate * x + y;
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
