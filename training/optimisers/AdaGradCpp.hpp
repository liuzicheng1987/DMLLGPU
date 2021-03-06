class AdaGradCpp : public OptimiserCpp
{

private:
  float learning_rate_;

  float offset_;

  thrust::device_vector<float> sum_dldw_squared_;

public:
  //Initialise the GradientDescent function
  AdaGradCpp(
      float _learning_rate,
      float _offset) : OptimiserCpp(/*size, rank*/)
  {

    //Store the input values
    this->learning_rate_ = _learning_rate;

    this->offset_ = _offset;

    this->epoch_num_ = 0;
  }

  //Destructor
  ~AdaGradCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};
