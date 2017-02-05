class AdaDeltaCpp : public OptimiserCpp
{

private:
  float gamma_;

  float offset_;

  thrust::device_vector<float> sum_dldw_squared_;

  thrust::device_vector<float> sum_updates_squared_;

public:
  //Initialise the GradientDescent function
  AdaDeltaCpp(
      float _gamma,
      float _offset) : OptimiserCpp(/*size, rank*/)
  {

    //Store the input values
    this->gamma_ = _gamma;

    this->offset_ = _offset;

    this->epoch_num_ = 0;
  }

  //Destructor
  ~AdaDeltaCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};
