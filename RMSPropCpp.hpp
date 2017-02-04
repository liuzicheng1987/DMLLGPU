class RMSPropCpp : public OptimiserCpp
{

private:
  //Learning rate of the optimiser (in the beginning)
  float learning_rate_;

  //Decay rate of the squared gradients
  float gamma_;

  //Squared gradients
  thrust::device_vector<float> sum_dldw_squared_;

  //Number of epochs we are currently in
  std::int32_t epoch_num_;

public:
  //Initialise the GradientDescent function
  RMSPropCpp(
      float _learning_rate, float _gamma) : OptimiserCpp(/*size, rank*/)
  {

    //Store the input values
    this->learning_rate_ = _learning_rate;

    this->gamma_ = _gamma;

    this->epoch_num_ = 0;
  }

  //Destructor
  ~RMSPropCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};
