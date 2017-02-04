class AdamCpp : public OptimiserCpp
{

private:
  //Learning rate of the optimiser (in the beginning)
  float learning_rate_;

  float decay_mom1_;

  float decay_mom2_;

  float offset_;

  //Squared gradients
  thrust::device_vector<float> sum_dldw_squared_;

  //Number of epochs we are currently in
  std::int32_t epoch_num_;

public:
  // Initialise Adam
  AdamCpp(
      float _learning_rate, float decay_mom1_, float decay_mom2_, float offset_) : OptimiserCpp(/*size, rank*/)
  {

    //Store the input values
    this->learning_rate_ = _learning_rate;

    this->decay_mom1_ = _decay_mom1;

    this->decay_mom2_ = _decay_mom2;

    this->offset_ = _offset;

    this->epoch_num_ = 0;
  }

  //Destructor
  ~AdamCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};
