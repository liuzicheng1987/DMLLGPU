class SGDCpp : public OptimiserCpp
{

private:
  //Learning rate of the optimiser (in the beginning)
  float learning_rate_;

  //Speed by which learning rate decays
  float learning_rate_power_;

  //momentum_ denotes the weight given to all previous updates
  float momentum_;
  
  //Number of epochs we are currently in
  std::int32_t epoch_num_;

  //Value, by which weights are actually updated by
  thrust::device_vector<float> update_;

public:
  //Initialise the GradientDescent function
  SGDCpp(
      float _learning_rate,
      float _learning_rate_power,
      float _momentum
      /*, const std::int32_t _size, const std::int32_t _rank*/
      ) : OptimiserCpp(/*size, rank*/)
  {

    //Store all of the input values
    this->learning_rate_ = _learning_rate;
    this->learning_rate_power_ = _learning_rate_power;
    this->momentum_ = momentum_;

    this->epoch_num_ = 0;
  }

  //Destructor
  ~SGDCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};
