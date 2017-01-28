class SGDCpp : public OptimiserCpp
{

private:
  float learning_rate_;

  float learning_rate_power_;

  std::int32_t epoch_num_;

public:
  //Initialise the GradientDescent function
  SGDCpp(
      float _learning_rate,
      float _learning_rate_power
      /*, const std::int32_t _size, const std::int32_t _rank*/
      ) : OptimiserCpp(/*size, rank*/)
  {

    //Store all of the input values
    this->learning_rate_ = _learning_rate;
    this->learning_rate_power_ = _learning_rate_power;

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
