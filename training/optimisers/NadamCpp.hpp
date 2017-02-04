class NadamCpp : public OptimiserCpp
{

private:
  //Learning rate of the optimiser (in the beginning)
  float learning_rate_;

  float beta_1_;

  float beta_2_;

  float schedule_decay_;

  float offset_;

  float m_schedule_;

  //Squared gradients
  thrust::device_vector<float> sum_dldw_squared_;

  //Number of epochs we are currently in
  std::int32_t epoch_num_;

  thrust::device_vector<float> est_mom1_b_;

  thrust::device_vector<float> est_mom2_b_;

  float momentum_cache_t_, momentum_cache_t_1_, m_schedule_new_, m_schedule_next_;

public:
  // Initialise Nadam
  NadamCpp(float _learning_rate, float _beta_1, float _beta_2, float _schedule_decay, float _offset) : OptimiserCpp(/*size, rank*/)
  {

    //Store the input values
    this->learning_rate_ = _learning_rate;

    this->beta_1_ = _beta_1;

    this->beta_2_ = _beta_2;

    this->schedule_decay_ = _schedule_decay;

    this->offset_ = _offset;

    this->epoch_num_ = 0;

    this-> m_schedule_ = 1.f;
  }

  //Destructor
  ~NadamCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};
