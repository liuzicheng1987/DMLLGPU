class OptimiserCpp
{

protected:
  //Pointer to W - for convenience
  float *w_ptr_;

  //Derivative of loss function for weight vector
  thrust::device_vector<float> dldw_;

  //Pointer to dldw - for convenience
  float *dldw_ptr_;

  //Sum of dLdw
  thrust::device_vector<float> sum_dldw_;

  //Number of samples
  std::int32_t num_samples_;

  //Number of batches
  std::int32_t num_batches_;

  //Sum of batch size in all process
  std::int32_t global_batch_size_;

  //Number of epochs we are currently in
  std::int32_t epoch_num_;

  /*

  std::int32_t global_batch_size;//global size of batches (meaning sum of local batch size in all processes)

  std::int32_t size;//Number of processes

  std::int32_t rank;//Rank of this proces

  */

  virtual void min(/*MPI_Comm comm,*/
                   NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
                   thrust::device_vector<float> &_W,
                   const float _tol,
                   const std::int32_t _max_num_epochs,
                   std::vector<float> &_sum_gradients)
  {
    throw std::invalid_argument("This shouldn't happen! You need to use an optimising algorithm (like SGD), not the base class!");
  } //Function to be accessed from minimise (does the actual work)!

  void calculate_and_record_dldw(std::int32_t _batch_num,
                                 NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm);

  //Returns true, if convergence condition is met
  bool record_sum_dldw_and_check_convergence(std::vector<float> &_sum_gradients, float _tol);

public:
  //Constructor
  OptimiserCpp(/*const std::int32_t _size, const std::int32_t _rank*/);

  virtual ~OptimiserCpp();

  void minimise(/*MPI_Comm comm,*/
                NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
                std::int32_t _num_samples,
                thrust::device_vector<float> &_W,
                std::int32_t _global_batch_size,
                const float _tol,
                const std::int32_t _max_num_epochs,
                std::vector<float> &_sum_gradients);
};
