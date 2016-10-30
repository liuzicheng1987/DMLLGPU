class OptimiserCpp {
	
protected:

  float *w_ptr_;//Pointer to W - for convenience

  thrust::device_vector<float> dldw_;//Derivative of loss function for weight vector
  float *dldw_ptr_;//Pointer to dLdw - for convenience

  thrust::device_vector<float> sum_dldw_;//Sum of dLdw

  std::int32_t I_;//Number of samples 

  std::int32_t num_batches_;//Number of batches

  std::int32_t global_batch_size_;//Sum of batch size in all process

  /*

  std::int32_t GlobalBatchSize;//global size of batches (meaning sum of local batch size in all processes)

  std::int32_t size;//Number of processes

  std::int32_t rank;//Rank of this proces

  */

  virtual void min(/*MPI_Comm comm,*/NeuralNetworkGPUCpp *_NeuralNet, thrust::device_vector<float> &_W, const float _tol, const std::int32_t _MaxNumEpochs, std::vector<float> &_SumGradients) 
  {throw std::invalid_argument("This shouldn't happen!\n You need to use an optimising algorithm, not the base class!");}//Function to be accessed from minimise (does the actual work)!

public:
	
  //Constructor
  OptimiserCpp(/*const std::int32_t _size, const std::int32_t _rank*/);

  virtual ~OptimiserCpp();
	
  void minimise (/*MPI_Comm comm,*/NeuralNetworkGPUCpp *_NeuralNet, std::int32_t _I, thrust::device_vector<float> &_W, std::int32_t _GlobalBatchSize, const float _tol, const std::int32_t _MaxNumEpochs, std::vector<float> &_SumGradients);//Minimise loss function (public function)
			       		
};
