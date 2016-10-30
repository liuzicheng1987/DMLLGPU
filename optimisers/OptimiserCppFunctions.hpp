OptimiserCpp::OptimiserCpp(/*const std::int32_t _size, const std::int32_t _rank*/) {
		
    /*this->size = _size;
      this->rank = _rank;*/
		
}

OptimiserCpp::~OptimiserCpp() {}
	
//dev_function_type is defined in file OptimiserCpp.hpp
void OptimiserCpp::minimise (/*MPI_Comm _comm,*/NeuralNetworkGPUCpp *_NeuralNet, std::int32_t _I, thrust::device_vector<float> &_W, std::int32_t _GlobalBatchSize, const float _tol, const std::int32_t _MaxNumEpochs, std::vector<float> &_SumGradients) {
	
  //Store all of the input values
  this->I_ = _I; 
  this->global_batch_size_ = _GlobalBatchSize;

  //Set this->Wptr
  this->w_ptr_ = thrust::raw_pointer_cast(_W.data());
				
  //Initialise dLdw and dLdwPtr
  this->dldw_ = thrust::device_vector<float>(_W.size());
  this->dldw_ptr_ = thrust::raw_pointer_cast(this->dldw_.data());
  
  //Initialise SumdLdw
  this->sum_dldw_ = thrust::device_vector<float>(_W.size());	
		
  //Calculate the number of batches needed
  this->num_batches_ = _NeuralNet->calc_num_batches(_I, _GlobalBatchSize);
					
  //Create the threads and pass the values they need
  this->min(/*comm,*/_NeuralNet, _W, _tol, _MaxNumEpochs, _SumGradients);

  //Clear this->dLdw and this->SumdLdw, so they don't take up space on the GPU
  this->dldw_.clear();
  this->sum_dldw_.clear();
		
}
