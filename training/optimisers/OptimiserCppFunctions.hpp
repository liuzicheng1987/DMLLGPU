OptimiserCpp::OptimiserCpp(/*const std::int32_t _size, const std::int32_t _rank*/) {
		
    /*this->size = _size;
      this->rank = _rank;*/
		
}

OptimiserCpp::~OptimiserCpp() {}
	
//dev_function_type is defined in file OptimiserCpp.hpp
void OptimiserCpp::minimise (/*MPI_Comm _comm,*/
			     NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm, 
			     std::int32_t                      _num_samples, 
			     thrust::device_vector<float>     &_W, 
			     std::int32_t                      _global_batch_size, 
			     const float                       _tol, 
			     const std::int32_t                _max_num_epochs, 
			     std::vector<float>               &_sum_gradients
			     ) {
	
  //Store all of the input values
  this->num_samples_ = _num_samples; 
  this->global_batch_size_ = _global_batch_size;

  //Set this->Wptr
  this->w_ptr_ = thrust::raw_pointer_cast(_W.data());
				
  //Initialise dLdw and dLdwPtr
  this->dldw_ = thrust::device_vector<float>(_W.size());
  this->dldw_ptr_ = thrust::raw_pointer_cast(this->dldw_.data());
  
  //Initialise sum_dldw
  this->sum_dldw_ = thrust::device_vector<float>(_W.size());	
		
  //Calculate the number of batches needed
  this->num_batches_ = _numerically_optimised_algorithm->calc_num_batches(
									  _num_samples, 
									  _global_batch_size
									  );
					
  //Create the threads and pass the values they need
  this->min( /*comm,*/
	    _numerically_optimised_algorithm,
	    _W, 
	    _tol, 
	    _max_num_epochs,
	    _sum_gradients
	     );

  //Clear this->dLdw and this->SumdLdw, so they don't take up space on the 
  this->dldw_.clear();
  this->sum_dldw_.clear();
		
}
