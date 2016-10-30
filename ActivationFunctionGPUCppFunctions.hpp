ActivationFunctionGPUCpp::ActivationFunctionGPUCpp(
						   std::int32_t  _NodeNumber, 
						   std::int32_t *_InputNodesFedIntoMeDense, 
						   std::int32_t  _InputNodesFedIntoMeDenseLength, 
						   std::int32_t *_InputNodesFedIntoMeSparse, 
						   std::int32_t  _InputNodesFedIntoMeSparseLength, 
						   std::int32_t *_HiddenNodesFedIntoMe, 
						   std::int32_t  _HiddenNodesFedIntoMeLength, 
						   std::int32_t  _IShareWeightsWith, 
						   bool          _NoWeightUpdates
						   ): NeuralNetworkNodeGPUCpp (
									      _NodeNumber,
									      _InputNodesFedIntoMeDense, 
									      _InputNodesFedIntoMeDenseLength, 
									      _InputNodesFedIntoMeSparse, 
									      _InputNodesFedIntoMeSparseLength, 
									      _HiddenNodesFedIntoMe, 
									      _HiddenNodesFedIntoMeLength, 
									      _IShareWeightsWith, 
									      _NoWeightUpdates
									      ) {

  //We are initialising these two vectors, so we can resize them later, if necessary
  this->output = thrust::device_vector<float>(1);
  this->delta = thrust::device_vector<float>(1);

}

ActivationFunctionGPUCpp::~ActivationFunctionGPUCpp() {};

std::int32_t ActivationFunctionGPUCpp::get_num_weights_required() {

  //This is necessary for every class that inherits from NeuralNetworkNodeGPUCpp
  return NumInputNodesCumulative + static_cast<std::int32_t>(HiddenNodesFedIntoMe.size()) + 1; 

};

void ActivationFunctionGPUCpp::calc_output(
					   const std::int32_t _BatchNum, 
					   const std::int32_t _BatchSize
					   ) {

  cublasStatus_t cstatus;//cuBLAS status variable - so we can check whether the cuBLAS operations were successful
  const float *w = this->W;//Pointer to weights
  
  std::int32_t J;//Number of columns in input data - for convenience

  //Needed for cuBLAS transformations
  const float alpha = 1.0; 
  const float beta = 1.0; 

  //Resize output and delta, if necessary
  //Output is stored in the NeuralNetworkNodeGPUCpp base class and stores the output of this node
  if (static_cast<std::int32_t>(this->output.size()) != _BatchSize) {
    
    //Resize output
    this->output.resize(_BatchSize);
    this->OutputPtr = thrust::raw_pointer_cast(this->output.data());

    //Resize delta
    this->delta.resize(_BatchSize);
    this->DeltaPtr = thrust::raw_pointer_cast(this->delta.data());

  }

  //Initialise output to zero
  thrust::fill(
	       this->output.begin(), 
	       this->output.end(), 
	       0.0f
	       );

  //Transform dense input nodes
  for (std::int32_t i: this->InputNodesFedIntoMeDense) {

    J = this->NeuralNet->get_dense_input_data(i, _BatchNum).J;//For convenience

    //Linear transformation of input data using cuBLAS
    //output = alpha*Xw + beta*output 
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemv
    cstatus = cublasSgemv(
			  this->NeuralNet->get_handle(), 
			  CUBLAS_OP_T, 
			  J,
			  _BatchSize,
			  &alpha, 
			  this->NeuralNet->get_dense_input_data(i, _BatchNum).Xptr, 
			  J, 
			  w, 
			  1, 
			  &beta, 
			  this->OutputPtr,
			  1
			  );

    //Make sure that matrix multiplication succeeded and throw error if it didn't!
    if (cstatus != CUBLAS_STATUS_SUCCESS)
      throw std::invalid_argument("Something went wrong during cuBLAS operation on dense input data!");
      
    w += J;//Increment w
    
  } 

  //Transform sparse input nodes
  for (std::int32_t i: this->InputNodesFedIntoMeSparse) {

    J = this->NeuralNet->get_sparse_input_data(i, _BatchNum).J;//For convenience
    
    //...to be implemented

    w += J;//Increment w
    
  } 
  
  //Transform hidden nodes fed into me and apply activation function
  this->forward_propagation(
			    w, 
			    this->HiddenNodesFedIntoMePtr.data(), 
			    this->HiddenNodesFedIntoMePtr.size(), 
			    this->output
			    );

}

void ActivationFunctionGPUCpp::calc_delta() {

  this->backpropagation(
			this->W + this->NumInputNodesCumulative, 
			this->HiddenNodesFedIntoMePtr.data(), 
			this->HiddenNodesFedIntoMePtr.size(), 
			this->output, 
			this->delta
			);

}

void ActivationFunctionGPUCpp::calc_dLdw(float *_dLdw, const std::int32_t _BatchNum, const std::int32_t _BatchSize) {

  //pointer _dLdw points to the beginning of the weights relevant for this node. This is achieved by the g(...) function.

  cublasStatus_t cstatus;//cuBLAS status variable - so we can check whether the cuBLAS operations were successful

  const float *w = this->W;//Pointer to weights
  float *dldw = _dLdw;//Pointer to derivatives

  std::int32_t J;//Number of columns in input data - for convenience

  //Needed for cuBLAS transformations
  const float alpha = 1.0; 
  const float beta = 0.0; 

  //Calculate derivatives for dense input nodes
  for (std::int32_t i: this->InputNodesFedIntoMeDense) {

    J = this->NeuralNet->get_dense_input_data(i, _BatchNum).J;//For convenience

    //Linear transformation of input data using cuBLAS
    //dldw = alpha*X(this->DeltaPtr) + beta*dldw 
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemv
    cstatus = cublasSgemv(
			  this->NeuralNet->get_handle(), 
			  CUBLAS_OP_N, 
			  J, 
			  _BatchSize, 
			  &alpha, 
			  this->NeuralNet->get_dense_input_data(i, _BatchNum).Xptr, 
			  J, 
			  this->DeltaPtr, 
			  1, 
			  &beta, 
			  dldw, 
			  1
			  );

    if (cstatus != CUBLAS_STATUS_SUCCESS) 
      throw std::runtime_error("Something went wrong during cuBLAS operation on input data while calculating derivatives!");

    w += J;//Increment w
    dldw += J;//Increment dldw
    
  } 

  //Calculate derivatives for sparse input nodes
  for (std::int32_t i: this->InputNodesFedIntoMeSparse) {

    J = this->NeuralNet->get_sparse_input_data(i, _BatchNum).J;//For convenience

    //...to be implemented

    w += J;//Increment w
    dldw += J;//Increment dldw
    
  }   

  //Calculate derivatives for hidden nodes fed into this node
  ActivationFunction::calc_dev_hidden(
				      dldw, 
				      this->HiddenNodesFedIntoMePtr.data(), 
				      this->HiddenNodesFedIntoMePtr.size(), 
				      this->delta
				      );
   
}
