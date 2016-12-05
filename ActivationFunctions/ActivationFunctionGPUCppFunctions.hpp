ActivationFunctionGPUCpp::ActivationFunctionGPUCpp(
						   std::int32_t    _node_number, 
						   std::int32_t    _dim,
						   std::int32_t   *_input_nodes_fed_into_me_dense, 
						   std::int32_t    _input_nodes_fed_into_me_dense_length, 
						   std::int32_t   *_input_nodes_fed_into_me_sparse, 
						   std::int32_t    _input_nodes_fed_into_me_sparse_length, 
						   std::int32_t   *_hidden_nodes_fed_into_me, 
						   std::int32_t    _hidden_nodes_fed_into_me_length, 
						   std::int32_t    _i_share_weights_with, 
						   bool            _no_weight_updates,
						   RegulariserCpp *_regulariser
						   ): NeuralNetworkNodeGPUCpp (
									      _node_number,
									      _dim,
									      _input_nodes_fed_into_me_dense, 
									      _input_nodes_fed_into_me_dense_length, 
									      _input_nodes_fed_into_me_sparse, 
									      _input_nodes_fed_into_me_sparse_length, 
									      _hidden_nodes_fed_into_me, 
									      _hidden_nodes_fed_into_me_length, 
									      _i_share_weights_with, 
									      _no_weight_updates,
									      _regulariser
									      ) {

  //We are initialising these two vectors, so we can resize them later, if necessary
  this->output = thrust::device_vector<float>(1);
  this->delta = thrust::device_vector<float>(1);

}

ActivationFunctionGPUCpp::~ActivationFunctionGPUCpp() {};

std::int32_t ActivationFunctionGPUCpp::get_num_weights_required() {

  std::int32_t num_weights_required = this->num_input_nodes_cumulative;

  for (auto hidden: this->hidden_nodes_fed_into_me_ptr)
    num_weights_required += hidden->get_dim();

  num_weights_required += 1;

  num_weights_required *= this->dim_;

  return num_weights_required; 

};

void ActivationFunctionGPUCpp::calc_output(
					   const std::int32_t _batch_num, 
					   const std::int32_t _batch_size
					   ) {

  //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
  cublasStatus_t cstatus_dense;

  //cuSPARSE status variable - so we can check whether the cuBLAS operations were successful
  cusparseStatus_t cstatus_sparse;
  
  //Pointer to weights
  const float *w = this->W;
  
  //Number of columns in input data - for convenience
  std::int32_t input_dim;

  //Needed for cuBLAS transformations
  const float alpha = 1.0; 
  float beta = 0.0; 

  //Resize output and delta, if necessary
  //Output is stored in the NeuralNetworkNodeGPUCpp base class and stores the output of this node
  if (static_cast<std::int32_t>(this->output.size()) != this->dim_*_batch_size) {
    
    //Resize output
    this->output.resize(this->dim_*_batch_size);
    this->output_ptr = thrust::raw_pointer_cast(this->output.data());

    //Resize delta
    this->delta.resize(this->dim_*_batch_size);
    this->delta_ptr = thrust::raw_pointer_cast(this->delta.data());

  }
  
  //Transform dense input nodes
  for (std::int32_t i: this->input_nodes_fed_into_me_dense) {

    input_dim = this->NeuralNet->get_dense_input_data(
						      i, 
						      _batch_num
						      ).dim;//For convenience

    //output = alpha*(WT)X + beta*output 
    //X: (input_dim X _batch_size)-matrix
    //NOTE: X is (input_dim X _batch_size) for input data,
    // but (_batch_size X input_dim) for hidden node data!
    //w: (input_dim X this->dim_)-matrix
    //output: (_batch_size X this->dim_)-matrix
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus_dense = cublasSgemm(
				this->NeuralNet->get_dense_handle(), //handle
				CUBLAS_OP_T, //transa
				CUBLAS_OP_N, //transb
				_batch_size, //m
				this->dim_, //n
				input_dim, //k           
				&alpha, //alpha
				this->NeuralNet->get_dense_input_data(
								      i,
								      _batch_num
								      ).X_ptr, //A
				input_dim, //lda
				w, //B
				input_dim, //ldb
				&beta, //beta
				this->output_ptr, //C 
				_batch_size //ldc
				);

    //Make sure that matrix multiplication succeeded and throw error if it didn't!
    if (cstatus_dense != CUBLAS_STATUS_SUCCESS)
      throw std::invalid_argument("Something went wrong during cuBLAS operation on dense input data!");
      
    w += this->dim_*input_dim;//Increment w to prepare it for next operation
    beta = 1.0;//Set beta to 1.0, so all subsequent matrix operations are added up.
    
  } 

  //Transform sparse input nodes
  for (std::int32_t i: this->input_nodes_fed_into_me_sparse) {

    input_dim = this->NeuralNet->get_sparse_input_data(i, _batch_num).dim;//For convenience
    
    //output = alpha*(WT)X + beta*output 
    //X: (_batch_size X input_dim)-CSR-matrix
    //w: (input_dim X this->dim_)-matrix
    //output: (_batch_size X this->dim_)-matrix
    cstatus_sparse = cusparseScsrmm(
				    this->NeuralNet->get_sparse_handle(),//handle
				    CUSPARSE_OPERATION_NON_TRANSPOSE,//transA
				    _batch_size,//m 
				    this->dim_,//n
				    input_dim, //k
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).num_non_zero, //nnz
				    &alpha, //alpha
				    this->NeuralNet->get_mat_descr(), //descrA 
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).X_data_ptr,//csrValA
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).X_indptr_ptr,//csrRowPtrA
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).X_indices_ptr,//csrColIndA
				    w, //B
				    input_dim, //ldb
				    &beta,//beta
				    this->output_ptr, //C 
				    _batch_size //ldc
				    );

    //Make sure that matrix multiplication succeeded and throw error if it didn't!
    if (cstatus_sparse != CUSPARSE_STATUS_SUCCESS)
      throw std::invalid_argument("Something went wrong during cuSPARSE operation on sparse input data!");
    
    w += this->dim_*input_dim;//Increment w
    beta = 1.0;//Set beta to 1.0, so all subsequent matrix operations are added up.
    
  } 
  
  //Transform hidden nodes
  //Calculate derivatives for hidden nodes
  for (auto node: this->hidden_nodes_fed_into_me_ptr) {

    input_dim = node->get_dim();//For convenience

    //Linear transformation of input data using cuBLAS
    //dldw = alpha*(X)deltaT + beta*dldw
    //X: (_batch_size X input_dim)-matrix 
    //NOTE: X is (input_dim X _batch_size) for input data,
    // but (_batch_size X input_dim) for hidden node data!
    //w: (input_dim X this->dim_)-matrix
    //output: (_batch_size X this->dim_)-matrix
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus_dense = cublasSgemm(
				this->NeuralNet->get_dense_handle(), //handle
				CUBLAS_OP_N, //transa
				CUBLAS_OP_N, //transb
				_batch_size, //m
				this->dim_, //n
				input_dim, //k           
				&alpha, //alpha
				node->get_output_ptr(),//A
				_batch_size, //lda
				w, //B
				input_dim, //ldb
				&beta, //beta
				this->output_ptr, //C 
				_batch_size //ldc
				);

    if (cstatus_dense != CUBLAS_STATUS_SUCCESS) 
      throw std::runtime_error("Something went wrong during cuBLAS operation on hidden node data while calculating derivatives!");

    w += this->dim_*input_dim;//Increment w to prepare it for next operation
    beta = 1.0;//Set beta to 1.0, so all subsequent matrix operations are added up.
    
  } 
  
  //This functions adds the bias and then applies the activation function
  this->forward_propagation(
			    _batch_size,
			    this->dim_,
			    w,
			    this->output
			    );

}

void ActivationFunctionGPUCpp::calc_delta(std::int32_t _batch_size) {

  //Begin by applying the backpropagation functor
  //This multiplies delta by the derivative of the activation function
  this->backpropagation(
			_batch_size,
			this->dim_,
			this->output,
			this->delta
			);

  //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
  cublasStatus_t cstatus;

  //Needed for the cuBLAS operations
  const float alpha = 1.0;
  const float beta = 1.0;

  //Dimension of the input matrices - for convenience
  std::int32_t input_dim;

  //Pointer to weights associated with hidden nodes
  const float *w = this->W + num_input_nodes_cumulative;

  //Calculate derivatives for hidden nodes
  for (auto node: this->hidden_nodes_fed_into_me_ptr) {

    input_dim = node->get_dim();//For convenience

    //dldw = alpha*(X)deltaT + beta*dldw
    //this->delta: (_batch_size X this->dim_)-matrix
    //w: (input_dim X this->dim_)-matrix
    //node->delta: (_batch_size X input_dim)-matrix
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus = cublasSgemm(
			  this->NeuralNet->get_dense_handle(), //handle
			  CUBLAS_OP_N, //transa
			  CUBLAS_OP_T, //transb
			  _batch_size, //m
			  input_dim, //n
			  this->dim_, //k           
			  &alpha, //alpha
			  this->delta_ptr,//A
			  _batch_size, //lda
			  w, //B
			  input_dim, //ldb
			  &beta, //beta
			  node->get_delta_ptr(), //C 
			  _batch_size //ldc
			  );

    if (cstatus != CUBLAS_STATUS_SUCCESS) 
    throw std::runtime_error("Something went wrong during cuBLAS operation on hidden node data while calculating delta!");

    w += this->dim_*input_dim;//Increment w to prepare it for next operation
    
  }
  
}

void ActivationFunctionGPUCpp::calc_dLdw(float *_dLdw, const std::int32_t _batch_num, const std::int32_t _batch_size) {

  //pointer _dLdw points to the beginning of the weights relevant for this node. This is achieved by the g(...) function.

  //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
  cublasStatus_t cstatus_dense;

  //cuSPARSE status variable - so we can check whether the cuBLAS operations were successful
  cusparseStatus_t cstatus_sparse;

  //Pointer to derivatives
  float *dldw = _dLdw;

  //Number of columns in input data - for convenience
  std::int32_t input_dim;

  //Needed for cuBLAS transformations
  //beta is set to 1.f, because of the possibility of weight sharing
  //This is why it is so important that the optimisers intialise
  //the derivatives to 0.f!
  const float alpha = 1.f; 
  const float beta = 1.f;

  //Calculate derivatives for dense input nodes
  for (std::int32_t i: this->input_nodes_fed_into_me_dense) {

    input_dim = this->NeuralNet->get_dense_input_data(i, _batch_num).dim;//For convenience

    //dldw = alpha*(X)delta + beta*dldw
    //X: (input_dim X _batch_size)-matrix 
    //NOTE: X is (input_dim X _batch_size) for input data,
    // but (_batch_size X input_dim) for hidden node data!
    //delta: (_batch_size X this->dim_)-matrix
    //dldw: (input_dim X this->dim_)-matrix (just like w)
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus_dense = cublasSgemm(
				this->NeuralNet->get_dense_handle(), //handle
				CUBLAS_OP_N, //transa
				CUBLAS_OP_N, //transb
				input_dim, //m
				this->dim_, //n
				_batch_size, //k           
				&alpha, //alpha
				this->NeuralNet->get_dense_input_data(
								      i, 
								      _batch_num
								      ).X_ptr, //A 
				input_dim, //lda
				this->delta_ptr,//B
				_batch_size, //ldb
				&beta, //beta
				dldw, //C 
				input_dim //ldc
				);

    if (cstatus_dense != CUBLAS_STATUS_SUCCESS) 
      throw std::runtime_error("Something went wrong during cuBLAS operation on input data while calculating derivatives!");

    dldw += this->dim_*input_dim;//Increment dldw to prepare it for next operation
    
  }

  //Calculate derivatives for sparse input nodes
  for (std::int32_t i: this->input_nodes_fed_into_me_sparse) {

    input_dim = this->NeuralNet->get_sparse_input_data(i, _batch_num).dim;//For convenience
    
    //dldw = alpha*(XT)delta + beta*dldw
    //X: (_batch_size X input_dim)-CSR-matrix
    //delta: (_batch_size X this->dim_)-matrix
    //dldw: (input_dim X this->dim_)-matrix (just like w)
    cstatus_sparse = cusparseScsrmm(
				    this->NeuralNet->get_sparse_handle(),//handle
				    CUSPARSE_OPERATION_TRANSPOSE,//transA
				    _batch_size,//m 
				    this->dim_,//n
				    input_dim, //k
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).num_non_zero, //nnz
				    &alpha, //alpha
				    this->NeuralNet->get_mat_descr(), //descrA 
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).X_data_ptr,//csrValA
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).X_indptr_ptr,//csrRowPtrA
				    this->NeuralNet->get_sparse_input_data(
									   i, 
									   _batch_num
									   ).X_indices_ptr,//csrColIndA
				    this->delta_ptr,//B
				    _batch_size, //ldb
				    &beta,//beta
				    dldw, //C 
				    input_dim //ldc
				    );

    //Make sure that matrix multiplication succeeded and throw error if it didn't!
    if (cstatus_sparse != CUSPARSE_STATUS_SUCCESS)
    throw std::invalid_argument("Something went wrong during cuSPARSE operation while calculating derivatives!");

    dldw += this->dim_*input_dim;//Increment dldw to prepare it for next operation
    
  } 

  //Calculate derivatives for hidden nodes
  for (auto node: this->hidden_nodes_fed_into_me_ptr) {

    input_dim = node->get_dim();//For convenience

    //dldw = alpha*(X)deltaT + beta*dldw
    //X: (_batch_size X input_dim)-matrix 
    //NOTE: X is (input_dim X _batch_size) for input data,
    // but (_batch_size X input_dim) for hidden node data!
    //delta: (_batch_size X this->dim_)-matrix
    //dldw: (input_dim X this->dim_)-matrix (just like w)
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus_dense = cublasSgemm(
				this->NeuralNet->get_dense_handle(), //handle
				CUBLAS_OP_T, //transa
				CUBLAS_OP_N, //transb
				input_dim, //m
				this->dim_, //n
				_batch_size, //k           
				&alpha, //alpha
				node->get_output_ptr(),//A 
				_batch_size, //lda
				this->delta_ptr,//B
				_batch_size, //ldb
				&beta, //beta
				dldw, //C 
				input_dim //ldc
				);

    if (cstatus_dense != CUBLAS_STATUS_SUCCESS) 
      throw std::runtime_error("Something went wrong during cuBLAS operation on hidden node data while calculating derivatives!");

    dldw += this->dim_*input_dim;//Increment dldw to prepare it for next operation
    
  } 

  //Resize ones, if necessary
  //Output is stored in the NeuralNetworkNodeGPUCpp base class and stores the output of this node
  if (static_cast<std::int32_t>(this->ones_.size()) < _batch_size) {
    
    this->ones_.resize(_batch_size);
    this->ones_ptr_ = thrust::raw_pointer_cast(this->ones_.data());

    thrust::fill(
		 this->ones_.begin(), 
		 this->ones_.begin() + _batch_size,
		 1.f
		 );

  }

  //Calculate derivative for bias
  //dldw = alpha*(X)deltaT + beta*dldw
  //X: (_batch_size X input_dim)-matrix 
  //NOTE: X is (input_dim X _batch_size) for input data,
  // but (_batch_size X input_dim) for hidden node data!
  //delta: (_batch_size X this->dim_)-matrix
  //dldw: (input_dim X this->dim_)-matrix (just like w)
  //NOTE: cuBLAS uses column-major order!
  //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
  //NOTE: There seems to be an error in the documentation -
  //ldb depends on transb, not transa (obviously)
  cstatus_dense = cublasSgemm(
			      this->NeuralNet->get_dense_handle(), //handle
			      CUBLAS_OP_N, //transa
			      CUBLAS_OP_N, //transb
			      1, //m
			      this->dim_, //n
			      _batch_size, //k           
			      &alpha, //alpha
			      this->ones_ptr_,//A 
			      1, //lda
			      this->delta_ptr,//B
			      _batch_size, //ldb
			      &beta, //beta
			      dldw, //C 
			      1//ldc
			      );

  if (cstatus_dense != CUBLAS_STATUS_SUCCESS) 
    throw std::runtime_error("Something went wrong during cuBLAS operation on hidden node data while calculating derivatives for the biases!");

  //Apply regulariser
  this->regulariser_->drdw(
			  this->NeuralNet->get_dense_handle(),
			  this->get_num_weights_required(),
			  static_cast<float>(_batch_size),
			  this->W,
			  _dLdw
			  );
    
}
