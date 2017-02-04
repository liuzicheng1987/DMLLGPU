ScatterCpp::ScatterCpp(
    std::int32_t _node_number,
    std::int32_t _dim,
    std::int32_t _target_node) : NeuralNetworkNodeCpp(_node_number,
						      _dim,
						      nullptr,
						      0,
						      nullptr,
						      0,
						      nullptr,
						      0,
						      -1,    //no weight sharing
						      false, //no weight updates
						      nullptr)
{

    this->target_node_ = _target_node;
};

ScatterCpp::~ScatterCpp()
{
}

//Calculate the output of the node
void ScatterCpp::calc_output(
    const std::int32_t _batch_num,
    const std::int32_t _batch_size)
{

    //Resize, if necessary
    if (static_cast<std::int32_t>(this->ones_.size()) < _batch_size)
    {

	//Resize output
	this->output_.resize(this->dim_ * _batch_size);
	this->output_ptr_ = thrust::raw_pointer_cast(this->output_.data());

	//Resize delta
	this->delta_.resize(this->dim_ * _batch_size);
	this->delta_ptr_ = thrust::raw_pointer_cast(this->delta_.data());

	//Resize ones
	this->ones_.resize(_batch_size);

	thrust::fill(this->ones_.begin(),
		     this->ones_.end(),
		     1.f);

	this->ones_ptr_ = thrust::raw_pointer_cast(this->ones_.data());
    }

    //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
    cublasStatus_t cstatus;

    //alpha and beta are needed for the cuBLAS operation
    const float alpha = 1.f;

    const float beta = 0.f;

    //Get target_node_output, for convenience
    float *target_node_output =
	this->relational_net_->get_output_network()->get_nodes()[this->target_node_]->get_output_ptr() + this->relational_net_->get_current_sample();

    //For convenienience
    std::int32_t batch_size_output_network = this->relational_net_->get_batch_size();

    //ones_ptr_: (_batch_size X 1)-matrix
    //target_node_output: (1 X this->dim_)-matrix (with stride)
    //this->output_ptr_: (_batch_size X this->dim_)-matrix
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus = cublasSgemm(this->neural_net_->get_dense_handle(), //handle
			  CUBLAS_OP_N,				 //transa
			  CUBLAS_OP_N,				 //transb
			  _batch_size,				 //m
			  this->dim_,				 //n
			  1,					 //k
			  &alpha,				 //alpha
			  this->ones_ptr_,			 //A
			  _batch_size,				 //lda
			  target_node_output,			 //B
			  batch_size_output_network,		 //ldb - note that there is a stride!
			  &beta,				 //beta
			  this->output_ptr_,			 //C
			  _batch_size				 //ldc
			  );

    //NOTE: Since delta in column-major order as well,
    //we set ldb to batch_size_output_network which results in a stride

    if (cstatus != CUBLAS_STATUS_SUCCESS)
	throw std::runtime_error(
	    "Something went wrong during cuBLAS operation while scattering!");
}

//Calculate the delta of the node (which is used for backpropagation)
void ScatterCpp::calc_delta(std::int32_t _batch_size)
{

    //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
    cublasStatus_t cstatus;

    //alpha and beta are needed for the cuBLAS operation
    const float alpha = 1.f;

    const float beta = 1.f;

    //Get target_node_output, for convenience
    float *target_node_delta =
	this->relational_net_->get_output_network()->get_nodes()[this->target_node_]->get_delta_ptr() + this->relational_net_->get_current_sample();

    //For convenienience
    std::int32_t batch_size_output_network = this->relational_net_->get_batch_size();

    //ones_ptr_: (1 X _batch_size)-matrix
    //this->delta_ptr_: (_batch_size X this->dim_)-matrix
    //target_node_delta: (1 X this->dim_)-matrix (with stride)
    //NOTE: cuBLAS uses column-major order!
    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
    //NOTE: There seems to be an error in the documentation -
    //ldb depends on transb, not transa (obviously)
    cstatus = cublasSgemm(this->neural_net_->get_dense_handle(), //handle
			  CUBLAS_OP_N,				 //transa
			  CUBLAS_OP_N,				 //transb
			  1,					 //m
			  this->dim_,				 //n
			  _batch_size,				 //k
			  &alpha,				 //alpha
			  this->ones_ptr_,			 //A
			  1,					 //lda
			  this->delta_ptr_,			 //B
			  _batch_size,				 //ldb
			  &beta,				 //beta
			  target_node_delta,			 //C
			  batch_size_output_network		 //ldc - note that there is a stride!
			  );

    if (cstatus != CUBLAS_STATUS_SUCCESS)
	throw std::runtime_error(
	    "Something went wrong during cuBLAS operation when calculating delta during scattering!");
}
