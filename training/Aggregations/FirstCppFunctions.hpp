FirstCpp::FirstCpp(
    std::int32_t _node_number,
    std::int32_t _dim,
    std::int32_t _input_network,
    bool _use_timestamps) : AggregationCpp(_node_number,
					   _dim,
					   _input_network,
					   _use_timestamps,
					   -1,   //no weight sharing
					   false //weight updates needed to
						 //    calculate derivatives for input networks
					   ){};
FirstCpp::~FirstCpp()
{
}

//Calculate the output of the node
void FirstCpp::calc_output(
    const std::int32_t _batch_num,
    const std::int32_t _batch_size)
{

    this->initialise(_batch_size);

    //For convenience
    std::int32_t *join_keys_left =
	this->relational_net_->get_join_keys_left_ptr(this->input_network_);

    //batch_size_aggregation_considering_timestamps_ denotes the number
    //of elements included in aggregation, after considering the
    //timestamps that are smaller than the output time stamp.
    //It is generally not identical to _batch_size
    this->calc_batch_size_aggregation_considering_timestamps(join_keys_left,
							     _batch_num, _batch_size);

    //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
    cublasStatus_t cstatus;

    //Output of the aggregation functions,
    //for convenience
    float *aggregation_output;

    //For convenience
    std::int32_t batch_size_input;

    //alpha and beta are needed for the cuBLAS operation
    const float alpha = 1.f;

    const float beta = 0.f;

    for (std::int32_t i = 0; i < _batch_size; ++i)
    {

	//This is needed, so ScatterCpp knows which sample to scatter
	this->relational_net_->set_current_sample(i);

	batch_size_input = this->batch_size_aggregation_considering_timestamps_[i];

	//If join_keys_left[i] (the batch number) is negative
	//or greater than the number of batches or there are no samples to aggregate, then
	//we do not need to apply any aggregations and we default
	//to zero (in method initialise())
	if (batch_size_input > 0 && join_keys_left[i] >= 0 && join_keys_left[i] < this->input_network_ptr_->get_num_batches())
	{

	    //_batch_num_input = join_keys_left[i]
	    //_batch_size_input = his->batch_size_aggregation_considering_timestamps_[i]
	    this->apply_forward_propagation_in_input_network(join_keys_left[i], batch_size_input);

	    //Resize this->included_in_aggregation_, if necessary
	    if (static_cast<std::int32_t>(this->included_in_aggregation_.size()) != 1)
	    {

		this->included_in_aggregation_.resize(1);

		thrust::fill(this->included_in_aggregation_.begin(),
			     this->included_in_aggregation_.end(),
			     1.f);

		this->included_in_aggregation_ptr_ = thrust::raw_pointer_cast(this->included_in_aggregation_.data());
	    }

	    //Get aggregation_output, for convenience
	    aggregation_output =
		this->input_network_ptr_->get_output_nodes()[0]->get_output_ptr();

	    //Apply aggregation
	    //output_ptr = alpha*included_in_aggregation_(aggregation_output) + beta*output_ptr
	    //included_in_aggregation_: (1 X 1)-matrix
	    //aggregation_output: (1 X this->dim_)-matrix
	    //output_ptr_ + i: (1 X this->dim_)-matrix
	    //NOTE: cuBLAS uses column-major order!
	    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
	    //NOTE: There seems to be an error in the documentation -
	    //ldb depends on transb, not transa (obviously)
	    cstatus = cublasSgemm(this->neural_net_->get_dense_handle(), //handle
				  CUBLAS_OP_N,				 //transa
				  CUBLAS_OP_N,				 //transb
				  1,					 //m
				  this->dim_,				 //n
				  1,					 //k
				  &alpha,				 //alpha
				  this->included_in_aggregation_ptr_,    //A
				  1,					 //lda
				  aggregation_output,			 //B
				  batch_size_input,			 //ldb - note that there is a stride
				  &beta,				 //beta
				  this->output_ptr_ + i,		 //C
				  _batch_size				 //ldc - note that there is a stride
				  );
	    //NOTE: Since output needs to be in column-major order as well,
	    //we set ldb to batch_size_input and ldc to _batch_size, which results in a stride.

	    if (cstatus != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error(
		    "Something went wrong during cuBLAS operation on first aggregation!");
	}
    }
}

//Calculate the delta of the node (which is used for backpropagation)
void FirstCpp::calc_delta(std::int32_t _batch_size)
{

    //For convenience
    std::int32_t *join_keys_left =
	this->relational_net_->get_join_keys_left_ptr(this->input_network_);

    //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
    cublasStatus_t cstatus;

    //alpha and beta are needed for the cuBLAS operation
    const float alpha = 1.f;

    const float beta = 0.f;

    //For convenience
    std::int32_t batch_size_input;

    float *delta_input_network;

    for (std::int32_t i = 0; i < _batch_size; ++i)
    {

	//This is needed, so ScatterCpp knows which sample to scatter
	this->relational_net_->set_current_sample(i);

	batch_size_input = this->batch_size_aggregation_considering_timestamps_[i];

	//If join_keys_left[i] (the batch number) is negative
	//or greater than the number of batches or there are no samples to aggregate, then
	//we do not need to calculate any delta
	if (join_keys_left[i] >= 0 && join_keys_left[i] < this->input_network_ptr_->get_num_batches() && batch_size_input > 0)
	{

	    //_batch_num_input = join_keys_left[i]
	    //_batch_size_input = his->batch_size_aggregation_considering_timestamps_[i]
	    this->apply_forward_propagation_in_input_network(join_keys_left[i],
							     batch_size_input);

	    //Sets all deltas in input network to zero
	    this->init_delta_in_input_network();

	    //For convenience
	    delta_input_network = this->input_network_ptr_->get_output_nodes()[0]->get_delta_ptr();

	    //Transfer delta to input network
	    //delta_input_network = alpha*included_in_aggregation_(this->delta_ptr_ + i) + beta*delta_input_network
	    //included_in_aggregation_: (1 X 1)-matrix
	    //this->delta_ptr_ + i: (1 X this->dim_)-matrix
	    //delta_input_network: (1 X this->dim_)-matrix
	    //NOTE: cuBLAS uses column-major order!
	    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
	    //NOTE: There seems to be an error in the documentation -
	    //ldb depends on transb, not transa (obviously)
	    cstatus = cublasSgemm(this->neural_net_->get_dense_handle(), //handle
				  CUBLAS_OP_N,				 //transa
				  CUBLAS_OP_N,				 //transb
				  1,					 //m
				  this->dim_,				 //n
				  1,					 //k
				  &alpha,				 //alpha
				  this->included_in_aggregation_ptr_,    //A
				  1,					 //lda
				  this->delta_ptr_ + i,			 //B
				  _batch_size,				 //ldb - note that there is a stride!
				  &beta,				 //beta
				  delta_input_network,			 //C
				  batch_size_input			 //ldc
				  );
	    //NOTE: Since output needs to be in column-major order as well,
	    //we set ldb to _batch_size and ldc to batch_size_input, which results in a stride.

	    if (cstatus != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error(
		    "Something went wrong during cuBLAS operation when calculating delta during first aggregation!");

	    //_batch_num_input = join_keys_left[i]
	    //_batch_size_input = this->batch_size_aggregation_considering_timestamps_[i]
	    this->backpropagate_and_calculate_dldw_in_input_network(join_keys_left[i],
								    batch_size_input);
	}
    }
}
