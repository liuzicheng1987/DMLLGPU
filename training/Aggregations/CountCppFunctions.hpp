CountCpp::CountCpp(
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
CountCpp::~CountCpp()
{
}

//Calculate the output of the node
void CountCpp::calc_output(
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

    //batch_size_input, expressed as a float value
    float batch_size_input_float;

    //alpha and beta are needed for the cuBLAS operation
    const float alpha = 1.f;

    const float beta = 0.f;

    for (std::int32_t i = 0; i < _batch_size; ++i)
    {

	batch_size_input_float = static_cast<float>(this->batch_size_aggregation_considering_timestamps_[i]);

	//If join_keys_left[i] (the batch number) is negative
	//or greater than the number of batches or there are no samples to aggregate, then
	//we do not need to apply any aggregations and we default
	//to zero (in method initialise())
	if (batch_size_input > 0 && join_keys_left[i] >= 0 && join_keys_left[i] < this->input_network_ptr_->get_num_batches())
	{

	    //Resize this->included_in_aggregation_, if necessary
	    if (static_cast<std::int32_t>(this->included_in_aggregation_.size()) != this->dim_)
	    {

		this->included_in_aggregation_.resize(this->dim_);

		thrust::fill(this->included_in_aggregation_.begin(),
			     this->included_in_aggregation_.end(),
			     1.f);

		this->included_in_aggregation_ptr_ = thrust::raw_pointer_cast(this->included_in_aggregation_.data());
	    }

	    //Get aggregation_output, for convenience
	    aggregation_output =
		this->input_network_ptr_->get_output_nodes()[0]->get_output_ptr();

	    //Apply count aggregation
	    //output_ptr = alpha*included_in_aggregation_(aggregation_output) + beta*output_ptr
	    //included_in_aggregation_: (this->dim_ X 1)-matrix
	    //batch_size_input_float: (1 X 1)-matrix
	    //output_ptr_ + i: (this->dim_ X 1)-matrix
	    //NOTE: cuBLAS uses column-major order!
	    //http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
	    //NOTE: There seems to be an error in the documentation -
	    //ldb depends on transb, not transa (obviously)
	    cstatus = cublasSgemm(this->neural_net_->get_dense_handle(), //handle
				  CUBLAS_OP_N,				 //transa
				  CUBLAS_OP_N,				 //transb
				  this->dim_,				 //m
				  1,					 //n
				  1,					 //k
				  &alpha,				 //alpha
				  this->included_in_aggregation_ptr_,    //A
				  1,					 //lda
				  aggregation_output,			 //B
				  batch_size_input,			 //ldb
				  &beta,				 //beta
				  this->output_ptr_ + i,		 //C
				  _batch_size				 //ldc
				  );

	    if (cstatus != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error(
		    "Something went wrong during cuBLAS operation on sum aggregation!");
	}
    }
}
