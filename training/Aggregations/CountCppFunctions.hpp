CountCpp::CountCpp(
    std::int32_t _node_number,
    std::int32_t _input_network,
    bool _use_timestamps) : AggregationCpp(_node_number,
					   1,//_dim = 1
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

    //We need to transform the number of elements included in aggregation to a float and then
    //transfer them to the GPU
    std::vector<float> batch_size_input_float_host(this->batch_size_aggregation_considering_timestamps_.size());

    std::transform(this->batch_size_aggregation_considering_timestamps_.begin(),
		   this->batch_size_aggregation_considering_timestamps_.end(),
		   batch_size_input_float_host.begin(),
		   [](std::int32_t i) { return static_cast<float>(i); });

    thrust::device_vector<float>
	batch_size_input_float_device(batch_size_input_float_host.begin(), batch_size_input_float_host.end());

    //Also, we need a pointer to the counts on the GPU
    float *batch_size_input_float_device_ptr = thrust::raw_pointer_cast(batch_size_input_float_device.data());

    //cuBLAS status variable - so we can check whether the cuBLAS operations were successful
    cublasStatus_t cstatus;

    //alpha and beta are needed for the cuBLAS operation
    const float alpha = 1.f;

    const float beta = 0.f;

    //Resize this->included_in_aggregation_, if necessary
	//Note that in the case of CountCpp, this->dim_ always equals 1!
    if (static_cast<std::int32_t>(this->included_in_aggregation_.size()) != this->dim_)
    {

	this->included_in_aggregation_.resize(this->dim_);

	thrust::fill(this->included_in_aggregation_.begin(),
		     this->included_in_aggregation_.end(),
		     1.f);

	this->included_in_aggregation_ptr_ = thrust::raw_pointer_cast(this->included_in_aggregation_.data());
    }

    //Apply count aggregation
    //batch_size_input_float_device_ptr: (_batch_size X 1)-matrix
    //this->included_in_aggregation_: (1 X this->dim_)-matrix
    //output_ptr_: (_batch_size X this->dim_)-matrix
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
			  batch_size_input_float_device_ptr,     //A
			  _batch_size,				 //lda
			  this->included_in_aggregation_ptr_,    //B
			  1,					 //ldb
			  &beta,				 //beta
			  this->output_ptr_,			 //C
			  _batch_size				 //ldc
			  );

    if (cstatus != CUBLAS_STATUS_SUCCESS)
	throw std::runtime_error(
	    "Something went wrong during cuBLAS operation on count aggregation!");
}
