SumCpp::SumCpp(
		std::int32_t _node_number,
		std::int32_t _dim,
		std::int32_t _input_network
		) : AggregationCpp(
				_node_number,
				_dim,
				_input_network,
				-1,//no weight sharing
				true//no weight updates needed
		) {};

SumCpp::~SumCpp() {}

//Calculate the output of the node
void calc_output(
		const std::int32_t _batch_num,
		const std::int32_t _batch_size
		) {

	//Resize output and delta, if necessary
	//Both output and delta are stored in NeuralNetworkNodeCpp
	//base class!
	if (static_cast<std::int32_t>(
			this->output_.size()
			) != this->dim_*_batch_size) {

		//Resize output
	    this->output_.resize(this->dim_*_batch_size);
	    this->output_ptr_ = thrust::raw_pointer_cast(this->output_.data());

	    //Resize delta
	    this->delta_.resize(this->dim_*_batch_size);
	    this->delta_ptr_ = thrust::raw_pointer_cast(this->delta_.data());

	  }

	//batch_size_aggregation denotes the number
	//of elements included in aggregation,
	//which is generally not identical to _batch_size
	std::int32_t batch_size_aggregation;

	std::int32_t *join_keys_left =
			this->relational_net_->get_join_keys_left_ptr(
					this->input_network_
					);

	//cuBLAS status variable - so we can check whether the cuBLAS operations were successful
	cublasStatus_t cstatus;

	//Output of the aggregation functions,
	//for convenience
	float *aggregation_output;

	//alpha and beta are needed for the cuBLAS operation
	const float alpha = 1.f;

	const float beta = 0.f;

	for (std::int32_t i = 0; i < _batch_size; ++i) {

		//Get batch_size_aggregation
		batch_size_aggregation =
				this->relational_net_->get_batch_size_aggregation(
						  this->input_network_,
						  join_keys_left[i]//batch_num
						                 );

		  //Resize included_in_aggregation_, if necessary
		  if (static_cast<std::int32_t>(
				  this->included_in_aggregation_.size()
				  ) < batch_size_aggregation) {

			    this->included_in_aggregation_.resize(batch_size_aggregation);
			    this->included_in_aggregation_ptr_ = thrust::raw_pointer_cast(
			    		this->included_in_aggregation_.data()
			    		);

			    //This is temporary - remove once timestamps are properly implemented
			    thrust::fill(
					 this->included_in_aggregation_.begin(),
					 this->included_in_aggregation_.begin() + batch_size_aggregation,
					 1.f
					 );

		  }

		//Apply feedforward propagation in input network
		for (node: input_network_ptr_->get_nodes())
			node->calc_output(
					join_keys_left[i],//batch_num
					batch_size_aggregation//batch_size
			);

		//Get aggregation_output
		aggregation_output =
				this->input_network_ptr_->get_output_nodes(
						)[0]->get_output_ptr();

		//Apply sum aggregation
		//output_ptr = alpha*included_in_aggregation_(aggregation_output) + beta*output_ptr
		//ones: (1 X batch_size_aggregation)-matrix
		//aggregation_output: (batch_size_aggregation X this->dim_)-matrix
		//output_ptr_: (1 X this->dim_)-matrix
		//NOTE: cuBLAS uses column-major order!
		//http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
		//NOTE: There seems to be an error in the documentation -
		//ldb depends on transb, not transa (obviously)
		cstatus = cublasSgemm(
				this->relational_net_->get_dense_handle(), //handle
				CUBLAS_OP_N, //transa
				CUBLAS_OP_N, //transb
				1, //m
				this->dim_, //n
				batch_size_aggregation, //k
				&alpha, //alpha
				this->included_in_aggregation_ptr_,//A
				1, //lda
				aggregation_output,//B
				batch_size_aggregation, //ldb
				&beta, //beta
				this->output_ptr_ + this->dim_*i, //C
				1//ldc
		);

		if (cstatus != CUBLAS_STATUS_SUCCESS)
			throw std::runtime_error("Something went wrong during cuBLAS operation on sum aggregation!");

	}

}

//Calculate the delta of the node (which is used for backpropagation)
void calc_delta(std::int32_t _batch_size){



}
