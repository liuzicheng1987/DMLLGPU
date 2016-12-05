DropoutCpp::DropoutCpp(
		       std::int32_t    _node_number,
		       float           _dropout_probability,
		       std::int32_t    _numbers_per_kernel,
		       std::int32_t    _num_kernels,
		       std::int32_t   *_hidden_nodes_fed_into_me, 
		       std::int32_t    _hidden_nodes_fed_into_me_length 
		       ): NeuralNetworkNodeGPUCpp (
						   _node_number,
						   0,
						   nullptr, 
						   0,
						   nullptr, 
						   0, 
						   _hidden_nodes_fed_into_me, 
						   _hidden_nodes_fed_into_me_length, 
						   -1, 
						   false,
						   nullptr
						   ) {

  //Initialise values
  this->skip_ = 0;
  this->dropout_probability_ = _dropout_probability;
  this->numbers_per_kernel_ = _numbers_per_kernel;
  this->num_kernels_ = _num_kernels;

  this->random_numbers_ = thrust::device_vector<float>(
						       this->numbers_per_kernel_
						       *this->num_kernels_
						       );

  this->random_numbers_ptr_ = thrust::raw_pointer_cast(
						       this->random_numbers_.data()
						       );

  thrust::for_each(
		   thrust::make_counting_iterator(0),
		   thrust::make_counting_iterator(0)
		   + this->num_kernels_,
		   DropoutFunctors::GenerateRandomNumbers(
							  this->numbers_per_kernel_,
							  this->random_seed_(),
							  this->random_numbers_ptr_
							  )
		   );

}

DropoutCpp::~DropoutCpp() {};

std::int32_t DropoutCpp::get_num_weights_required() {

  //Number of weights required is always 0, but we use
  //this function to calculate the dimensionality
  this->dim_ = 0;

  for (auto node: this->hidden_nodes_fed_into_me_ptr)
    this->dim_ += node->get_dim();

  return 0; 

};

void DropoutCpp::calc_output(
			     const std::int32_t _batch_num, 
			     const std::int32_t _batch_size
			     ) {
  
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

  std::int32_t input_dim;//Dimension of input nodes, for convenience

  std::int32_t output_begin = 0;//Marks beginning of output, note that output is in column-major order

  if (this->NeuralNet->get_sample() == true) {

    //Transform hidden nodes
    //Calculate derivatives for hidden nodes
    for (auto node: this->hidden_nodes_fed_into_me_ptr) {

      input_dim = node->get_dim();//For convenience

      //Generate random numbers, if necessary
      if (
	  this->skip_ + input_dim*_batch_size > 
	  this->numbers_per_kernel_*this->num_kernels_
	  ) {
      
	thrust::for_each(
			 thrust::make_counting_iterator(0),
			 thrust::make_counting_iterator(0)
			 + this->num_kernels_,
			 DropoutFunctors::GenerateRandomNumbers(
								this->numbers_per_kernel_,
								this->random_seed_(),
								this->random_numbers_ptr_
								)
			 );
	
	this->skip_ = 0;
      
      }
    
      //Apply dropout
      thrust::transform(
			thrust::make_counting_iterator(this->skip_),
			thrust::make_counting_iterator(this->skip_)
			+ input_dim*_batch_size,
			node->get_output().begin(),
			this->output.begin() + output_begin,
			DropoutFunctors::StandardDropout(
							 this->dropout_probability_,
							 this->random_numbers_ptr_
							 )
			);

    }

    this->skip_ += input_dim*_batch_size;
    output_begin += input_dim*_batch_size;
  
  } else {//If sample is not true
    
    for (auto node: this->hidden_nodes_fed_into_me_ptr) {

      input_dim = node->get_dim();//For convenience

      //Multiply by dropout probability
      thrust::transform(
			thrust::make_constant_iterator(this->dropout_probability_),
			thrust::make_constant_iterator(this->dropout_probability_)
			+ input_dim*_batch_size,
			node->get_output().begin(),
			this->output.begin() + output_begin,
		        thrust::multiplies<float>()
			);

      output_begin += input_dim*_batch_size;
    
    }

  }

  
}

void DropoutCpp::calc_delta(std::int32_t _batch_size) {

  std::int32_t input_dim;//Dimension of input nodes, for convenience

  std::int32_t output_begin = 0;//Marks beginning of output, note that output is in column-major order

  if (this->NeuralNet->get_sample() == true) {
  
    for (auto node: this->hidden_nodes_fed_into_me_ptr) {

      input_dim = node->get_dim();//For convenience

      thrust::for_each(
		       thrust::make_zip_iterator(
						 thrust::make_tuple(
								    this->output.begin()
								    + output_begin,
								    this->delta.begin()
								    + output_begin,
								    node->get_delta().begin()
								    )
						 ),
		       thrust::make_zip_iterator(
						 thrust::make_tuple(
								    this->output.begin()
								    + output_begin
								    + input_dim*_batch_size,
								    this->delta.begin()
								    + output_begin
								    + input_dim*_batch_size,
								    node->get_delta().begin()
								    + input_dim*_batch_size
								    )
						 ),
		       DropoutFunctors::DropoutCalcDelta() 
		       );

      output_begin += input_dim*_batch_size;

    }

  } else {//If sample is not true

    for (auto node: this->hidden_nodes_fed_into_me_ptr) {

      input_dim = node->get_dim();//For convenience
      
      //Multiply by dropout probability
      thrust::transform(
			thrust::make_constant_iterator(this->dropout_probability_),
			thrust::make_constant_iterator(this->dropout_probability_)
			+ input_dim*_batch_size,
			this->delta.begin() + output_begin,
			node->get_delta().begin(),
		        thrust::multiplies<float>()
			);

      output_begin += input_dim*_batch_size;

    }

  }

}
