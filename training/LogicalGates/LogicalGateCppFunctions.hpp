LogicalGateCpp::LogicalGateCpp (
				std::int32_t  _node_number, 
				std::int32_t  _dim,
				std::int32_t *_hidden_nodes_fed_into_me, 
				std::int32_t  _hidden_nodes_fed_into_me_length
				): NeuralNetworkNodeCpp (
							    _node_number,
							    _dim,
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
  
  //Make sure that _hidden_nodes_fed_into_me_length is between 2 and 4
  if (
      _hidden_nodes_fed_into_me_length < 2 ||
      _hidden_nodes_fed_into_me_length > 4
      ) {
    
    throw std::invalid_argument("Number of hidden nodes fed into gate must be between 2 and 4!");
    
  }
      
  
}

LogicalGateCpp::~LogicalGateCpp() {};

std::int32_t LogicalGateCpp::get_num_weights_required() {
  
  //Logical gates have no weights, 
  //but we use this function to make sure all dimensions match!
  if (
      std::any_of(
		  this->hidden_nodes_fed_into_me_ptr.begin(), 
		  this->hidden_nodes_fed_into_me_ptr.end(), 
		  [this](NeuralNetworkNodeCpp* node) {
		    return node->get_dim() != this->dim_;
		  }
		  )
      ) 
    throw std::invalid_argument("Dimension of all gates fed into logical gate must match output dimension!");

  return 0; 

};

void LogicalGateCpp::calc_output(
				 const std::int32_t _batch_num, 
				 const std::int32_t _batch_size
				 ) {

  //Resize output and delta, if necessary
  //Output is stored in the NeuralNetworkNodeCpp base class and stores the output of this node
  if (static_cast<std::int32_t>(this->output.size()) < this->dim_*_batch_size) {
    
    //Resize output
    this->output.resize(this->dim_*_batch_size);
    this->output_ptr = thrust::raw_pointer_cast(this->output.data());
    
    //Resize delta
    this->delta.resize(this->dim_*_batch_size);
    this->delta_ptr = thrust::raw_pointer_cast(this->delta.data());
    
  }

  //Feeding input nodes into gates is not allowed
  //- we go straight to the hidden nodes.

  //Recall the following output function:
  //output: a_ + b_*(c_*input1 + d)*(c_*input2 + d)*...

  switch (this->hidden_nodes_fed_into_me_ptr.size()) {

  case 2:
    
    thrust::for_each(
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin(),
								  this->output.begin()
								  )
					       ),
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->output.begin()
								  + this->dim_*_batch_size
								  )
					       ),
		     LogicalGateFunctors::LogicalGateForwardPropagation2(
									 this->a_,
									 this->b_,
									 this->c_,
									 this->d_
									 )
		     );
    break;

  case 3:

    thrust::for_each(
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), 
								  this->output.begin()
								  )
					       ),
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->output.begin()
								  + this->dim_*_batch_size
								  )
					       ),
		     LogicalGateFunctors::LogicalGateForwardPropagation3(
									 this->a_,
									 this->b_,
									 this->c_,
									 this->d_
									 )
		     );

    break;
    
  case 4:
    
    thrust::for_each(
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[3]->get_output().begin(),								  
								  this->output.begin()
								  )
					       ),
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[3]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->output.begin()
								  + this->dim_*_batch_size
								  )
					       ),
		     LogicalGateFunctors::LogicalGateForwardPropagation4(
									 this->a_,
									 this->b_,
									 this->c_,
									 this->d_
									 )
		     );
    
    break;

  }

}

void LogicalGateCpp::calc_delta(std::int32_t _batch_size) {

  switch (this->hidden_nodes_fed_into_me_ptr.size()) {

  case 2:
    thrust::for_each(
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->delta.begin(),
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin(),								 
								  this->hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_delta().begin()
								  )
					       ),
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->delta.begin()
								  + this->dim_*_batch_size,						 
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[0]->get_delta().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_delta().begin()
								  + this->dim_*_batch_size						    
								  )
					       ),
		     LogicalGateFunctors::LogicalGateBackpropagation2(
								      this->a_,
								      this->b_,
								      this->c_,
								      this->d_
								      )
		     );
    break;
    
  case 3:
    thrust::for_each(
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->delta.begin(),
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin(),								 
								  this->hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(),
								  this->hidden_nodes_fed_into_me_ptr[2]->get_delta().begin()
								  )
					       ),
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->delta.begin()
								  + this->dim_*_batch_size,						 
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[0]->get_delta().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_delta().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[2]->get_delta().begin()
								  + this->dim_*_batch_size						    
								  )
					       ),
		     LogicalGateFunctors::LogicalGateBackpropagation3(
								      this->a_,
								      this->b_,
								      this->c_,
								      this->d_
								      )
		     );

    break;
    
  case 4:
    thrust::for_each(
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->delta.begin(),
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin(),
								  this->hidden_nodes_fed_into_me_ptr[3]->get_output().begin(),								  
								  this->hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(),
								  this->hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(),
								  this->hidden_nodes_fed_into_me_ptr[2]->get_delta().begin(),
								  this->hidden_nodes_fed_into_me_ptr[3]->get_delta().begin()
								  )
					       ),
		     thrust::make_zip_iterator(
					       thrust::make_tuple(
								  this->delta.begin()
								  + this->dim_*_batch_size,						 
								  this->hidden_nodes_fed_into_me_ptr[0]->get_output().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[2]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[3]->get_output().begin() 
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[0]->get_delta().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[1]->get_delta().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[2]->get_delta().begin()
								  + this->dim_*_batch_size,
								  this->hidden_nodes_fed_into_me_ptr[3]->get_delta().begin()
								  + this->dim_*_batch_size
								  )
					       ),
		     LogicalGateFunctors::LogicalGateBackpropagation4(
								      this->a_,
								      this->b_,
								      this->c_,
								      this->d_
								      )
		     );
    
    break;

  }

}
