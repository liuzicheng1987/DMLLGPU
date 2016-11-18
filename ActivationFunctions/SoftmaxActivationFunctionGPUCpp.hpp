class SoftmaxActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {

private:
  
  std::int32_t num_vars_;//Number of discrete variables

  std::int32_t num_states_per_var_;//Number of states per discrete variable

  thrust::device_vector<float> linear_transformation_;//Stores the linear transformation 
                                                     //(necessary to avoid race conditions)

  float *linear_transformation_ptr_;//Pointer to linear_transformation_

  
public:

  SoftmaxActivationFunctionGPUCpp (
				    std::int32_t    _node_number,
				    std::int32_t    _num_vars,
				    std::int32_t    _num_states_per_var,
				    std::int32_t   *_input_nodes_fed_into_me_dense,
				    std::int32_t    _input_nodes_fed_into_me_dense_length,
				    std::int32_t   *_input_nodes_fed_into_me_sparse,
				    std::int32_t    _input_nodes_fed_into_me_sparse_length,
				    std::int32_t   *_hidden_nodes_fed_into_me,
				    std::int32_t    _hidden_nodes_fed_into_me_length,
				    std::int32_t    _i_share_weights_with,
				    bool            _no_weight_updates,
				    RegulariserCpp *_regulariser			   
				    ): ActivationFunctionGPUCpp(
								_node_number,
								_num_vars*_num_states_per_var,
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
    this->num_vars_ = _num_vars;
    this->num_states_per_var_ = _num_states_per_var;
    
  
  };
	
  ~SoftmaxActivationFunctionGPUCpp() {};

  void forward_propagation (
			    const std::int32_t            _batch_size,
			    const std::int32_t            _dim,
			    const float                  *_bias,
			    thrust::device_vector<float> &_output
			   ) {

  if (static_cast<std::int32_t>(this->linear_transformation_.size()) < this->dim_*_batch_size) {
    
    this->linear_transformation_.resize(this->dim_*_batch_size);
    this->linear_transformation_ptr_ = 
      thrust::raw_pointer_cast(this->linear_transformation_.data());

  }

  thrust::copy(
	       _output.begin(),
	       _output.begin() + _batch_size*_dim,
	       this->linear_transformation_.begin()
	       );
      
  thrust::transform(
		    thrust::make_counting_iterator(0),
		    thrust::make_counting_iterator(0) + _batch_size*_dim,
		    _output.begin(),
		    ActivationFunctions::SoftmaxForwardPropagation(
								   _bias,
								   _batch_size,
								   this->num_vars_,
								   this->num_states_per_var_,
								   this->linear_transformation_ptr_
								   )
		    );
       
  }


  void backpropagation (
			const std::int32_t            _batch_size,
			const std::int32_t            _dim,
			thrust::device_vector<float> &_output,
			thrust::device_vector<float> &_delta
			) {

    thrust::transform(
		      _output.begin(),
		      _output.begin() + _batch_size*_dim,
		      _delta.begin(),	      
		      _delta.begin(),
		      ActivationFunctions::SoftmaxBackpropagation()
		      );    
    
  }
  
};
