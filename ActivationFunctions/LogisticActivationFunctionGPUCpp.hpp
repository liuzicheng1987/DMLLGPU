class LogisticActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {

public:

  LogisticActivationFunctionGPUCpp (
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
				    ): ActivationFunctionGPUCpp(
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
								) {};
	
  ~LogisticActivationFunctionGPUCpp() {};

  void forward_propagation (
			    const std::int32_t            _batch_size,
			    const std::int32_t            _dim,
			    const float                  *_bias,
			    thrust::device_vector<float> &_output
			   ) {
    
    thrust::transform(
		      _output.begin(),
		      _output.begin() + _batch_size*_dim,
		      thrust::make_counting_iterator(0),
		      _output.begin(),
		      ActivationFunctions::LogisticForwardPropagation(
								      _bias, 
								      _batch_size
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
		      ActivationFunctions::LogisticBackpropagation()
		      );    
    
  }
  
};
