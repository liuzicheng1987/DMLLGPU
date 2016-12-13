class LinearActivationFunctionCpp: public ActivationFunctionCpp {

public:

  LinearActivationFunctionCpp (
				  std::int32_t    _node_number,
				  std::int32_t    _dim,
				  std::int32_t   *_input_nodes_fed_into_me_dense,
				  std::int32_t    _input_nodes_fed_into_me_dense_length,
				  std::int32_t   *_input_nodes_fed_into_me_sparse,
				  std::int32_t    _input_nodes_fed_into_me_sparse_length,
				  std::int32_t   *_hidden_nodes_fed_into_me,
				  std::int32_t    _hidden_nodes_fed_into_me_length,
				  std::int32_t    _i_share_weights_with
				  ): ActivationFunctionCpp(
							   _node_number,
							   _dim,
							   _input_nodes_fed_into_me_dense,
							   _input_nodes_fed_into_me_dense_length,
							   _input_nodes_fed_into_me_sparse,
							   _input_nodes_fed_into_me_sparse_length,
							   _hidden_nodes_fed_into_me,
							   _hidden_nodes_fed_into_me_length,
							   _i_share_weights_with   
							   ) {};
	
	
  ~LinearActivationFunctionCpp() {};


  void forward_propagation(
			   const std::int32_t  _batch_size,
			   const std::int32_t  _dim,
			   const float        *_bias,
			   std::vector<float> &_output
			   ) {

    //Add bias
    for (std::int32_t i = 0; i < _dim; ++i)
      for (std::int32_t j = 0; j < _batch_size; ++j)
	_output[_batch_size*i + j] += _bias[i];

  }

};
