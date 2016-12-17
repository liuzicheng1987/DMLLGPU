class ActivationFunctionCpp: public NeuralNetworkNodeCpp {

public:

  ActivationFunctionCpp(
			   std::int32_t    _node_number,
			   std::int32_t    _dim,
			   std::int32_t   *_input_nodes_fed_into_me_dense,
			   std::int32_t    _input_nodes_fed_into_me_dense_length,
			   std::int32_t   *_input_nodes_fed_into_me_sparse,
			   std::int32_t    _input_nodes_fed_into_me_sparse_length,
			   std::int32_t   *_hidden_nodes_fed_into_me,
			   std::int32_t    _hidden_nodes_fed_into_me_length,
			   std::int32_t    _i_share_weights_with
			   );

  ~ActivationFunctionCpp();

  //Used to get the number of weights required
  //(very important to finalise the neural network)
  std::int32_t get_num_weights_required();

  //Calculate the output of the node
  void calc_output(
		   const std::int32_t _batch_num,
		   const std::int32_t _batch_size
		   );

  //Virtual function meant to be overridden
  //Adds the bias and applies the activation function during forward propagation
  virtual void forward_propagation (
				    const std::int32_t  _batch_size,
				    const std::int32_t  _dim,
				    const float        *_bias,
				    std::vector<float> &_output
				    ) {};

};
