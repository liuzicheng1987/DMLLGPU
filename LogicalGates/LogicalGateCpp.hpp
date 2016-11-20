class LogicalGateCpp: public NeuralNetworkNodeGPUCpp {

protected:

  //The parameters a_, b_, c_ and d_ define the gate
  //output: a_ + b_*(c_*input1 + d)*(c_*input2 + d)*...
  //AND-Gate: a_ = 0.0, b_ = 1.0, c_ = 1.0, d_ = 0.0  
  float a_, b_, c_, d_;

public:

  LogicalGateCpp (
		  std::int32_t  _node_number, 
		  std::int32_t  _dim,
		  std::int32_t *_hidden_nodes_fed_into_me, 
		  std::int32_t  _hidden_nodes_fed_into_me_lengtho
		  );

  ~LogicalGateCpp ();

  //Used to get the number of weights required
  //(very important to finalise the neural network)
  std::int32_t get_num_weights_required();

  //Calculate the output of the node
  void calc_output(
		   const std::int32_t _batch_num,
		   const std::int32_t _batch_size
		   );

  //Calculate the delta of the node (which is used for backpropagation)
  void calc_delta(std::int32_t _batch_size);

};
