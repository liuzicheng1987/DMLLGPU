class AggregationCpp: public NeuralNetworkNodeCpp {

private:

  //Index signifying input network
  std::int32_t input_network_;

  //Pointer to input network
  NeuralNetworkCpp *input_network_ptr_;

  //Pointer to ones_
  float *ones_ptr_;

  //Pointer to relational network containing
  //aggregation
  RelationalNetworkCpp *relational_net_;

protected:

  //One, if element is included in aggregation,
  //zero otherwise
  thrust::device_vector<float> included_in_aggregation_;

public:

  AggregationCpp(
		 std::int32_t _node_number, 
		 std::int32_t _dim,
		 std::int32_t _input_network,			       
		 std::int32_t _i_share_weights_with, 
		 bool         _no_weight_updates
		 );

  ~AggregationCpp();

  //Used to get the number of weights required
  //(very important to finalise the neural network)
  virtual std::int32_t get_num_weights_required() {return 0;};

  //Calculate the output of the node
  virtual void calc_output(
			   const std::int32_t _batch_num,
			   const std::int32_t _batch_size
			   ) {};

  //Calculate the delta of the node (which is used for backpropagation)
  virtual void calc_delta(std::int32_t _batch_size) {};

  //Calculate the derivatives for the individual weights
  virtual void calc_dLdw(
			 float              *_dLdw,
			 const std::int32_t  _batch_num,
			 const std::int32_t  _batch_size
			 ) {};


};
