class NeuralNetworkCpp;


class NeuralNetworkNodeCpp {
	
  friend class NeuralNetworkCpp;
	
protected:

  //Number of dimensions of this nodes' output
  std::int32_t dim_;
	
  //Node number of this node
  std::int32_t node_number_;

  //Node number of node that this node shares weights with
  std::int32_t i_share_weights_with_;

  //Integer signifying matrix of dense input matrix fed into this node
  std::vector<std::int32_t> input_nodes_fed_into_me_dense_;

  //Integer signifying matrix of dense input matrix fed into this node
  std::vector<std::int32_t> input_nodes_fed_into_me_sparse_;

  //Accumulated number of input nodes as signified by the value dim in DenseMatrix and CSRMatrix
  std::int32_t num_input_nodes_cumulative_;
  
  //Node numbers of hidden nodes fed into this node
  std::vector<std::int32_t> hidden_nodes_fed_into_me_;

  //Pointers to hidden nodes fed into this node
  std::vector<NeuralNetworkNodeCpp*> hidden_nodes_fed_into_me_ptr_;

  //Where the node stores its output_
  std::vector<float> output_;

  //For convenience: Pointer to output
  float *output_ptr_;

  //Pointer to the weights for this neural network node (all weights are kept by the NeuralNetwork object)
  const float *W_;

  //Pointer to the neural network that containts this node
  NeuralNetworkCpp *neural_net_;

public:
			
  NeuralNetworkNodeCpp (
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
	
  virtual ~NeuralNetworkNodeCpp();

  //Used to get the number of weights required (very important to finalise the neural network)
  virtual std::int32_t get_num_weights_required() {
    return 0;
  };
	
  //All of these functions are overridden
  //Calculate the output_ of the node
  virtual void calc_output(
			   const std::int32_t _batch_num,
			   const std::int32_t _batch_size
			   ) {};	 
	
  //A bunch of getters
  std::vector<NeuralNetworkNodeCpp*>& get_hidden_nodes_fed_into_me_ptr() {
    return this->hidden_nodes_fed_into_me_ptr_;
  };

  std::vector<float>& get_output() {
    return this->output_;
  };

  float* get_output_ptr() {
    return this->output_ptr_;
  };

  std::int32_t get_dim() {
    return this->dim_;
  };	

		
};
