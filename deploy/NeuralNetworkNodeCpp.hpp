class NeuralNetworkCpp;


class NeuralNetworkNodeCpp {
	
  friend class NeuralNetworkCpp;
	
protected:

  std::int32_t dim_;//Number of dimensions of this nodes' output
	
  std::int32_t node_number; //Node number of this node

  std::int32_t i_share_weights_with; //Node number of node that this node shares weights with

  std::vector<std::int32_t> input_nodes_fed_into_me_dense;//Integer signifying matrix of dense input matrix fed into this node
  std::vector<std::int32_t> input_nodes_fed_into_me_sparse;//Integer signifying matrix of dense input matrix fed into this node

  std::int32_t num_input_nodes_cumulative;//Accumulated number of input nodes as signified by the value dim in DenseMatrix and CSRMatrix
  
  std::vector<std::int32_t> hidden_nodes_fed_into_me;//Node numbers of hidden nodes fed into this node
  std::vector<NeuralNetworkNodeCpp*> hidden_nodes_fed_into_me_ptr;//Pointers to hidden nodes fed into this node

  std::vector<float> output;//Where the node stores its output

  float *output_ptr;//For convenience: Pointer to output

  const float *W;//Pointer to the weights for this neural network node (all weights are kept by the NeuralNetwork object)

  NeuralNetworkCpp *NeuralNet;//Pointer to the neural network that containts this node

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
  //Calculate the output of the node
  virtual void calc_output(
			   const std::int32_t _batch_num,
			   const std::int32_t _batch_size
			   ) {};	 
	
  //A bunch of getters
  std::vector<NeuralNetworkNodeCpp*>& get_hidden_nodes_fed_into_me_ptr() {
    return this->hidden_nodes_fed_into_me_ptr;
  };

  std::vector<float>& get_output() {
    return this->output;
  };

  float* get_output_ptr() {
    return this->output_ptr;
  };

  std::int32_t get_dim() {
    return this->dim_;
  };	

		
};
