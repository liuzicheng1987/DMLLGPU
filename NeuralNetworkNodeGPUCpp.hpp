class NeuralNetworkGPUCpp;


class NeuralNetworkNodeGPUCpp {
	
  friend class NeuralNetworkGPUCpp;
	
protected:

  std::int32_t dim_;//Number of dimensions of this nodes' output
	
  std::int32_t node_number; //Node number of this node

  std::int32_t i_share_weights_with; //Node number of node that this node shares weights with

  bool no_weight_updates; //Sometimes we want parameters to stay constants during training. num_samplesn this case, we set this variable to true.

  std::vector<std::int32_t> input_nodes_fed_into_me_dense;//Integer signifying matrix of dense input matrix fed into this node
  std::vector<std::int32_t> input_nodes_fed_into_me_sparse;//Integer signifying matrix of dense input matrix fed into this node

  std::int32_t num_input_nodes_cumulative;//Accumulated number of input nodes as signified by the value dim in DenseMatrix and CSRMatrix
  
  std::vector<std::int32_t> hidden_nodes_fed_into_me;//Node numbers of hidden nodes fed into this node
  std::vector<NeuralNetworkNodeGPUCpp*> hidden_nodes_fed_into_me_ptr;//Pointers to hidden nodes fed into this node

  thrust::device_vector<float> output;//Where the node stores its output
  thrust::device_vector<float> delta;//Where the node stores the derivatives from the backpropagation procedure

  float *output_ptr;//For convenience: Pointer to output
  float *delta_ptr;//For convenience: Pointer to delta

  const float *W;//Pointer to the weights for this neural network node (all weights are kept by the NeuralNetwork object)

  NeuralNetworkGPUCpp *NeuralNet;//Pointer to the neural network that containts this node

  RegulariserCpp *regulariser_;//Pointer to the regulariser

public:
			
  NeuralNetworkNodeGPUCpp (
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
			   );
	
  virtual ~NeuralNetworkNodeGPUCpp();

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

  //Calculate the delta of the node (which is used for backpropagation)
  virtual void calc_delta(std::int32_t _batch_size) {};

  //Calculate the derivatives for the individual weights
  virtual void calc_dLdw(
			 float *_dLdw,
			 const std::int32_t _batch_num,
			 const std::int32_t _batch_size
			 ) {};	
	
  //Neural network nodes also need a set_params() and get_params() function! This is very helpful for pretraining or duplicating parts of neural networks.
	
  //set_params sets the weights
  /*void set_params(float *_W, std::int32_t _lengthW) {
				
    if (_lengthW != this->NumWeightsRequired) throw std::invalid_argument("_length of provided W does not match lengthW!");
    if (this->W == NULL) throw std::invalid_argument("Neural network not finalised!");
		
    for (std::int32_t i=0; i<_lengthW; ++i) this->W[i] = _W[i];
		
    } */	
	
  //get_params gets the weights
  /*void get_params(float *_W, std::int32_t _lengthW) {
				
    if (_lengthW != this->NumWeightsRequired) throw std::invalid_argument("_length of provided W does not match lengthW!");
    if (this->W  == NULL) throw std::invalid_argument("Neural network not finalised!");
		
    for (std::int32_t i=0; i<_lengthW; ++i) _W[i] = this->W[i];
		
    } */		 
	
  //A bunch of getters
  std::vector<NeuralNetworkNodeGPUCpp*>& get_hidden_nodes_fed_into_me_ptr() {
    return this->hidden_nodes_fed_into_me_ptr;
  };

  thrust::device_vector<float>& get_output() {
    return this->output;
  };

  float* get_output_ptr() {
    return this->output_ptr;
  };

  thrust::device_vector<float>& get_delta() {
    return this->delta;
  };	

  float* get_delta_ptr() {
    return this->delta_ptr;
  };

  std::int32_t get_dim() {
    return this->dim_;
  };	

		
};
