class NeuralNetworkGPUCpp;


class NeuralNetworkNodeGPUCpp {
	
  friend class NeuralNetworkGPUCpp;
	
protected:
	
  std::int32_t NodeNumber; //Node number of this node
  std::int32_t IShareWeightsWith; //Node number of node that this node shares weight with
  bool NoWeightUpdates; //Sometimes we want parameters to stay constants during training. In this case, we set this variable to true.

  std::vector<std::int32_t> InputNodesFedIntoMeDense;//Integer signifying matrix of dense input matrix fed into this node
  std::vector<std::int32_t> InputNodesFedIntoMeSparse;//Integer signifying matrix of dense input matrix fed into this node
  std::int32_t NumInputNodesCumulative;//Accumulated number of input nodes as signified by the value J in DenseInputStruct and SparseInputStruct
  
  std::vector<std::int32_t> HiddenNodesFedIntoMe;//Node numbers of hidden nodes fed into this node
  std::vector<NeuralNetworkNodeGPUCpp*> HiddenNodesFedIntoMePtr;//Pointers to hidden nodes fed into this node

  thrust::device_vector<float> output;//Where the node stores its output
  thrust::device_vector<float> delta;//Where the node stores the derivatives from the backpropagation procedure

  float *OutputPtr;//For convenience: Pointer to output
  float *DeltaPtr;//For convenience: Pointer to delta

  const float *W;//Pointer to the weights for this neural network node (all weights are kept by the NeuralNetwork object)

  NeuralNetworkGPUCpp *NeuralNet;//Pointer to the neural network that containts this node

public:
			
  NeuralNetworkNodeGPUCpp (std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates);
	
  virtual ~NeuralNetworkNodeGPUCpp();

  virtual std::int32_t get_num_weights_required() {return 0;};//Used to get the number of weights required (very important to finalise the neural network)
	
  //All of these functions are overridden
  virtual void calc_output(const std::int32_t _BatchNum, const std::int32_t _BatchSize) {};//Calculate the output of the node	   
  virtual void calc_delta() {};//Calculate the delta of the node (which is used for backpropagation)
  virtual void calc_dLdw(float *_dLdw, const std::int32_t _BatchNum, const std::int32_t _BatchSize) {};//Calculate the derivatives for the individual weights	
	
  //Neural network nodes also need a set_params() and get_params() function! This is very helpful for pretraining or duplicating parts of neural networks.
	
  //set_params sets the weights
  /*void set_params(float *_W, std::int32_t _lengthW) {
				
    if (_lengthW != this->NumWeightsRequired) throw std::invalid_argument("Length of provided W does not match lengthW!");
    if (this->W == NULL) throw std::invalid_argument("Neural network not finalised!");
		
    for (std::int32_t i=0; i<_lengthW; ++i) this->W[i] = _W[i];
		
    } */	
	
  //get_params gets the weights
  /*void get_params(float *_W, std::int32_t _lengthW) {
				
    if (_lengthW != this->NumWeightsRequired) throw std::invalid_argument("Length of provided W does not match lengthW!");
    if (this->W  == NULL) throw std::invalid_argument("Neural network not finalised!");
		
    for (std::int32_t i=0; i<_lengthW; ++i) _W[i] = this->W[i];
		
    } */		 
	
  //A bunch of getters
  std::vector<NeuralNetworkNodeGPUCpp*>& get_hidden_nodes_fed_into_me_ptr() {return this->HiddenNodesFedIntoMePtr;};
  thrust::device_vector<float>& get_output() {return this->output;};
  thrust::device_vector<float>& get_delta() {return this->delta;};	
		
};
