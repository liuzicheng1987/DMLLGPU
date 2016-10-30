class ActivationFunctionGPUCpp: public NeuralNetworkNodeGPUCpp {

public:

  ActivationFunctionGPUCpp(
			   std::int32_t  _NodeNumber,
			   std::int32_t *_InputNodesFedIntoMeDense,
			   std::int32_t  _InputNodesFedIntoMeDenseLength,
			   std::int32_t *_InputNodesFedIntoMeSparse,
			   std::int32_t  _InputNodesFedIntoMeSparseLength,
			   std::int32_t *_HiddenNodesFedIntoMe,
			   std::int32_t  _HiddenNodesFedIntoMeLength,
			   std::int32_t  _IShareWeightsWith,
			   bool          _NoWeightUpdates
			   );

  ~ActivationFunctionGPUCpp();

  //Forward propagation and backpropagation
  //These are function pointers that call on the
  //appropriate kernel functions to do the actual processing

  //Function pointer, which contains the activation function
  std::function<void(
		     const float                   *W,
		     NeuralNetworkNodeGPUCpp      **HiddenNodesFedIntoMePtr,
		     std::size_t                    HiddenNodesFedIntoMePtrSize,
		     thrust::device_vector<float>  &output
		     )> forward_propagation;

  //Function pointer, which contains the derivative of the activation function
  std::function<void(
		     const float                   *W,
		     NeuralNetworkNodeGPUCpp      **HiddenNodesFedIntoMePtr,
		     std::size_t                    HiddenNodesFedIntoMePtrSize,
		     thrust::device_vector<float>  &output,
		     thrust::device_vector<float>  &delta
		     )> backpropagation;

  //Used to get the number of weights required
  //(very important to finalise the neural network)
  std::int32_t get_num_weights_required();

  //Calculate the output of the node
  void calc_output(
		   const std::int32_t _BatchNum,
		   const std::int32_t _BatchSize
		   );

  //Calculate the delta of the node (which is used for backpropagation)
  void calc_delta();

  //Calculate the derivatives for the individual weights
  void calc_dLdw(
		 float              *_dLdw,
		 const std::int32_t  _BatchNum,
		 const std::int32_t  _BatchSize
		 );

};
