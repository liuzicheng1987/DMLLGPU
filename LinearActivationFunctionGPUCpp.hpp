class LinearActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {

public:

  LinearActivationFunctionGPUCpp(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates): ActivationFunctionGPUCpp(_NodeNumber, _InputNodesFedIntoMeDense, _InputNodesFedIntoMeDenseLength, _InputNodesFedIntoMeSparse, _InputNodesFedIntoMeSparseLength, _HiddenNodesFedIntoMe, _HiddenNodesFedIntoMeLength, _IShareWeightsWith, _NoWeightUpdates) {

    this->forward_propagation = ActivationFunction::Linear_forward_propagation;
    this->backpropagation = ActivationFunction::Linear_backpropagation;
    
};
	
  ~LinearActivationFunctionGPUCpp() {};

};
