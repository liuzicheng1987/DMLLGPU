NeuralNetworkNodeGPUCpp::NeuralNetworkNodeGPUCpp(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates) {
		
  this->NodeNumber = _NodeNumber;
      		
  //Make sure that the nodes pointed to are smaller than the node number
  for (std::int32_t i=0; i<_HiddenNodesFedIntoMeLength; ++i) 
    if (_HiddenNodesFedIntoMe[i] >= this->NodeNumber) 
      throw std::invalid_argument("HiddenNodesFedIntoMe must always be smaller than node number!");
				
  //Make sure that IShareWeightsWith is smaller than the node number
  if (_IShareWeightsWith >= this->NodeNumber) 
    throw std::invalid_argument("IShareWeightsWith must always be smaller than node number!");
		
  //Declare this->InputNodesFedIntoMeDense and transfer parameters
  this->InputNodesFedIntoMeDense = std::vector<std::int32_t>( _InputNodesFedIntoMeDenseLength);
  for (std::int32_t i=0; i<_InputNodesFedIntoMeDenseLength; ++i) this->InputNodesFedIntoMeDense[i] = _InputNodesFedIntoMeDense[i];    

  //Declare this->InputNodesFedIntoMeSparse and transfer parameters
  this->InputNodesFedIntoMeSparse = std::vector<std::int32_t>( _InputNodesFedIntoMeSparseLength);
  for (std::int32_t i=0; i<_InputNodesFedIntoMeSparseLength; ++i) this->InputNodesFedIntoMeSparse[i] = _InputNodesFedIntoMeSparse[i];    

  //Declare this->HiddenNodesFedIntoMe and transfer parameters
  this->HiddenNodesFedIntoMe = std::vector<std::int32_t>(_HiddenNodesFedIntoMeLength);
  for (std::int32_t i=0; i<_HiddenNodesFedIntoMeLength; ++i) this->HiddenNodesFedIntoMe[i] = _HiddenNodesFedIntoMe[i];    

  this->IShareWeightsWith = _IShareWeightsWith;
  this->NoWeightUpdates = _NoWeightUpdates;
		
  //All objects that inherit from NeuralNetworkNodeCpp must also calculate this->NumWeightsRequired!
		
};

NeuralNetworkNodeGPUCpp::~NeuralNetworkNodeGPUCpp(){};
