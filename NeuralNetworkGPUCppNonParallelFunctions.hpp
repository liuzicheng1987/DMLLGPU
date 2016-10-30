//Private member functions (keep this comment - it is necessary for the automatic generation of the script)

//Protected member functions (keep this comment - it is necessary for the automatic generation of the script)

//Public member functions (keep this comment - it is necessary for the automatic generation of the script)

NeuralNetworkGPUCpp::NeuralNetworkGPUCpp(std::int32_t *_NumInputNodesDense, std::int32_t _NumInputNodesDenseLength, std::int32_t *_NumInputNodesSparse, std::int32_t _NumInputNodesSparseLength, std::int32_t _NumOutputNodesDense, std::int32_t _NumOutputNodesSparse, LossFunctionCpp *loss/*, RegulariserCpp *regulariser*/) {

  //Make that the input is reasonable
  if (_NumInputNodesDenseLength + _NumInputNodesSparseLength <= 0) throw std::invalid_argument("You must provide at least some input nodes!");
  if (_NumOutputNodesDense + _NumOutputNodesSparse <= 0) throw std::invalid_argument("You must provide at least some output nodes!");

  if (std::any_of(_NumInputNodesDense, _NumInputNodesDense + _NumInputNodesDenseLength, [](int i){return i <= 0;})) throw std::invalid_argument("Width of all input matrices must be greater than 0!");
  if (std::any_of(_NumInputNodesSparse, _NumInputNodesSparse + _NumInputNodesSparseLength, [](int i){return i <= 0;})) throw std::invalid_argument("Width of all input matrices must be greater than 0!");

  //Init NumHiddenNodes
  this->NumHiddenNodes = (std::size_t)0;
  
  //Init NumOutputNodes
  this->NumOutputNodesDense = _NumOutputNodesDense; 
  this->NumOutputNodesSparse = _NumOutputNodesSparse;
  this->NumOutputNodes = _NumOutputNodesDense + _NumOutputNodesSparse;

  //Set up input data and target data
  this->DenseInputData = std::vector<std::vector<DenseInputStruct>>(_NumInputNodesDenseLength);
  this->SparseInputData = std::vector<std::vector<SparseInputStruct>>(_NumInputNodesSparseLength);
  this->DenseTargets = std::vector<std::vector<DenseInputStruct>>(_NumOutputNodesDense);
  this->SparseTargets = std::vector<std::vector<SparseInputStruct>>(_NumOutputNodesSparse);

  this->DenseInputDataJ = std::vector<std::int32_t>(_NumInputNodesDenseLength);
  this->SparseInputDataJ = std::vector<std::int32_t>(_NumInputNodesSparseLength);
  this->DenseTargetsJ = std::vector<std::int32_t>(_NumOutputNodesDense);
  this->SparseTargetsJ = std::vector<std::int32_t>(_NumOutputNodesSparse);

  //Transfer number of input nodes
  std::copy(_NumInputNodesDense, _NumInputNodesDense + _NumInputNodesDenseLength, this->DenseInputDataJ.data());
  std::copy(_NumInputNodesSparse, _NumInputNodesSparse + _NumInputNodesSparseLength, this->SparseInputDataJ.data());
  
  this->loss = loss;
  //this->regulariser = regulariser;
		
  this->nodes = std::vector<NeuralNetworkNodeGPUCpp*>(this->NumOutputNodes);		
  this->OutputNodes = nodes.data();

  //Initialise to nullptr\
  std::fill(this->nodes.begin(), this->nodes.end(), nullptr);
				
  //Since neural network has not been finalised, set finalised to false
  this->finalised = false;

}
					 
NeuralNetworkGPUCpp::~NeuralNetworkGPUCpp()  {};

void NeuralNetworkGPUCpp::init_hidden_node(NeuralNetworkNodeGPUCpp *_HiddenNode) {
	
  //Make sure that the neural network has not already been finalised!
  if (this->finalised) throw std::invalid_argument("Neural network has already been finalised!");

  if (_HiddenNode->NodeNumber >= this->NumHiddenNodes) {	

    std::int32_t NumAdditionalNodes = _HiddenNode->NodeNumber + 1 - this->NumHiddenNodes;

    //Extend hidden nodes vector
    std::vector<NeuralNetworkNodeGPUCpp*>::iterator it = this->nodes.begin() + this->nodes.size();
    this->nodes.insert(it, NumAdditionalNodes, nullptr);	

    //Increase NumHiddenNodes and reset pointer OutputNodes
    this->NumHiddenNodes += NumAdditionalNodes;
    this->OutputNodes = nodes.data() + this->NumHiddenNodes;

    //Increase NodeNumber of OutputNodes
    for (std::int32_t i=0; i<this->NumOutputNodes; ++i) if (this->OutputNodes[i] != nullptr) this->OutputNodes[i]->NodeNumber += NumAdditionalNodes;
  }

  this->nodes[_HiddenNode->NodeNumber] = _HiddenNode;
					
};

void NeuralNetworkGPUCpp::init_output_node(NeuralNetworkNodeGPUCpp *_OutputNode) {
		
  //Make sure that the neural network has not already been finalised!
  if (this->finalised) 
    throw std::invalid_argument("Neural network has already been finalised!");
  
  //Make sure that node number is in range
  if (_OutputNode->NodeNumber >= (std::int32_t)(this->nodes.size()) || _OutputNode->NodeNumber < 0) 
    throw std::invalid_argument("Output node: Node number out of range!");
				
  this->nodes[_OutputNode->NodeNumber] = _OutputNode;
						
};

void NeuralNetworkGPUCpp::finalise(/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/float _WeightInitRange) {

  //Make sure that neural net has not been finalised already
  if (this->finalised == true)
    throw std::invalid_argument("Neural network has already been finalised!");

  //Make sure that all nodes were initialised
  if (std::any_of(this->nodes.begin(), this->nodes.end(), [](NeuralNetworkNodeGPUCpp *node) {return node == nullptr;}))
      throw std::invalid_argument("Not all nodes have been initialised!");
    
  //Calculate pointer to hidden nodes fed into me
  for (auto node: this->nodes) {

    node->HiddenNodesFedIntoMePtr.clear();
    for (auto i: node->HiddenNodesFedIntoMe) node->HiddenNodesFedIntoMePtr.push_back(this->nodes[i]);

  }
   
  //Transfer number fo input nodes to nodes, so we can calculate the number of weights needed
  for (auto node: this->nodes) {

    //Set initial value to zero
    node->NumInputNodesCumulative = 0;

    //Add dense input
    for (auto dense: node->InputNodesFedIntoMeDense) node->NumInputNodesCumulative += this->DenseInputDataJ[dense];

    //Add sparse input
    for (auto sparse: node->InputNodesFedIntoMeSparse) node->NumInputNodesCumulative += this->SparseInputDataJ[sparse];
     
  }

  //Transfer number of output nodes to targets
  for (int i=0; i < this->NumOutputNodesDense; ++i) DenseTargetsJ[i] = 1;//Temporary solution until layers are implemented! 
  for (int i=0; i < this->NumOutputNodesSparse; ++i) SparseTargetsJ[i] = 1;//Temporary solution until layers are implemented! 

  //Calculate CumulativeNumWeightsRequired and initialise W
  std::int32_t lengthW = 0;
  std::vector<std::int32_t> CumulativeNumWeightsRequiredHost;
  for (auto node: this->nodes) {

    node->NeuralNet = this;
    CumulativeNumWeightsRequiredHost.push_back(lengthW);
    lengthW += node->get_num_weights_required();

  }

  CumulativeNumWeightsRequiredHost.push_back(lengthW);

  //Transfer CumulativeNumWeightsRequired to device vector
  this->CumulativeNumWeightsRequired = thrust::device_vector<std::int32_t>(CumulativeNumWeightsRequiredHost.data(), CumulativeNumWeightsRequiredHost.data() + CumulativeNumWeightsRequiredHost.size());

  //Init Whost
  std::vector<float> Whost(lengthW);
  
  std::mt19937 gen(1);//Note that we deliberately choose a constant seed to get the same output every time we call the function
  std::uniform_real_distribution<float> dist(_WeightInitRange*(-1.0f), _WeightInitRange);

  //Initialise weight vector
  //The vector of weights associated with the input nodes cannot be a csr_matrix. The solution is to keep a set of weights that always assume the value of 0.0.
  for (auto node: this->nodes) {
         
    for (std::int32_t i = CumulativeNumWeightsRequiredHost[node->NodeNumber]; i < CumulativeNumWeightsRequiredHost[node->NodeNumber + 1]; ++i) Whost[i] = dist(gen);
            
  }
  
  //Transfor to device vector
  this->W = thrust::device_vector<float>(Whost.data(), Whost.data() + Whost.size());

  //Set finalised to true so we know we can now fit the neural network
  this->finalised = true;
  
}

std::int32_t NeuralNetworkGPUCpp::get_length_params() {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");

  return (std::int32_t)(this->W.size());
		
};

void NeuralNetworkGPUCpp::get_params(float *_W, std::int32_t _LengthW) {

  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  
  for (std::int32_t i=0; i<_LengthW; ++i) _W[i] = this->W[i];

}

std::int32_t NeuralNetworkGPUCpp::get_input_nodes_fed_into_me_dense_length(std::int32_t _NodeNumber) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_NodeNumber < 0 || _NodeNumber >= (std::int32_t)(nodes.size())) std::invalid_argument("NodeNumber out of bounds!");

  return (std::int32_t)(this->nodes[_NodeNumber]->InputNodesFedIntoMeDense.size());
		
};

void NeuralNetworkGPUCpp::get_input_nodes_fed_into_me_dense(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_NodeNumber < 0 || _NodeNumber >= (std::int32_t)(nodes.size())) std::invalid_argument("NodeNumber out of bounds!");
		
  for (std::int32_t i=0; i<_InputNodesFedIntoMeDenseLength; ++i) _InputNodesFedIntoMeDense[i] = this->nodes[_NodeNumber]->InputNodesFedIntoMeDense[i];
  		
};

std::int32_t NeuralNetworkGPUCpp::get_input_nodes_fed_into_me_sparse_length(std::int32_t _NodeNumber) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_NodeNumber < 0 || _NodeNumber >= (std::int32_t)(nodes.size())) std::invalid_argument("NodeNumber out of bounds!");

  return (std::int32_t)(this->nodes[_NodeNumber]->InputNodesFedIntoMeSparse.size());
		
};

void NeuralNetworkGPUCpp::get_input_nodes_fed_into_me_sparse(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_NodeNumber < 0 || _NodeNumber >= (std::int32_t)(nodes.size())) std::invalid_argument("NodeNumber out of bounds!");
		
  for (std::int32_t i=0; i<_InputNodesFedIntoMeSparseLength; ++i) _InputNodesFedIntoMeSparse[i] = this->nodes[_NodeNumber]->InputNodesFedIntoMeSparse[i];
  		
};
     
std::int32_t NeuralNetworkGPUCpp::get_hidden_nodes_fed_into_me_length(std::int32_t _NodeNumber) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_NodeNumber < 0 || _NodeNumber >= (std::int32_t)(nodes.size())) std::invalid_argument("NodeNumber out of bounds!");

  return (std::int32_t)(this->nodes[_NodeNumber]->HiddenNodesFedIntoMe.size());
		
};

void NeuralNetworkGPUCpp::get_hidden_nodes_fed_into_me(std::int32_t _NodeNumber, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _LengthHiddenNodesFedIntoMe) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_NodeNumber < 0 || _NodeNumber >= (std::int32_t)(nodes.size())) std::invalid_argument("NodeNumber out of bounds!");
		
  for (std::int32_t i=0; i<_LengthHiddenNodesFedIntoMe; ++i) _HiddenNodesFedIntoMe[i] = this->nodes[_NodeNumber]->HiddenNodesFedIntoMe[i];
  		
};

//BatchBegin and BatchEnd are used to share the burden evenly among the processes
void NeuralNetworkGPUCpp::calc_batch_begin_end (std::int32_t &_BatchBegin, std::int32_t &_BatchEnd, std::int32_t &_BatchSize, const std::int32_t _BatchNum, const std::int32_t _I, const std::int32_t _NumBatches) {
									
  //Calculate _BatchBegin
  _BatchBegin = _BatchNum*(_I/_NumBatches);
		
  //Calculate _BatchSize
  if (_BatchNum < _NumBatches-1) _BatchSize = _I/_NumBatches;
  else _BatchSize = _I - _BatchBegin;
		
  //Calculate _BatchEnd
  _BatchEnd = _BatchBegin + _BatchSize;
	
}

void NeuralNetworkGPUCpp::load_dense(std::vector<DenseInputStruct> &data, float *_X, std::int32_t _I, std::int32_t _J, std::int32_t _NumBatches) {

  std::int32_t BatchBegin, BatchEnd, BatchSize;

  for (std::int32_t BatchNum = 0; BatchNum<_NumBatches; ++BatchNum) {

    this->calc_batch_begin_end(BatchBegin, BatchEnd, BatchSize, BatchNum, _I, _NumBatches);

    //Transfer _I and _J
    data[BatchNum].BatchSize = BatchSize;
    data[BatchNum].J = _J;

    //Transfer X to GPU and set Xptr
    data[BatchNum].X = thrust::device_vector<float>(_X + BatchBegin*_J, _X + BatchEnd*_J);
    data[BatchNum].Xptr = thrust::raw_pointer_cast(data[BatchNum].X.data());

  } 

}

void NeuralNetworkGPUCpp::load_dense_data(std::int32_t _NumInputNode, float *_X, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize) {

  if (_NumInputNode >= (std::int32_t)(this->DenseInputData.size()) || _NumInputNode < 0) throw std::invalid_argument("NumInputNode out of bounds!");
  if (_J != this->DenseInputDataJ[_NumInputNode]) throw std::invalid_argument("Width J of array provided does not match the width that has been set when initialising the network!");

  std::int32_t NumBatches = calc_num_batches (/*MPI_Comm comm,*/ _I, _GlobalBatchSize);//Calculates the number of batches needed

  std::cout << "NumBatches: " << NumBatches << "\n";

  this->DenseInputData[_NumInputNode] = std::vector<DenseInputStruct>(NumBatches);
  
  this->load_dense(this->DenseInputData[_NumInputNode], _X, _I, _J, NumBatches);

}

void NeuralNetworkGPUCpp::load_dense_targets(std::int32_t _NumOutputNode, float *_Y, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize) {

  if (_NumOutputNode >= (std::int32_t)(this->DenseTargets.size()) || _NumOutputNode < 0) throw std::invalid_argument("NumOutputNode out of bounds!");
  if (_J != this->DenseTargetsJ[_NumOutputNode]) throw std::invalid_argument("Width J of array provided does not match the width that has been set when initialising the network!");
  
  std::int32_t NumBatches = calc_num_batches (/*MPI_Comm comm,*/ _I, _GlobalBatchSize);//Calculates the number of batches needed

  this->DenseTargets[_NumOutputNode] = std::vector<DenseInputStruct>(NumBatches);

  this->load_dense(this->DenseTargets[_NumOutputNode], _Y, _I, _J, NumBatches);

}

void NeuralNetworkGPUCpp::load_sparse(std::vector<SparseInputStruct> &data, float *_XData, std::int32_t _XDataLength,  std::int32_t *_XIndices, std::int32_t _XIndicesLength, std::int32_t *_XIndptr, std::int32_t _XIndptrLength, std::int32_t _I, std::int32_t _J, std::int32_t _NumBatches) {

  //Transfer _I and _J
  //data.I = _I;
  //data.J = _J;

  //Transfer XData to GPU and set XDataPtr  
  //data.XData = thrust::device_vector<float>(_XData, _XData + _XDataLength);
  // data.XDataPtr = thrust::raw_pointer_cast(data.XData.data()); 

  //Transfer XIndices to GPU and set XIndicesPtr
  //data.XIndices = thrust::device_vector<std::int32_t>(_XIndices, _XIndices + _XIndicesLength);
  //data.XIndicesPtr = thrust::raw_pointer_cast(data.XIndices.data()); 

  //Transfer XIndptr to GPU and set XIndptrPtr
  //data.XIndptr = thrust::device_vector<std::int32_t>(_XIndptr, _XIndptr + _XIndptrLength);
  //data.XIndptrPtr = thrust::raw_pointer_cast(data.XIndptr.data()); 

}

void NeuralNetworkGPUCpp::load_sparse_data(std::int32_t _NumInputNode, float *_XData, std::int32_t _XDataLength,  std::int32_t *_XIndices, std::int32_t _XIndicesLength, std::int32_t *_XIndptr, std::int32_t _XIndptrLength, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize) {

  if (_NumInputNode >= (std::int32_t)(this->SparseInputData.size()) || _NumInputNode < 0) throw std::invalid_argument("NumInputNode out of bounds!");
  if (_J != this->SparseInputDataJ[_NumInputNode]) throw std::invalid_argument("Width J of array provided does not match the width that has been set when initialising the network!");

  std::int32_t NumBatches = calc_num_batches (/*MPI_Comm comm,*/ _I, _GlobalBatchSize);//Calculates the number of batches needed

  this->SparseInputData[_NumInputNode] = std::vector<SparseInputStruct>(NumBatches);

  this->load_sparse(this->SparseInputData[_NumInputNode], _XData, _XDataLength, _XIndices, _XIndicesLength, _XIndptr, _XIndptrLength, _I, _J, NumBatches);

}

void NeuralNetworkGPUCpp::load_sparse_targets(std::int32_t _NumOutputNode, float *_YData, std::int32_t _YDataLength,  std::int32_t *_YIndices, std::int32_t _YIndicesLength, std::int32_t *_YIndptr, std::int32_t _YIndptrLength, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize) {

  if (_NumOutputNode >= (std::int32_t)(this->SparseTargets.size()) || _NumOutputNode < 0) throw std::invalid_argument("NumOutputNode out of bounds!");
  if (_J != this->SparseTargetsJ[_NumOutputNode]) throw std::invalid_argument("Width J of array provided does not match the width that has been set when initialising the network!");

  std::int32_t NumBatches = calc_num_batches (/*MPI_Comm comm,*/ _I, _GlobalBatchSize);//Calculates the number of batches needed

  this->SparseTargets[_NumOutputNode] = std::vector<SparseInputStruct>(NumBatches);

  this->load_sparse(this->SparseTargets[_NumOutputNode], _YData, _YDataLength, _YIndices, _YIndicesLength, _YIndptr, _YIndptrLength, _I, _J, NumBatches);

}

void NeuralNetworkGPUCpp::delete_data() {

  for (auto& data: this->DenseInputData) data.clear();
  for (auto& data: this->DenseTargets) data.clear();
  for (auto& data: this->SparseInputData) data.clear();
  for (auto& data: this->SparseTargets) data.clear();

}
	           
void NeuralNetworkGPUCpp::transform(float *_Yhat, std::int32_t _IY2, std::int32_t _JY2, bool _sample, std::int32_t _SampleSize, bool _GetHiddenNodes) {
		
  //Make sure that neural network has been finalised!
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  
  //Get BatchSize
  std::vector<std::int32_t> BatchSize;
  if (this->DenseInputData.size() > 0) {

    for (auto data: this->DenseInputData[0])
      BatchSize.push_back(data.BatchSize);

  } else if (this->SparseInputData.size() > 0) {

    for (auto data: this->SparseInputData[0])
      BatchSize.push_back(data.BatchSize);

  } else throw std::invalid_argument("No input data provided!");

  //Get NumBatches
  std::int32_t NumBatches = (std::int32_t)(BatchSize.size());

  //Make sure that the BatchSizes are identical for all matrices provided!
  for (auto DataVector: this->DenseInputData) {

    if (DataVector.size() != BatchSize.size()) throw std::invalid_argument("All input matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) if (DataVector[i].BatchSize != BatchSize[i]) throw std::invalid_argument("All input matrices must have the exact same number of samples!");

  }

  //Make sure that the BatchSizes are identical for all matrices provided!
  for (auto DataVector: this->SparseInputData) {

    if (DataVector.size() != BatchSize.size()) throw std::invalid_argument("All input matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) if (DataVector[i].BatchSize != BatchSize[i]) throw std::invalid_argument("All input matrices must have the exact same number of samples!");

  }

  //Store input values
  this->I = _IY2;
  this->sample = _sample;
  if (!_sample) _SampleSize = 1;

  const double SampleAvg = 1.0/((double)_SampleSize);

  //Set pointers contained in the NeuralNetworkNodes class
  for (std::size_t n=0; n<this->nodes.size(); ++n) this->nodes[n]->W = thrust::raw_pointer_cast(this->W.data()) + this->CumulativeNumWeightsRequired[n];
  
  //Init YhatTemp
  thrust::device_vector<float> YhatTemp(_Yhat, _Yhat + _IY2*_JY2);
  
  //Init cuBLAS handle
  cublasCreate(&(this->handle));

  //Calculate output
  std::int32_t BatchBegin = 0;
  for (std::int32_t BatchNum=0; BatchNum<NumBatches; ++BatchNum, BatchBegin += BatchSize[BatchNum]) {
    						
    //Calculate nodes
    for (auto node: this->nodes) node->calc_output(BatchNum, BatchSize[BatchNum]);

    //Add to YhatTemp
    for (std::int32_t n=0; n<this->NumOutputNodes; ++n) thrust::transform(this->OutputNodes[n]->output.begin(), this->OutputNodes[n]->output.end(), YhatTemp.begin() + _IY2*n + BatchBegin, YhatTemp.begin() + _IY2*n + BatchBegin, thrust::plus<float>());

  }	
  
  //Get data from YhatTemp and transpose
  for (std::int32_t i=0; i<_IY2; ++i) for (std::int32_t j=0; j<_JY2; ++j) _Yhat[i*_JY2 + j] = YhatTemp[j*_IY2 + i];

  //Destroy cuBLAS handle
  cublasDestroy(this->handle);
			
  //Clear data, so it does not unnecessarily take up space on the GPU
  this->delete_data();
		
};
