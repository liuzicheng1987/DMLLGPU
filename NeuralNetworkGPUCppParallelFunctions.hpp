//Calculates the number of batches needed
std::int32_t NeuralNetworkGPUCpp::calc_num_batches (/*MPI_Comm _comm, */std::int32_t _I, std::int32_t _GlobalBatchSize) {
	
  std::int32_t GlobalI;
	
  //Add all local I and store the result in GlobalI
  //MPI_Allreduce(&I, &GlobalI, 1, MPI_INT, MPI_SUM, comm);
  //MPI_Barrier(_comm);

  GlobalI = _I;
	
  if (_GlobalBatchSize < 1 || _GlobalBatchSize > GlobalI) _GlobalBatchSize = GlobalI;		
			
  //Calculate the number of batches needed to divide GlobalI such that the sum of all local batches approximately equals GlobalBatchSize
  if (GlobalI % _GlobalBatchSize == 0) return GlobalI/_GlobalBatchSize; 
  else return GlobalI/_GlobalBatchSize + 1;

  //MPI_Barrier(_comm);
				
}

void NeuralNetworkGPUCpp::dfdw(/*MPI_Comm comm,*/float *_dLdw, const float *_W, const std::int32_t _BatchBegin, const std::int32_t _BatchEnd, const std::int32_t _BatchSize, const std::int32_t _BatchNum, const std::int32_t _EpochNum) {

  //Set pointers contained in the NeuralNetworkNodes class
  for (std::size_t n=0; n<this->nodes.size(); ++n) 
    this->nodes[n]->W = _W + this->CumulativeNumWeightsRequired[n];
   
  //Forward propagation
  for (auto node: this->nodes) 
    node->calc_output(_BatchNum, _BatchSize);

  //Initialise delta
  //Needs to be after forward propagation, 
  //because forward propagation might resize delta
  for (auto node: this->nodes) 
    thrust::fill(
		 node->delta.begin(),
		 node->delta.end(),
		 0.f
	       );

  //Calculate loss
  for (std::size_t n=0; n<DenseTargets.size(); ++n)
    this->loss->dLossdYhat (
			    /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ 
			    this->DenseTargets[n][_BatchNum].BatchSize,
			    this->DenseTargets[n][_BatchNum].J, 
			    this->OutputNodes[n]->delta, 
			    this->DenseTargets[n][_BatchNum].X,
			    this->OutputNodes[n]->output
			    );
      
  //Backpropagation
  for (std::size_t n=1; n<=this->nodes.size(); ++n) 
    this->nodes[this->nodes.size()-n]->calc_delta();

  //Calculate derivative
  for (std::size_t n=0; n<this->nodes.size(); ++n) 
    this->nodes[n]->calc_dLdw(
			      _dLdw + this->CumulativeNumWeightsRequired[n], 
			      _BatchNum, 
			      _BatchSize
			      );

  //Add all localdZdW and store the result in dZdW
  //MPI_Allreduce(localdZdW, this->optimiser->dZdW, this->lengthW, MPI_DOUBLE, MPI_SUM, comm);						
  //Barrier: Wait until all processes have reached this point
  //MPI_Barrier(comm);							

  //Apply regulariser
  //this->regulariser->g(this->optimiser->dZdW, W, 0, this->lengthW, this->lengthW, (double)BatchSize); 		


}

void NeuralNetworkGPUCpp::fit (/*MPI_Comm comm,*/ OptimiserCpp *_optimiser, std::int32_t _GlobalBatchSize, const float _tol, const std::int32_t _MaxNumEpochs, const std::int32_t _MinibatchSizeStandard, const bool _sample) {

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

  //Calculate this->I
  this->I = std::accumulate(BatchSize.begin(), BatchSize.end(), 0);

  //Get NumBatches
  std::int32_t NumBatches = (std::int32_t)(BatchSize.size());

  //Make sure that the BatchSizes are identical for all matrices provided!
  for (auto DataVector: this->DenseInputData) {

    if (DataVector.size() != BatchSize.size()) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) if (DataVector[i].BatchSize != BatchSize[i]) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  //Make sure that the BatchSizes are identical for all matrices provided!
  for (auto DataVector: this->DenseTargets) {

    if (DataVector.size() != BatchSize.size()) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) if (DataVector[i].BatchSize != BatchSize[i]) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  //Make sure that the BatchSizes are identical for all matrices provided!
  for (auto DataVector: this->SparseInputData) {

    if (DataVector.size() != BatchSize.size()) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) if (DataVector[i].BatchSize != BatchSize[i]) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  for (auto DataVector: this->SparseTargets) {

    if (DataVector.size() != BatchSize.size()) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) if (DataVector[i].BatchSize != BatchSize[i]) throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  this->optimiser = _optimiser;
  this->sample = _sample;

  //Init cuBLAS handle
  cublasCreate(&(this->handle));

  //Do the actual optimisation
  this->optimiser->minimise(this, this->I, this->W, _GlobalBatchSize, _tol, _MaxNumEpochs, this->SumGradients); 
  
  //Destroy cuBLAS handle
  cublasDestroy(this->handle);
			
  //Clear data. so it does not unnecessarily take up space on the GPU
  this->delete_data();

}

/*	
void NeuralNetworkCpp::FitDense (MPI_Comm comm, double *X, std::int32_t I, std::int32_t J, double *Y, std::int32_t IY2, std::int32_t JY2, OptimiserCpp *optimiser, std::int32_t GlobalBatchSize, const double tol, const std::int32_t MaxNumIterations, const std::int32_t MinibatchSizeStandard, const bool sample) {
		
  std::int32_t NumBatches;
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
				
  //Check input values
  if (J != this->NumInputNodes) throw std::invalid_argument("NeuralNetwork: Number of attributes does not match NumInputNodes that has been defined when declaring the class!");
  if (this->loss->IsSupervised && JY2 != this->NumOutputNodes) throw std::invalid_argument("NeuralNetwork: Width of Y does not match NumInputNodes that has been defined when declaring the class!");
  if (I != IY2) throw std::invalid_argument("NeuralNetwork: Length of Y does not match length of X!");

  //Store input values (which we need for f() and g())
  this->optimiser = optimiser;
  this->X = X;
  this->Y = Y;
  this->I = I;
  
  //Activate/disactivate sampling
  this->sample = sample;
		
  //Calculate the number of batches
  //This function is contained in the optimiser class
  optimiser->CalcNumBatches (comm, this->I, GlobalBatchSize, NumBatches);	

  //Transpose input values for Y
  this->Ytrans.reset(new double[this->NumOutputNodes*I]);
		
  //Pass GlobalBatchSize and this->NumBatches to loss function and nodes
  //Unfortunately, SWIG does not allow for static variables!
  for (std::int32_t j=0; j<this->NumHiddenNodes + this->NumOutputNodes; ++j) {
    this->HiddenNodes[j]->GlobalBatchSize = GlobalBatchSize; 
    this->HiddenNodes[j]->NumBatches = NumBatches; 
  }
		
  this->loss->GlobalBatchSize = GlobalBatchSize; 
  this->loss->NumBatches = NumBatches;
		
  //this->MinibatchSizeStandard determines the number of samples that are calculated at once when calling the functions.
  //It must be optimised for cache efficiency
  this->MinibatchSizeStandard = MinibatchSizeStandard;
		
  //Allocate this->delta, this->LossDelta, this->hidden and this->output
  this->delta.reset(new double[(this->NumHiddenNodes + this->NumOutputNodes)*this->MinibatchSizeStandard]);	
  if (this->NumHiddenNodes > 0) this->hidden.reset(new double[this->NumHiddenNodes*this->MinibatchSizeStandard]);
  this->output.reset(new double[this->NumOutputNodes*this->MinibatchSizeStandard]);
		
  //Optimise
  this->optimiser->minimise(comm, this, I, lengthW, GlobalBatchSize, tol, MaxNumIterations);
				
  //Set X back to NULL, so we still have the possibility to apply the algorithm to CSR_matrices
  this->X = NULL;

  //Reset Ytrans
  this->Ytrans.reset(nullptr);
				
}
}
*/
