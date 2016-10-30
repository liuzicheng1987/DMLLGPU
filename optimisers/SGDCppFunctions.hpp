void SGDCpp::min(/*MPI_Comm comm,*/NeuralNetworkGPUCpp *_NeuralNet, thrust::device_vector<float> &_W, const float _tol, const std::int32_t _MaxNumEpochs, std::vector<float> &_SumGradients) {
		
  std::int32_t BatchBegin, BatchEnd, BatchSize, GlobalBatchSize;

  float CurrentLearningRate;

  float SumGradients;//The sum of all sum of gradients - will be recorded in _SumGradients
			
  for (; 
       this->EpochNum < _MaxNumEpochs; 
       ++(this->EpochNum)
       ) {//EpochNum layer
			
    //Init this->sum_dldw_
    thrust::fill(
		 this->sum_dldw_.begin(), 
		 this->sum_dldw_.end(), 
		 0.f
		 );

    //this->num_batches_ is inherited from the Optimiser class
    for (
	 std::int32_t BatchNum = 0; 
	 BatchNum < this->num_batches_; 
	 ++BatchNum
	 ) {//BatchNum layer
				
      //We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
      _NeuralNet->calc_batch_begin_end(
				       BatchBegin, 
				       BatchEnd, 
				       BatchSize, 
				       BatchNum, 
				       this->I_, 
				       this->num_batches_
				       );

      //Calculate GlobalBatchSize
      GlobalBatchSize = BatchSize;
					
      //Barrier: Wait until all processes have reached this point
      //MPI_Barrier(comm);													 	
      //Call dfdw()
      //Note that it is the responsibility of whoever writes the underlying algorithm to make sure that this->dLdW and this->SumdLdW are passed to ALL processes
      //It is, however, your responsibility to place a barrier after that, if required
      _NeuralNet->dfdw(
		       /*comm,*/
		       dldw_ptr_, 
		       w_ptr_, 
		       BatchBegin, 
		       BatchEnd, 
		       BatchSize, 
		       BatchNum,
		       EpochNum
		       );
				
      //Add all BatchSize and store the result in GlobalBatchSize
      //MPI_Allreduce(&BatchSize, &GlobalBatchSize, 1, MPI_INT, MPI_SUM, comm);		
      //GlobalBatchSizeFloat = (float)GlobalBatchSize;	
												
      //Barrier: Wait until all processes have reached this point
      //MPI_Barrier(comm);

      //Record sum_dldw for SumGradients
      thrust::transform(
			dldw_.begin(), 
			dldw_.end(), 
			this->sum_dldw_.begin(), 
			this->sum_dldw_.begin(), 
			thrust::plus<float>()
			);
     
      //Calculate current learning rate
      //Learning rates are always divided by the sample size				
      CurrentLearningRate = 
	(this->LearningRate/pow(static_cast<float>(EpochNum + 1), this->LearningRatePower))
	/(static_cast<float>(GlobalBatchSize));

      //Update W
      thrust::transform(
			dldw_.begin(), 
			dldw_.end(),
			_W.begin(), 
			_W.begin(), 
			utils::saxpy<float>((-1.f)*CurrentLearningRate)
			);
            
    }//BatchNum layer
			
    //Record sum_dldw
    SumGradients = thrust::transform_reduce(
					    this->sum_dldw_.begin(), 
					    this->sum_dldw_.end(),
					    utils::square<float>(), 
					    0.f, 
					    thrust::plus<float>()
					    );

    //SumGradients != SumGradients means SumGradients is nan
    if (SumGradients != SumGradients) 
      throw std::invalid_argument(
				  "The gradients seem to have spun out of control! You might want to reduce the learning rate!"
				  );
    
    _SumGradients.push_back(SumGradients);
    		
    //Check whether convergence condition is met. If yes, break
    if (SumGradients/(static_cast<float>(_W.size())) < _tol) 
      break;
			
  }//EpochNum layer
				
}
