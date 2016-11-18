//Calculates the number of batches needed
std::int32_t NeuralNetworkGPUCpp::calc_num_batches (/*MPI_Comm _comm, */std::int32_t _num_samples, std::int32_t _global_batch_size) {
	
  std::int32_t GlobalI;
	
  //Add all local num_samples and store the result in GlobalI
  //MPI_Allreduce(&I, &GlobalI, 1, MPI_INT, MPI_SUM, comm);
  //MPI_Barrier(_comm);

  GlobalI = _num_samples;
	
  if (_global_batch_size < 1 || _global_batch_size > GlobalI) _global_batch_size = GlobalI;		
			
  //Calculate the number of batches needed to divide GlobalI such that the sum of all local batches approximately equals global_batch_size
  if (GlobalI % _global_batch_size == 0) return GlobalI/_global_batch_size; 
  else return GlobalI/_global_batch_size + 1;

  //MPI_Barrier(_comm);
				
}

void NeuralNetworkGPUCpp::dfdw(/*MPI_Comm comm,*/
			       float             *_dLdw, 
			       const float       *_W, 
			       const std::int32_t _batch_begin, 
			       const std::int32_t _batch_end, 
			       const std::int32_t _batch_size, 
			       const std::int32_t _batch_num, 
			       const std::int32_t _epoch_num
			       ) {
  
  //Set pointers contained in the NeuralNetworkNodes class
  for (std::size_t n=0; n<this->nodes.size(); ++n) 
    this->nodes[n]->W = _W + this->cumulative_num_weights_required[n];
   
  //Forward propagation
  for (auto node: this->nodes) 
    node->calc_output(_batch_num, _batch_size);

  //Initialise delta
  //Needs to be after forward propagation, 
  //because forward propagation might resize delta
  for (auto node: this->nodes) 
    thrust::fill(
		 node->delta.begin(),
		 node->delta.end(),
		 0.f
	       );

  //Calculate loss for dense targets
  for (std::int32_t n=0; n<this->num_output_nodes_dense; ++n)
    this->loss->dloss_dyhat_dense (
				   /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ 
				   this->dense_targets[n][_batch_num],
				   this->output_nodes_dense[n]->output,
				   this->output_nodes_dense[n]->output_ptr,
				   this->output_nodes_dense[n]->delta
				   );

  //Calculate loss for sparse targets
  for (std::int32_t n=0; n<this->num_output_nodes_sparse; ++n)
    this->loss->dloss_dyhat_sparse (
				    /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ 
				    this->sparse_targets[n][_batch_num],
				    this->output_nodes_sparse[n]->output,
				    this->output_nodes_sparse[n]->output_ptr,
				    this->output_nodes_sparse[n]->delta
				    );
    
  //Backpropagation
  for (std::size_t n=1; n<=this->nodes.size(); ++n) 
    this->nodes[this->nodes.size()-n]->calc_delta(_batch_size);

  //Calculate derivative
  for (std::size_t n=0; n<this->nodes.size(); ++n) 
    this->nodes[n]->calc_dLdw(
			      _dLdw + this->cumulative_num_weights_required[n], 
			      _batch_num, 
			      _batch_size
			      );

  //Add all localdZdW and store the result in dZdW
  //MPI_Allreduce(localdZdW, this->optimiser->dZdW, this->lengthW, MPI_DOUBLE, MPI_SUM, comm);						
  //Barrier: Wait until all processes have reached this point
  //MPI_Barrier(comm);							

  //Apply regulariser
  //this->regulariser->g(this->optimiser->dZdW, W, 0, this->lengthW, this->lengthW, (double)batch_size); 		


}

void NeuralNetworkGPUCpp::fit (/*MPI_Comm comm,*/ 
			       OptimiserCpp      *_optimiser, 
			       std::int32_t       _global_batch_size, 
			       const float        _tol, 
			       const std::int32_t _max_num_epochs, 
			       const std::int32_t _MinibatchSizeStandard,
			       const bool         _sample
			       ) {

  //Make sure that neural network has been finalised!
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");

  //Get batch_size
  std::vector<std::int32_t> batch_size;
  if (this->dense_input_data.size() > 0) {

    for (auto data: this->dense_input_data[0])
      batch_size.push_back(data.batch_size);

  } else if (this->sparse_input_data.size() > 0) {

    for (auto data: this->sparse_input_data[0])
      batch_size.push_back(data.batch_size);

  } else throw std::invalid_argument("No input data provided!");

  //Calculate this->num_samples
  this->num_samples = std::accumulate(batch_size.begin(), batch_size.end(), 0);

  //Get num_batches
  std::int32_t num_batches = (std::int32_t)(batch_size.size());

  //Make sure that the batch_sizes are identical for all matrices provided!
  for (auto DataVector: this->dense_input_data) {

    if (DataVector.size() != batch_size.size()) 
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) 
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  //Make sure that the batch_sizes are identical for all matrices provided!
  for (auto DataVector: this->dense_targets) {

    if (DataVector.size() != batch_size.size())
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i)
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  //Make sure that the batch_sizes are identical for all matrices provided!
  for (auto DataVector: this->sparse_input_data) {

    if (DataVector.size() != batch_size.size())
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) 
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  for (auto DataVector: this->sparse_targets) {

    if (DataVector.size() != batch_size.size()) 
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) 
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  this->optimiser = _optimiser;
  this->sample = _sample;

  //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
  cublasCreate(&(this->dense_handle_));
  cusparseCreate(&(this->sparse_handle_));
  cusparseCreateMatDescr(&(this->mat_descr_));
  cusparseSetMatType(this->mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);

  //Do the actual optimisation
  this->optimiser->minimise(
			    this, 
			    this->num_samples, 
			    this->W, 
			    _global_batch_size, 
			    _tol, 
			    _max_num_epochs, 
			    this->sum_gradients
			    ); 
  
  //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
  cublasDestroy(this->dense_handle_);
  cusparseDestroyMatDescr(this->mat_descr_);
  cusparseDestroy(this->sparse_handle_);
			
  //Clear data. so it does not unnecessarily take up space on the GPU
  this->delete_data();

}
