void NeuralNetworkCpp::dfdw(/*MPI_Comm comm,*/
			       float                     *_dLdw,
			       const float         *_W,
			       const std::int32_t _batch_begin, 
			       const std::int32_t _batch_end, 
			       const std::int32_t _batch_size, 
			       const std::int32_t _batch_num, 
			       const std::int32_t _epoch_num
			       ) {
  
  //Set pointers contained in the NeuralNetworkNodes class
  for (std::size_t n=0; n<this->nodes_.size(); ++n) 
    this->nodes_[n]->W_ = _W + this->cumulative_num_weights_required_[n];
   
  //Forward propagation
  for (auto node: this->nodes_) 
    node->calc_output(_batch_num, _batch_size);

  //Initialise delta
  //Needs to be after forward propagation, 
  //because forward propagation might resize delta
  for (auto node: this->nodes_) 
    thrust::fill(
		 node->delta_.begin(),
		 node->delta_.end(),
		 0.f
	       );

  //Calculate loss for dense targets
  for (std::int32_t n=0; n<this->num_output_nodes_dense_; ++n)
    this->loss_->dloss_dyhat_dense (
				   /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ 
				   this->dense_targets_[n][_batch_num],
				   this->output_nodes_dense_[n]->output_,
				   this->output_nodes_dense_[n]->output_ptr_,
				   this->output_nodes_dense_[n]->delta_
				   );

  //Calculate loss for sparse targets
  for (std::int32_t n=0; n<this->num_output_nodes_sparse_; ++n)
    this->loss_->dloss_dyhat_sparse (
				    /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ 
				    this->sparse_targets_[n][_batch_num],
				    this->output_nodes_sparse_[n]->output_,
				    this->output_nodes_sparse_[n]->output_ptr_,
				    this->output_nodes_sparse_[n]->delta_
				    );
    
  //Backpropagation
  for (std::size_t n=1; n<=this->nodes_.size(); ++n) 
    this->nodes_[this->nodes_.size()-n]->calc_delta(_batch_size);

  //Calculate derivative
  for (std::size_t n=0; n<this->nodes_.size(); ++n) 
    if (this->nodes_[n]->no_weight_updates_ == false)
      this->nodes_[n]->calc_dLdw(
				_dLdw + this->cumulative_num_weights_required_[n], 
				_batch_num, 
				_batch_size
				);
  
  //Add all localdZdW and store the result in dZdW
  //MPI_Allreduce(localdZdW, this->optimiser->dZdW, this->lengthW, MPI_DOUBLE, MPI_SUM, comm);						
  //Barrier: Wait until all processes have reached this point
  //MPI_Barrier(comm);							


}

void NeuralNetworkCpp::fit (/*MPI_Comm comm,*/ 
			       OptimiserCpp      *_optimiser, 
			       std::int32_t       _global_batch_size, 
			       const float        _tol, 
			       const std::int32_t _max_num_epochs, 
			       const std::int32_t _MinibatchSizeStandard,
			       const bool         _sample
			       ) {

  //Make sure that neural network has been finalised!
  if (!this->finalised_) throw std::invalid_argument("Neural network has not been finalised!");

  //Get batch_size
  std::vector<std::int32_t> batch_size;
  if (this->dense_input_data_.size() > 0) {

    for (auto data: this->dense_input_data_[0])
      batch_size.push_back(data.batch_size);

  } else if (this->sparse_input_data_.size() > 0) {

    for (auto data: this->sparse_input_data_[0])
      batch_size.push_back(data.batch_size);

  } else throw std::invalid_argument("No input data provided!");

  //Calculate this->num_samples
  this->num_samples_ = std::accumulate(batch_size.begin(), batch_size.end(), 0);

  //Get num_batches
  std::int32_t num_batches = (std::int32_t)(batch_size.size());

  //Make sure that the batch_sizes are identical for all matrices provided!
  for (auto DataVector: this->dense_input_data_) {

    if (DataVector.size() != batch_size.size()) 
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) 
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  //Make sure that the batch_sizes are identical for all matrices provided!
  for (auto DataVector: this->dense_targets_) {

    if (DataVector.size() != batch_size.size())
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i)
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  //Make sure that the batch_sizes are identical for all matrices provided!
  for (auto DataVector: this->sparse_input_data_) {

    if (DataVector.size() != batch_size.size())
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) 
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  for (auto DataVector: this->sparse_targets_) {

    if (DataVector.size() != batch_size.size()) 
      throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

    for (std::size_t i=0; i<DataVector.size(); ++i) 
      if (DataVector[i].batch_size != batch_size[i]) 
	throw std::invalid_argument("All input and output matrices must have the exact same number of samples!");

  }

  this->optimiser_ = _optimiser;
  this->sample_ = _sample;

  //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
  cublasCreate(&(this->dense_handle_));
  cusparseCreate(&(this->sparse_handle_));
  cusparseCreateMatDescr(&(this->mat_descr_));
  cusparseSetMatType(this->mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(this->mat_descr_, CUSPARSE_INDEX_BASE_ZERO);

  //Do the actual optimisation
  this->optimiser_->minimise(
			    this, 
			    this->num_samples_, 
			    this->W_, 
			    _global_batch_size, 
			    _tol, 
			    _max_num_epochs, 
			    this->sum_gradients_
			    ); 
  
  //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
  cublasDestroy(this->dense_handle_);
  cusparseDestroyMatDescr(this->mat_descr_);
  cusparseDestroy(this->sparse_handle_);
			
  //Clear data. so it does not unnecessarily take up space on the 
  this->delete_data();

}
