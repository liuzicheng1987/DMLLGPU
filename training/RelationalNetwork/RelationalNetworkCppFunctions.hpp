//Finalises the relational network
void RelationalNetworkCpp::finalise() {};
	  
//The purpose of this function is to calculate the gradient of the weights
void RelationalNetworkCpp::dfdw(/*MPI_Comm comm,*/
				float             *_dLdw, 
				const float       *_W, 
				const std::int32_t _batch_begin, 
				const std::int32_t _batch_end, 
				const std::int32_t _batch_size, 
				const std::int32_t _batch_num, 
				const std::int32_t _epoch_num
				) {



};

//The purpose of this function is to calculate the batch size
//and make sure that the batch sizes in all samples are coherent
std::vector<std::int32_t> RelationalNetworkCpp::calculate_batch_size_and_ensure_coherence() {

	//Get batch_size
    std::vector<std::int32_t> batch_size;

    if (this->dense_input_data_.size() > 0) {

      for (auto data: this->dense_input_data_[0])
    	  batch_size.push_back(data.batch_size);

    } else if (this->sparse_input_data_.size() > 0) {

      for (auto data: this->sparse_input_data_[0])
    	  batch_size.push_back(data.batch_size);

    } else {

      throw std::invalid_argument("No input data provided!");

    }

    //Make sure that the batch_sizes are identical for all matrices provided!
    for (auto data_vector: this->dense_input_data_) {

      if (data_vector.size() != batch_size.size())
	throw std::invalid_argument("All input matrices must have the exact same number of samples!");

      for (std::size_t i=0; i<data_vector.size(); ++i)
	if (data_vector[i].batch_size != batch_size[i])
	  throw std::invalid_argument("All input matrices must have the exact same number of samples!");

    }

    //Make sure that the batch_sizes are identical for all matrices provided!
    for (auto data_vector: this->sparse_input_data_) {

      if (data_vector.size() != batch_size.size())
	throw std::invalid_argument("All input matrices must have the exact same number of samples!");

      for (std::size_t i=0; i<data_vector.size(); ++i)
	if (data_vector[i].batch_size != batch_size[i])
	  throw std::invalid_argument("All input matrices must have the exact same number of samples!");

    }

    return batch_size;

}

//The purpose of this function is to generate a prediction through the fitted network
void RelationalNetworkCpp::transform(
		 float       *_Yhat,
		 std::int32_t _Y2_num_samples,
		 std::int32_t _Y2_dim,
		 bool         _sample,
		 std::int32_t _sample_size,
		 bool         _get_hidden_nodes
		 ) {

    //Make sure that relational network has been finalised!
    if (!this->finalised_) throw std::invalid_argument(
    		"Relational network has not been finalised!"
    		);

    std::vector<std::int32_t> batch_size
    		= this->calculate_batch_size_and_ensure_coherence();

    //Get num_batches
    std::int32_t num_batches = static_cast<std::int32_t>(
    		batch_size.size()
    		);

    //Get batch_size_aggregation
    this->batch_size_aggregation_.clear();

    for (auto& input_network: this->input_networks_) {

    	this->batch_size_aggregation_.push_back(
    			input_network.calculate_batch_size_and_ensure_coherence();
    	);

    };

    //Store input values
    this->num_samples_ = _Y2_num_samples;
    this->sample_ = _sample;
    if (!_sample) _sample_size = 1;

    const double SampleAvg = 1.0/((double)_sample_size);

    //Set pointers contained in the NeuralNetworkNodes class
    for (std::size_t n=0; n<this->nodes_.size(); ++n)
      this->nodes_[n]->W_ = thrust::raw_pointer_cast(this->W_.data())
	+ this->cumulative_num_weights_required_[n];

    //Init YhatTemp
    thrust::device_vector<float> YhatTemp(
					  _Yhat,
					  _Yhat + _Y2_num_samples*_Y2_dim
					  );

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    cublasCreate(&(this->dense_handle_));


}
