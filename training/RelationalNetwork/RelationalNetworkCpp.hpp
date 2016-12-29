class RelationalNetworkCpp: public NumericallyOptimisedAlgorithmCpp {
  
private:

  //Output of the relational network
  //- contains aggregation nodes that
  //link to the input networks
  NeuralNetworkCpp *output_network_;

  //Input to to the aggregations contained in
  //the output network
  std::vector<NeuralNetworkCpp*> input_networks_;

  //Keys by which the input data is merged
  //with the output data, must be in form of
  //integers, from 0 to number of unique keys
  //minus one
  std::vector<std::vector<std::int32_t>> join_keys_left_;

  //Signifies which of the join_keys_left
  //each input_network_ uses
  std::vector<std::int32_t> join_key_used_;

  //Vector containing weights of ALL neural networks 
  thrust::device_vector<float> W_;

  //Pointer to W_
  float *w_ptr_;

  //Batch sizes of the aggregations of each
  //of the input networks
  std::vector<std::vector<std::int32_t>> batch_size_aggregation_;

  //Beginning of the current batch
  std::int32_t batch_begin;

  //This handle is needed for the cuBLAS library.
  cublasHandle_t dense_handle_;

public:
	
  RelationalNetworkCpp(
  		) : NumericallyOptimisedAlgorithmCpp()
  		{};

  ~RelationalNetworkCpp() {};

  //Finalises the relational network
  void finalise();
  
  //The purpose of this function is to calculate the batch size
  //and make sure that the batch sizes in all samples are coherent
  std::vector<std::int32_t> calculate_batch_size_and_ensure_coherence();

  //The purpose of this function is to calculate the gradient of the weights
  void dfdw(/*MPI_Comm comm,*/
	    float             *_dLdw, 
	    const float       *_W, 
	    const std::int32_t _batch_begin, 
	    const std::int32_t _batch_end, 
	    const std::int32_t _batch_size, 
	    const std::int32_t _batch_num, 
	    const std::int32_t _epoch_num
	    );

  //The purpose of this function is to generate a prediction through the fitted network
  void transform(
		 float       *_Yhat,
		 std::int32_t _Y2_num_samples,
		 std::int32_t _Y2_dim,
		 bool         _sample,
		 std::int32_t _sample_size,
		 bool         _get_hidden_nodes
		 );

  //A bunch of getters and setters

  void set_input_network(
	NeuralNetworkCpp *_input_network
  ) {
	  this->input_networks_.push_back(_input_network);
  }

  void set_output_network(
	NeuralNetworkCpp *_output_network
  ) {
	  this->output_network_ = _output_network;
  }

  void set_join_keys_left(
	std::int32_t *_join_keys_left,
	std::int32_t  _join_keys_left_length
  );

  void set_join_keys_used(
		  std::int32_t *_join_keys_used,
		  std::int32_t  _join_keys_used_length
  );

  std::vector<std::int32_t>& get_join_keys_left(std::int32_t i) {
   
    return this->join_keys_left_[i];
    
  }

  std::int32_t* get_join_keys_left_ptr(std::int32_t _input_network) {

	return this->join_keys_left_[
	                            this->join_key_used_[_input_network]
	                                                 ].data()
	                                                 + this->batch_begin;

  }

  std::int32_t get_batch_size_aggregation(
		  std::int32_t _input_network,
		  std::int32_t _batch_num
		  ) {

	return this->batch_size_aggregation_[_input_network][_batch_num];

  }

  cublasHandle_t& get_dense_handle() {
    return this->dense_handle_;
  };

};
