
class RelationalNetworkCpp: public NumericallyOptimisedAlgorithmCpp {
  
private:

  NeuralNetworkCpp *output_network_;//Output of the relational network 
                                    //- contains aggregation nodes that 
                                    //link to the input networks

  std::vector<*NeuralNetworkCpp> input_networks_;//Input to to the aggregations contained in
                                                 //the output network

  std::vector<std::vector<std::int32_t>> join_keys_left_;//Keys by which the input data is merged
                                                         //with the output data, must be in form of
                                                         //integers

  std::vector<std::int32_t> join_key_used_;//Signifies which of the join_keys_left
                                           //each input_network_ uses

  thrust::device_vector<float> W_;//Vector containing weights of ALL neural networks 

  float *w_ptr_;//Pointer to W_
  	    
public:
	
  RelationalNetworkCpp () {};

  ~RelationalNetworkCpp() {};

  //Finalises the relational network
  void finalise();
  
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

  //A bunch of getters and setters

  std::vector<std::int32_t>& get_join_keys_left(std::int32_t i) {
   
    return this->join_keys_left_[i];
    
  }

};
