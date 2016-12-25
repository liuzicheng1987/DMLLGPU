AggregationCpp::AggregationCpp(
			       std::int32_t _node_number, 
			       std::int32_t _dim,
			       std::int32_t _input_network,			       
			       std::int32_t _i_share_weights_with, 
			       bool         _no_weight_updates
			       ) : NeuralNetworkNodeCpp(
							_node_number, 
							_dim,
							nullptr, 
							0, 
						        nullptr, 
							0,
							nullptr, 
							0,
							_i_share_weights_with, 
							_no_weight_updates,
							nullptr
							) {
  
  this->input_network_ = _input_network;
  
};

NeuralNetworkNodeCpp::~NeuralNetworkNodeCpp(){};
