//----------------------------------------------------------------------------------------------
//class DropoutCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class DropoutCpp: public NeuralNetworkNodeGPUCpp {

public:

  DropoutCpp (
	      std::int32_t    _node_number,
	      float           _dropout_probability,
	      std::int32_t    _numbers_per_kernel,
	      std::int32_t    _num_kernels,
	      std::int32_t   *_hidden_nodes_fed_into_me,
	      std::int32_t    _hidden_nodes_fed_into_me_length
	      );

};

//----------------------------------------------------------------------------------------------
//class NodeSamplerCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class NodeSamplerCpp: public NeuralNetworkNodeGPUCpp {

public:

  NodeSamplerCpp (
		  std::int32_t    _node_number,
		  std::int32_t    _numbers_per_kernel,
		  std::int32_t    _num_kernels,
		  std::int32_t   *_hidden_nodes_fed_into_me,
		  std::int32_t    _hidden_nodes_fed_into_me_length
		  );

};


//----------------------------------------------------------------------------------------------
