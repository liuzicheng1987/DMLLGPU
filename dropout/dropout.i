//----------------------------------------------------------------------------------------------
//class DropoutCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class DropoutCpp: public NeuralNetworkNodeGPUCpp {

public:

  DropoutCpp (
	      std::int32_t    _node_number,
	      float           _dropout_probability,
	      std::int32_t    _numbers_per_kernel,
	      std::int32_t    _num_kernels,
	      std::int32_t   *_input_nodes_fed_into_me_dense,
	      std::int32_t    _input_nodes_fed_into_me_dense_length,
	      std::int32_t   *_input_nodes_fed_into_me_sparse,
	      std::int32_t    _input_nodes_fed_into_me_sparse_length,
	      std::int32_t   *_hidden_nodes_fed_into_me,
	      std::int32_t    _hidden_nodes_fed_into_me_length
	      );

};

//----------------------------------------------------------------------------------------------
