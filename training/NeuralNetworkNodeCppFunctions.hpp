NeuralNetworkNodeCpp::NeuralNetworkNodeCpp(std::int32_t _node_number,
		std::int32_t _dim, std::int32_t *_input_nodes_fed_into_me_dense,
		std::int32_t _input_nodes_fed_into_me_dense_length,
		std::int32_t *_input_nodes_fed_into_me_sparse,
		std::int32_t _input_nodes_fed_into_me_sparse_length,
		std::int32_t *_hidden_nodes_fed_into_me,
		std::int32_t _hidden_nodes_fed_into_me_length,
		std::int32_t _i_share_weights_with, bool _no_weight_updates,
		RegulariserCpp *_regulariser) {

	//Transfer input parameters
	this->node_number_ = _node_number;
	this->dim_ = _dim;
	this->i_share_weights_with_ = _i_share_weights_with;
	this->no_weight_updates_ = _no_weight_updates;
	this->regulariser_ = _regulariser;

	//Make sure that the nodes pointed to are smaller than the node number
	for (std::int32_t i = 0; i < _hidden_nodes_fed_into_me_length; ++i)
		if (_hidden_nodes_fed_into_me[i] >= this->node_number_)
			throw std::invalid_argument(
					"hidden_nodes_fed_into_me must always be smaller than node number!");

	//Make sure that i_share_weights_with is smaller than the node number
	if (i_share_weights_with_ >= this->node_number_)
		throw std::invalid_argument(
				"i_share_weights_with must always be smaller than node number!");

	//Declare this->input_nodes_fed_into_me_dense and transfer parameters
	this->input_nodes_fed_into_me_dense_ = std::vector < std::int32_t
			> (_input_nodes_fed_into_me_dense_length);
	for (std::int32_t i = 0; i < _input_nodes_fed_into_me_dense_length; ++i)
		this->input_nodes_fed_into_me_dense_[i] =
				_input_nodes_fed_into_me_dense[i];

	//Declare this->input_nodes_fed_into_me_sparse and transfer parameters
	this->input_nodes_fed_into_me_sparse_ = std::vector < std::int32_t
			> (_input_nodes_fed_into_me_sparse_length);
	for (std::int32_t i = 0; i < _input_nodes_fed_into_me_sparse_length; ++i)
		this->input_nodes_fed_into_me_sparse_[i] =
				_input_nodes_fed_into_me_sparse[i];

	//Declare this->hidden_nodes_fed_into_me and transfer parameters
	this->hidden_nodes_fed_into_me_ = std::vector < std::int32_t
			> (_hidden_nodes_fed_into_me_length);
	for (std::int32_t i = 0; i < _hidden_nodes_fed_into_me_length; ++i)
		this->hidden_nodes_fed_into_me_[i] = _hidden_nodes_fed_into_me[i];

	//All objects that inherit from NeuralNetworkNodeCpp must also calculate this->NumWeightsRequired!

}

NeuralNetworkNodeCpp::~NeuralNetworkNodeCpp() {
}
