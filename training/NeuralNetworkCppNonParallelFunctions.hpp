NeuralNetworkCpp::NeuralNetworkCpp(std::int32_t *_num_input_nodes_dense,
				   std::int32_t _num_input_nodes_dense_length,
				   std::int32_t *_num_input_nodes_sparse,
				   std::int32_t _num_input_nodes_sparse_length,
				   std::int32_t _num_output_nodes_dense,
				   std::int32_t _num_output_nodes_sparse, LossFunctionCpp *_loss) : NumericallyOptimisedAlgorithmCpp()
{

    //Make that the input is reasonable
    if (_num_input_nodes_dense_length + _num_input_nodes_sparse_length <= 0)
	throw std::invalid_argument(
	    "You must provide at least some input nodes!");

    if (_num_output_nodes_dense + _num_output_nodes_sparse <= 0)
	throw std::invalid_argument(
	    "You must provide at least some output nodes!");

    if (std::any_of(_num_input_nodes_dense,
		    _num_input_nodes_dense + _num_input_nodes_dense_length,
		    [](int i) { return i <= 0; }))
	throw std::invalid_argument(
	    "Width of all input matrices must be greater than 0!");

    if (std::any_of(_num_input_nodes_sparse,
		    _num_input_nodes_sparse + _num_input_nodes_sparse_length,
		    [](int i) { return i <= 0; }))
	throw std::invalid_argument(
	    "Width of all input matrices must be greater than 0!");

    //Init num_hidden_nodes
    this->num_hidden_nodes_ = (std::size_t)0;

    //Init num_output_nodes
    this->num_output_nodes_dense_ = _num_output_nodes_dense;
    this->num_output_nodes_sparse_ = _num_output_nodes_sparse;
    this->num_output_nodes_ = _num_output_nodes_dense + _num_output_nodes_sparse;

    //Set up input data and target data
    this->dense_input_data_ = std::vector<std::vector<matrix::DenseMatrix>>(_num_input_nodes_dense_length);

    this->sparse_input_data_ = std::vector<std::vector<matrix::CSRMatrix>>(_num_input_nodes_sparse_length);

    this->dense_targets_ = std::vector<std::vector<matrix::DenseMatrix>>(_num_output_nodes_dense);

    this->sparse_targets_ = std::vector<std::vector<matrix::COOVector>>(_num_output_nodes_sparse);

    this->dense_input_data_dim_ = std::vector<std::int32_t>(_num_input_nodes_dense_length);

    this->sparse_input_data_dim_ = std::vector<std::int32_t>(_num_input_nodes_sparse_length);

    this->dense_targets_dim_ = std::vector<std::int32_t>(_num_output_nodes_dense);

    this->sparse_targets_dim_ = std::vector<std::int32_t>(_num_output_nodes_sparse);

    //Transfer number of input nodes
    std::copy(_num_input_nodes_dense,
	      _num_input_nodes_dense + _num_input_nodes_dense_length,
	      this->dense_input_data_dim_.data());

    std::copy(_num_input_nodes_sparse,
	      _num_input_nodes_sparse + _num_input_nodes_sparse_length,
	      this->sparse_input_data_dim_.data());

    this->loss_ = _loss;
    this->loss_->set_neural_net(this);

    this->nodes_ = std::vector<NeuralNetworkNodeCpp *>(this->num_output_nodes_);
    this->output_nodes_ = nodes_.data();
    this->output_nodes_dense_ = this->output_nodes_;
    this->output_nodes_sparse_ = this->output_nodes_ + _num_output_nodes_dense;

    //Initialise to nullptr
    std::fill(this->nodes_.begin(), this->nodes_.end(), nullptr);

    //Since neural network has not been finalised, set finalised to false
    this->finalised_ = false;
}

NeuralNetworkCpp::~NeuralNetworkCpp(){};

void NeuralNetworkCpp::init_hidden_node(NeuralNetworkNodeCpp *_hidden_node)
{

    //Make sure that the neural network has not already been finalised!
    if (this->finalised_)
	throw std::invalid_argument(
	    "Neural network has already been finalised!");

    if (_hidden_node->node_number_ >= this->num_hidden_nodes_)
    {

	std::int32_t num_additional_nodes = _hidden_node->node_number_ + 1 - this->num_hidden_nodes_;

	//Extend hidden nodes vector
	std::vector<NeuralNetworkNodeCpp *>::iterator it = this->nodes_.begin() + this->nodes_.size();
	this->nodes_.insert(it, num_additional_nodes, nullptr);

	//Increase num_hidden_nodes and reset pointers output_nodes, output_nodes_dense
	//and output_nodes_sparse
	this->num_hidden_nodes_ += num_additional_nodes;
	this->output_nodes_ = nodes_.data() + this->num_hidden_nodes_;
	this->output_nodes_dense_ = this->output_nodes_;
	this->output_nodes_sparse_ = this->output_nodes_ + this->num_output_nodes_dense_;

	//Increase node_number of output_nodes
	for (std::int32_t i = 0; i < this->num_output_nodes_; ++i)
	    if (this->output_nodes_[i] != nullptr)
		this->output_nodes_[i]->node_number_ += num_additional_nodes;
    }

    this->nodes_[_hidden_node->node_number_] = _hidden_node;
};

void NeuralNetworkCpp::init_output_node(NeuralNetworkNodeCpp *_output_node)
{

    //Make sure that the neural network has not already been finalised!
    if (this->finalised_)
	throw std::invalid_argument(
	    "Neural network has already been finalised!");

    //Make sure that node number is in range
    if (_output_node->node_number_ >= (std::int32_t)(this->nodes_.size()) || _output_node->node_number_ < 0)
	throw std::invalid_argument("Output node: Node number out of range!");

    this->nodes_[_output_node->node_number_] = _output_node;
};

void NeuralNetworkCpp::finalise(
    /*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/ float _weight_init_range)
{

    //Make sure that neural net has not been finalised already
    if (this->finalised_ == true)
	throw std::invalid_argument(
	    "Neural network has already been finalised!");

    //Make sure that all nodes were initialised
    if (std::any_of(this->nodes_.begin(), this->nodes_.end(),
		    [](NeuralNetworkNodeCpp *node) { return node == nullptr; }))
	throw std::invalid_argument("Not all nodes have been initialised!");

    //Calculate pointer to hidden nodes fed into me
    for (auto node : this->nodes_)
    {

	node->hidden_nodes_fed_into_me_ptr_.clear();
	for (auto i : node->hidden_nodes_fed_into_me_)
	    node->hidden_nodes_fed_into_me_ptr_.push_back(this->nodes_[i]);
    }

    //Transfer number to input nodes to nodes, so we can calculate the number of weights needed
    for (auto node : this->nodes_)
    {

	//Set initial value to zero
	node->num_input_nodes_cumulative_ = 0;

	//Make sure dense input is in range
	if (std::any_of(node->input_nodes_fed_into_me_dense_.begin(),
			node->input_nodes_fed_into_me_dense_.end(),
			[this](std::int32_t i) {
			    return (i < 0) ||
				   (i >= static_cast<std::int32_t>(this->dense_input_data_dim_.size()));
			}))
	    throw std::invalid_argument("input_dense out of bounds!");

	//Make sure sparse input is in range
	if (std::any_of(node->input_nodes_fed_into_me_sparse_.begin(),
			node->input_nodes_fed_into_me_sparse_.end(),
			[this](std::int32_t i) {
			    return (i < 0) ||
				   (i >= static_cast<std::int32_t>(this->sparse_input_data_dim_.size()));
			}))
	    throw std::invalid_argument("input_sparse out of bounds!");

	//Add dense input
	for (auto dense : node->input_nodes_fed_into_me_dense_)
	    node->num_input_nodes_cumulative_ +=
		this->dense_input_data_dim_[dense];

	//Add sparse input
	for (auto sparse : node->input_nodes_fed_into_me_sparse_)
	    node->num_input_nodes_cumulative_ +=
		this->sparse_input_data_dim_[sparse];
    }

    //Transfer number of output nodes to targets
    for (int i = 0; i < this->num_output_nodes_dense_; ++i)
	dense_targets_dim_[i] = this->output_nodes_[i]->dim_;

    for (int i = 0; i < this->num_output_nodes_sparse_; ++i)
	sparse_targets_dim_[i] =
	    this->output_nodes_[num_output_nodes_dense_ + i]->dim_;

    //Calculate cumulative_num_weights_required and initialise W
    std::int32_t length_W = 0;

    this->cumulative_num_weights_required_.clear();

    for (auto node : this->nodes_)
    {

	node->neural_net_ = this;

	if (node->i_share_weights_with_ < 0)
	{

	    //If node does not share weights with another node, then count lengthW
	    this->cumulative_num_weights_required_.push_back(length_W);
	    length_W += node->get_num_weights_required();
	}
	else
	{

	    //If node does share weights with another node, then make sure num_weights_required match
	    if (node->get_num_weights_required() != this->nodes_[node->i_share_weights_with_]->get_num_weights_required())
		std::invalid_argument(
		    "Number of weights of nodes must match for weight sharing to be possible!");

	    this->cumulative_num_weights_required_.push_back(
		this->cumulative_num_weights_required_[node->i_share_weights_with_]);
	}
    }

    this->cumulative_num_weights_required_.push_back(length_W);

    //Init Whost
    std::vector<float> Whost(length_W);

    //Note that we deliberately choose a constant seed to get the same output every time we call the function
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dist(_weight_init_range * (-1.0f),
					       _weight_init_range);

    //Initialise weight vector
    for (std::int32_t i = 0;
	 i < length_W;
	 ++i)
    {
	Whost[i] = dist(gen);
    }

    //Transfor to device vector
    this->W_ = thrust::device_vector<float>(Whost.data(),
					    Whost.data() + Whost.size());

    //Set finalised to true so we know we can now fit the neural network
    this->finalised_ = true;
}

std::int32_t NeuralNetworkCpp::get_length_params()
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");

    return (std::int32_t)(this->W_.size());
};

void NeuralNetworkCpp::get_params(float *_W, std::int32_t _length_W)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");

    for (std::int32_t i = 0; i < _length_W; ++i)
	_W[i] = this->W_[i];
}

void NeuralNetworkCpp::set_params(float *_W, std::int32_t _length_W)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");

    if (_length_W != static_cast<std::int32_t>(this->W_.size()))
	throw std::invalid_argument(
	    "Length of provided weight vector does not match expected size!");

    for (std::int32_t i = 0; i < _length_W; ++i)
	this->W_[i] = _W[i];
}

std::int32_t NeuralNetworkCpp::get_input_nodes_fed_into_me_dense_length(
    std::int32_t _node_number)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");
    if (_node_number < 0 || _node_number >= (std::int32_t)(nodes_.size()))
	std::invalid_argument("node_number out of bounds!");

    return (std::int32_t)(
	this->nodes_[_node_number]->input_nodes_fed_into_me_dense_.size());
};

void NeuralNetworkCpp::get_input_nodes_fed_into_me_dense(
    std::int32_t _node_number, std::int32_t *_input_nodes_fed_into_me_dense,
    std::int32_t _input_nodes_fed_into_me_dense_length)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");
    if (_node_number < 0 || _node_number >= (std::int32_t)(nodes_.size()))
	std::invalid_argument("node_number out of bounds!");

    for (std::int32_t i = 0; i < _input_nodes_fed_into_me_dense_length; ++i)
	_input_nodes_fed_into_me_dense[i] =
	    this->nodes_[_node_number]->input_nodes_fed_into_me_dense_[i];
};

std::int32_t NeuralNetworkCpp::get_input_nodes_fed_into_me_sparse_length(
    std::int32_t _node_number)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");

    if (_node_number < 0 || _node_number >= (std::int32_t)(nodes_.size()))
	std::invalid_argument("node_number out of bounds!");

    return (std::int32_t)(
	this->nodes_[_node_number]->input_nodes_fed_into_me_sparse_.size());
};

void NeuralNetworkCpp::get_input_nodes_fed_into_me_sparse(
    std::int32_t _node_number,
    std::int32_t *_input_nodes_fed_into_me_sparse,
    std::int32_t _input_nodes_fed_into_me_sparse_length)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");
    if (_node_number < 0 || _node_number >= (std::int32_t)(nodes_.size()))
	std::invalid_argument("node_number out of bounds!");

    for (std::int32_t i = 0; i < _input_nodes_fed_into_me_sparse_length; ++i)
	_input_nodes_fed_into_me_sparse[i] =
	    this->nodes_[_node_number]->input_nodes_fed_into_me_sparse_[i];
};

std::int32_t NeuralNetworkCpp::get_hidden_nodes_fed_into_me_length(
    std::int32_t _node_number)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");
    if (_node_number < 0 || _node_number >= (std::int32_t)(nodes_.size()))
	std::invalid_argument("node_number out of bounds!");

    return (std::int32_t)(
	this->nodes_[_node_number]->hidden_nodes_fed_into_me_.size());
};

void NeuralNetworkCpp::get_hidden_nodes_fed_into_me(std::int32_t _node_number,
						    std::int32_t *_hidden_nodes_fed_into_me,
						    std::int32_t __lengthhidden_nodes_fed_into_me)
{

    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");
    if (_node_number < 0 || _node_number >= (std::int32_t)(nodes_.size()))
	std::invalid_argument("node_number out of bounds!");

    for (std::int32_t i = 0; i < __lengthhidden_nodes_fed_into_me; ++i)
	_hidden_nodes_fed_into_me[i] =
	    this->nodes_[_node_number]->hidden_nodes_fed_into_me_[i];
};

void NeuralNetworkCpp::load_dense(std::vector<matrix::DenseMatrix> &_data,
				  float *_X, std::int32_t _num_samples, std::int32_t _dim,
				  std::int32_t _num_batches, bool _transpose)
{

    std::int32_t batch_begin, batch_end, batch_size;

    for (std::int32_t batch_num = 0; batch_num < _num_batches; ++batch_num)
    {

	this->calc_batch_begin_end(batch_begin, batch_end, batch_size,
				   batch_num, _num_samples, _num_batches);

	//Transfer _num_samples and _dim
	_data[batch_num].batch_size = batch_size;
	_data[batch_num].dim = _dim;

	std::vector<float> X_transpose;

	//Target vectors need to be transposed
	if (_transpose)
	{

	    X_transpose = std::vector<float>(batch_size * _dim);

	    //Transpose target
	    for (std::int32_t i = 0; i < batch_size; ++i)
		for (std::int32_t j = 0; j < _dim; ++j)
		    X_transpose[j * batch_size + i] = _X[(batch_begin + i) * _dim + j];

	    //Transfer to
	    _data[batch_num].X = thrust::device_vector<float>(
		X_transpose.begin(), X_transpose.end());
	}
	else
	{

	    //Transfer X to
	    _data[batch_num].X = thrust::device_vector<float>(
		_X + batch_begin * _dim, _X + batch_end * _dim);
	}

	//Set X_ptr
	_data[batch_num].X_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X.data());
    }
}

//This function is especially for relational networks
void NeuralNetworkCpp::load_dense(std::vector<matrix::DenseMatrix> &_data,
				  float *_X, std::int32_t _num_samples, std::int32_t _dim,
				  std::int32_t _num_batches, bool _transpose, std::int32_t *_indptr,
				  std::int32_t _indptr_length)
{

    std::int32_t batch_begin, batch_end, batch_size;

    if (_num_batches != _indptr_length - 1)
	throw std::invalid_argument(
	    "_num_batches and _indptr_length do not match!");

    _data = std::vector<matrix::DenseMatrix>(_num_batches);

    for (std::int32_t batch_num = 0; batch_num < _num_batches; ++batch_num)
    {

	//For readability
	batch_begin = _indptr[batch_num];
	batch_end = _indptr[batch_num + 1];
	batch_size = batch_end - batch_begin;

	//Transfer _num_samples and _dim
	_data[batch_num].batch_size = batch_size;
	_data[batch_num].dim = _dim;

	std::vector<float> X_transpose;

	//Target vectors need to be transposed
	if (_transpose)
	{

	    X_transpose = std::vector<float>(batch_size * _dim);

	    //Transpose target
	    for (std::int32_t i = 0; i < batch_size; ++i)
		for (std::int32_t j = 0; j < _dim; ++j)
		    X_transpose[j * batch_size + i] = _X[(batch_begin + i) * _dim + j];

	    //Transfer to
	    _data[batch_num].X = thrust::device_vector<float>(
		X_transpose.begin(), X_transpose.end());
	}
	else
	{

	    //Transfer X to
	    _data[batch_num].X = thrust::device_vector<float>(
		_X + batch_begin * _dim, _X + batch_end * _dim);
	}

	//Set X_ptr
	_data[batch_num].X_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X.data());
    }
}

void NeuralNetworkCpp::load_dense_data(std::int32_t _num_input_node, float *_X,
				       std::int32_t _num_samples, std::int32_t _dim,
				       std::int32_t _global_batch_size)
{

    if (_num_input_node >= (std::int32_t)(this->dense_input_data_.size()) || _num_input_node < 0)
	throw std::invalid_argument("num_input_node out of bounds!");

    if (_dim != this->dense_input_data_dim_[_num_input_node])
	throw std::invalid_argument(
	    "Width dim of array provided does not match the width that has been set when initialising the network!");

    std::int32_t num_batches = calc_num_batches(
	/*MPI_Comm comm,*/
	_num_samples, _global_batch_size);

    this->dense_input_data_[_num_input_node] = std::vector<matrix::DenseMatrix>(num_batches);

    this->load_dense(this->dense_input_data_[_num_input_node], _X, _num_samples,
		     _dim, num_batches, false //do not transpose
		     );
}

void NeuralNetworkCpp::load_dense_targets(std::int32_t _num_output_node,
					  float *_Y, std::int32_t _num_samples, std::int32_t _dim,
					  std::int32_t _global_batch_size)
{

    if (_num_output_node >= (std::int32_t)(this->dense_targets_.size()) || _num_output_node < 0)
	throw std::invalid_argument("num_output_node out of bounds!");

    if (_dim != this->dense_targets_dim_[_num_output_node])
	throw std::invalid_argument(
	    "Width dim of array provided does not match the width that has been set when initialising the network!");

    //Calculates the number of batches needed
    std::int32_t num_batches = calc_num_batches(
	/*MPI_Comm comm,*/
	_num_samples, _global_batch_size);

    this->dense_targets_[_num_output_node] = std::vector<matrix::DenseMatrix>(num_batches);

    this->load_dense(this->dense_targets_[_num_output_node], _Y, _num_samples,
		     _dim, num_batches, true //do transpose
		     );
}

void NeuralNetworkCpp::load_csr(std::vector<matrix::CSRMatrix> &_data,
				float *_X_data, std::int32_t _X_data_length, std::int32_t *_X_indices,
				std::int32_t _X_indices_length, std::int32_t *_X_indptr,
				std::int32_t _X_indptr_length, std::int32_t _num_samples,
				std::int32_t _dim, std::int32_t _num_batches)
{

    std::int32_t batch_begin, batch_end, batch_size;

    for (std::int32_t batch_num = 0; batch_num < _num_batches; ++batch_num)
    {

	this->calc_batch_begin_end(batch_begin, batch_end, batch_size,
				   batch_num, _num_samples, _num_batches);

	//Transfer _num_samples and _dim
	_data[batch_num].batch_size = batch_size;
	_data[batch_num].dim = _dim;
	_data[batch_num].num_non_zero = _X_indptr[batch_end] - _X_indptr[batch_begin];

	//Transfer X_data to  and set X_data_ptr
	_data[batch_num].X_data = thrust::device_vector<float>(
	    _X_data + _X_indptr[batch_begin],
	    _X_data + _X_indptr[batch_end]);

	_data[batch_num].X_data_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_data.data());

	//Transfer X_indices to  and set X_indices_ptr
	_data[batch_num].X_indices = thrust::device_vector<std::int32_t>(_X_indices + _X_indptr[batch_begin], _X_indices + _X_indptr[batch_end]);

	_data[batch_num].X_indices_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_indices.data());

	//Transfer X_indptr to  and set X_indptr_ptr
	//Do not forget the last element - it is important
	_data[batch_num].X_indptr = thrust::device_vector<std::int32_t>(&_X_indptr[batch_begin], &_X_indptr[batch_end] + 1
									//_X_indptr has size batch_size + 1
									);

	_data[batch_num].X_indptr_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_indptr.data());

	//Substract value of first elements from all elements in X_indptr
	thrust::for_each(_data[batch_num].X_indptr.begin(),
			 _data[batch_num].X_indptr.end(), thrust::placeholders::_1 -=
							  _X_indptr[batch_begin]);
    }
}

void NeuralNetworkCpp::load_csr(std::vector<matrix::CSRMatrix> &_data,
				float *_X_data, std::int32_t _X_data_length, std::int32_t *_X_indices,
				std::int32_t _X_indices_length, std::int32_t *_X_indptr,
				std::int32_t _X_indptr_length, std::int32_t _num_samples,
				std::int32_t _dim, std::int32_t _num_batches, std::int32_t *_indptr,
				std::int32_t _indptr_length)
{

    std::int32_t batch_begin, batch_end, batch_size;

    if (_num_batches != _indptr_length - 1)
	throw std::invalid_argument(
	    "_num_batches and _indptr_length do not match!");

    for (std::int32_t batch_num = 0; batch_num < _num_batches; ++batch_num)
    {

	//For readability
	batch_begin = _indptr[batch_num];
	batch_end = _indptr[batch_num + 1];
	batch_size = batch_end - batch_begin;

	//Transfer _num_samples and _dim
	_data[batch_num].batch_size = batch_size;
	_data[batch_num].dim = _dim;
	_data[batch_num].num_non_zero = _X_indptr[batch_end] - _X_indptr[batch_begin];

	//Transfer X_data to  and set X_data_ptr
	_data[batch_num].X_data = thrust::device_vector<float>(
	    _X_data + _X_indptr[batch_begin],
	    _X_data + _X_indptr[batch_end]);

	_data[batch_num].X_data_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_data.data());

	//Transfer X_indices to  and set X_indices_ptr
	_data[batch_num].X_indices = thrust::device_vector<std::int32_t>(_X_indices + _X_indptr[batch_begin], _X_indices + _X_indptr[batch_end]);

	_data[batch_num].X_indices_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_indices.data());

	//Transfer X_indptr to  and set X_indptr_ptr
	//Do not forget the last element - it is important
	_data[batch_num].X_indptr = thrust::device_vector<std::int32_t>(&_X_indptr[batch_begin], &_X_indptr[batch_end] + 1
									//_X_indptr has size batch_size + 1
									);

	_data[batch_num].X_indptr_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_indptr.data());

	//Substract value of first elements from all elements in X_indptr
	thrust::for_each(_data[batch_num].X_indptr.begin(),
			 _data[batch_num].X_indptr.end(), thrust::placeholders::_1 -=
							  _X_indptr[batch_begin]);
    }
}

//This function expects a CSR matrix, but transforms it into a COO vector on the GPU
void NeuralNetworkCpp::load_coo(std::vector<matrix::COOVector> &_data,
				float *_X_data, std::int32_t _X_data_length, std::int32_t *_X_indices,
				std::int32_t _X_indices_length, std::int32_t *_X_indptr,
				std::int32_t _X_indptr_length, std::int32_t _num_samples,
				std::int32_t _dim, std::int32_t _num_batches)
{

    std::int32_t batch_begin, batch_end, batch_size;

    thrust::device_vector<std::int32_t> X_col;

    for (std::int32_t batch_num = 0; batch_num < _num_batches; ++batch_num)
    {

	this->calc_batch_begin_end(batch_begin, batch_end, batch_size,
				   batch_num, _num_samples, _num_batches);

	//Transfer _num_samples and _dim
	_data[batch_num].batch_size = batch_size;
	_data[batch_num].dim = _dim;
	_data[batch_num].num_non_zero = _X_indptr[batch_end] - _X_indptr[batch_begin];

	//Transfer X_data to  and set X_data_ptr
	_data[batch_num].X_data = thrust::device_vector<float>(
	    _X_data + _X_indptr[batch_begin],
	    _X_data + _X_indptr[batch_end]);

	_data[batch_num].X_data_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_data.data());

	//Transfer X_indices to  (which we rename X_row)
	//Remember that the output of the neural net will be in column-major
	//order, so we have to multiply by batch_size!
	_data[batch_num].X_indices = thrust::device_vector<std::int32_t>(_X_indices + _X_indptr[batch_begin], _X_indices + _X_indptr[batch_end]);

	_data[batch_num].X_indices_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_indices.data());

	//Multiply by batch_size - remember that the output is in column major order!
	thrust::for_each(_data[batch_num].X_indices.begin(),
			 _data[batch_num].X_indices.end(), thrust::placeholders::_1 *=
							   batch_size);

	//Transfer X_indptr to
	//X_indptr is transformed into X_col
	X_col = thrust::device_vector<std::int32_t>(_X_indptr[batch_end] - _X_indptr[batch_begin]);

	//Transform the X_indptr vector into a column vector
	for (std::int32_t i = 0; i < batch_size; ++i)
	    thrust::fill(
		X_col.begin() + _X_indptr[batch_begin + i] - _X_indptr[batch_begin],
		X_col.begin() + _X_indptr[batch_begin + i + 1] - _X_indptr[batch_begin], i);

	//Now, add X_col to X_indices
	thrust::transform(_data[batch_num].X_indices.begin(),
			  _data[batch_num].X_indices.end(), X_col.begin(),
			  _data[batch_num].X_indices.begin(),
			  thrust::plus<std::int32_t>());
    }
}

//This function expects a CSR matrix, but transforms it into a COO vector on the GPU
void NeuralNetworkCpp::load_coo(std::vector<matrix::COOVector> &_data,
				float *_X_data, std::int32_t _X_data_length, std::int32_t *_X_indices,
				std::int32_t _X_indices_length, std::int32_t *_X_indptr,
				std::int32_t _X_indptr_length, std::int32_t _num_samples,
				std::int32_t _dim, std::int32_t _num_batches, std::int32_t *_indptr,
				std::int32_t _indptr_length)
{

    std::int32_t batch_begin, batch_end, batch_size;

    if (_num_batches != _indptr_length - 1)
	throw std::invalid_argument(
	    "_num_batches and _indptr_length do not match!");

    thrust::device_vector<std::int32_t> X_col;

    for (std::int32_t batch_num = 0; batch_num < _num_batches; ++batch_num)
    {

	//For readability
	batch_begin = _indptr[batch_num];
	batch_end = _indptr[batch_num + 1];
	batch_size = batch_end - batch_begin;

	//Transfer _num_samples and _dim
	_data[batch_num].batch_size = batch_size;
	_data[batch_num].dim = _dim;
	_data[batch_num].num_non_zero = _X_indptr[batch_end] - _X_indptr[batch_begin];

	//Transfer X_data to  and set X_data_ptr
	_data[batch_num].X_data = thrust::device_vector<float>(
	    _X_data + _X_indptr[batch_begin],
	    _X_data + _X_indptr[batch_end]);

	_data[batch_num].X_data_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_data.data());

	//Transfer X_indices to  (which we rename X_row)
	//Remember that the output of the neural net will be in column-major
	//order, so we have to multiply by batch_size!
	_data[batch_num].X_indices = thrust::device_vector<std::int32_t>(_X_indices + _X_indptr[batch_begin], _X_indices + _X_indptr[batch_end]);

	_data[batch_num].X_indices_ptr = thrust::raw_pointer_cast(
	    _data[batch_num].X_indices.data());

	//Multiply by batch_size - remember that the output is in column major order!
	thrust::for_each(_data[batch_num].X_indices.begin(),
			 _data[batch_num].X_indices.end(), thrust::placeholders::_1 *=
							   batch_size);

	//Transfer X_indptr to
	//X_indptr is transformed into X_col
	X_col = thrust::device_vector<std::int32_t>(_X_indptr[batch_end] - _X_indptr[batch_begin]);

	//Transform the X_indptr vector into a column vector
	for (std::int32_t i = 0; i < batch_size; ++i)
	    thrust::fill(
		X_col.begin() + _X_indptr[batch_begin + i] - _X_indptr[batch_begin],
		X_col.begin() + _X_indptr[batch_begin + i + 1] - _X_indptr[batch_begin], i);

	//Now, add X_col to X_indices
	thrust::transform(_data[batch_num].X_indices.begin(),
			  _data[batch_num].X_indices.end(), X_col.begin(),
			  _data[batch_num].X_indices.begin(),
			  thrust::plus<std::int32_t>());
    }
}

void NeuralNetworkCpp::load_sparse_data(std::int32_t _num_input_node,
					float *_X_data, std::int32_t _X_data_length, std::int32_t *_X_indices,
					std::int32_t _X_indices_length, std::int32_t *_X_indptr,
					std::int32_t _X_indptr_length, std::int32_t _num_samples,
					std::int32_t _dim, std::int32_t _global_batch_size)
{

    if (_num_input_node >= (std::int32_t)(this->sparse_input_data_.size()) || _num_input_node < 0)
	throw std::invalid_argument("num_input_node out of bounds!");

    if (_dim != this->sparse_input_data_dim_[_num_input_node])
	throw std::invalid_argument(
	    "Width dim of array provided does not match the width that has been set when initialising the network!");

    std::int32_t num_batches = calc_num_batches(
	/*MPI_Comm comm,*/
	_num_samples, _global_batch_size);

    this->sparse_input_data_[_num_input_node] = std::vector<matrix::CSRMatrix>(num_batches);

    this->load_csr(this->sparse_input_data_[_num_input_node], _X_data,
		   _X_data_length, _X_indices, _X_indices_length, _X_indptr,
		   _X_indptr_length, _num_samples, _dim, num_batches);
}

void NeuralNetworkCpp::load_sparse_targets(std::int32_t _num_output_node,
					   float *_Y_data, std::int32_t _Y_data_length, std::int32_t *_Y_indices,
					   std::int32_t _Y_indices_length, std::int32_t *_Y_indptr,
					   std::int32_t _Y_indptr_length, std::int32_t _num_samples,
					   std::int32_t _dim, std::int32_t _global_batch_size)
{

    if (_num_output_node >= (std::int32_t)(this->sparse_targets_.size()) || _num_output_node < 0)
	throw std::invalid_argument("num_output_node out of bounds!");

    if (_dim != this->sparse_targets_dim_[_num_output_node])
	throw std::invalid_argument(
	    "Width dim of array provided does not match the width that has been set when initialising the network!");

    std::int32_t num_batches = calc_num_batches(
	/*MPI_Comm comm,*/
	_num_samples, _global_batch_size); //Calculates the number of batches needed

    this->sparse_targets_[_num_output_node] = std::vector<matrix::COOVector>(num_batches);

    this->load_coo(this->sparse_targets_[_num_output_node], _Y_data,
		   _Y_data_length, _Y_indices, _Y_indices_length, _Y_indptr,
		   _Y_indptr_length, _num_samples, _dim, num_batches);
}

void NeuralNetworkCpp::delete_data()
{

    for (auto &data : this->dense_input_data_)
	data.clear();
    for (auto &data : this->dense_targets_)
	data.clear();
    for (auto &data : this->sparse_input_data_)
	data.clear();
    for (auto &data : this->sparse_targets_)
	data.clear();
}

std::vector<std::int32_t> NeuralNetworkCpp::calculate_batch_size_and_ensure_coherence()
{

    //Get batch_size
    std::vector<std::int32_t> batch_size;

    if (this->dense_input_data_.size() > 0)
    {

	for (auto data : this->dense_input_data_[0])
	    batch_size.push_back(data.batch_size);
    }
    else if (this->sparse_input_data_.size() > 0)
    {

	for (auto data : this->sparse_input_data_[0])
	    batch_size.push_back(data.batch_size);
    }
    else
	throw std::invalid_argument("No input data provided!");

    //Make sure that the batch_sizes are identical for all matrices provided!
    for (auto data_vector : this->dense_input_data_)
    {

	if (data_vector.size() != batch_size.size())
	    throw std::invalid_argument(
		"All input and output matrices must have the exact same number of samples!");

	for (std::size_t i = 0; i < data_vector.size(); ++i)
	    if (data_vector[i].batch_size != batch_size[i])

		throw std::invalid_argument(
		    "All input and output matrices must have the exact same number of samples!");
    }

    //Make sure that the batch_sizes are identical for all matrices provided
    for (auto data_vector : this->sparse_input_data_)
    {

	if (data_vector.size() != batch_size.size())
	    throw std::invalid_argument(
		"All input and output matrices must have the exact same number of samples!");

	for (std::size_t i = 0; i < data_vector.size(); ++i)
	    if (data_vector[i].batch_size != batch_size[i])

		throw std::invalid_argument(
		    "All input and output matrices must have the exact same number of samples!");
    }

    return batch_size;
};

void NeuralNetworkCpp::ensure_coherence_for_target_values(std::vector<std::int32_t> &_batch_size)
{

    for (auto data_vector : this->dense_targets_)
    {

	if (data_vector.size() != _batch_size.size())
	    throw std::invalid_argument(
		"All input and output matrices must have the exact same number of samples!");

	for (std::size_t i = 0; i < data_vector.size(); ++i)
	    if (data_vector[i].batch_size != _batch_size[i])
		throw std::invalid_argument(
		    "All input and output matrices must have the exact same number of samples!");
    }

    for (auto data_vector : this->sparse_targets_)
    {

	if (data_vector.size() != _batch_size.size())
	    throw std::invalid_argument(
		"All input and output matrices must have the exact same number of samples!");

	for (std::size_t i = 0; i < data_vector.size(); ++i)
	    if (data_vector[i].batch_size != _batch_size[i])
		throw std::invalid_argument(
		    "All input and output matrices must have the exact same number of samples!");
    }
}

void NeuralNetworkCpp::transform(float *_Yhat, std::int32_t _Y2_num_samples,
				 std::int32_t _Y2_dim, bool _sample, std::int32_t _sample_size,
				 bool _get_hidden_nodes)
{

    //Make sure that neural network has been finalised!
    if (!this->finalised_)
	throw std::invalid_argument("Neural network has not been finalised!");

    //Get batch_size
    std::vector<std::int32_t> batch_size =
	this->calculate_batch_size_and_ensure_coherence();

    std::int32_t num_batches = static_cast<std::int32_t>(batch_size.size());

    //Store input values
    this->num_samples_ = _Y2_num_samples;
    this->sample_ = _sample;
    if (!_sample)
	_sample_size = 1;

    const double SampleAvg = 1.0 / ((double)_sample_size);

    //Set pointers contained in the NeuralNetworkNodes class
    this->set_node_weights(thrust::raw_pointer_cast(this->W_.data()));

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    this->init_cublas_cusparse_handles();

    //Init YhatTemp
    thrust::device_vector<float> Yhat_temp(_Yhat,
					   _Yhat + _Y2_num_samples * _Y2_dim);

    //Calculate output
    std::int32_t batch_begin = 0;
    for (std::int32_t batch_num = 0; batch_num < num_batches;
	 ++batch_num, batch_begin += batch_size[batch_num])
	for (std::int32_t iteration = 0; iteration < _sample_size;
	     ++iteration)
	{

	    //Calculate nodes
	    for (auto node : this->nodes_)
		node->calc_output(batch_num, batch_size[batch_num]);

	    //Add to YhatTemp
	    std::int32_t col_num = 0;
	    for (std::int32_t node_num = 0; node_num < this->num_output_nodes_;
		 ++node_num)
		for (std::int32_t dim = 0;
		     dim < this->output_nodes_[node_num]->dim_; ++dim)
		{

		    thrust::transform(
			this->output_nodes_[node_num]->output_.begin() + dim * batch_size[batch_num],

			this->output_nodes_[node_num]->output_.begin() + (dim + 1) * batch_size[batch_num],

			Yhat_temp.begin() + _Y2_num_samples * col_num + batch_begin,

			Yhat_temp.begin() + _Y2_num_samples * col_num + batch_begin,

			thrust::plus<float>()

			    );

		    ++col_num;
		}
	}

    //Get data from YhatTemp and transpose
    for (std::int32_t i = 0; i < _Y2_num_samples; ++i)
	for (std::int32_t j = 0; j < _Y2_dim; ++j)
	    _Yhat[i * _Y2_dim + j] = Yhat_temp[j * _Y2_num_samples + i];

    //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
    this->destroy_cublas_cusparse_handles();

    //Clear data, so it does not unnecessarily take up space on the
    this->delete_data();
};
