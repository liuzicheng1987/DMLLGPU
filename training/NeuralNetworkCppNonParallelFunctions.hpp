NeuralNetworkCpp::NeuralNetworkCpp(
				   std::int32_t    *_num_input_nodes_dense, 
				   std::int32_t     _num_input_nodes_dense_length, 
				   std::int32_t    *_num_input_nodes_sparse, 
				   std::int32_t     _num_input_nodes_sparse_length, 
				   std::int32_t     _num_output_nodes_dense, 
				   std::int32_t     _num_output_nodes_sparse, 
				   LossFunctionCpp *_loss
				   ): NumericallyOptimisedAlgorithmCpp() {

  //Make that the input is reasonable
  if (_num_input_nodes_dense_length + _num_input_nodes_sparse_length <= 0) 
    throw std::invalid_argument("You must provide at least some input nodes!");

  if (_num_output_nodes_dense + _num_output_nodes_sparse <= 0) 
    throw std::invalid_argument("You must provide at least some output nodes!");

  if (std::any_of(_num_input_nodes_dense, 
		  _num_input_nodes_dense + _num_input_nodes_dense_length, 
		  [](int i){return i <= 0;}
		  )
      ) 
    throw std::invalid_argument("Width of all input matrices must be greater than 0!");

  if (std::any_of(
		  _num_input_nodes_sparse, 
		  _num_input_nodes_sparse + _num_input_nodes_sparse_length, 
		  [](int i){return i <= 0;}
		  )
      ) 
    throw std::invalid_argument("Width of all input matrices must be greater than 0!");

  //Init num_hidden_nodes
  this->num_hidden_nodes = (std::size_t)0;
  
  //Init num_output_nodes
  this->num_output_nodes_dense = _num_output_nodes_dense; 
  this->num_output_nodes_sparse = _num_output_nodes_sparse;
  this->num_output_nodes = _num_output_nodes_dense + _num_output_nodes_sparse;

  //Set up input data and target data
  this->dense_input_data = std::vector<std::vector<matrix::DenseMatrix>>(_num_input_nodes_dense_length);
  this->sparse_input_data = std::vector<std::vector<matrix::CSRMatrix>>(_num_input_nodes_sparse_length);
  this->dense_targets = std::vector<std::vector<matrix::DenseMatrix>>(_num_output_nodes_dense);
  this->sparse_targets = std::vector<std::vector<matrix::COOVector>>(_num_output_nodes_sparse);

  this->dense_input_data_dim = std::vector<std::int32_t>(_num_input_nodes_dense_length);
  this->sparse_input_data_dim = std::vector<std::int32_t>(_num_input_nodes_sparse_length);
  this->dense_targets_dim = std::vector<std::int32_t>(_num_output_nodes_dense);
  this->sparse_targets_dim = std::vector<std::int32_t>(_num_output_nodes_sparse);

  //Transfer number of input nodes
  std::copy(
	    _num_input_nodes_dense, 
	    _num_input_nodes_dense + _num_input_nodes_dense_length, 
	    this->dense_input_data_dim.data()
	    );

  std::copy(
	    _num_input_nodes_sparse, 
	    _num_input_nodes_sparse + _num_input_nodes_sparse_length, 
	    this->sparse_input_data_dim.data()
	    );
  
  this->loss = _loss;
  this->loss->set_neural_net(this);
		
  this->nodes = std::vector<NeuralNetworkNodeCpp*>(this->num_output_nodes);		
  this->output_nodes = nodes.data();
  this->output_nodes_dense = this->output_nodes;
  this->output_nodes_sparse = this->output_nodes + _num_output_nodes_dense;

  //Initialise to nullptr
  std::fill(this->nodes.begin(), this->nodes.end(), nullptr);
				
  //Since neural network has not been finalised, set finalised to false
  this->finalised = false;

}
					 
NeuralNetworkCpp::~NeuralNetworkCpp()  {};

void NeuralNetworkCpp::init_hidden_node(NeuralNetworkNodeCpp *_hidden_node) {
	
  //Make sure that the neural network has not already been finalised!
  if (this->finalised) throw std::invalid_argument("Neural network has already been finalised!");

  if (_hidden_node->node_number >= this->num_hidden_nodes) {	

    std::int32_t num_additional_nodes = _hidden_node->node_number + 1 - this->num_hidden_nodes;

    //Extend hidden nodes vector
    std::vector<NeuralNetworkNodeCpp*>::iterator it = this->nodes.begin() + this->nodes.size();
    this->nodes.insert(it, num_additional_nodes, nullptr);	

    //Increase num_hidden_nodes and reset pointers output_nodes, output_nodes_dense 
    //and output_nodes_sparse
    this->num_hidden_nodes += num_additional_nodes;
    this->output_nodes = nodes.data() + this->num_hidden_nodes;
    this->output_nodes_dense = this->output_nodes;
    this->output_nodes_sparse = this->output_nodes + this->num_output_nodes_dense;

    //Increase node_number of output_nodes
    for (std::int32_t i=0; i<this->num_output_nodes; ++i)
      if (this->output_nodes[i] != nullptr) 
	this->output_nodes[i]->node_number += num_additional_nodes;
  
  }

  this->nodes[_hidden_node->node_number] = _hidden_node;
					
};

void NeuralNetworkCpp::init_output_node(NeuralNetworkNodeCpp *_output_node) {
		
  //Make sure that the neural network has not already been finalised!
  if (this->finalised) 
    throw std::invalid_argument("Neural network has already been finalised!");
  
  //Make sure that node number is in range
  if (_output_node->node_number >= (std::int32_t)(this->nodes.size()) || _output_node->node_number < 0) 
    throw std::invalid_argument("Output node: Node number out of range!");
				
  this->nodes[_output_node->node_number] = _output_node;
						
};

void NeuralNetworkCpp::finalise(/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/float _weight_init_range) {

  //Make sure that neural net has not been finalised already
  if (this->finalised == true)
    throw std::invalid_argument("Neural network has already been finalised!");

  //Make sure that all nodes were initialised
  if (std::any_of(this->nodes.begin(), this->nodes.end(), [](NeuralNetworkNodeCpp *node) {return node == nullptr;}))
    throw std::invalid_argument("Not all nodes have been initialised!");
    
  //Calculate pointer to hidden nodes fed into me
  for (auto node: this->nodes) {

    node->hidden_nodes_fed_into_me_ptr.clear();
    for (auto i: node->hidden_nodes_fed_into_me) 
      node->hidden_nodes_fed_into_me_ptr.push_back(this->nodes[i]);

  }
   
  //Transfer number to input nodes to nodes, so we can calculate the number of weights needed
  for (auto node: this->nodes) {

    //Set initial value to zero
    node->num_input_nodes_cumulative = 0;

    //Make sure dense input is in range
    if (std::any_of (
		     node->input_nodes_fed_into_me_dense.begin(),
		     node->input_nodes_fed_into_me_dense.end(),
		     [this](std::int32_t i) {
		       return 
			 (i < 0) || 
			 (
			  i >= static_cast<std::int32_t>(this->dense_input_data_dim.size())
			  );
		     }
		     )
	)
      throw std::invalid_argument("input_dense out of bounds!");

    //Make sure sparse input is in range
    if (std::any_of (
		     node->input_nodes_fed_into_me_sparse.begin(),
		     node->input_nodes_fed_into_me_sparse.end(),
		     [this](std::int32_t i) {
		       return 
			 (i < 0) || 
			 (
			  i >= static_cast<std::int32_t>(this->sparse_input_data_dim.size())
			  );
		     }
		     )
	)
      throw std::invalid_argument("input_sparse out of bounds!");

    //Add dense input
    for (auto dense: node->input_nodes_fed_into_me_dense) 
      node->num_input_nodes_cumulative 
	+= this->dense_input_data_dim[dense];
    
    //Add sparse input
    for (auto sparse: node->input_nodes_fed_into_me_sparse) 
      node->num_input_nodes_cumulative 
	+= this->sparse_input_data_dim[sparse];
     
  }

  //Transfer number of output nodes to targets
  for (int i=0; i < this->num_output_nodes_dense; ++i) 
    dense_targets_dim[i] = this->output_nodes[i]->dim_; 
  
  for (int i=0; i < this->num_output_nodes_sparse; ++i) 
    sparse_targets_dim[i] = this->output_nodes[num_output_nodes_dense + i]->dim_;

  //Calculate cumulative_num_weights_required and initialise W
  std::int32_t lengthW = 0;
  
  this->cumulative_num_weights_required_.clear();

  for (auto node: this->nodes) {

    node->NeuralNet = this;

    if (node->i_share_weights_with < 0) {

      //If node does not share weights with another node, then count lengthW
      this->cumulative_num_weights_required_.push_back(lengthW);
      lengthW += node->get_num_weights_required();

    } else {

      //If node does share weights with another node, then make sure num_weights_required match
      if (
	  node->get_num_weights_required() !=
	  this->nodes[node->i_share_weights_with]->get_num_weights_required()
	  )
	std::invalid_argument("Number of weights of nodes must match for weight sharing to be possible!");
	
      this->cumulative_num_weights_required_.push_back(
						       this->cumulative_num_weights_required_[node->i_share_weights_with]
						       );
      

    }

  }

  this->cumulative_num_weights_required_.push_back(lengthW);

  //Init Whost
  std::vector<float> Whost(lengthW);
  
  std::mt19937 gen(1);//Note that we deliberately choose a constant seed to get the same output every time we call the function
  std::uniform_real_distribution<float> dist(_weight_init_range*(-1.0f), _weight_init_range);

  //Initialise weight vector
  //The vector of weights associated with the input nodes cannot be a csr_matrix. The solution is to keep a set of weights that always assume the value of 0.0.
  for (auto node: this->nodes) {
         
    for (
	 std::int32_t i = cumulative_num_weights_required_[node->node_number]; 
	 i < cumulative_num_weights_required_[node->node_number + 1]; 
	 ++i
	 ) Whost[i] = dist(gen);
            
  }
  
  //Transfor to device vector
  this->W = thrust::device_vector<float>(Whost.data(), Whost.data() + Whost.size());

  //Set finalised to true so we know we can now fit the neural network
  this->finalised = true;
  
}

std::int32_t NeuralNetworkCpp::get_length_params() {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");

  return (std::int32_t)(this->W.size());
		
};

void NeuralNetworkCpp::get_params(float *_W, std::int32_t _length_W) {

  if (!this->finalised) 
    throw std::invalid_argument("Neural network has not been finalised!");
  
  for (std::int32_t i=0; i<_length_W; ++i) 
    _W[i] = this->W[i];

}

void NeuralNetworkCpp::set_params(float *_W, std::int32_t _length_W) {

  if (!this->finalised) 
    throw std::invalid_argument("Neural network has not been finalised!");

  if (_length_W != static_cast<std::int32_t>(this->W.size())) 
    throw std::invalid_argument("Length of provided weight vector does not match expected size!");
  
  for (std::int32_t i=0; i<_length_W; ++i) 
    this->W[i] = _W[i];

}

std::int32_t NeuralNetworkCpp::get_input_nodes_fed_into_me_dense_length(
									   std::int32_t _node_number
									   ) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_node_number < 0 || _node_number >= (std::int32_t)(nodes.size())) std::invalid_argument("node_number out of bounds!");

  return (std::int32_t)(this->nodes[_node_number]->input_nodes_fed_into_me_dense.size());
		
};

void NeuralNetworkCpp::get_input_nodes_fed_into_me_dense(
							    std::int32_t  _node_number,
							    std::int32_t *_input_nodes_fed_into_me_dense, 
							    std::int32_t  _input_nodes_fed_into_me_dense_length
							    ) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (
      _node_number < 0 || 
      _node_number >= (std::int32_t)(nodes.size())
      ) 
    std::invalid_argument("node_number out of bounds!");
		
  for (std::int32_t i=0; i<_input_nodes_fed_into_me_dense_length; ++i) 
    _input_nodes_fed_into_me_dense[i] 
      = this->nodes[_node_number]->input_nodes_fed_into_me_dense[i];
  		
};

std::int32_t NeuralNetworkCpp::get_input_nodes_fed_into_me_sparse_length(std::int32_t _node_number) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_node_number < 0 || _node_number >= (std::int32_t)(nodes.size())) std::invalid_argument("node_number out of bounds!");

  return (std::int32_t)(this->nodes[_node_number]->input_nodes_fed_into_me_sparse.size());
		
};

void NeuralNetworkCpp::get_input_nodes_fed_into_me_sparse(
							     std::int32_t  _node_number, 
							     std::int32_t *_input_nodes_fed_into_me_sparse, 
							     std::int32_t _input_nodes_fed_into_me_sparse_length
							     ) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_node_number < 0 || _node_number >= (std::int32_t)(nodes.size())) std::invalid_argument("node_number out of bounds!");
		
  for (std::int32_t i=0; i<_input_nodes_fed_into_me_sparse_length; ++i) _input_nodes_fed_into_me_sparse[i] = this->nodes[_node_number]->input_nodes_fed_into_me_sparse[i];
  		
};
     
std::int32_t NeuralNetworkCpp::get_hidden_nodes_fed_into_me_length(std::int32_t _node_number) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_node_number < 0 || _node_number >= (std::int32_t)(nodes.size())) std::invalid_argument("node_number out of bounds!");

  return (std::int32_t)(this->nodes[_node_number]->hidden_nodes_fed_into_me.size());
		
};

void NeuralNetworkCpp::get_hidden_nodes_fed_into_me(
						       std::int32_t  _node_number,
						       std::int32_t *_hidden_nodes_fed_into_me,
						       std::int32_t __lengthhidden_nodes_fed_into_me
						       ) {
		
  if (!this->finalised) throw std::invalid_argument("Neural network has not been finalised!");
  if (_node_number < 0 || _node_number >= (std::int32_t)(nodes.size())) std::invalid_argument("node_number out of bounds!");
		
  for (std::int32_t i=0; i<__lengthhidden_nodes_fed_into_me; ++i) _hidden_nodes_fed_into_me[i] = this->nodes[_node_number]->hidden_nodes_fed_into_me[i];
  		
};

//batch_begin and batch_end are used to share the burden evenly among the processes
void NeuralNetworkCpp::calc_batch_begin_end (
						std::int32_t      &_batch_begin,
						std::int32_t      &_batch_end, 
						std::int32_t      &_batch_size, 
						const std::int32_t _batch_num,
						const std::int32_t _num_samples,
						const std::int32_t _num_batches
						) {
									
  //Calculate _batch_begin
  _batch_begin = _batch_num*(_num_samples/_num_batches);
		
  //Calculate _batch_size
  if (_batch_num < _num_batches-1) _batch_size = _num_samples/_num_batches;
  else _batch_size = _num_samples - _batch_begin;
		
  //Calculate _batch_end
  _batch_end = _batch_begin + _batch_size;
	
}

void NeuralNetworkCpp::load_dense(
				     std::vector<matrix::DenseMatrix> &_data,
				     float                    *_X,
				     std::int32_t              _num_samples,
				     std::int32_t              _dim,
				     std::int32_t              _num_batches,
				     bool                      _transpose
				     ) {

  std::int32_t batch_begin, batch_end, batch_size;

  for (std::int32_t batch_num = 0; batch_num<_num_batches; ++batch_num) {

    this->calc_batch_begin_end(
			       batch_begin,
			       batch_end,
			       batch_size,
			       batch_num,
			       _num_samples,
			       _num_batches
			       );

    //Transfer _num_samples and _dim
    _data[batch_num].batch_size = batch_size;
    _data[batch_num].dim = _dim;

    std::vector<float> X_transpose;

    //Target vectors need to be transposed
    if (_transpose) {
      
      X_transpose = std::vector<float>(batch_size*_dim);

      //Transpose target
      for (std::int32_t i=0; i<batch_size; ++i)
	for (std::int32_t j=0; j<_dim; ++j)
	  X_transpose[j*batch_size + i] = 
	    _X[(batch_begin + i)*_dim + j];

      //Transfer to 
      _data[batch_num].X = thrust::device_vector<float>(
							X_transpose.begin(),
							X_transpose.end()
							);
      
    } else {

      //Transfer X to 
      _data[batch_num].X = thrust::device_vector<float>(
							_X + batch_begin*_dim,
							_X + batch_end*_dim
							);
    }
    
    //Set X_ptr
    _data[batch_num].X_ptr = thrust::raw_pointer_cast(_data[batch_num].X.data());

  } 

}

void NeuralNetworkCpp::load_dense_data(
					  std::int32_t _num_input_node,
					  float       *_X,
					  std::int32_t _num_samples,
					  std::int32_t _dim,
					  std::int32_t _global_batch_size
					  ) {

  if (
      _num_input_node >= (std::int32_t)(this->dense_input_data.size()) ||
      _num_input_node < 0
      )
    throw std::invalid_argument("num_input_node out of bounds!");


  if (_dim != this->dense_input_data_dim[_num_input_node])
    throw std::invalid_argument("Width dim of array provided does not match the width that has been set when initialising the network!");

  std::int32_t num_batches = calc_num_batches (
					       /*MPI_Comm comm,*/
					       _num_samples,
					       _global_batch_size
					       );

  this->dense_input_data[_num_input_node] = std::vector<matrix::DenseMatrix>(num_batches);
  
  this->load_dense(
		   this->dense_input_data[_num_input_node],
		   _X,
		   _num_samples,
		   _dim,
		   num_batches,
		   false //do not transpose
		   );

}

void NeuralNetworkCpp::load_dense_targets(
					     std::int32_t _num_output_node,
					     float       *_Y,
					     std::int32_t _num_samples,
					     std::int32_t _dim,
					     std::int32_t _global_batch_size
					     ) {

  if (_num_output_node >= (std::int32_t)(this->dense_targets.size()) || _num_output_node < 0) 
    throw std::invalid_argument("num_output_node out of bounds!");
  
  if (_dim != this->dense_targets_dim[_num_output_node]) 
    throw std::invalid_argument("Width dim of array provided does not match the width that has been set when initialising the network!");
 
  //Calculates the number of batches needed
  std::int32_t num_batches = calc_num_batches (
					       /*MPI_Comm comm,*/
					       _num_samples,
					       _global_batch_size
					       );

  this->dense_targets[_num_output_node] = 
    std::vector<matrix::DenseMatrix>(num_batches);

  this->load_dense(
		   this->dense_targets[_num_output_node],
		   _Y,
		   _num_samples,
		   _dim,
		   num_batches,
		   true //do transpose
		   );

}

void NeuralNetworkCpp::load_csr(
				   std::vector<matrix::CSRMatrix> &_data,
				   float                          *_X_data,
				   std::int32_t                    _X_data_length,
				   std::int32_t                   *_X_indices,
				   std::int32_t                    _X_indices_length,
				   std::int32_t                   *_X_indptr,
				   std::int32_t                    _X_indptr_length,
				   std::int32_t                    _num_samples,
				   std::int32_t                    _dim,
				   std::int32_t                    _num_batches
				   ) {

  std::int32_t batch_begin, batch_end, batch_size;

  for (std::int32_t batch_num = 0; batch_num<_num_batches; ++batch_num) {

    this->calc_batch_begin_end(
			       batch_begin,
			       batch_end,
			       batch_size,
			       batch_num,
			       _num_samples,
			       _num_batches
			       );

    //Transfer _num_samples and _dim
    _data[batch_num].batch_size = batch_size;
    _data[batch_num].dim = _dim;
    _data[batch_num].num_non_zero = 
      _X_indptr[batch_end] - _X_indptr[batch_begin];

    //Transfer X_data to  and set X_data_ptr
    _data[batch_num].X_data =
      thrust::device_vector<float>(
				   _X_data + _X_indptr[batch_begin],
				   _X_data + _X_indptr[batch_end]
				   );
    
    _data[batch_num].X_data_ptr =
      thrust::raw_pointer_cast(_data[batch_num].X_data.data());

    //Transfer X_indices to  and set X_indices_ptr   
    _data[batch_num].X_indices = 
      thrust::device_vector<std::int32_t>(
					  _X_indices + _X_indptr[batch_begin],
					  _X_indices + _X_indptr[batch_end]
					  );
    
    _data[batch_num].X_indices_ptr =
      thrust::raw_pointer_cast(_data[batch_num].X_indices.data());


    //Transfer X_indptr to  and set X_indptr_ptr
    //Do not forget the last element - it is important
    _data[batch_num].X_indptr = 
      thrust::device_vector<std::int32_t>(
					  &_X_indptr[batch_begin],
					  &_X_indptr[batch_end] + 1 
					  //_X_indptr has size batch_size + 1
					  );
    
    _data[batch_num].X_indptr_ptr =
      thrust::raw_pointer_cast(_data[batch_num].X_indptr.data());

    //Substract value of first elements from all elements in X_indptr
    thrust::for_each(
		     _data[batch_num].X_indptr.begin(),
		     _data[batch_num].X_indptr.end(),
		     thrust::placeholders::_1 -= _X_indptr[batch_begin]
		     );
 
  }
}

//This function expects a CSR matrix, but transforms it into a COO vector on the 
void NeuralNetworkCpp::load_coo(
				   std::vector<matrix::COOVector> &_data,
				   float                  *_X_data,
				   std::int32_t            _X_data_length,
				   std::int32_t           *_X_indices,
				   std::int32_t            _X_indices_length,
				   std::int32_t           *_X_indptr,
				   std::int32_t            _X_indptr_length,
				   std::int32_t            _num_samples,
				   std::int32_t            _dim,
				   std::int32_t            _num_batches
				   ) {

  std::int32_t batch_begin, batch_end, batch_size;

  thrust::device_vector<std::int32_t> X_col;

  for (std::int32_t batch_num = 0; batch_num<_num_batches; ++batch_num) {

    this->calc_batch_begin_end(
			       batch_begin,
			       batch_end,
			       batch_size,
			       batch_num,
			       _num_samples,
			       _num_batches
			       );

    //Transfer _num_samples and _dim
    _data[batch_num].batch_size = batch_size;
    _data[batch_num].dim = _dim;
    _data[batch_num].num_non_zero = 
      _X_indptr[batch_end] - _X_indptr[batch_begin];

    //Transfer X_data to  and set X_data_ptr
    _data[batch_num].X_data =
      thrust::device_vector<float>(
				   _X_data + _X_indptr[batch_begin],
				   _X_data + _X_indptr[batch_end]
				   );
    
    _data[batch_num].X_data_ptr =
      thrust::raw_pointer_cast(_data[batch_num].X_data.data());

    //Transfer X_indices to  (which we rename X_row)
    //Remember that the output of the neural net will be in column-major
    //order, so we have to multiply by batch_size!
    _data[batch_num].X_indices = 
      thrust::device_vector<std::int32_t>(
					   _X_indices + _X_indptr[batch_begin], 
					   _X_indices + _X_indptr[batch_end]
					  );
   
    _data[batch_num].X_indices_ptr =
      thrust::raw_pointer_cast(_data[batch_num].X_indices.data());

    //Multiply by batch_size - remember that the output is in column major order!
    thrust::for_each(
    		     _data[batch_num].X_indices.begin(),
    		     _data[batch_num].X_indices.end(),
    		     thrust::placeholders::_1 *= batch_size		     
    		     );

    //Transfer X_indptr to 
    //X_indptr is transformed into X_col
    X_col = 
      thrust::device_vector<std::int32_t>(
					  _X_indptr[batch_end] 
					  - _X_indptr[batch_begin]
					  );

    //Transform the X_indptr vector into a column vector
    for (
         std::int32_t i = 0; 
	 i < batch_size;
	 ++i
        )
      thrust::fill (
                    X_col.begin() 
		     + _X_indptr[batch_begin + i] - _X_indptr[batch_begin],
		    X_col.begin() +
		     _X_indptr[batch_begin + i+1] - _X_indptr[batch_begin],
		    i
		    );
     
    //Now, add X_col to X_indices
    thrust::transform(
		      _data[batch_num].X_indices.begin(),
		      _data[batch_num].X_indices.end(),
		      X_col.begin(),
		      _data[batch_num].X_indices.begin(),
		      thrust::plus<std::int32_t>()
		      );

	 
  }
}



    void NeuralNetworkCpp::load_sparse_data(
                                               std::int32_t  _num_input_node,
					       float        *_X_data,
					       std::int32_t  _X_data_length,
					       std::int32_t *_X_indices,
                                               std::int32_t  _X_indices_length,
                                               std::int32_t *_X_indptr,
                                               std::int32_t  _X_indptr_length,
                                               std::int32_t  _num_samples,
                                               std::int32_t  _dim,
                                               std::int32_t  _global_batch_size
      ) {

    if (
    _num_input_node >= (std::int32_t)(this->sparse_input_data.size()) || 
      _num_input_node < 0
			) throw std::invalid_argument("num_input_node out of bounds!");
  
    if (_dim != this->sparse_input_data_dim[_num_input_node]) 
      throw std::invalid_argument("Width dim of array provided does not match the width that has been set when initialising the network!");

    std::int32_t num_batches = calc_num_batches (
    /*MPI_Comm comm,*/
    _num_samples,
      _global_batch_size
      );

    this->sparse_input_data[_num_input_node] 
      = std::vector<matrix::CSRMatrix>(num_batches);

    this->load_csr(
    this->sparse_input_data[_num_input_node],
      _X_data,
      _X_data_length,
      _X_indices,
      _X_indices_length,
      _X_indptr,
      _X_indptr_length,
      _num_samples,
      _dim, 
      num_batches
      );

  }

    void NeuralNetworkCpp::load_sparse_targets (
     std::int32_t  _num_output_node,
      float        *_Y_data,
      std::int32_t  _Y_data_length,
      std::int32_t *_Y_indices,
      std::int32_t  _Y_indices_length,
      std::int32_t *_Y_indptr,
      std::int32_t  _Y_indptr_length,
      std::int32_t  _num_samples,
      std::int32_t  _dim,
      std::int32_t  _global_batch_size
      ) {

    if (
    _num_output_node >= (std::int32_t)(this->sparse_targets.size()) ||
      _num_output_node < 0
			 ) 
      throw std::invalid_argument("num_output_node out of bounds!");

    if (_dim != this->sparse_targets_dim[_num_output_node]) 
      throw std::invalid_argument("Width dim of array provided does not match the width that has been set when initialising the network!");

    std::int32_t num_batches = calc_num_batches (
    /*MPI_Comm comm,*/
    _num_samples,
      _global_batch_size
      );//Calculates the number of batches needed

    this->sparse_targets[_num_output_node] = std::vector<matrix::COOVector>(num_batches);

    this->load_coo(
      this->sparse_targets[_num_output_node],
      _Y_data,
      _Y_data_length,
      _Y_indices,
      _Y_indices_length,
      _Y_indptr,
      _Y_indptr_length,
      _num_samples,
      _dim,
      num_batches
      );

  }

    void NeuralNetworkCpp::delete_data() {

    for (auto& data: this->dense_input_data) data.clear();
    for (auto& data: this->dense_targets) data.clear();
    for (auto& data: this->sparse_input_data) data.clear();
    for (auto& data: this->sparse_targets) data.clear();

  }
	           
    void NeuralNetworkCpp::transform(
                                        float       *_Yhat,
                                        std::int32_t _Y2_num_samples, 
					std::int32_t _Y2_dim, 
					bool         _sample, 
					std::int32_t _sample_size, 
					bool         _Gethidden_nodes
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

    } else {

      throw std::invalid_argument("No input data provided!");

    }

    //Get num_batches
    std::int32_t num_batches = (std::int32_t)(batch_size.size());

    //Make sure that the batch_sizes are identical for all matrices provided!
    for (auto DataVector: this->dense_input_data) {

      if (DataVector.size() != batch_size.size()) 
	throw std::invalid_argument("All input matrices must have the exact same number of samples!");

      for (std::size_t i=0; i<DataVector.size(); ++i) 
	if (DataVector[i].batch_size != batch_size[i]) 
	  throw std::invalid_argument("All input matrices must have the exact same number of samples!");

    }

    //Make sure that the batch_sizes are identical for all matrices provided!
    for (auto DataVector: this->sparse_input_data) {

      if (DataVector.size() != batch_size.size()) 
	throw std::invalid_argument("All input matrices must have the exact same number of samples!");

      for (std::size_t i=0; i<DataVector.size(); ++i) 
	if (DataVector[i].batch_size != batch_size[i]) 
	  throw std::invalid_argument("All input matrices must have the exact same number of samples!");

    }

    //Store input values
    this->num_samples = _Y2_num_samples;
    this->sample = _sample;
    if (!_sample) _sample_size = 1;

    const double SampleAvg = 1.0/((double)_sample_size);

    //Set pointers contained in the NeuralNetworkNodes class
    for (std::size_t n=0; n<this->nodes.size(); ++n) 
      this->nodes[n]->W = thrust::raw_pointer_cast(this->W.data()) 
	+ this->cumulative_num_weights_required_[n];
  
    //Init YhatTemp
    thrust::device_vector<float> YhatTemp(
					  _Yhat, 
					  _Yhat + _Y2_num_samples*_Y2_dim
					  );
  
    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    cublasCreate(&(this->dense_handle_));
    cusparseCreate(&(this->sparse_handle_));
    cusparseCreateMatDescr(&(this->mat_descr_));
    cusparseSetMatType(this->mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(this->mat_descr_, CUSPARSE_INDEX_BASE_ZERO);
    
    //Calculate output
    std::int32_t batch_begin = 0;
    for (
	 std::int32_t batch_num = 0; 
	 batch_num < num_batches; 
	 ++batch_num, batch_begin += batch_size[batch_num]
	 ) 
      for (
	   std::int32_t iteration = 0;
	   iteration < _sample_size;
	   ++iteration
	   ) {
    						
	//Calculate nodes
	for (auto node: this->nodes) 
	  node->calc_output(
			    batch_num, 
			    batch_size[batch_num]
			    );

	//Add to YhatTemp
	std::int32_t col_num = 0;
	for (
	     std::int32_t node_num = 0; 
	     node_num < this->num_output_nodes; 
	     ++node_num
	     ) 
	  for (
	       std::int32_t dim = 0; 
	       dim < this->output_nodes[node_num]->dim_; 
	       ++dim
	       ) {
		
	    thrust::transform(
			      this->output_nodes[node_num]->output.begin() 
			      + dim*batch_size[batch_num], 
				  
			      this->output_nodes[node_num]->output.begin() 
			      + (dim+1)*batch_size[batch_num], 
				  
			      YhatTemp.begin() + _Y2_num_samples*col_num + batch_begin, 

			      YhatTemp.begin() + _Y2_num_samples*col_num + batch_begin, 

			      thrust::plus<float>()
				  
			      );
		
	    ++col_num;
	  }
      }
  
    //Get data from YhatTemp and transpose
    for (std::int32_t i=0; i<_Y2_num_samples; ++i) 
      for (std::int32_t j=0; j<_Y2_dim; ++j) 
	_Yhat[i*_Y2_dim + j] = YhatTemp[j*_Y2_num_samples + i];

    //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
    cublasDestroy(this->dense_handle_);
    cusparseDestroyMatDescr(this->mat_descr_);
    cusparseDestroy(this->sparse_handle_);
			
    //Clear data, so it does not unnecessarily take up space on the 
    this->delete_data();
		
  };
