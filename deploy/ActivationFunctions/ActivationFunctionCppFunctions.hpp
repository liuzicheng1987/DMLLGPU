ActivationFunctionCpp::ActivationFunctionCpp(
						   std::int32_t    _node_number, 
						   std::int32_t    _dim,
						   std::int32_t   *_input_nodes_fed_into_me_dense, 
						   std::int32_t    _input_nodes_fed_into_me_dense_length, 
						   std::int32_t   *_input_nodes_fed_into_me_sparse, 
						   std::int32_t    _input_nodes_fed_into_me_sparse_length, 
						   std::int32_t   *_hidden_nodes_fed_into_me, 
						   std::int32_t    _hidden_nodes_fed_into_me_length,
						   std::int32_t    _i_share_weights_with
						   ): NeuralNetworkNodeCpp (
									      _node_number,
									      _dim,
									      _input_nodes_fed_into_me_dense, 
									      _input_nodes_fed_into_me_dense_length, 
									      _input_nodes_fed_into_me_sparse, 
									      _input_nodes_fed_into_me_sparse_length, 
									      _hidden_nodes_fed_into_me, 
									      _hidden_nodes_fed_into_me_length,
									      _i_share_weights_with
									      ) {}

ActivationFunctionCpp::~ActivationFunctionCpp() {};

std::int32_t ActivationFunctionCpp::get_num_weights_required() {

  std::int32_t num_weights_required = this->num_input_nodes_cumulative;

  for (auto hidden: this->hidden_nodes_fed_into_me_ptr)
    num_weights_required += hidden->get_dim();

  num_weights_required += 1;

  num_weights_required *= this->dim_;

  return num_weights_required; 

};

void ActivationFunctionCpp::calc_output(
					   const std::int32_t _batch_num, 
					   const std::int32_t _batch_size
					   ) {
  
  //Pointer to weights
  const float *w = this->W;
  
  //Number of columns in input data - for convenience
  std::int32_t input_dim;

  //Needed for transformations
  const float alpha = 1.0; 
  float beta = 0.0; 

  //Resize output and delta, if necessary
  //Output is stored in the NeuralNetworkNodeCpp base class and stores the output of this node
  if (static_cast<std::int32_t>(this->output.size()) != this->dim_*_batch_size) {
    
    //Resize output
    this->output.resize(this->dim_*_batch_size);
    this->output_ptr = this->output.data();

  }
  
  //Transform dense input nodes
  for (std::int32_t i: this->input_nodes_fed_into_me_dense) {

    input_dim = this->NeuralNet->get_dense_input_data(
						      i, 
						      _batch_num
						      ).dim;//For convenience

    //output = alpha*(WT)X + beta*output 
    //X: (input_dim X _batch_size)-matrix
    //NOTE: X is (input_dim X _batch_size) for input data,
    // but (_batch_size X input_dim) for hidden node data!
    //w: (input_dim X this->dim_)-matrix
    //output: (_batch_size X this->dim_)-matrix
    //NOTE: We use column-major order!
    matrix::matrix_multiplication(
				  true, //transa
				  _batch_size, //dim0
				  this->dim_, //dim1
				  input_dim, //dim2           
				  1.f, //alpha
				  this->NeuralNet->get_dense_input_data(
								      i,
								      _batch_num
								      ).X_ptr, //A
				  w, //B
				  beta, //beta
				  this->output_ptr //C 
				  );
 
    w += this->dim_*input_dim;//Increment w to prepare it for next operation
    beta = 1.0;//Set beta to 1.0, so all subsequent matrix operations are added up.
    
  } 

  //Transform sparse input nodes
  for (std::int32_t i: this->input_nodes_fed_into_me_sparse) {

    input_dim = this->NeuralNet->get_sparse_input_data(i, _batch_num).dim;//For convenience
    
    //output = alpha*(WT)X + beta*output 
    //X: (_batch_size X input_dim)-CSR-matrix
    //w: (input_dim X this->dim_)-matrix
    //output: (_batch_size X this->dim_)-matrix
    matrix::matrix_multiplication_sparse(
					 _batch_size,//dim0
					 this->dim_,//dim1
					 input_dim, //dim2
					 this->NeuralNet->get_sparse_input_data(
										i, 
										_batch_num
										).num_non_zero, //nnz
					 1.f, //alpha
					 this->NeuralNet->get_sparse_input_data(
										i, 
										_batch_num
										).X_data_ptr,//data
					 this->NeuralNet->get_sparse_input_data(
										i, 
										_batch_num
										).X_indptr_ptr,//indptr
					 this->NeuralNet->get_sparse_input_data(
										i, 
										_batch_num
										).X_indices_ptr,//indices
					 w, //B
					 beta,//beta
					 this->output_ptr //C 
					 );

    
    w += this->dim_*input_dim;//Increment w
    beta = 1.0;//Set beta to 1.0, so all subsequent matrix operations are added up.
    
  } 
  
  //Transform hidden nodes
  //Calculate derivatives for hidden nodes
  for (auto node: this->hidden_nodes_fed_into_me_ptr) {

    input_dim = node->get_dim();//For convenience

    //Linear transformation of input data using cuBLAS
    //dldw = alpha*(X)deltaT + beta*dldw
    //X: (_batch_size X input_dim)-matrix 
    //NOTE: X is (input_dim X _batch_size) for input data,
    // but (_batch_size X input_dim) for hidden node data!
    //w: (input_dim X this->dim_)-matrix
    //output: (_batch_size X this->dim_)-matrix
    //NOTE: We use column-major order!
    matrix::matrix_multiplication(
				  false, //transa
				  _batch_size, //dim0
				  this->dim_, //dim1
				  input_dim, //dim2           
				  1.f, //alpha
				  node->get_output_ptr(),//A
				  w, //B
				  beta,//beta
				  this->output_ptr //C 
				  );

    w += this->dim_*input_dim;//Increment w to prepare it for next operation
    beta = 1.0;//Set beta to 1.0, so all subsequent matrix operations are added up.
    
  } 
  
  //This functions adds the bias and then applies the activation function
  this->forward_propagation(
			    _batch_size,
			    this->dim_,
			    w,
			    this->output
			    );

}
