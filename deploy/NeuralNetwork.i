//----------------------------------------------------------------------------------------------
//class NeuralNetworkNodeCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class NeuralNetworkNodeCpp {
	
  //friend class NeuralNetworkCpp;
	
 public://This is temporary - change to protected later
			
  NeuralNetworkNodeCpp (
			   std::int32_t    _node_number, 
			   std::int32_t    _dim,
			   std::int32_t   *_input_nodes_fed_into_me_dense, 
			   std::int32_t    _input_nodes_fed_into_me_dense_length, 
			   std::int32_t   *_input_nodes_fed_into_me_sparse, 
			   std::int32_t    _input_nodes_fed_into_me_sparse_length,
			   std::int32_t   *_hidden_nodes_fed_into_me, 
			   std::int32_t    _hidden_nodes_fed_into_me_length,
			   std::int32_t    _i_share_weights_with
			   );
	
  virtual ~NeuralNetworkNodeCpp();

	  		
};

//----------------------------------------------------------------------------------------------
//class ActivationFunctionCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class ActivationFunctionCpp: public NeuralNetworkNodeCpp {
	
public:
	
  ActivationFunctionCpp(
			   std::int32_t    _node_number,
			   std::int32_t    _dim,
			   std::int32_t   *_input_nodes_fed_into_me_dense,
			   std::int32_t    _input_nodes_fed_into_me_dense_length,
			   std::int32_t   *_input_nodes_fed_into_me_sparse,
			   std::int32_t    _input_nodes_fed_into_me_sparse_length,
			   std::int32_t   *_hidden_nodes_fed_into_me,
			   std::int32_t    _hidden_nodes_fed_into_me_length,
			   std::int32_t    _i_share_weights_with
			   );
	
  ~ActivationFunctionCpp();

};

//----------------------------------------------------------------------------------------------
//class LogisticActivationFunctionCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class LogisticActivationFunctionCpp: public ActivationFunctionCpp {
	
public:

  LogisticActivationFunctionCpp (
				    std::int32_t    _node_number,
				    std::int32_t    _dim,
				    std::int32_t   *_input_nodes_fed_into_me_dense,
				    std::int32_t    _input_nodes_fed_into_me_dense_length,
				    std::int32_t   *_input_nodes_fed_into_me_sparse,
				    std::int32_t    _input_nodes_fed_into_me_sparse_length,
				    std::int32_t   *_hidden_nodes_fed_into_me,
				    std::int32_t    _hidden_nodes_fed_into_me_length,
				    std::int32_t    _i_share_weights_with
				    );
    
  ~LogisticActivationFunctionCpp();

};

//----------------------------------------------------------------------------------------------
//class LinearActivationFunctionCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class LinearActivationFunctionCpp: public ActivationFunctionCpp {
	
public:
	
  LinearActivationFunctionCpp (
				  std::int32_t    _node_number,
				  std::int32_t    _dim,
				  std::int32_t   *_input_nodes_fed_into_me_dense,
				  std::int32_t    _input_nodes_fed_into_me_dense_length,
				  std::int32_t   *_input_nodes_fed_into_me_sparse,
				  std::int32_t    _input_nodes_fed_into_me_sparse_length,
				  std::int32_t   *_hidden_nodes_fed_into_me,
				  std::int32_t    _hidden_nodes_fed_into_me_length,
				  std::int32_t    _i_share_weights_with
				  );

  ~LinearActivationFunctionCpp();

};
/*
//----------------------------------------------------------------------------------------------
//class SoftmaxActivationFunctionCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class SoftmaxActivationFunctionCpp: public ActivationFunctionCpp {

public:

  SoftmaxActivationFunctionCpp (
				    std::int32_t    _node_number,
				    std::int32_t    _num_vars,
				    std::int32_t    _num_states_per_var,
				    std::int32_t   *_input_nodes_fed_into_me_dense,
				    std::int32_t    _input_nodes_fed_into_me_dense_length,
				    std::int32_t   *_input_nodes_fed_into_me_sparse,
				    std::int32_t    _input_nodes_fed_into_me_sparse_length,
				    std::int32_t   *_hidden_nodes_fed_into_me,
				    std::int32_t    _hidden_nodes_fed_into_me_length
				   );
	
  ~SoftmaxActivationFunctionCpp();

};
*/
//----------------------------------------------------------------------------------------------
//class NeuralNetworkCpp

%apply (float* IN_ARRAY1, int DIM1) {(float *_W, std::int32_t _length_W)};

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_num_input_nodes_dense, std::int32_t _num_input_nodes_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_num_input_nodes_sparse, std::int32_t _num_input_nodes_sparse_length)};

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t __lengthhidden_nodes_fed_into_me)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_Yhat, std::int32_t _Y2_num_samples, std::int32_t _Y2_dim)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_X, std::int32_t _num_samples, std::int32_t _dim)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_X_data, std::int32_t _X_data_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_X_indices, std::int32_t _X_indices_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_X_indptr, std::int32_t _X_indptr_length)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_Y, std::int32_t _num_samples, std::int32_t _dim)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_Y_data, std::int32_t _Y_data_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_Y_indices, std::int32_t _Y_indices_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_Y_indptr, std::int32_t _Y_indptr_length)};

%apply (float* IN_ARRAY1, int DIM1) {(float *_sum_gradients, std::int32_t _sum_gradients_length)};


class NeuralNetworkCpp/*: public NumericallyOptimisedMLAlgorithmCpp*/ {
		
public:
	
  NeuralNetworkCpp (
		       std::int32_t *_num_input_nodes_dense, 
		       std::int32_t _num_input_nodes_dense_length, 
		       std::int32_t *_num_input_nodes_sparse,
		       std::int32_t _num_input_nodes_sparse_length, 
		       std::int32_t _num_output_nodes_dense, 
		       std::int32_t _num_output_nodes_sparse
		       );
	
  ~NeuralNetworkCpp();

  void init_hidden_node(
			NeuralNetworkNodeCpp *_hidden_node
			);
	
  void init_output_node(
			NeuralNetworkNodeCpp *_output_node
			);

  std::int32_t get_length_params();

  void get_params(
		  float       *_W, 
		  std::int32_t _length_W
		  );

  void set_params(
		  float       *_W, 
		  std::int32_t _length_W
		  );

  std::int32_t get_input_nodes_fed_into_me_dense_length(
							std::int32_t _node_number
							);
	
  void get_input_nodes_fed_into_me_dense(
					 std::int32_t  _node_number, 
					 std::int32_t *_input_nodes_fed_into_me_dense, 
					 std::int32_t  _input_nodes_fed_into_me_dense_length
					 );

  std::int32_t get_input_nodes_fed_into_me_sparse_length(
							 std::int32_t _node_number
							 );
	
  void get_input_nodes_fed_into_me_sparse(
					  std::int32_t  _node_number, 
					  std::int32_t *_input_nodes_fed_into_me_sparse, 
					  std::int32_t  _input_nodes_fed_into_me_sparse_length
					  );
	
  std::int32_t get_hidden_nodes_fed_into_me_length(
						   std::int32_t _node_number
						   );

  void get_hidden_nodes_fed_into_me(
				    std::int32_t _node_number, 
				    std::int32_t *_hidden_nodes_fed_into_me, 
				    std::int32_t __lengthhidden_nodes_fed_into_me
				    );

  //This function prepares the neural network for training
  void finalise(
		/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/ 
		float _weight_init_range
		);
	
  //This function loads the provided dataset into the 
  void load_dense_data(
		       std::int32_t _num_input_node, 
		       float       *_X, 
		       std::int32_t _num_samples, 
		       std::int32_t _dim, 
		       std::int32_t _global_batch_size
		       );


  //This function loads the provided dataset into the 
  void load_sparse_data(
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
			);

  //This functions transform inputs data into predictions
  void transform(
		 float       *_Yhat, 
		 std::int32_t _Y2_num_samples, 
		 std::int32_t _Y2_dim, 
		 bool         _sample, 
		 std::int32_t _sample_size,
		 bool         _Gethidden_nodes
		 );
  
  //This functions returns the sum of the dimensionalities of all output nodes
  std::int32_t get_sum_output_dim();
	
};

//----------------------------------------------------------------------------------------------
