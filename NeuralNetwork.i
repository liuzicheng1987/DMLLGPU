//----------------------------------------------------------------------------------------------
//class NeuralNetworkNodeGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class NeuralNetworkNodeGPUCpp {
	
  //friend class NeuralNetworkGPUCpp;
	
 public://This is temporary - change to protected later
			
  NeuralNetworkNodeGPUCpp (
			   std::int32_t    _node_number, 
			   std::int32_t    _dim,
			   std::int32_t   *_input_nodes_fed_into_me_dense, 
			   std::int32_t    _input_nodes_fed_into_me_dense_length, 
			   std::int32_t   *_input_nodes_fed_into_me_sparse, 
			   std::int32_t    _input_nodes_fed_into_me_sparse_length,
			   std::int32_t   *_hidden_nodes_fed_into_me, 
			   std::int32_t    _hidden_nodes_fed_into_me_length,
			   std::int32_t    _i_share_weights_with, 
			   bool            _no_weight_updates,
			   RegulariserCpp *_regulariser
			   );
	
  virtual ~NeuralNetworkNodeGPUCpp();

	  		
};

//----------------------------------------------------------------------------------------------
//class ActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class ActivationFunctionGPUCpp: public NeuralNetworkNodeGPUCpp {
	
public:
	
  ActivationFunctionGPUCpp(
			   std::int32_t    _node_number,
			   std::int32_t    _dim,
			   std::int32_t   *_input_nodes_fed_into_me_dense,
			   std::int32_t    _input_nodes_fed_into_me_dense_length,
			   std::int32_t   *_input_nodes_fed_into_me_sparse,
			   std::int32_t    _input_nodes_fed_into_me_sparse_length,
			   std::int32_t   *_hidden_nodes_fed_into_me,
			   std::int32_t    _hidden_nodes_fed_into_me_length,
			   std::int32_t    _i_share_weights_with,
			   bool            _no_weight_updates,
			   RegulariserCpp *_regulariser			   
			   );
	
  ~ActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class LogisticActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class LogisticActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {
	
public:

  LogisticActivationFunctionGPUCpp (
				    std::int32_t    _node_number,
				    std::int32_t    _dim,
				    std::int32_t   *_input_nodes_fed_into_me_dense,
				    std::int32_t    _input_nodes_fed_into_me_dense_length,
				    std::int32_t   *_input_nodes_fed_into_me_sparse,
				    std::int32_t    _input_nodes_fed_into_me_sparse_length,
				    std::int32_t   *_hidden_nodes_fed_into_me,
				    std::int32_t    _hidden_nodes_fed_into_me_length,
				    std::int32_t    _i_share_weights_with,
				    bool            _no_weight_updates,
				    RegulariserCpp *_regulariser			   
				    );
    
  ~LogisticActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class LinearActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class LinearActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {
	
public:
	
  LinearActivationFunctionGPUCpp (
				  std::int32_t    _node_number,
				  std::int32_t    _dim,
				  std::int32_t   *_input_nodes_fed_into_me_dense,
				  std::int32_t    _input_nodes_fed_into_me_dense_length,
				  std::int32_t   *_input_nodes_fed_into_me_sparse,
				  std::int32_t    _input_nodes_fed_into_me_sparse_length,
				  std::int32_t   *_hidden_nodes_fed_into_me,
				  std::int32_t    _hidden_nodes_fed_into_me_length,
				  std::int32_t    _i_share_weights_with,
				  bool            _no_weight_updates,
				  RegulariserCpp *_regulariser			   
				  );

  ~LinearActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class SoftmaxActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_dense, std::int32_t _input_nodes_fed_into_me_dense_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_input_nodes_fed_into_me_sparse, std::int32_t _input_nodes_fed_into_me_sparse_length)};

class SoftmaxActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {

public:

  SoftmaxActivationFunctionGPUCpp (
				    std::int32_t    _node_number,
				    std::int32_t    _num_vars,
				    std::int32_t    _num_states_per_var,
				    std::int32_t   *_input_nodes_fed_into_me_dense,
				    std::int32_t    _input_nodes_fed_into_me_dense_length,
				    std::int32_t   *_input_nodes_fed_into_me_sparse,
				    std::int32_t    _input_nodes_fed_into_me_sparse_length,
				    std::int32_t   *_hidden_nodes_fed_into_me,
				    std::int32_t    _hidden_nodes_fed_into_me_length,
				    std::int32_t    _i_share_weights_with,
				    bool            _no_weight_updates,
				    RegulariserCpp *_regulariser			   
				   );
	
  ~SoftmaxActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class NeuralNetworkGPUCpp

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


class NeuralNetworkGPUCpp/*: public NumericallyOptimisedMLAlgorithmCpp*/ {
		
public:
	
  NeuralNetworkGPUCpp (
		       std::int32_t *_num_input_nodes_dense, 
		       std::int32_t _num_input_nodes_dense_length, 
		       std::int32_t *_num_input_nodes_sparse,
		       std::int32_t _num_input_nodes_sparse_length, 
		       std::int32_t _num_output_nodes_dense, 
		       std::int32_t _num_output_nodes_sparse,
		       LossFunctionCpp *_loss/*, _RegulariserCpp *_regulariser*/
		       );
	
  ~NeuralNetworkGPUCpp();

  void init_hidden_node(
			NeuralNetworkNodeGPUCpp *_hidden_node
			);
	
  void init_output_node(
			NeuralNetworkNodeGPUCpp *_output_node
			);

  std::int32_t get_length_params();

  void get_params(
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
	
  //This function loads the provided dataset into the GPU
  void load_dense_data(
		       std::int32_t _num_input_node, 
		       float       *_X, 
		       std::int32_t _num_samples, 
		       std::int32_t _dim, 
		       std::int32_t _global_batch_size
		       );

  //This function loads the provided targets into the GPU
  void load_dense_targets(
			  std::int32_t num_output_node, 
			  float       *_Y, 
			  std::int32_t _num_samples, 
			  std::int32_t _dim, 
			  std::int32_t _global_batch_size
			  );

  //This function loads the provided dataset into the GPU
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

  //This functions loads the provided targets into the GPU
  void load_sparse_targets(
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
			   );

  //The purpose of this function is to fit the neural network
  void fit (/*MPI_Comm comm,*/ 
	    OptimiserCpp      *_optimiser, 
	    std::int32_t       _global_batch_size,
	    const float        _tol,
	    const std::int32_t _max_num_epochs,
	    const std::int32_t _MinibatchSizeStandard,
	    const bool         _sample
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

  //This functions returns the length of sum of the gradients during each training epoch
  //Identical to the number of epochs
  std::int32_t get_sum_gradients_length();

  //This functions returns the sum of the gradients during each training epoch
  void get_sum_gradients(
			 float        *_sum_gradients,
			 std::int32_t  _sum_gradients_length
			 );
  
  //This functions returns the sum of the dimensionalities of all output nodes
  std::int32_t get_sum_output_dim();
	
};

//----------------------------------------------------------------------------------------------
