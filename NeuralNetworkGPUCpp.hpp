class NeuralNetworkNodeGPUCpp;
class OptimiserCpp;
//class DropoutCpp;
//class BayesianDropoutCpp;


//Because the neural network can consist of a number of input matrices, both dense and sparse, we keep them in a vector of structs
struct DenseInputStruct {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions

  thrust::device_vector<float> X;//Vector containing data

  float *X_ptr;//Pointer to data contained in X, for convenience and readability
  
};

struct SparseInputStruct {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions
  std::int32_t num_non_zero;//Number of non-zero elements in matrix

  thrust::device_vector<float> X_data;//Data vector (for sparse datasets)
  thrust::device_vector<std::int32_t> X_indices;//indices for data (for sparse datasets)
  thrust::device_vector<std::int32_t> X_indptr;//indptr for sata (for sparse datasets)

  float *X_data_ptr;//Pointer to data contained in X_data, for convenience and readability
  std::int32_t *X_indices_ptr;//Pointer to data contained in X_indices, for convenience and readability
  std::int32_t *X_indptr_ptr;//Pointer to data contained in X_indptr, for convenience and readability

};

class NeuralNetworkGPUCpp {

  friend class NeuralNetworkNodeGPUCpp;
  //friend class DropoutCpp;
  //friend class BayesianDropoutCpp;
  
private:
	
  std::vector<NeuralNetworkNodeGPUCpp*> nodes;//Vector containing pointers to the neural network nodes
  NeuralNetworkNodeGPUCpp** output_nodes; //Raw pointer to the output nodes (which are also contained in nodes)

  thrust::device_vector<std::int32_t> cumulative_num_weights_required;//Accumulated number of weights required for each neural network node

  std::vector<std::vector<DenseInputStruct>> dense_input_data;//Dense input data
  std::vector<std::vector<SparseInputStruct>> sparse_input_data;//Sparse input data

  std::vector<std::vector<DenseInputStruct>> dense_targets;//Dense target data
  std::vector<std::vector<SparseInputStruct>> sparse_targets;//Sparse target data

  std::vector<std::int32_t> dense_input_data_dim;//Number of dimensions in dense input data
  std::vector<std::int32_t> sparse_input_data_dim;//Number of dimensions in sparse input data

  std::vector<std::int32_t> dense_targets_dim;//Number of dimensions in dense targets
  std::vector<std::int32_t> sparse_targets_dim;//Number of dimensions in sparse targets

  thrust::device_vector<float> W;//Weights for the neural network

  LossFunctionCpp *loss;//Pointer to loss function 
  //RegulariserCpp *regulariser;//Pointer to regulariser

  OptimiserCpp *optimiser;//Pointer to optimiser

  std::vector<float> sum_gradients;//Sum of squared gradients for each epoch

  std::size_t num_output_nodes_dense;//Number of output nodes with dense targets
  std::size_t num_output_nodes_sparse;//Number of output nodes with sparse targets
  std::size_t num_output_nodes;//Number of output nodes (=num_output_nodes_dense + num_output_nodes_sparse), for convenience
  
  std::size_t num_hidden_nodes;//Number of hidden nodes

  bool finalised;//Neural network can not be trained unless neural network is finalised

  bool sample;//Some nodes have a random component, this is used to activate sampling

  std::int32_t num_samples;//Number of samples

  std::int32_t global_batch_size;//Approximate number of samples used for updating the weights in each iteration

  cublasHandle_t dense_handle_;//This handle is needed for the cuBLAS library.
  
  cusparseHandle_t sparse_handle_;//This handle is needed for the cuSPARSE library.

  cusparseMatDescr_t mat_descr_;//This matrix descriptor is needed for the cuSPARSE library.
	       
public:
	
  NeuralNetworkGPUCpp (
		       std::int32_t    *_num_input_nodes_dense, 
		       std::int32_t     _num_input_nodes_dense_length,
		       std::int32_t    *_num_input_nodes_sparse,
		       std::int32_t     _num_input_nodes_sparse_length, 
		       std::int32_t     _num_output_nodes_dense, 
		       std::int32_t     _num_output_nodes_sparse, 
		       LossFunctionCpp *_loss/*, _RegulariserCpp *_regulariser*/
		       );
	
  ~NeuralNetworkGPUCpp();

  //Functions are ordered by order in which they would be executed (roughly)
	
  void init_hidden_node(NeuralNetworkNodeGPUCpp *_hidden_node);//Initialise an input node
	
  void init_output_node(NeuralNetworkNodeGPUCpp *_output_node);//Initialise an output node

  //This has nothing to do with "finalize" Java. It simply calculates a number of parameters in the neural net 
  void finalise(/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/ float _weight_init_range);

  //Returns the number of weights of the neural network
  std::int32_t get_length_params();

  //Returns the weight of the neural network
  void get_params(float *_W, std::int32_t _length_W);

  //Returns an integer signifying the number of hidden nodes fed into node _node_number
  std::int32_t get_input_nodes_fed_into_me_dense_length(std::int32_t _node_number);
	
  //Returns an integer vector signifying the hidden nodes fed into _node_number
  void get_input_nodes_fed_into_me_dense(
					 std::int32_t  _node_number, 
					 std::int32_t *_input_nodes_fed_into_me_dense, 
					 std::int32_t  _input_nodes_fed_into_me_dense_length
					 );
  //Returns an integer signifying the number of hidden nodes fed into node _node_number
  std::int32_t get_input_nodes_fed_into_me_sparse_length(std::int32_t _node_number);
	
  //Returns an integer vector signifying the hidden nodes fed into node  _node_number
  void get_input_nodes_fed_into_me_sparse(
					  std::int32_t  _node_number,
					  std::int32_t *_input_nodes_fed_into_me_sparse, 
					  std::int32_t  _input_nodes_fed_into_me_sparse_length
					  );
  //Returns an integer signifying the number of hidden nodes fed into node _node_number
  std::int32_t get_hidden_nodes_fed_into_me_length(std::int32_t _node_number);
	
  //Returns an integer vector signifying the hidden nodes fed into node  _node_number
  void get_hidden_nodes_fed_into_me(
				    std::int32_t  _node_number, 
				    std::int32_t *_hidden_nodes_fed_into_me, 
				    std::int32_t  __lengthhidden_nodes_fed_into_me
				    );

  //Calculates the number of batches needed
  std::int32_t calc_num_batches (/*MPI_Comm comm,*/
				 std::int32_t _num_samples, 
				 std::int32_t _global_batch_size
				 );
  
  //Calculate beginning and end of each batch
  void calc_batch_begin_end (
			     std::int32_t      &_batch_begin, 
			     std::int32_t      &_batch_end, 
			     std::int32_t      &_batch_size, 
			     const std::int32_t _batch_num, 
			     const std::int32_t _num_samples, 
			     const std::int32_t _num_batches
			     );
  
  //load_dense_data and load_dense_targets are actually wrappers, which simply call this method
  void load_dense(
		  std::vector<DenseInputStruct> &data, 
		  float                         *_X, 
		  std::int32_t                   _num_samples, 
		  std::int32_t                   _dim, 
		  std::int32_t                   _num_batches
		  );
  
  //This functions loads the provided dataset into the GPU
  void load_dense_data(
		       std::int32_t _num_input_node, 
		       float       *_X, 
		       std::int32_t _num_samples,
		       std::int32_t _dim,
		       std::int32_t _global_batch_size
		       );

  //This functions loads the provided targets into the GPU
  void load_dense_targets(
			  std::int32_t num_output_node, 
			  float       *_Y, 
			  std::int32_t _num_samples, 
			  std::int32_t _dim, 
			  std::int32_t _global_batch_size
			  );
  
  //load_sparse_data and load_sparse_targets are actually wrappers, which simply call this method
  void load_sparse(
		   std::vector<SparseInputStruct> &data, 
		   float                          *_X_data, 
		   std::int32_t                    _X_data_length, 
		   std::int32_t                   *_X_indices,
		   std::int32_t                    _X_indices_length,
		   std::int32_t                   *_X_indptr, 
		   std::int32_t                    _X_indptr_length, 
		   std::int32_t                    _num_samples,
		   std::int32_t                    _dim, 
		   std::int32_t                    _num_batches
		   );
  
  //This functions loads the provided sparse data into the GPU
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

  //The purpose of this function is to calculate the gradient of the weights
  void dfdw(/*MPI_Comm comm,*/
	    float *_dLdw, 
	    const float *_W, 
	    const std::int32_t _batch_begin, 
	    const std::int32_t _batch_end, 
	    const std::int32_t _batch_size, 
	    const std::int32_t _batch_num, 
	    const std::int32_t _epoch_num
	    );

  //The purpose of this function is to generate a prediction through the fitted network
  void transform(
		 float       *_Yhat,
		 std::int32_t _Y2_num_samples,
		 std::int32_t _Y2_dim,
		 bool         _sample,
		 std::int32_t _sample_size,
		 bool         _Gethidden_nodes
		 );
  
  //The purpose of this function is to delete the input data used for fitting or transforming after it is no longer needed, so it doesn't take up space on the GPU
  void delete_data();

  //The following functions are a bunch of getters and setters

  //The nodes need to be able to access the cuBLAS handle
  cublasHandle_t& get_dense_handle() {
    return this->dense_handle_;
  };

  //The nodes need to be able to access the cuSPARSE handle
  cusparseHandle_t& get_sparse_handle() {
    return this->sparse_handle_;
  };

  //The nodes need to be able to access the cuSPARSE matrix descriptior
  cusparseMatDescr_t& get_mat_descr() {
    return this->mat_descr_;
  };

  //The nodes need to be able to access the private input data.
  DenseInputStruct& get_dense_input_data(
					 std::int32_t i, 
					 std::int32_t _batch_num
					 ) {
    return this->dense_input_data[i][_batch_num];
  };

  //The nodes need to be able to access the private input data.
  SparseInputStruct& get_sparse_input_data(
					   std::int32_t i, 
					   std::int32_t _batch_num
					   ) {
    return this->sparse_input_data[i][_batch_num];
  };

  //This functions returns the length of sum of the gradients during each training epoch
  //Identical to the number of epochs
  std::int32_t get_sum_gradients_length() {
    return static_cast<std::int32_t>(this->sum_gradients.size());
  };
  
  //This functions returns the sum of the gradients during each training epoch
  void get_sum_gradients(
			 float        *_sum_gradients,
			 std::int32_t  _sum_gradients_size
			 ) {
    std::copy(this->sum_gradients.begin(),
	      this->sum_gradients.end(),
	      _sum_gradients
	      );
  };

	
};
