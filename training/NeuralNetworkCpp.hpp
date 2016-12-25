class NeuralNetworkNodeCpp;
class OptimiserCpp;
class LossFunctionCpp;

class NeuralNetworkCpp: public NumericallyOptimisedAlgorithmCpp {

  friend class NeuralNetworkNodeCpp;
  
private:
	
  //Vector containing pointers to the neural network nodes
  std::vector<NeuralNetworkNodeCpp*> nodes_;

  //Raw pointer to the output nodes (which are also contained in nodes)
  NeuralNetworkNodeCpp** output_nodes_;

  //Raw pointer to output nodes with dense targets
  NeuralNetworkNodeCpp** output_nodes_dense_;

  //Raw pointer to output nodes with sparse targets
  NeuralNetworkNodeCpp** output_nodes_sparse_;

  //Accumulated number of weights required for each neural network node
  std::vector<std::int32_t> cumulative_num_weights_required_;

  //Dense input data
  std::vector<std::vector<matrix::DenseMatrix>> dense_input_data_;

  //Sparse input data
  std::vector<std::vector<matrix::CSRMatrix>> sparse_input_data_;

  //Dense target data
  std::vector<std::vector<matrix::DenseMatrix>> dense_targets_;

  //Sparse target data
  std::vector<std::vector<matrix::COOVector>> sparse_targets_;

  //Number of dimensions in dense input data
  std::vector<std::int32_t> dense_input_data_dim_;

  //Number of dimensions in sparse input data
  std::vector<std::int32_t> sparse_input_data_dim_;

  //Number of dimensions in dense targets
  std::vector<std::int32_t> dense_targets_dim_;

  //Number of dimensions in sparse targets
  std::vector<std::int32_t> sparse_targets_dim_;

  //Weights for the neural network
  thrust::device_vector<float> W_;

  //Pointer to loss function
  LossFunctionCpp *loss_;

  //Pointer to optimiser_
  OptimiserCpp *optimiser_;

  //Sum of squared gradients for each epoch
  std::vector<float> sum_gradients_;

  //Number of output nodes with dense targets
  std::size_t num_output_nodes_dense_;

  //Number of output nodes with sparse targets
  std::size_t num_output_nodes_sparse_;

  //Number of output nodes (=num_output_nodes_dense_
  //+ num_output_nodes_sparse_), for convenience
  std::size_t num_output_nodes_;
  
  //Number of hidden nodes
  std::size_t num_hidden_nodes_;

  //Neural network can not be trained unless neural network is finalised
  bool finalised_;

  //Some nodes have a random component, this is used to activate sampling
  bool sample_;

  //Number of samples
  std::int32_t num_samples_;

  //Approximate number of samples used for updating the weights
  //in each iteration
  std::int32_t global_batch_size;

  //This handle is needed for the cuBLAS library.
  cublasHandle_t dense_handle_;
  
  //This handle is needed for the cuSPARSE library.
  cusparseHandle_t sparse_handle_;

  //This matrix descriptor is needed for the cuSPARSE library.
  cusparseMatDescr_t mat_descr_;
	       
public:
	
  NeuralNetworkCpp (
		       std::int32_t    *_num_input_nodes_dense, 
		       std::int32_t     _num_input_nodes_dense_length,
		       std::int32_t    *_num_input_nodes_sparse,
		       std::int32_t     _num_input_nodes_sparse_length, 
		       std::int32_t     _num_output_nodes_dense, 
		       std::int32_t     _num_output_nodes_sparse, 
		       LossFunctionCpp *_loss/*, _RegulariserCpp *_regulariser*/
		       );
	
  ~NeuralNetworkCpp();

  //Functions are ordered by order in which they would be executed (roughly)
	
  void init_hidden_node(NeuralNetworkNodeCpp *_hidden_node);//Initialise an input node
	
  void init_output_node(NeuralNetworkNodeCpp *_output_node);//Initialise an output node

  //This has nothing to do with "finalize" Java. It simply calculates a number of parameters in the neural net 
  void finalise(/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/ 
		float _weight_init_range
		);

  //Returns the number of weights of the neural network
  std::int32_t get_length_params();

  //Returns the weight of the neural network
  void get_params(float *_W, std::int32_t _length_W);

  //Sets the weight of the neural network
  void set_params(float *_W, std::int32_t _length_W);

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
  
  //load_dense_data and load_dense_targets are actually wrappers, which simply call this method
  void load_dense(
		  std::vector<matrix::DenseMatrix> &data, 
		  float                            *_X, 
		  std::int32_t                      _num_samples, 
		  std::int32_t                      _dim, 
		  std::int32_t                      _num_batches,
		  bool                              _transpose
		  );
  
  //This functions loads the provided dataset into the 
  void load_dense_data(
		       std::int32_t _num_input_node, 
		       float       *_X, 
		       std::int32_t _num_samples,
		       std::int32_t _dim,
		       std::int32_t _global_batch_size
		       );

  //This functions loads the provided targets into the 
  void load_dense_targets(
			  std::int32_t num_output_node, 
			  float       *_Y, 
			  std::int32_t _num_samples, 
			  std::int32_t _dim, 
			  std::int32_t _global_batch_size
			  );
  
  //load_sparse_data is a wrapper, which simply calls this method
  void load_csr(
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
		   );

  //load_sparse_targets is a wrapper, which simply calls this method
  void load_coo(
		   std::vector<matrix::COOVector> &_data, 
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
  
  //This functions loads the provided sparse data into the 
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
  
  //This functions loads the provided targets into the 
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
	    float             *_dLdw, 
	    const float       *_W, 
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
  
  //The purpose of this function is to delete the input data used for fitting or transforming after it is no longer needed, so it doesn't take up space on the 
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
  matrix::DenseMatrix& get_dense_input_data(
					    std::int32_t i, 
					    std::int32_t _batch_num
					    ) {
    return this->dense_input_data_[i][_batch_num];
  };

  //The nodes need to be able to access the private input data dimension.
  std::vector<std::int32_t>& get_dense_input_data_dim() {
    return this->dense_input_data_dim_;
  };

  //The nodes need to be able to access the private input data dimension.
  std::vector<std::int32_t>& get_sparse_input_data_dim() {
    return this->sparse_input_data_dim_;
  };

  //The nodes need to be able to access the private input data.
  matrix::CSRMatrix& get_sparse_input_data(
					   std::int32_t i, 
					   std::int32_t _batch_num
					   ) {
    return this->sparse_input_data_[i][_batch_num];
  };

  //This functions returns the length of sum of the gradients during each training epoch
  //Identical to the number of epochs
  std::int32_t get_sum_gradients_length() {
    return static_cast<std::int32_t>(this->sum_gradients_.size());
  };
  
  //This functions returns the sum of the gradients during each training epoch
  void get_sum_gradients(
			 float        *_sum_gradients,
			 std::int32_t  _sum_gradients_size
			 ) {
    std::copy(this->sum_gradients_.begin(),
	      this->sum_gradients_.end(),
	      _sum_gradients
	      );
  };


  //This functions returns the sum of the dimensionalities of all output nodes
  std::int32_t get_sum_output_dim() {
    
    std::int32_t output_dim = 0;

    for (auto dim: dense_targets_dim_)
      output_dim += dim;

    for (auto dim: sparse_targets_dim_)
      output_dim += dim;

    return output_dim;

  };

  //This functions returns the boolean that determines whether we should sample
  bool get_sample() {

    return this->sample_;

  };

  
};
