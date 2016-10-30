class NeuralNetworkNodeGPUCpp;
class OptimiserCpp;
//class DropoutCpp;
//class BayesianDropoutCpp;


//Because the neural network can consist of a number of input matrices, both dense and sparse, we keep them in a vector of structs
struct DenseInputStruct {

  std::int32_t BatchSize;//Number of samples
  std::int32_t J;//Number of dimensions

  thrust::device_vector<float> X;//Vector containing data

  float *Xptr;//Pointer to data contained in X, for convenience and readability
  
};

struct SparseInputStruct {

  std::int32_t BatchSize;//Number of samples
  std::int32_t J;//Number of dimensions

  thrust::device_vector<float> XData;//Data vector (for sparse datasets)
  thrust::device_vector<std::int32_t> XIndices;//indices for data (for sparse datasets)
  thrust::device_vector<std::int32_t> XIndptr;//indptr for sata (for sparse datasets)

  float *XDataPtr;//Pointer to data contained in XData, for convenience and readability
  std::int32_t *XIndicesPtr;//Pointer to data contained in XIndices, for convenience and readability
  std::int32_t *XIndptrPtr;//Pointer to data contained in XIndptr, for convenience and readability

};

class NeuralNetworkGPUCpp {

  friend class NeuralNetworkNodeGPUCpp;
  //friend class DropoutCpp;
  //friend class BayesianDropoutCpp;
  
private:
	
  std::vector<NeuralNetworkNodeGPUCpp*> nodes;//Vector containing pointers to the neural network nodes
  NeuralNetworkNodeGPUCpp** OutputNodes; //Raw pointer to the output nodes (which are also contained in nodes)
  thrust::device_vector<std::int32_t> CumulativeNumWeightsRequired;//Accumulated number of weights required for each neural network node

  std::vector<std::vector<DenseInputStruct>> DenseInputData;//Dense input data
  std::vector<std::vector<SparseInputStruct>> SparseInputData;//Sparse input data

  std::vector<std::vector<DenseInputStruct>> DenseTargets;//Dense target data
  std::vector<std::vector<SparseInputStruct>> SparseTargets;//Sparse target data

  std::vector<std::int32_t> DenseInputDataJ;//Number of dimensions in dense input data
  std::vector<std::int32_t> SparseInputDataJ;//Number of dimensions in sparse input data

  std::vector<std::int32_t> DenseTargetsJ;//Number of dimensions in dense targets
  std::vector<std::int32_t> SparseTargetsJ;//Number of dimensions in sparse targets

  thrust::device_vector<float> W;//Weights for the neural network

  LossFunctionCpp *loss;//Pointer to loss function 
  //RegulariserCpp *regulariser;//Pointer to regulariser

  OptimiserCpp *optimiser;//Pointer to optimiser

  std::vector<float> SumGradients;//Sum of squared gradients for each epoch

  std::size_t NumOutputNodesDense;//Number of output nodes with dense targets
  std::size_t NumOutputNodesSparse;//Number of output nodes with sparse targets
  std::size_t NumOutputNodes;//Number of output nodes (=NumOutputNodesDense + NumOutputNodesSparse), for convenience
  
  std::size_t NumHiddenNodes;//Number of hidden nodes

  bool finalised;//Neural network can not be trained unless neural network is finalised

  bool sample;//Some nodes have a random component, this is used to activate sampling

  std::int32_t I;//Number of samples

  std::int32_t GlobalBatchSize;//Approximate number of samples used for updating the weights in each iteration

  cublasHandle_t handle;//This handle is needed for the cuBLAS library.
		
public:
	
  NeuralNetworkGPUCpp (std::int32_t *_NumInputNodesDense, std::int32_t _NumInputNodesDenseLength, std::int32_t *_NumInputNodesSparse, std::int32_t _NumInputNodesSparseLength, std::int32_t _NumOutputNodesDense, std::int32_t _NumOutputNodesSparse, LossFunctionCpp *_loss/*, _RegulariserCpp *_regulariser*/);
	
  ~NeuralNetworkGPUCpp();

  //Functions are ordered by order in which they would be executed (roughly)
	
  void init_hidden_node(NeuralNetworkNodeGPUCpp *_HiddenNode);//Initialise an input node
	
  void init_output_node(NeuralNetworkNodeGPUCpp *_OutputNode);//Initialise an output node

  void finalise(/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/ float _WeightInitRange);//This has nothing to do with "finalize" in Java. It simply calculates a number of parameters in the neural net 

  std::int32_t get_length_params();//Returns the number of weights of the neural network

  void get_params(float *_W, std::int32_t _LengthW);//Returns the weight of the neural network

  std::int32_t get_input_nodes_fed_into_me_dense_length(std::int32_t _NodeNumber);//Returns an integer signifying the number of hidden nodes fed into node _NodeNumber
	
  void get_input_nodes_fed_into_me_dense(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength);//Returns an integer vector signifying the hidden nodes fed into node  _NodeNumber

  std::int32_t get_input_nodes_fed_into_me_sparse_length(std::int32_t _NodeNumber);//Returns an integer signifying the number of hidden nodes fed into node _NodeNumber
	
  void get_input_nodes_fed_into_me_sparse(std::int32_t _NodeNumber,std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength);//Returns an integer vector signifying the hidden nodes fed into node  _NodeNumber
	
  std::int32_t get_hidden_nodes_fed_into_me_length(std::int32_t _NodeNumber);//Returns an integer signifying the number of hidden nodes fed into node _NodeNumber
	
  void get_hidden_nodes_fed_into_me(std::int32_t _NodeNumber, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _LengthHiddenNodesFedIntoMe);//Returns an integer vector signifying the hidden nodes fed into node  _NodeNumber

  std::int32_t calc_num_batches (/*MPI_Comm comm,*/std::int32_t _I, std::int32_t _GlobalBatchSize);//Calculates the number of batches needed
		
  void calc_batch_begin_end (std::int32_t &_BatchBegin, std::int32_t &_BatchEnd, std::int32_t &_BatchSize, const std::int32_t _BatchNum, const std::int32_t _I, const std::int32_t _NumBatches);//Calculates beginning and end of batch at each iteration

  void load_dense(std::vector<DenseInputStruct> &data, float *_X, std::int32_t _I, std::int32_t _J, std::int32_t _NumBatches);//load_dense_data and load_dense_targets are actually wrappers, which simply call this method
		
  void load_dense_data(std::int32_t _NumInputNode, float *_X, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize);//This functions loads the provided dataset into the GPU

  void load_dense_targets(std::int32_t NumOutputNode, float *_Y, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize);//This functions loads the provided targets into the GPU

  void load_sparse(std::vector<SparseInputStruct> &data, float *_XData, std::int32_t _XDataLength,  std::int32_t *_XIndices, std::int32_t _XIndicesLength, std::int32_t *_XIndptr, std::int32_t _XIndptrLength, std::int32_t _I, std::int32_t _J, std::int32_t _NumBatches);//load_sparse_data and load_sparse_targets are actually wrappers, which simply call this method

  void load_sparse_data(std::int32_t _NumInputNode, float *_XData, std::int32_t _XDataLength,  std::int32_t *_XIndices, std::int32_t _XIndicesLength, std::int32_t *_XIndptr, std::int32_t _XIndptrLength, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize);//This functions loads the provided dataset into the GPU

  void load_sparse_targets(std::int32_t _NumOutputNode, float *_YData, std::int32_t _YDataLength,  std::int32_t *_YIndices, std::int32_t _YIndicesLength, std::int32_t *_YIndptr, std::int32_t _YIndptrLength, std::int32_t _I, std::int32_t _J, std::int32_t _GlobalBatchSize);//This functions loads the provided targets into the GPU

  void fit (/*MPI_Comm comm,*/ OptimiserCpp *_optimiser, std::int32_t _GlobalBatchSize, const float _tol, const std::int32_t _MaxNumEpochs, const std::int32_t _MinibatchSizeStandard, const bool _sample);//The purpose of this function is to fit the neural network

  void dfdw(/*MPI_Comm comm,*/float *_dLdw, const float *_W, const std::int32_t _BatchBegin, const std::int32_t _BatchEnd, const std::int32_t _BatchSize, const std::int32_t _BatchNum, const std::int32_t _EpochNum);//The purpose of this function is to calculate the gradient of the weights

  void transform(float *_Yhat, std::int32_t _IY2, std::int32_t _JY2, bool _sample, std::int32_t _SampleSize, bool _GetHiddenNodes);//The purpose of this function is to generate a prediction through the fitted network
  
  void delete_data();//The purpose of this function is to delete the input data used for fitting or transforming after it is no longer needed, so it doesn't take up space on the GPU

  //The following functions are a bunch of getters and setters

  //The nodes need to be able to acces the cuBLAS handle
  cublasHandle_t& get_handle() {
    return this->handle;
  };

  //The nodes need to be able to access the private input data.
  DenseInputStruct& get_dense_input_data(
					 std::int32_t i, 
					 std::int32_t _BatchNum
					 ) {
    return this->DenseInputData[i][_BatchNum];
  };

  //The nodes need to be able to access the private input data.
  SparseInputStruct& get_sparse_input_data(
					   std::int32_t i, 
					   std::int32_t _BatchNum
					   ) {
    return this->SparseInputData[i][_BatchNum];
  };

  //This functions returns the length of sum of the gradients during each training epoch
  //Identical to the number of epochs
  std::int32_t get_sum_gradients_length() {
    return static_cast<std::int32_t>(this->SumGradients.size());
  };
  
  //This functions returns the sum of the gradients during each training epoch
  void get_sum_gradients(
			 float        *_sum_gradients,
			 std::int32_t  _sum_gradients_size
			 ) {
    std::copy(this->SumGradients.begin(),
	      this->SumGradients.end(),
	      _sum_gradients
	      );
  };

	
};
