//----------------------------------------------------------------------------------------------
//class NeuralNetworkNodeGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength)};

class NeuralNetworkNodeGPUCpp {
	
  //friend class NeuralNetworkGPUCpp;
	
 public://This is temporary - change to protected later
			
  NeuralNetworkNodeGPUCpp (std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates);
	
  virtual ~NeuralNetworkNodeGPUCpp();

	  		
};

//----------------------------------------------------------------------------------------------
//class ActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength)};

class ActivationFunctionGPUCpp: public NeuralNetworkNodeGPUCpp {
	
public:
	
  ActivationFunctionGPUCpp(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates);
	
  ~ActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class LogisticActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength)};

class LogisticActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {
	
public:

  LogisticActivationFunctionGPUCpp(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates);
    
  ~LogisticActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class LinearActivationFunctionGPUCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength)};

class LinearActivationFunctionGPUCpp: public ActivationFunctionGPUCpp {
	
public:
	
  LinearActivationFunctionGPUCpp(std::int32_t _NodeNumber, std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength, std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength, std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _HiddenNodesFedIntoMeLength, std::int32_t _IShareWeightsWith, bool _NoWeightUpdates);
	
  ~LinearActivationFunctionGPUCpp();

};

//----------------------------------------------------------------------------------------------
//class NeuralNetworkGPUCpp

%apply (float* IN_ARRAY1, int DIM1) {(float *_W, std::int32_t _LengthW)};

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_NumInputNodesDense, std::int32_t _NumInputNodesDenseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_NumInputNodesSparse, std::int32_t _NumInputNodesSparseLength)};

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeDense, std::int32_t _InputNodesFedIntoMeDenseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_InputNodesFedIntoMeSparse, std::int32_t _InputNodesFedIntoMeSparseLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_HiddenNodesFedIntoMe, std::int32_t _LengthHiddenNodesFedIntoMe)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_Yhat, std::int32_t _IY2, std::int32_t _JY2)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_X, std::int32_t _I, std::int32_t _J)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_XData, std::int32_t _XDataLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_XIndices, std::int32_t _XIndicesLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_XIndptr, std::int32_t _XIndptrLength)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_Y, std::int32_t _I, std::int32_t _J)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_YData, std::int32_t _YDataLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_YIndices, std::int32_t _YIndicesLength)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_YIndptr, std::int32_t _YIndptrLength)};

%apply (float* IN_ARRAY1, int DIM1) {(float *_sum_gradients, std::int32_t _sum_gradients_length)};


class NeuralNetworkGPUCpp/*: public NumericallyOptimisedMLAlgorithmCpp*/ {
		
public:
	
  NeuralNetworkGPUCpp (std::int32_t *_NumInputNodesDense, std::int32_t _NumInputNodesDenseLength, std::int32_t *_NumInputNodesSparse, std::int32_t _NumInputNodesSparseLength, std::int32_t _NumOutputNodesDense, std::int32_t _NumOutputNodesSparse, LossFunctionCpp *_loss/*, _RegulariserCpp *_regulariser*/);
	
  ~NeuralNetworkGPUCpp();

  void init_hidden_node(
			NeuralNetworkNodeGPUCpp *_HiddenNode
			);
	
  void init_output_node(
			NeuralNetworkNodeGPUCpp *_OutputNode
			);

  std::int32_t get_length_params();

  void get_params(
		  float       *_W, 
		  std::int32_t _LengthW
		  );

  std::int32_t get_input_nodes_fed_into_me_dense_length(
							std::int32_t _NodeNumber
							);
	
  void get_input_nodes_fed_into_me_dense(
					 std::int32_t  _NodeNumber, 
					 std::int32_t *_InputNodesFedIntoMeDense, 
					 std::int32_t  _InputNodesFedIntoMeDenseLength
					 );

  std::int32_t get_input_nodes_fed_into_me_sparse_length(
							 std::int32_t _NodeNumber
							 );
	
  void get_input_nodes_fed_into_me_sparse(
					  std::int32_t  _NodeNumber, 
					  std::int32_t *_InputNodesFedIntoMeSparse, 
					  std::int32_t  _InputNodesFedIntoMeSparseLength
					  );
	
  std::int32_t get_hidden_nodes_fed_into_me_length(
						   std::int32_t _NodeNumber
						   );

  void get_hidden_nodes_fed_into_me(
				    std::int32_t _NodeNumber, 
				    std::int32_t *_HiddenNodesFedIntoMe, 
				    std::int32_t _LengthHiddenNodesFedIntoMe
				    );

  //This function prepares the neural network for training
  void finalise(
		/*MPI_Comm comm, std::int32_t rank, std::int32_t size,*/ 
		float _WeightInitRange
		);
	
  //This function loads the provided dataset into the GPU
  void load_dense_data(
		       std::int32_t _NumInputNode, 
		       float       *_X, 
		       std::int32_t _I, 
		       std::int32_t _J, 
		       std::int32_t _GlobalBatchSize
		       );

  //This function loads the provided targets into the GPU
  void load_dense_targets(
			  std::int32_t NumOutputNode, 
			  float       *_Y, 
			  std::int32_t _I, 
			  std::int32_t _J, 
			  std::int32_t _GlobalBatchSize
			  );

  //This function loads the provided dataset into the GPU
  void load_sparse_data(
			std::int32_t  _NumInputNode, 
			float        *_XData, 
			std::int32_t  _XDataLength,  
			std::int32_t *_XIndices, 
			std::int32_t  _XIndicesLength,
			std::int32_t *_XIndptr, 
			std::int32_t  _XIndptrLength, 
			std::int32_t  _I, 
			std::int32_t  _J, 
			std::int32_t  _GlobalBatchSize
			);

  //This functions loads the provided targets into the GPU
  void load_sparse_targets(
			   std::int32_t  _NumOutputNode, 
			   float        *_YData, 
			   std::int32_t  _YDataLength,  
			   std::int32_t *_YIndices, 
			   std::int32_t  _YIndicesLength, 
			   std::int32_t *_YIndptr, 
			   std::int32_t  _YIndptrLength, 
			   std::int32_t  _I, 
			   std::int32_t  _J, 
			   std::int32_t  _GlobalBatchSize
			   );

  //The purpose of this function is to fit the neural network
  void fit (/*MPI_Comm comm,*/ 
	    OptimiserCpp      *_optimiser, 
	    std::int32_t       _GlobalBatchSize,
	    const float        _tol,
	    const std::int32_t _MaxNumEpochs,
	    const std::int32_t _MinibatchSizeStandard,
	    const bool         _sample
	    );

  //This functions transform inputs data into predictions
  void transform(
		 float       *_Yhat, 
		 std::int32_t _IY2, 
		 std::int32_t _JY2, 
		 bool         _sample, 
		 std::int32_t _SampleSize,
		 bool         _GetHiddenNodes
		 );

  //This functions returns the length of sum of the gradients during each training epoch
  //Identical to the number of epochs
  std::int32_t get_sum_gradients_length();

  //This functions returns the sum of the gradients during each training epoch
  void get_sum_gradients(
			 float        *_sum_gradients,
			 std::int32_t  _sum_gradients_length
			 );
	
};

//----------------------------------------------------------------------------------------------
