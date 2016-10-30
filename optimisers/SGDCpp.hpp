class SGDCpp: public OptimiserCpp {
	
private:

  float LearningRate, LearningRatePower;

  std::int32_t EpochNum;

public:
		
  //Initialise the GradientDescent function
  SGDCpp (float _LearningRate, float _LearningRatePower/*, const std::int32_t _size, const std::int32_t _rank*/): OptimiserCpp(/*size, rank*/) {
	
    //Store all of the input values
    this->LearningRate = _LearningRate; 
    this->LearningRatePower = _LearningRatePower;

    this->EpochNum = 0;//Initialise number of epochs to zero
		
  }

  //Destructor		
  ~SGDCpp() {}
		
   //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/NeuralNetworkGPUCpp *_NeuralNet, thrust::device_vector<float> &_W, const float _tol, const std::int32_t _MaxNumEpochs, std::vector<float> &_SumGradients);
		
};

