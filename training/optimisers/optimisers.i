//----------------------------------------------------------------------------------------------
//class NumericallyOptimisedAlgorithmCpp

class NumericallyOptimisedAlgorithmCpp {
	
public:

  NumericallyOptimisedAlgorithmCpp();

  virtual ~NumericallyOptimisedAlgorithmCpp();

};

//----------------------------------------------------------------------------------------------
//class OptimiserCpp

class OptimiserCpp {
	
public:

  OptimiserCpp();

  virtual ~OptimiserCpp();

};

//----------------------------------------------------------------------------------------------
//class SGDCpp


class SGDCpp: public OptimiserCpp {
	
public:
		
  SGDCpp (float _learning_rate, 
          float _learning_rate_power, 
          float _momentum) : OptimiserCpp();

  ~SGDCpp();
				
};

//----------------------------------------------------------------------------------------------
//class AdaGradCpp


class AdaGradCpp: public OptimiserCpp {
	
public:
		
  AdaGradCpp (float _learning_rate) : OptimiserCpp();

  ~AdaGradCpp();

};

//----------------------------------------------------------------------------------------------
//class RMSPropCpp
  
class RMSPropCpp : public OptimiserCpp
{


public:
  //Initialise the GradientDescent function
  RMSPropCpp(
      float _learning_rate, float _gamma) : OptimiserCpp(/*size, rank*/)
  {

    //Store the input values
    this->learning_rate_ = _learning_rate;

    this->gamma_ = _gamma;

    this->epoch_num_ = 0;
  }

  //Destructor
  ~RMSPropCpp() {}

  //dev_function_type is defined in OptimiserCpp.hpp!
  void min(/*MPI_Comm comm,*/
           NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
           thrust::device_vector<float> &_W,
           const float _tol,
           const std::int32_t _max_num_epochs,
           std::vector<float> &_sum_gradients);
};


				





