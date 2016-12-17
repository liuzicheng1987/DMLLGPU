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
		
  SGDCpp (float _learning_rate, float _learning_rate_power) : OptimiserCpp();

  ~SGDCpp();
				
};

//----------------------------------------------------------------------------------------------
//class AdaGradCpp


class AdaGradCpp: public OptimiserCpp {
	
public:
		
  AdaGradCpp (float _learning_rate) : OptimiserCpp();

  ~AdaGradCpp();
				
};




