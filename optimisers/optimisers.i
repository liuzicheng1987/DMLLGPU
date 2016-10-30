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
		
  SGDCpp (float _LearningRate, float _LearningRatePower):OptimiserCpp();

  ~SGDCpp();
				
};



