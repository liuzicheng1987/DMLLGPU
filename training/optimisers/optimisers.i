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
		
  AdaGradCpp (float _learning_rate, float _offset) : OptimiserCpp();

  ~AdaGradCpp();

};


//----------------------------------------------------------------------------------------------
//class AdaDeltaCpp

class AdaDeltaCpp : public OptimiserCpp
{

public:

  AdaDeltaCpp(
      float _gamma,
      float _offset) : OptimiserCpp();

  ~AdaDeltaCpp() {}

};


//----------------------------------------------------------------------------------------------
//class RMSPropCpp
  
class RMSPropCpp : public OptimiserCpp
{


public:
  //Initialise the GradientDescent function
  RMSPropCpp(
      float _learning_rate, float _gamma) : OptimiserCpp(/*size, rank*/);

  //Destructor
  ~RMSPropCpp();

};

//----------------------------------------------------------------------------------------------
//class AdamCpp
  
class AdamCpp : public OptimiserCpp
{


public:
  //Initialise the GradientDescent function
  AdamCpp(
    float _learning_rate, float decay_mom1_, float decay_mom2_, float offset_) : OptimiserCpp(/*size, rank*/);

  //Destructor
  ~AdamCpp();

};

//----------------------------------------------------------------------------------------------
//class NadamCpp
  
class NadamCpp : public OptimiserCpp
{


public:
  //Initialise the GradientDescent function
  NadamCpp(
    float _learning_rate, float beta_1_, float beta_2_, float schedule_decay_, float offset_) : OptimiserCpp(/*size, rank*/);

  //Destructor
  ~NadamCpp();

};



				





