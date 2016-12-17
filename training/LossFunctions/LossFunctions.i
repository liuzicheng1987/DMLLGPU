class LossFunctionCpp {
	
public:	
  LossFunctionCpp ();
  ~LossFunctionCpp();

};

class SquareLossCpp: public LossFunctionCpp {
	
public:	
  SquareLossCpp (): LossFunctionCpp();
  ~SquareLossCpp();

};


