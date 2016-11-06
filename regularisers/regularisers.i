//----------------------------------------------------------------------------------------------
//class RegulariserCpp

class RegulariserCpp {
	
public:

  RegulariserCpp() {};

  virtual ~RegulariserCpp() {};

};

//----------------------------------------------------------------------------------------------
//class L2RegulariserCpp


class L2RegulariserCpp: public RegulariserCpp {
	
public:
		
  L2RegulariserCpp(float _alpha);

  ~L2RegulariserCpp();
			
};



