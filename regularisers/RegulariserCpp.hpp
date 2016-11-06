class RegulariserCpp {
	
public:
	
  //Constructor
  RegulariserCpp() {};

  virtual ~RegulariserCpp() {};
	
  //Calculate derivative of regulariser for w
  virtual void drdw (/*MPI_Comm comm,*/
		     cublasHandle_t    &_handle,
		     const std::int32_t _length_w,
		     const float        _batch_size_float,
		     const float       *_w,
		     float             *_dldw
		     ) {};
			       		
};
