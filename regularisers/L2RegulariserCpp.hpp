class L2RegulariserCpp: public RegulariserCpp {

private:
  
  //Constant that is multiplied with derivative
  float alpha_;

public:
	
  //Constructor
  L2RegulariserCpp(float _alpha): RegulariserCpp() {
    
    this->alpha_ = _alpha;
    
  };

  ~L2RegulariserCpp() {};
	
  //Calculate derivative of regulariser for w
  void drdw (/*MPI_Comm comm,*/
	     cublasHandle_t    &_handle,
	     const std::int32_t _length_w,
	     const float        _batch_size_float,
	     const float       *_w,
	     float             *_dldw
	     ) {
    
    cublasStatus_t cstatus;

    float alpha = this->alpha_*2.f*_batch_size_float;

    cstatus = cublasSaxpy(
			  _handle, //handle 
			  _length_w, //n
			  &alpha, //alpha
			  _w, //x
			  1, //incx
			  _dldw, //y
			  1 //incy
			  );

    if (cstatus != CUBLAS_STATUS_SUCCESS) 
      throw std::runtime_error("Something went wrong during cuBLAS saxpy operation in L2Regulariser!");

      
      
  };
			       		
};
