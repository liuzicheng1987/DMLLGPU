class LossFunctionCpp {
	
 public:	
	
  //Constructor
  LossFunctionCpp() {}

  //Virtual destructor		
  virtual ~LossFunctionCpp() {}
			
  virtual void dLossdYhat (/*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ const std::int32_t I, const std::int32_t J, thrust::device_vector<float> &dlossdYhat, thrust::device_vector<float> &Y, thrust::device_vector<float> &Yhat) {};
	
};
