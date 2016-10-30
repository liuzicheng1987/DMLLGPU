//Because this loss function is so simple, we refrain from the usual practice of keeping its functions in a separate file.
class SquareLossCpp: public LossFunctionCpp {
	
 public:	
	
  //Constructor
  SquareLossCpp (): LossFunctionCpp() {}

  //Destructor		
  ~SquareLossCpp() {}
			
  void dLossdYhat (/*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/ const std::int32_t I, const std::int32_t J, thrust::device_vector<float> &dlossdYhat, thrust::device_vector<float> &Y, thrust::device_vector<float> &Yhat) {
	
    thrust::transform(Yhat.begin(), Yhat.begin() + I, Y.begin(), dlossdYhat.begin(), thrust::minus<float>());
		
  }
	
};
