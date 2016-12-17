//The class NumericallyOptimisedAlgorithmCpp serves as a base class for every
//algorithm that is to be numerically optimised (such as neural networks)

class NumericallyOptimisedAlgorithmCpp {
	       
public:
  
  NumericallyOptimisedAlgorithmCpp() {};
	
  ~NumericallyOptimisedAlgorithmCpp() {};

  //Calculates the number of batches needed
  virtual std::int32_t calc_num_batches (/*MPI_Comm comm,*/
					 std::int32_t _num_samples, 
					 std::int32_t _global_batch_size
					 ) {};
  
  //Calculate beginning and end of each batch
  virtual void calc_batch_begin_end (
				     std::int32_t      &_batch_begin, 
				     std::int32_t      &_batch_end, 
				     std::int32_t      &_batch_size, 
				     const std::int32_t _batch_num, 
				     const std::int32_t _num_samples, 
				     const std::int32_t _num_batches
				     ) {};

  //The purpose of this function is to calculate the gradient of the weights
  virtual void dfdw(/*MPI_Comm comm,*/
		    float             *_dLdw, 
		    const float       *_W, 
		    const std::int32_t _batch_begin, 
		    const std::int32_t _batch_end, 
		    const std::int32_t _batch_size, 
		    const std::int32_t _batch_num, 
		    const std::int32_t _epoch_num
		    ) {};
  
};
