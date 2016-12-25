//The class NumericallyOptimisedAlgorithmCpp serves as a base class for every
//algorithm that is to be numerically optimised (such as neural networks)

class NumericallyOptimisedAlgorithmCpp {
	       
public:
  
  NumericallyOptimisedAlgorithmCpp() {};
	
  ~NumericallyOptimisedAlgorithmCpp() {};

  //Calculates the number of batches needed
  std::int32_t calc_num_batches (/*MPI_Comm comm,*/
				 std::int32_t _num_samples, 
				 std::int32_t _global_batch_size
				 ) {
    
    std::int32_t GlobalI;
	
    //Add all local num_samples and store the result in GlobalI
    //MPI_Allreduce(&I, &GlobalI, 1, MPI_INT, MPI_SUM, comm);
    //MPI_Barrier(_comm);

    GlobalI = _num_samples;
	
    if (_global_batch_size < 1 || _global_batch_size > GlobalI) _global_batch_size = GlobalI;		
			
    //Calculate the number of batches needed to divide GlobalI such that the sum of all local batches approximately equals global_batch_size
    if (GlobalI % _global_batch_size == 0) return GlobalI/_global_batch_size; 
    else return GlobalI/_global_batch_size + 1;

    //MPI_Barrier(_comm);
        
  };
  
  //Calculate beginning and end of each batch
  void calc_batch_begin_end (
			     std::int32_t      &_batch_begin, 
			     std::int32_t      &_batch_end, 
			     std::int32_t      &_batch_size, 
			     const std::int32_t _batch_num, 
			     const std::int32_t _num_samples, 
			     const std::int32_t _num_batches
			     ) {

    //Calculate _batch_begin
    _batch_begin = _batch_num*(_num_samples/_num_batches);
		
    //Calculate _batch_size
    if (_batch_num < _num_batches-1) _batch_size = _num_samples/_num_batches;
    else _batch_size = _num_samples - _batch_begin;
		
    //Calculate _batch_end
    _batch_end = _batch_begin + _batch_size;

  };

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
