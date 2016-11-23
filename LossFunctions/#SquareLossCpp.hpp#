//Because this loss function is so simple, we refrain from the usual practice of keeping its functions in a separate file.
class SquareLossCpp: public LossFunctionCpp {
	
public:	
	
  //Constructor
  SquareLossCpp (): LossFunctionCpp() {}

  //Destructor		
  ~SquareLossCpp() {}
			
  void dloss_dyhat_dense (
			  DenseMatrix                  &_target, 
			  thrust::device_vector<float> &_output,
			  float                        *_output_ptr,
			  thrust::device_vector<float> &_dlossdoutput
			  ) {
	
    thrust::transform(
		      _output.begin(), 
		      _output.begin() + _target.batch_size*_target.dim, 
		      _target.X.begin(), 
		      _dlossdoutput.begin(), 
		      thrust::minus<float>()
		      );
		
  }

  void dloss_dyhat_sparse (
			   COOVector                    &_target,
			   thrust::device_vector<float> &_output,
			   float                        *_output_ptr,
			   thrust::device_vector<float> &_dlossdoutput
			   ) {
    
    thrust::copy(
		 _output.begin(),
		 _output.begin() + _target.batch_size*_target.dim, 
		 _dlossdoutput.begin()
		 );

    const float alpha = -1.f;
    
    cusparseSaxpyi(
		   this->NeuralNet_->get_sparse_handle(),//handle
		   _target.num_non_zero,//nnz
		   &alpha,//alpha
		   _target.X_data_ptr,//xVal
		   _target.X_indices_ptr,//xInd
		   thrust::raw_pointer_cast(_dlossdoutput.data()),//y
		   CUSPARSE_INDEX_BASE_ZERO//idxBase
		   );
     
  };
	
};
