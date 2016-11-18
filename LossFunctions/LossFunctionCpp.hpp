class LossFunctionCpp {

protected:
  
  NeuralNetworkGPUCpp *NeuralNet_;
	
public:	
	
  //Constructor
  LossFunctionCpp() {}

  //Virtual destructor		
  virtual ~LossFunctionCpp() {}
			
  virtual void dloss_dyhat_dense (
				  DenseMatrix                  &_target, 
				  thrust::device_vector<float> &_output,
				  float                        *_output_ptr,
				  thrust::device_vector<float> &_dlossdoutput
				  ) {};
    						
  virtual void dloss_dyhat_sparse (
				   COOVector                    &_target,
				   thrust::device_vector<float> &_output,
				   float                        *_output_ptr,
				   thrust::device_vector<float> &_dlossdoutput			  
				   ) {};

  void set_neural_net(NeuralNetworkGPUCpp *_NeuralNet) {

    this->NeuralNet_ = _NeuralNet;
    
  }
  
 	
};
