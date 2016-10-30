namespace ActivationFunction {

__device__ float Linear(float x) {

  return x; 
            
}

__device__ float dLineardw(float y) {

  return 1.f; 
            
}

__device__ float Logistic(float x) {

  return 1.f/(1.f + expf((-1.f)*x)); 
            
}

__device__ float dLogisticdw(float y) {

  return y*(1.f - y);
            
}

}
/*
template<Activation activation> 
__device__ float dActivationFunctiondw(float y) {

     float dydw;     

     switch(activation) {
     
     case logistic: 
       dydw = y*(1.0f - y);
       break;
       
     case linear: 
       dydw = 1.0f; 
       break;

     }

    return dydw;
            
}

template __device__ float dActivationFunctiondw<logistic>(float y);
template __device__ float dActivationFunctiondw<linear>(float y);*/
