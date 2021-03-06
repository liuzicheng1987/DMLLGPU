import os
os.chdir("/home/patrick/Dropbox/Programs/NeuralNetworkGPU/")


ActivationFunctionsList = ["Linear", "Logistic"]

for activation in ActivationFunctionsList:
    template = """struct $ACTIVATIONForwardPropagation$ITER {

  const float *w;

  $ACTIVATIONForwardPropagation$ITER(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {
      
    thrust::get<$ITER>(t) = $ACTIVATION($LINEARTRANSFORMATIONthrust::get<$ITER>(t) + w[$ITER]);
           
  }

};

"""
    
    EquationTemplate = "thrust::get<$ITER>(t)*w[$ITER] + "
    
    output = """namespace ActivationFunction {\n\n"""
    
    for i in range(10):
        equation = ""
        for j in range(i):
            equation += EquationTemplate.replace("$ITER", str(j))
        output += template.replace("$ITER", str(i)).replace("$LINEARTRANSFORMATION", equation)

    NonLinearTransformation = """

void $ACTIVATION_forward_propagation(const float *W, NeuralNetworkNodeGPUCpp **HiddenNodesFedIntoMePtr, std::size_t HiddenNodesFedIntoMePtrSize, thrust::device_vector<float> &output) {

  switch(HiddenNodesFedIntoMePtrSize) {
    
    $CASES
  }
}
"""
    
    cases = ""
    case = """  case $ITER:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple($HIDDENBEGINoutput.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple($HIDDENENDoutput.end())),
		     $ACTIVATIONForwardPropagation$ITER(W));
    break;

"""
    
    begin = "HiddenNodesFedIntoMePtr[$ITER]->get_output().begin(), "
    end = "HiddenNodesFedIntoMePtr[$ITER]->get_output().end(), "
        
    for i in range(10):
        hiddenbegin = "" 
        hiddenend = ""
        for j in range(i):
            hiddenbegin += begin.replace("$ITER", str(j))
            hiddenend += end.replace("$ITER", str(j))
        cases += case.replace("$ITER", str(i)).replace("$HIDDENBEGIN", hiddenbegin).replace("$HIDDENEND", hiddenend)
    
    output += NonLinearTransformation.replace("$CASES", cases)
    output += "}"
    output = output.replace("$ACTIVATION", activation)
    
    open(activation + "ForwardPropagation.hpp", "wb").write(output)
