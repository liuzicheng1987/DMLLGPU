struct LogisticBackpropagation1 {

  const float *w;

  LogisticBackpropagation1(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {
    
    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    
    thrust::get<2>(t) = d*w[0];

  }

};

void Logistic_backpropagation(const float *W, std::vector<NeuralNetworkNodeGPUCpp*> &HiddenNodesFedIntoMePtr, thrust::device_vector<float> &output, thrust::device_vector<float> &delta) {

  switch(HiddenNodesFedIntoMePtr.size()) {

  case 0:
    break;

  case 1:
    thrust::for_each(thrust::make_zip_iterator(output.begin(), delta.begin(), thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_delta().begin())),
		     thrust::make_zip_iterator(output.end(), delta.end(), thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_delta().end())),
		     LogisticBackpropagation1(W));
    break;

  }
}
