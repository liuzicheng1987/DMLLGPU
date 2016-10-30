struct NonLinearTransformation0 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation0(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<0>(t) = ActivationFunction(thrust::get<0>(t) + w[0]);

  }

};

struct NonLinearTransformation1 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation1(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<1>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t) + w[1]);

  }

};

struct NonLinearTransformation2 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation2(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<2>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t) + w[2]);

  }

};

struct NonLinearTransformation3 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation3(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<3>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t) + w[3]);

  }

};

struct NonLinearTransformation4 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation4(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<4>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t) + w[4]);

  }

};

struct NonLinearTransformation5 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation5(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<5>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t) + w[5]);

  }

};

struct NonLinearTransformation6 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation6(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<6>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t) + w[6]);

  }

};

struct NonLinearTransformation7 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation7(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<7>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t) + w[7]);

  }

};

struct NonLinearTransformation8 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation8(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<8>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t)*w[7] + thrust::get<8>(t) + w[8]);

  }

};

struct NonLinearTransformation9 {

  const float *w;
  ActivationFunctionGPUCpp *activation;

  NonLinearTransformation9(const float* _w, ActivationFunctionGPUCpp *_activation) : w(_w), activation(_activation) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<9>(t) = ActivationFunction(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t)*w[7] + thrust::get<8>(t)*w[8] + thrust::get<9>(t) + w[9]);

  }

};



void NonLinearTransformation(const float *W, std::vector<NeuralNetworkNodeGPUCpp*> &HiddenNodesFedIntoMePtr, thrust::device_vector<float> &output, ActivationFunctionGPUCpp *activation) {

  switch(HiddenNodesFedIntoMePtr.size()) {

      case 0:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end())),
		     NonLinearTransformation0(W, activation));
    break;

  case 1:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), output.end())),
		     NonLinearTransformation1(W, activation));
    break;

  case 2:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), output.end())),
		     NonLinearTransformation2(W, activation));
    break;

  case 3:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), output.end())),
		     NonLinearTransformation3(W, activation));
    break;

  case 4:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), output.end())),
		     NonLinearTransformation4(W, activation));
    break;

  case 5:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), output.end())),
		     NonLinearTransformation5(W, activation));
    break;

  case 6:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), output.end())),
		     NonLinearTransformation6(W, activation));
    break;

  case 7:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), HiddenNodesFedIntoMePtr[6]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), HiddenNodesFedIntoMePtr[6]->get_output().end(), output.end())),
		     NonLinearTransformation7(W, activation));
    break;

  case 8:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), HiddenNodesFedIntoMePtr[6]->get_output().begin(), HiddenNodesFedIntoMePtr[7]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), HiddenNodesFedIntoMePtr[6]->get_output().end(), HiddenNodesFedIntoMePtr[7]->get_output().end(), output.end())),
		     NonLinearTransformation8(W, activation));
    break;

  case 9:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), HiddenNodesFedIntoMePtr[6]->get_output().begin(), HiddenNodesFedIntoMePtr[7]->get_output().begin(), HiddenNodesFedIntoMePtr[8]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), HiddenNodesFedIntoMePtr[6]->get_output().end(), HiddenNodesFedIntoMePtr[7]->get_output().end(), HiddenNodesFedIntoMePtr[8]->get_output().end(), output.end())),
		     NonLinearTransformation9(W, activation));
    break;


  }
}
