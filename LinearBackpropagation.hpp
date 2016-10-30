namespace ActivationFunction {

struct LinearBackpropagation1 {

  const float *w;

  LinearBackpropagation1(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];


  }

};

struct LinearBackpropagation2 {

  const float *w;

  LinearBackpropagation2(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];


  }

};

struct LinearBackpropagation3 {

  const float *w;

  LinearBackpropagation3(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];


  }

};

struct LinearBackpropagation4 {

  const float *w;

  LinearBackpropagation4(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];


  }

};

struct LinearBackpropagation5 {

  const float *w;

  LinearBackpropagation5(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];
    thrust::get<6>(t) += d*w[4];


  }

};

struct LinearBackpropagation6 {

  const float *w;

  LinearBackpropagation6(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];
    thrust::get<6>(t) += d*w[4];
    thrust::get<7>(t) += d*w[5];


  }

};

struct LinearBackpropagation7 {

  const float *w;

  LinearBackpropagation7(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];
    thrust::get<6>(t) += d*w[4];
    thrust::get<7>(t) += d*w[5];
    thrust::get<8>(t) += d*w[6];


  }

};

struct LinearBackpropagation8 {

  const float *w;

  LinearBackpropagation8(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLineardw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];
    thrust::get<6>(t) += d*w[4];
    thrust::get<7>(t) += d*w[5];
    thrust::get<8>(t) += d*w[6];
    thrust::get<9>(t) += d*w[7];


  }

};



void Linear_backpropagation(const float *W, NeuralNetworkNodeGPUCpp **HiddenNodesFedIntoMePtr, std::size_t HiddenNodesFedIntoMePtrSize, thrust::device_vector<float> &output, thrust::device_vector<float> &delta) {

  switch(HiddenNodesFedIntoMePtrSize) {

  case 0:
    break;

  case 1:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end())),
		     LinearBackpropagation1(W));
    break;

  case 2:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end())),
		     LinearBackpropagation2(W));
    break;

  case 3:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin(), HiddenNodesFedIntoMePtr[2]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end(), HiddenNodesFedIntoMePtr[2]->get_delta().end())),
		     LinearBackpropagation3(W));
    break;

  case 4:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin(), HiddenNodesFedIntoMePtr[2]->get_delta().begin(), HiddenNodesFedIntoMePtr[3]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end(), HiddenNodesFedIntoMePtr[2]->get_delta().end(), HiddenNodesFedIntoMePtr[3]->get_delta().end())),
		     LinearBackpropagation4(W));
    break;

  case 5:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin(), HiddenNodesFedIntoMePtr[2]->get_delta().begin(), HiddenNodesFedIntoMePtr[3]->get_delta().begin(), HiddenNodesFedIntoMePtr[4]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end(), HiddenNodesFedIntoMePtr[2]->get_delta().end(), HiddenNodesFedIntoMePtr[3]->get_delta().end(), HiddenNodesFedIntoMePtr[4]->get_delta().end())),
		     LinearBackpropagation5(W));
    break;

  case 6:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin(), HiddenNodesFedIntoMePtr[2]->get_delta().begin(), HiddenNodesFedIntoMePtr[3]->get_delta().begin(), HiddenNodesFedIntoMePtr[4]->get_delta().begin(), HiddenNodesFedIntoMePtr[5]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end(), HiddenNodesFedIntoMePtr[2]->get_delta().end(), HiddenNodesFedIntoMePtr[3]->get_delta().end(), HiddenNodesFedIntoMePtr[4]->get_delta().end(), HiddenNodesFedIntoMePtr[5]->get_delta().end())),
		     LinearBackpropagation6(W));
    break;

  case 7:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin(), HiddenNodesFedIntoMePtr[2]->get_delta().begin(), HiddenNodesFedIntoMePtr[3]->get_delta().begin(), HiddenNodesFedIntoMePtr[4]->get_delta().begin(), HiddenNodesFedIntoMePtr[5]->get_delta().begin(), HiddenNodesFedIntoMePtr[6]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end(), HiddenNodesFedIntoMePtr[2]->get_delta().end(), HiddenNodesFedIntoMePtr[3]->get_delta().end(), HiddenNodesFedIntoMePtr[4]->get_delta().end(), HiddenNodesFedIntoMePtr[5]->get_delta().end(), HiddenNodesFedIntoMePtr[6]->get_delta().end())),
		     LinearBackpropagation7(W));
    break;

  case 8:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), HiddenNodesFedIntoMePtr[0]->get_delta().begin(), HiddenNodesFedIntoMePtr[1]->get_delta().begin(), HiddenNodesFedIntoMePtr[2]->get_delta().begin(), HiddenNodesFedIntoMePtr[3]->get_delta().begin(), HiddenNodesFedIntoMePtr[4]->get_delta().begin(), HiddenNodesFedIntoMePtr[5]->get_delta().begin(), HiddenNodesFedIntoMePtr[6]->get_delta().begin(), HiddenNodesFedIntoMePtr[7]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), HiddenNodesFedIntoMePtr[0]->get_delta().end(), HiddenNodesFedIntoMePtr[1]->get_delta().end(), HiddenNodesFedIntoMePtr[2]->get_delta().end(), HiddenNodesFedIntoMePtr[3]->get_delta().end(), HiddenNodesFedIntoMePtr[4]->get_delta().end(), HiddenNodesFedIntoMePtr[5]->get_delta().end(), HiddenNodesFedIntoMePtr[6]->get_delta().end(), HiddenNodesFedIntoMePtr[7]->get_delta().end())),
		     LinearBackpropagation8(W));
    break;



  }
}
}