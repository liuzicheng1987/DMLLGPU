namespace ActivationFunction {

struct LinearForwardPropagation0 {

  const float *w;

  LinearForwardPropagation0(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<0>(t) = Linear(thrust::get<0>(t) + w[0]);

  }

};

struct LinearForwardPropagation1 {

  const float *w;

  LinearForwardPropagation1(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<1>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t) + w[1]);

  }

};

struct LinearForwardPropagation2 {

  const float *w;

  LinearForwardPropagation2(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<2>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t) + w[2]);

  }

};

struct LinearForwardPropagation3 {

  const float *w;

  LinearForwardPropagation3(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<3>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t) + w[3]);

  }

};

struct LinearForwardPropagation4 {

  const float *w;

  LinearForwardPropagation4(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<4>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t) + w[4]);

  }

};

struct LinearForwardPropagation5 {

  const float *w;

  LinearForwardPropagation5(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<5>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t) + w[5]);

  }

};

struct LinearForwardPropagation6 {

  const float *w;

  LinearForwardPropagation6(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<6>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t) + w[6]);

  }

};

struct LinearForwardPropagation7 {

  const float *w;

  LinearForwardPropagation7(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<7>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t) + w[7]);

  }

};

struct LinearForwardPropagation8 {

  const float *w;

  LinearForwardPropagation8(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<8>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t)*w[7] + thrust::get<8>(t) + w[8]);

  }

};

struct LinearForwardPropagation9 {

  const float *w;

  LinearForwardPropagation9(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<9>(t) = Linear(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t)*w[7] + thrust::get<8>(t)*w[8] + thrust::get<9>(t) + w[9]);

  }

};



void Linear_forward_propagation(const float *W, NeuralNetworkNodeGPUCpp **HiddenNodesFedIntoMePtr, std::size_t HiddenNodesFedIntoMePtrSize, thrust::device_vector<float> &output) {

  switch(HiddenNodesFedIntoMePtrSize) {

      case 0:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end())),
		     LinearForwardPropagation0(W));
    break;

  case 1:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), output.end())),
		     LinearForwardPropagation1(W));
    break;

  case 2:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), output.end())),
		     LinearForwardPropagation2(W));
    break;

  case 3:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), output.end())),
		     LinearForwardPropagation3(W));
    break;

  case 4:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), output.end())),
		     LinearForwardPropagation4(W));
    break;

  case 5:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), output.end())),
		     LinearForwardPropagation5(W));
    break;

  case 6:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), output.end())),
		     LinearForwardPropagation6(W));
    break;

  case 7:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), HiddenNodesFedIntoMePtr[6]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), HiddenNodesFedIntoMePtr[6]->get_output().end(), output.end())),
		     LinearForwardPropagation7(W));
    break;

  case 8:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), HiddenNodesFedIntoMePtr[6]->get_output().begin(), HiddenNodesFedIntoMePtr[7]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), HiddenNodesFedIntoMePtr[6]->get_output().end(), HiddenNodesFedIntoMePtr[7]->get_output().end(), output.end())),
		     LinearForwardPropagation8(W));
    break;

  case 9:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().begin(), HiddenNodesFedIntoMePtr[1]->get_output().begin(), HiddenNodesFedIntoMePtr[2]->get_output().begin(), HiddenNodesFedIntoMePtr[3]->get_output().begin(), HiddenNodesFedIntoMePtr[4]->get_output().begin(), HiddenNodesFedIntoMePtr[5]->get_output().begin(), HiddenNodesFedIntoMePtr[6]->get_output().begin(), HiddenNodesFedIntoMePtr[7]->get_output().begin(), HiddenNodesFedIntoMePtr[8]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(HiddenNodesFedIntoMePtr[0]->get_output().end(), HiddenNodesFedIntoMePtr[1]->get_output().end(), HiddenNodesFedIntoMePtr[2]->get_output().end(), HiddenNodesFedIntoMePtr[3]->get_output().end(), HiddenNodesFedIntoMePtr[4]->get_output().end(), HiddenNodesFedIntoMePtr[5]->get_output().end(), HiddenNodesFedIntoMePtr[6]->get_output().end(), HiddenNodesFedIntoMePtr[7]->get_output().end(), HiddenNodesFedIntoMePtr[8]->get_output().end(), output.end())),
		     LinearForwardPropagation9(W));
    break;


  }
}
}