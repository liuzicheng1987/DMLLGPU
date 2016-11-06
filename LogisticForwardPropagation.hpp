namespace ActivationFunction {

struct LogisticForwardPropagation0 {

  const float *w;

  LogisticForwardPropagation0(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<0>(t) = Logistic(thrust::get<0>(t) + w[0]);

  }

};

struct LogisticForwardPropagation1 {

  const float *w;

  LogisticForwardPropagation1(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<1>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t) + w[1]);

  }

};

struct LogisticForwardPropagation2 {

  const float *w;

  LogisticForwardPropagation2(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<2>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t) + w[2]);

  }

};

struct LogisticForwardPropagation3 {

  const float *w;

  LogisticForwardPropagation3(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<3>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t) + w[3]);

  }

};

struct LogisticForwardPropagation4 {

  const float *w;

  LogisticForwardPropagation4(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<4>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t) + w[4]);

  }

};

struct LogisticForwardPropagation5 {

  const float *w;

  LogisticForwardPropagation5(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<5>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t) + w[5]);

  }

};

struct LogisticForwardPropagation6 {

  const float *w;

  LogisticForwardPropagation6(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<6>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t) + w[6]);

  }

};

struct LogisticForwardPropagation7 {

  const float *w;

  LogisticForwardPropagation7(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<7>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t) + w[7]);

  }

};

struct LogisticForwardPropagation8 {

  const float *w;

  LogisticForwardPropagation8(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<8>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t)*w[7] + thrust::get<8>(t) + w[8]);

  }

};

struct LogisticForwardPropagation9 {

  const float *w;

  LogisticForwardPropagation9(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    thrust::get<9>(t) = Logistic(thrust::get<0>(t)*w[0] + thrust::get<1>(t)*w[1] + thrust::get<2>(t)*w[2] + thrust::get<3>(t)*w[3] + thrust::get<4>(t)*w[4] + thrust::get<5>(t)*w[5] + thrust::get<6>(t)*w[6] + thrust::get<7>(t)*w[7] + thrust::get<8>(t)*w[8] + thrust::get<9>(t) + w[9]);

  }

};



void Logistic_forward_propagation(const float *W, NeuralNetworkNodeGPUCpp **hidden_nodes_fed_into_me_ptr, std::size_t hidden_nodes_fed_into_me_ptrSize, thrust::device_vector<float> &output) {

  switch(hidden_nodes_fed_into_me_ptrSize) {

      case 0:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end())),
		     LogisticForwardPropagation0(W));
    break;

  case 1:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), output.end())),
		     LogisticForwardPropagation1(W));
    break;

  case 2:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), output.end())),
		     LogisticForwardPropagation2(W));
    break;

  case 3:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), output.end())),
		     LogisticForwardPropagation3(W));
    break;

  case 4:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), output.end())),
		     LogisticForwardPropagation4(W));
    break;

  case 5:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), output.end())),
		     LogisticForwardPropagation5(W));
    break;

  case 6:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), output.end())),
		     LogisticForwardPropagation6(W));
    break;

  case 7:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), hidden_nodes_fed_into_me_ptr[6]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), hidden_nodes_fed_into_me_ptr[6]->get_output().end(), output.end())),
		     LogisticForwardPropagation7(W));
    break;

  case 8:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), hidden_nodes_fed_into_me_ptr[6]->get_output().begin(), hidden_nodes_fed_into_me_ptr[7]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), hidden_nodes_fed_into_me_ptr[6]->get_output().end(), hidden_nodes_fed_into_me_ptr[7]->get_output().end(), output.end())),
		     LogisticForwardPropagation8(W));
    break;

  case 9:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), hidden_nodes_fed_into_me_ptr[6]->get_output().begin(), hidden_nodes_fed_into_me_ptr[7]->get_output().begin(), hidden_nodes_fed_into_me_ptr[8]->get_output().begin(), output.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), hidden_nodes_fed_into_me_ptr[6]->get_output().end(), hidden_nodes_fed_into_me_ptr[7]->get_output().end(), hidden_nodes_fed_into_me_ptr[8]->get_output().end(), output.end())),
		     LogisticForwardPropagation9(W));
    break;


  }
}
}