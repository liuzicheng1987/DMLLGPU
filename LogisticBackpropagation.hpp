namespace ActivationFunctions {

struct LogisticBackpropagation1 {

  const float *w;

  LogisticBackpropagation1(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];


  }

};

struct LogisticBackpropagation2 {

  const float *w;

  LogisticBackpropagation2(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];


  }

};

struct LogisticBackpropagation3 {

  const float *w;

  LogisticBackpropagation3(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];


  }

};

struct LogisticBackpropagation4 {

  const float *w;

  LogisticBackpropagation4(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];


  }

};

struct LogisticBackpropagation5 {

  const float *w;

  LogisticBackpropagation5(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];
    thrust::get<6>(t) += d*w[4];


  }

};

struct LogisticBackpropagation6 {

  const float *w;

  LogisticBackpropagation6(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
    thrust::get<1>(t) = d;//Transform delta, which is needed to calculate dLdw

    thrust::get<2>(t) += d*w[0];
    thrust::get<3>(t) += d*w[1];
    thrust::get<4>(t) += d*w[2];
    thrust::get<5>(t) += d*w[3];
    thrust::get<6>(t) += d*w[4];
    thrust::get<7>(t) += d*w[5];


  }

};

struct LogisticBackpropagation7 {

  const float *w;

  LogisticBackpropagation7(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
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

struct LogisticBackpropagation8 {

  const float *w;

  LogisticBackpropagation8(const float* _w) : w(_w) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float d = thrust::get<1>(t)*dLogisticdw(thrust::get<0>(t));
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



void Logistic_backpropagation(const float *W, NeuralNetworkNodeGPUCpp **hidden_nodes_fed_into_me_ptr, std::size_t hidden_nodes_fed_into_me_ptrSize, thrust::device_vector<float> &output, thrust::device_vector<float> &delta) {

  switch(hidden_nodes_fed_into_me_ptrSize) {

  case 0:
    break;

  case 1:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end())),
		     LogisticBackpropagation1(W));
    break;

  case 2:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end())),
		     LogisticBackpropagation2(W));
    break;

  case 3:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[2]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end(), hidden_nodes_fed_into_me_ptr[2]->get_delta().end())),
		     LogisticBackpropagation3(W));
    break;

  case 4:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[2]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[3]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end(), hidden_nodes_fed_into_me_ptr[2]->get_delta().end(), hidden_nodes_fed_into_me_ptr[3]->get_delta().end())),
		     LogisticBackpropagation4(W));
    break;

  case 5:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[2]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[3]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[4]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end(), hidden_nodes_fed_into_me_ptr[2]->get_delta().end(), hidden_nodes_fed_into_me_ptr[3]->get_delta().end(), hidden_nodes_fed_into_me_ptr[4]->get_delta().end())),
		     LogisticBackpropagation5(W));
    break;

  case 6:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[2]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[3]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[4]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[5]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end(), hidden_nodes_fed_into_me_ptr[2]->get_delta().end(), hidden_nodes_fed_into_me_ptr[3]->get_delta().end(), hidden_nodes_fed_into_me_ptr[4]->get_delta().end(), hidden_nodes_fed_into_me_ptr[5]->get_delta().end())),
		     LogisticBackpropagation6(W));
    break;

  case 7:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[2]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[3]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[4]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[5]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[6]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end(), hidden_nodes_fed_into_me_ptr[2]->get_delta().end(), hidden_nodes_fed_into_me_ptr[3]->get_delta().end(), hidden_nodes_fed_into_me_ptr[4]->get_delta().end(), hidden_nodes_fed_into_me_ptr[5]->get_delta().end(), hidden_nodes_fed_into_me_ptr[6]->get_delta().end())),
		     LogisticBackpropagation7(W));
    break;

  case 8:
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(output.begin(), delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[1]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[2]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[3]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[4]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[5]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[6]->get_delta().begin(), hidden_nodes_fed_into_me_ptr[7]->get_delta().begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(output.end(), delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_delta().end(), hidden_nodes_fed_into_me_ptr[1]->get_delta().end(), hidden_nodes_fed_into_me_ptr[2]->get_delta().end(), hidden_nodes_fed_into_me_ptr[3]->get_delta().end(), hidden_nodes_fed_into_me_ptr[4]->get_delta().end(), hidden_nodes_fed_into_me_ptr[5]->get_delta().end(), hidden_nodes_fed_into_me_ptr[6]->get_delta().end(), hidden_nodes_fed_into_me_ptr[7]->get_delta().end())),
		     LogisticBackpropagation8(W));
    break;



  }
}
}
