namespace ActivationFunction {

struct CustomTuple0 {


    float bias;
};

struct TransformationCustomTuple0 {

  template <typename Tuple>
  __device__ const CustomTuple0 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple0 result;


    result.bias = d;

    return result;
  }

};

struct SumCustomTuple0 {

  __device__ const CustomTuple0 operator()(const CustomTuple0& lhs, const CustomTuple0& rhs) {

    CustomTuple0 result;


    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple1 {

    float val0;

    float bias;
};

struct TransformationCustomTuple1 {

  template <typename Tuple>
  __device__ const CustomTuple1 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple1 result;

    result.val0 = d*thrust::get<1>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple1 {

  __device__ const CustomTuple1 operator()(const CustomTuple1& lhs, const CustomTuple1& rhs) {

    CustomTuple1 result;

    result.val0 = lhs.val0 + rhs.val0;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple2 {

    float val0;
    float val1;

    float bias;
};

struct TransformationCustomTuple2 {

  template <typename Tuple>
  __device__ const CustomTuple2 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple2 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple2 {

  __device__ const CustomTuple2 operator()(const CustomTuple2& lhs, const CustomTuple2& rhs) {

    CustomTuple2 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple3 {

    float val0;
    float val1;
    float val2;

    float bias;
};

struct TransformationCustomTuple3 {

  template <typename Tuple>
  __device__ const CustomTuple3 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple3 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple3 {

  __device__ const CustomTuple3 operator()(const CustomTuple3& lhs, const CustomTuple3& rhs) {

    CustomTuple3 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple4 {

    float val0;
    float val1;
    float val2;
    float val3;

    float bias;
};

struct TransformationCustomTuple4 {

  template <typename Tuple>
  __device__ const CustomTuple4 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple4 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);
    result.val3 = d*thrust::get<4>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple4 {

  __device__ const CustomTuple4 operator()(const CustomTuple4& lhs, const CustomTuple4& rhs) {

    CustomTuple4 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;
    result.val3 = lhs.val3 + rhs.val3;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple5 {

    float val0;
    float val1;
    float val2;
    float val3;
    float val4;

    float bias;
};

struct TransformationCustomTuple5 {

  template <typename Tuple>
  __device__ const CustomTuple5 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple5 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);
    result.val3 = d*thrust::get<4>(t);
    result.val4 = d*thrust::get<5>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple5 {

  __device__ const CustomTuple5 operator()(const CustomTuple5& lhs, const CustomTuple5& rhs) {

    CustomTuple5 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;
    result.val3 = lhs.val3 + rhs.val3;
    result.val4 = lhs.val4 + rhs.val4;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple6 {

    float val0;
    float val1;
    float val2;
    float val3;
    float val4;
    float val5;

    float bias;
};

struct TransformationCustomTuple6 {

  template <typename Tuple>
  __device__ const CustomTuple6 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple6 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);
    result.val3 = d*thrust::get<4>(t);
    result.val4 = d*thrust::get<5>(t);
    result.val5 = d*thrust::get<6>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple6 {

  __device__ const CustomTuple6 operator()(const CustomTuple6& lhs, const CustomTuple6& rhs) {

    CustomTuple6 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;
    result.val3 = lhs.val3 + rhs.val3;
    result.val4 = lhs.val4 + rhs.val4;
    result.val5 = lhs.val5 + rhs.val5;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple7 {

    float val0;
    float val1;
    float val2;
    float val3;
    float val4;
    float val5;
    float val6;

    float bias;
};

struct TransformationCustomTuple7 {

  template <typename Tuple>
  __device__ const CustomTuple7 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple7 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);
    result.val3 = d*thrust::get<4>(t);
    result.val4 = d*thrust::get<5>(t);
    result.val5 = d*thrust::get<6>(t);
    result.val6 = d*thrust::get<7>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple7 {

  __device__ const CustomTuple7 operator()(const CustomTuple7& lhs, const CustomTuple7& rhs) {

    CustomTuple7 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;
    result.val3 = lhs.val3 + rhs.val3;
    result.val4 = lhs.val4 + rhs.val4;
    result.val5 = lhs.val5 + rhs.val5;
    result.val6 = lhs.val6 + rhs.val6;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple8 {

    float val0;
    float val1;
    float val2;
    float val3;
    float val4;
    float val5;
    float val6;
    float val7;

    float bias;
};

struct TransformationCustomTuple8 {

  template <typename Tuple>
  __device__ const CustomTuple8 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple8 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);
    result.val3 = d*thrust::get<4>(t);
    result.val4 = d*thrust::get<5>(t);
    result.val5 = d*thrust::get<6>(t);
    result.val6 = d*thrust::get<7>(t);
    result.val7 = d*thrust::get<8>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple8 {

  __device__ const CustomTuple8 operator()(const CustomTuple8& lhs, const CustomTuple8& rhs) {

    CustomTuple8 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;
    result.val3 = lhs.val3 + rhs.val3;
    result.val4 = lhs.val4 + rhs.val4;
    result.val5 = lhs.val5 + rhs.val5;
    result.val6 = lhs.val6 + rhs.val6;
    result.val7 = lhs.val7 + rhs.val7;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};

struct CustomTuple9 {

    float val0;
    float val1;
    float val2;
    float val3;
    float val4;
    float val5;
    float val6;
    float val7;
    float val8;

    float bias;
};

struct TransformationCustomTuple9 {

  template <typename Tuple>
  __device__ const CustomTuple9 operator()(const Tuple &t) {

    float d = thrust::get<0>(t);//d is the delta value
    CustomTuple9 result;

    result.val0 = d*thrust::get<1>(t);
    result.val1 = d*thrust::get<2>(t);
    result.val2 = d*thrust::get<3>(t);
    result.val3 = d*thrust::get<4>(t);
    result.val4 = d*thrust::get<5>(t);
    result.val5 = d*thrust::get<6>(t);
    result.val6 = d*thrust::get<7>(t);
    result.val7 = d*thrust::get<8>(t);
    result.val8 = d*thrust::get<9>(t);

    result.bias = d;

    return result;
  }

};

struct SumCustomTuple9 {

  __device__ const CustomTuple9 operator()(const CustomTuple9& lhs, const CustomTuple9& rhs) {

    CustomTuple9 result;

    result.val0 = lhs.val0 + rhs.val0;
    result.val1 = lhs.val1 + rhs.val1;
    result.val2 = lhs.val2 + rhs.val2;
    result.val3 = lhs.val3 + rhs.val3;
    result.val4 = lhs.val4 + rhs.val4;
    result.val5 = lhs.val5 + rhs.val5;
    result.val6 = lhs.val6 + rhs.val6;
    result.val7 = lhs.val7 + rhs.val7;
    result.val8 = lhs.val8 + rhs.val8;

    result.bias = lhs.bias + rhs.bias;

    return result;
  }

};



void calc_dev_hidden(float *dLdw, NeuralNetworkNodeGPUCpp **hidden_nodes_fed_into_me_ptr, std::size_t hidden_nodes_fed_into_me_ptrSize, thrust::device_vector<float> &delta) {

  switch(hidden_nodes_fed_into_me_ptrSize) {

  case 0:
    CustomTuple0 res0;
    float dldw_host0[1];


    res0.bias = 0.0f;

    res0 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end())),
                                                    TransformationCustomTuple0(),
                                                    res0,
                                                    SumCustomTuple0()
                                                    );

    dldw_host0[0] = res0.bias;

    cudaMemcpy(dLdw, dldw_host0, 1*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 1:
    CustomTuple1 res1;
    float dldw_host1[2];

    res1.val0 = 0.0f;

    res1.bias = 0.0f;

    res1 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end())),
                                                    TransformationCustomTuple1(),
                                                    res1,
                                                    SumCustomTuple1()
                                                    );
    dldw_host1[0] = res1.val0;

    dldw_host1[1] = res1.bias;

    cudaMemcpy(dLdw, dldw_host1, 2*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 2:
    CustomTuple2 res2;
    float dldw_host2[3];

    res2.val0 = 0.0f;
    res2.val1 = 0.0f;

    res2.bias = 0.0f;

    res2 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end())),
                                                    TransformationCustomTuple2(),
                                                    res2,
                                                    SumCustomTuple2()
                                                    );
    dldw_host2[0] = res2.val0;
    dldw_host2[1] = res2.val1;

    dldw_host2[2] = res2.bias;

    cudaMemcpy(dLdw, dldw_host2, 3*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 3:
    CustomTuple3 res3;
    float dldw_host3[4];

    res3.val0 = 0.0f;
    res3.val1 = 0.0f;
    res3.val2 = 0.0f;

    res3.bias = 0.0f;

    res3 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end())),
                                                    TransformationCustomTuple3(),
                                                    res3,
                                                    SumCustomTuple3()
                                                    );
    dldw_host3[0] = res3.val0;
    dldw_host3[1] = res3.val1;
    dldw_host3[2] = res3.val2;

    dldw_host3[3] = res3.bias;

    cudaMemcpy(dLdw, dldw_host3, 4*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 4:
    CustomTuple4 res4;
    float dldw_host4[5];

    res4.val0 = 0.0f;
    res4.val1 = 0.0f;
    res4.val2 = 0.0f;
    res4.val3 = 0.0f;

    res4.bias = 0.0f;

    res4 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end())),
                                                    TransformationCustomTuple4(),
                                                    res4,
                                                    SumCustomTuple4()
                                                    );
    dldw_host4[0] = res4.val0;
    dldw_host4[1] = res4.val1;
    dldw_host4[2] = res4.val2;
    dldw_host4[3] = res4.val3;

    dldw_host4[4] = res4.bias;

    cudaMemcpy(dLdw, dldw_host4, 5*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 5:
    CustomTuple5 res5;
    float dldw_host5[6];

    res5.val0 = 0.0f;
    res5.val1 = 0.0f;
    res5.val2 = 0.0f;
    res5.val3 = 0.0f;
    res5.val4 = 0.0f;

    res5.bias = 0.0f;

    res5 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end())),
                                                    TransformationCustomTuple5(),
                                                    res5,
                                                    SumCustomTuple5()
                                                    );
    dldw_host5[0] = res5.val0;
    dldw_host5[1] = res5.val1;
    dldw_host5[2] = res5.val2;
    dldw_host5[3] = res5.val3;
    dldw_host5[4] = res5.val4;

    dldw_host5[5] = res5.bias;

    cudaMemcpy(dLdw, dldw_host5, 6*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 6:
    CustomTuple6 res6;
    float dldw_host6[7];

    res6.val0 = 0.0f;
    res6.val1 = 0.0f;
    res6.val2 = 0.0f;
    res6.val3 = 0.0f;
    res6.val4 = 0.0f;
    res6.val5 = 0.0f;

    res6.bias = 0.0f;

    res6 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end())),
                                                    TransformationCustomTuple6(),
                                                    res6,
                                                    SumCustomTuple6()
                                                    );
    dldw_host6[0] = res6.val0;
    dldw_host6[1] = res6.val1;
    dldw_host6[2] = res6.val2;
    dldw_host6[3] = res6.val3;
    dldw_host6[4] = res6.val4;
    dldw_host6[5] = res6.val5;

    dldw_host6[6] = res6.bias;

    cudaMemcpy(dLdw, dldw_host6, 7*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 7:
    CustomTuple7 res7;
    float dldw_host7[8];

    res7.val0 = 0.0f;
    res7.val1 = 0.0f;
    res7.val2 = 0.0f;
    res7.val3 = 0.0f;
    res7.val4 = 0.0f;
    res7.val5 = 0.0f;
    res7.val6 = 0.0f;

    res7.bias = 0.0f;

    res7 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), hidden_nodes_fed_into_me_ptr[6]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), hidden_nodes_fed_into_me_ptr[6]->get_output().end())),
                                                    TransformationCustomTuple7(),
                                                    res7,
                                                    SumCustomTuple7()
                                                    );
    dldw_host7[0] = res7.val0;
    dldw_host7[1] = res7.val1;
    dldw_host7[2] = res7.val2;
    dldw_host7[3] = res7.val3;
    dldw_host7[4] = res7.val4;
    dldw_host7[5] = res7.val5;
    dldw_host7[6] = res7.val6;

    dldw_host7[7] = res7.bias;

    cudaMemcpy(dLdw, dldw_host7, 8*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 8:
    CustomTuple8 res8;
    float dldw_host8[9];

    res8.val0 = 0.0f;
    res8.val1 = 0.0f;
    res8.val2 = 0.0f;
    res8.val3 = 0.0f;
    res8.val4 = 0.0f;
    res8.val5 = 0.0f;
    res8.val6 = 0.0f;
    res8.val7 = 0.0f;

    res8.bias = 0.0f;

    res8 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), hidden_nodes_fed_into_me_ptr[6]->get_output().begin(), hidden_nodes_fed_into_me_ptr[7]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), hidden_nodes_fed_into_me_ptr[6]->get_output().end(), hidden_nodes_fed_into_me_ptr[7]->get_output().end())),
                                                    TransformationCustomTuple8(),
                                                    res8,
                                                    SumCustomTuple8()
                                                    );
    dldw_host8[0] = res8.val0;
    dldw_host8[1] = res8.val1;
    dldw_host8[2] = res8.val2;
    dldw_host8[3] = res8.val3;
    dldw_host8[4] = res8.val4;
    dldw_host8[5] = res8.val5;
    dldw_host8[6] = res8.val6;
    dldw_host8[7] = res8.val7;

    dldw_host8[8] = res8.bias;

    cudaMemcpy(dLdw, dldw_host8, 9*sizeof(float), cudaMemcpyHostToDevice);
    break;

  case 9:
    CustomTuple9 res9;
    float dldw_host9[10];

    res9.val0 = 0.0f;
    res9.val1 = 0.0f;
    res9.val2 = 0.0f;
    res9.val3 = 0.0f;
    res9.val4 = 0.0f;
    res9.val5 = 0.0f;
    res9.val6 = 0.0f;
    res9.val7 = 0.0f;
    res9.val8 = 0.0f;

    res9.bias = 0.0f;

    res9 = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(delta.begin(), hidden_nodes_fed_into_me_ptr[0]->get_output().begin(), hidden_nodes_fed_into_me_ptr[1]->get_output().begin(), hidden_nodes_fed_into_me_ptr[2]->get_output().begin(), hidden_nodes_fed_into_me_ptr[3]->get_output().begin(), hidden_nodes_fed_into_me_ptr[4]->get_output().begin(), hidden_nodes_fed_into_me_ptr[5]->get_output().begin(), hidden_nodes_fed_into_me_ptr[6]->get_output().begin(), hidden_nodes_fed_into_me_ptr[7]->get_output().begin(), hidden_nodes_fed_into_me_ptr[8]->get_output().begin())),
                                                    thrust::make_zip_iterator(thrust::make_tuple(delta.end(), hidden_nodes_fed_into_me_ptr[0]->get_output().end(), hidden_nodes_fed_into_me_ptr[1]->get_output().end(), hidden_nodes_fed_into_me_ptr[2]->get_output().end(), hidden_nodes_fed_into_me_ptr[3]->get_output().end(), hidden_nodes_fed_into_me_ptr[4]->get_output().end(), hidden_nodes_fed_into_me_ptr[5]->get_output().end(), hidden_nodes_fed_into_me_ptr[6]->get_output().end(), hidden_nodes_fed_into_me_ptr[7]->get_output().end(), hidden_nodes_fed_into_me_ptr[8]->get_output().end())),
                                                    TransformationCustomTuple9(),
                                                    res9,
                                                    SumCustomTuple9()
                                                    );
    dldw_host9[0] = res9.val0;
    dldw_host9[1] = res9.val1;
    dldw_host9[2] = res9.val2;
    dldw_host9[3] = res9.val3;
    dldw_host9[4] = res9.val4;
    dldw_host9[5] = res9.val5;
    dldw_host9[6] = res9.val6;
    dldw_host9[7] = res9.val7;
    dldw_host9[8] = res9.val8;

    dldw_host9[9] = res9.bias;

    cudaMemcpy(dLdw, dldw_host9, 10*sizeof(float), cudaMemcpyHostToDevice);
    break;



  }
}
}