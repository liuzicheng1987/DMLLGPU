namespace OptimiserFunctors
{

//---------------------------------------------------------------------------
//AdaGradFunctor

struct AdaGradFunctor
{

  const float learning_rate;

  AdaGradFunctor(
      const float _learning_rate) : learning_rate(_learning_rate)
  {
  }

  template <typename Tuple>
  __device__ void operator()(Tuple t)
  {

    //t<0> dldw
    //t<1> sum_dldw_squared
    //t<2> W
    thrust::get<1>(t) += thrust::get<0>(t) * thrust::get<0>(t);
    thrust::get<2>(t) -= ((thrust::get<1>(t) == 0.f) ? (0.f) : (
                                                                   learning_rate * thrust::get<0>(t) /
                                                                   sqrt(thrust::get<1>(t))));
  }
};

//---------------------------------------------------------------------------
//RMSPropFunctor

struct RMSPropFunctor
{

  const float learning_rate;

  const float gamma;

  RMSPropFunctor(
      const float _learning_rate,
      const float _gamma) : learning_rate(_learning_rate),
                            gamma(_gamma)

  {
  }

  template <typename Tuple>
  __device__ void operator()(Tuple t)
  {

    //t<0> dldw
    //t<1> sum_dldw_squared
    //t<2> W
    thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<0>(t) * (1.f - gamma) + thrust::get<1>(t) * gamma;
    thrust::get<2>(t) -= ((thrust::get<1>(t) == 0.f) ? (0.f) : (
                                                                   learning_rate * thrust::get<0>(t) /
                                                                   sqrt(thrust::get<1>(t))));
  }
};

//---------------------------------------------------------------------------
}
