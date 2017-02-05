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
//AdamFunctor

struct AdamFunctor
{

  const std::int32_t epoch_num;

  const float learning_rate;

  const float decay_mom1;

  const float decay_mom2;

  const float offset;

  AdamFunctor(
      const std::int32_t _epoch_num,
      const float _learning_rate,
      const float _decay_mom1,
      const float _decay_mom2,
      const float _offset) : epoch_num(_epoch_num),
                             learning_rate(_learning_rate),
                             decay_mom1(_decay_mom1),
                             decay_mom2(_decay_mom2),
                             offset(_offset)

  {
  }

  template <typename Tuple>
  __device__ void operator()(Tuple t)
  {

    //t<0> dldw
    //t<1> v_t sum_dldw_squared
    //t<2> W
    //t<3> m_t

    // m_t = \beta_1 * m_{1-1} + (1 - \beta_1) * \frac{\partial l}{\partial w}
    thrust::get<3>(t) = decay_mom1 * thrust::get<3>(t) + (1.f - decay_mom1) * thrust::get<0>(t);

    // v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * \frac{\partial l}{\partial w} ** 2
    thrust::get<1>(t) = decay_mom2 * thrust::get<1>(t) + (1.f - decay_mom2) * thrust::get<0>(t) * thrust::get<0>(t);

    // \Theta_t = \Theta_(t-1) - \alpha \frac{m_t * (1 - \beta_1^t) ** -1}{\sqrt{v_t * (1 - \beta_2^t) ** -1} + \epsilon}
    thrust::get<2>(t) -= learning_rate * (thrust::get<3>(t) / (1 - powf(decay_mom1, epoch_num))) /
                         (sqrt((thrust::get<1>(t) / (1 - powf(decay_mom2, epoch_num)))) + offset);
  }
};

//---------------------------------------------------------------------------
//NadamFunctor

struct NadamFunctor
{

  const std::int32_t epoch_num;

  const float learning_rate;

  const float beta_1;

  const float beta_2;

  const float momentum_cache_t;

  const float momentum_cache_t_1;

  const float m_schedule_new;

  const float m_schedule_next;

  const float offset;

  NadamFunctor(
      const std::int32_t _epoch_num,
      const float _learning_rate,
      const float _beta_1,
      const float _beta_2,
      const float _momentum_cache_t,
      const float _momentum_cache_t_1,
      const float _m_schedule_new,
      const float _m_schedule_next,
      const float _offset) : epoch_num(_epoch_num),
                             learning_rate(_learning_rate),
                             beta_1(_beta_1),
                             beta_2(_beta_2),
                             momentum_cache_t(_momentum_cache_t),
                             momentum_cache_t_1(_momentum_cache_t_1),
                             m_schedule_new(_m_schedule_new),
                             m_schedule_next(_m_schedule_next),
                             offset(_offset)

  {
  }

  template <typename Tuple>
  __device__ void operator()(Tuple t)
  {

    //t<0> dldw
    //t<1> ms / est_mom1_b
    //t<2> vs / est_mom2_b
    //t<3> W

    // m_t = \beta_1 * m_{1-1} + (1 - \beta_1) * \frac{\partial l}{\partial w}
    thrust::get<1>(t) = beta_1 * thrust::get<1>(t) + (1.f - beta_1) * thrust::get<0>(t);

    // v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * \frac{\partial l}{\partial w} ** 2
    thrust::get<2>(t) = beta_2 * thrust::get<2>(t) + (1.f - beta_2) * thrust::get<0>(t) * thrust::get<0>(t);

    // \Theta_t = \Theta_(t-1) - ...
    thrust::get<3>(t) -= (learning_rate / (sqrt(thrust::get<2>(t) * (1 - powf(beta_2, (epoch_num - 1)))) + offset)) *
                         (1.f - momentum_cache_t) * thrust::get<0>(t) / (1.f - m_schedule_new) +
                          momentum_cache_t_1 * thrust::get<1>(t) / (1.f - m_schedule_next);
  }
};

//---------------------------------------------------------------------------
}
