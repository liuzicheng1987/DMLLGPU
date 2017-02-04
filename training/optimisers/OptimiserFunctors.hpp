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

  const float step;

  AdamFunctor(
      const std::int32_t _epoch_num,
      const float _learning_rate,
      const float _decay_mom1,
      const float _decay_mom2,
      const float _offset) : epoch_num(_epoch_num)
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
    thrust::get<3>(t) = decay_mom1 * thrust::get<3>(t) + (1.f - decay_mom1) * thrust::get<0>;

    // v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * \frac{\partial l}{\partial w} ** 2
    thrust::get<1>(t) = decay_mom2 * thrust::get<0>(t) + (1.f - decay_mom2) * thrust::get<0> ** 2;

    // \Theta_t = \Theta_(t-1) - \alpha \frac{m_t * (1 - \beta_1^t) ** -1}{\sqrt{v_t * (1 - \beta_2^t) ** -1} + \epsilon}
    thrust::get<2>(t) -= learning_rate * (thrust::get<3>(t) / (1 - decay_mom1 ** epoch_num))/
                                         (sqrt((thrust::get<1>(t) / (1 - decay_mom2 ** epoch_num) )) + offset);
  }
};

//---------------------------------------------------------------------------
//NadamFunctor

struct NadamFunctor
{

  const std::int32_t epoch_num;

  // initialise m_schedule to 1
  std::float m_schedule;

  std::float momentum_cache_t;

  std::float momentum_cache_t_1;

  std::float m_schedule_new;

  std::float m_schedule_next;

  const float learning_rate;

  const float beta_1;

  const float beta_2;

  const float schedule_decay;

  const float offset;

  const float step;

  NadamFunctor(
      const std::int32_t _epoch_num,
      const float _learning_rate,
      const float _beta_1,
      const float _beta_2,
      const float _schedule_decay,
      const float _offset) : epoch_num(_epoch_num)
                            learning_rate(_learning_rate),
                            beta_1(_beta_1),
                            beta_2(_beta_2),
                            schedule_decay(_schedule_decay),
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
    
    

    // grads = self.get_gradients(loss, params)
    // self.updates = [K.update_add(self.iterations, 1)]

    // t = self.iterations + 1

    // Due to the recommendations in [2], i.e. warming momentum schedule
    momentum_cache_t = beta_1 * (1.f - 0.5 * (t * self.schedule_decay) ** 0.96);
    momentum_cache_t_1 = beta_1 * (1.f - 0.5 * (0.96, (t + 1) * schedule_decay) ** 0.96);
    m_schedule_new = m_schedule * momentum_cache_t;
    m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1;

    // update/return m_schedule for next iteration
    m_schedule = m_schedule_new;

    // shapes = [K.get_variable_shape(p) for p in params]
    // ms = [K.zeros(shape) for shape in shapes]
    // vs = [K.zeros(shape) for shape in shapes]

    // self.weights = [self.iterations] + ms + vs

    // for p, g, m, v in zip(params, grads, ms, vs):
    //     # the following equations given in [1]
    //     g_prime = g / (1. - m_schedule_new)
    //     m_t = self.beta_1 * m + (1. - self.beta_1) * g
    //     m_t_prime = m_t / (1. - m_schedule_next)
    //     v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
    //     v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
    //     m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

    //     self.updates.append(K.update(m, m_t))
    //     self.updates.append(K.update(v, v_t))

    //     p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
    //     new_p = p_t


    // m_t = \beta_1 * m_{1-1} + (1 - \beta_1) * \frac{\partial l}{\partial w}
    thrust::get<1>(t) = beta_1 * thrust::get<1>(t) + (1.f - beta_1) * thrust::get<0>

    // v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * \frac{\partial l}{\partial w} ** 2
    thrust::get<2>(t) = beta_2 * thrust::get<2>(t) + (1.f - beta_2) * thrust::get<0> ** 2

    // \Theta_t = \Theta_(t-1) - ...
    thrust::get<3>(t) -= (learning_rate / (sqrt(thrust::get<2>(t) * (1 - beta_2 ** epoch_num) ** -1) + offset)) *
                          (1.f - momentum_cache_t) * thrust::get<0>(t) / (1.f - m_schedule_new) + 
                          momentum_cache_t_1 * thrust::get<1>(t) / (1.f - m_schedule_next);
  }
};

//---------------------------------------------------------------------------
}
