namespace ActivationFunctions
{

//---------------------------------------------------------------------------

struct LinearForwardPropagation
{

  const float *bias;
  const std::int32_t batch_size;

  LinearForwardPropagation(
      const float *_bias,
      const std::int32_t _batch_size) : bias(_bias), batch_size(_batch_size) {}

  __device__ float operator()(float x, int i)
  {
    return x + bias[i / batch_size];
  }
};

//Backpropagation functions is not needed

//---------------------------------------------------------------------------

struct LogisticForwardPropagation
{

  const float *bias;
  const std::int32_t batch_size;

  LogisticForwardPropagation(
      const float *_bias,
      const std::int32_t _batch_size) : bias(_bias),
                                        batch_size(_batch_size)
  {
  }

  __device__ float operator()(float x, int i)
  {
    float xplusbias = x + bias[i / batch_size];
    return 1.f / (1.f + expf((-1.f) * xplusbias));
  }
};

struct LogisticBackpropagation
{

  LogisticBackpropagation() {}

  __device__ float operator()(const float output, const float delta)
  {
    return delta * output * (1.f - output);
  }
};

//---------------------------------------------------------------------------

struct SoftmaxCalculateSum
{

  const float *bias;
  const std::int32_t batch_size;
  const std::int32_t num_vars;
  const std::int32_t num_states_per_var;
  const float *output;

  SoftmaxCalculateSum(
      const float *_bias,
      const std::int32_t _batch_size,
      const std::int32_t _num_vars,
      const std::int32_t _num_states_per_var,
      const float *_output) : bias(_bias),
                              batch_size(_batch_size),
                              num_vars(_num_vars),
                              num_states_per_var(_num_states_per_var),
                              output(_output)
  {
  }

  __device__ float operator()(const std::int32_t i)
  {

    //Total size of array is batch_size*num_vars*num_states_per_var
    //Leading dimension is batch_size, then num_states_per_var, then num_vars
    const std::int32_t sample_num = i % batch_size;
    const std::int32_t var_num = i / batch_size;

    float sum_states_per_var = 0.f;

    //Calculate sum of all states belonging to this variable and this sample
    for (
        std::int32_t state = 0;
        state < num_states_per_var;
        ++state)
      sum_states_per_var += expf(
          output[batch_size * var_num * num_states_per_var + sample_num + state * batch_size] +
          bias[var_num * num_states_per_var + state]);

    //Finally, return sum
    return sum_states_per_var;
  }
};

struct SoftmaxForwardPropagation
{

  const float *bias;
  const std::int32_t batch_size;
  const std::int32_t num_vars;
  const std::int32_t num_states_per_var;
  const float *sum_states_per_var;

  SoftmaxForwardPropagation(
      const float *_bias,
      const std::int32_t _batch_size,
      const std::int32_t _num_vars,
      const std::int32_t _num_states_per_var,
      const float *_sum_states_per_var) : bias(_bias),
                                          batch_size(_batch_size),
                                          num_vars(_num_vars),
                                          num_states_per_var(_num_states_per_var),
                                          sum_states_per_var(_sum_states_per_var)
  {
  }

  __device__ float operator()(
      const float x,
      const std::int32_t i)
  {

    //Total size of array is batch_size*num_vars*num_states_per_var
    //Leading dimension is batch_size, then num_states_per_var, then num_vars
    const std::int32_t sample_num = i % batch_size;
    const std::int32_t var_num = i / (batch_size * num_states_per_var);

    float xplusbias = x + bias[i / batch_size];

    //Return normalised output
    return expf(xplusbias) / sum_states_per_var[var_num * batch_size + sample_num];
  }
};

struct SoftmaxBackpropagation
{

  SoftmaxBackpropagation() {}

  __device__ float operator()(const float output, const float delta)
  {
    return delta * output * (1.f - output);
  }
};

//---------------------------------------------------------------------------
}
