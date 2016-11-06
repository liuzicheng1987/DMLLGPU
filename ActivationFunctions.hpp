namespace ActivationFunctions {

//---------------------------------------------------------------------------

struct LinearForwardPropagation {
  
  const float *bias;
  const std::int32_t batch_size;

  LinearForwardPropagation(
			   const float       *_bias,
			   const std::int32_t _batch_size
			   ) : bias(_bias), batch_size(_batch_size) {}

  __device__
  float operator()(float x, int i) { 
    return x + bias[i/batch_size]; 
  }

};

  //Backpropagation functions is not needed


//---------------------------------------------------------------------------


struct LogisticForwardPropagation {

  const float *bias;
  const std::int32_t batch_size;

  LogisticForwardPropagation(
			   const float       *_bias,
			   const std::int32_t _batch_size
			   ) : bias(_bias), batch_size(_batch_size) {}

  __device__
  float operator()(float x, int i) { 
    float xplusbias = x + bias[i/batch_size];
    return 1.f/(1.f + expf((-1.f)*xplusbias)); 
  }

};

struct LogisticBackpropagation {

  LogisticBackpropagation() {}

  __device__
  float operator()(const float output, const float delta) { 
    return delta*output*(1.f - output);
  }

};

//---------------------------------------------------------------------------

}
