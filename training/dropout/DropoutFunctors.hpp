namespace DropoutFunctors {

//---------------------------------------------------------------------------
//Generate random numbers


struct GenerateRandomNumbers {

  const std::int32_t numbers_per_kernel;
  std::uint32_t      seed;
  float             *random_numbers;

  GenerateRandomNumbers (
			 std::int32_t  _numbers_per_kernel,
			 std::uint32_t _seed,
			 float        *_random_numbers
			 ) : numbers_per_kernel(_numbers_per_kernel),
			     seed(_seed),
			     random_numbers(_random_numbers)
  {}

  __device__
  void operator()(const std::int32_t i) {

    thrust::default_random_engine rng(seed);
    
    thrust::uniform_real_distribution<float> dist(0.f, 1.f);
    
    rng.discard(i*numbers_per_kernel);

    for (std::int32_t j = 0; j < numbers_per_kernel; ++j)
      random_numbers[i*numbers_per_kernel + j] = dist(rng);
  
  }
};

//---------------------------------------------------------------------------
//Calculate output for standard dropout

struct StandardDropout {

  const float  dropout_probability;
  const float *random_numbers;

  StandardDropout (
		   const float  _dropout_probability,
		   const float *_random_numbers
		   ) : dropout_probability(_dropout_probability),
		       random_numbers(_random_numbers)
  {}

  __device__
  float operator()(
		   const std::int32_t i,
		   const float input
		   ) {

    return (
	    (random_numbers[i] > dropout_probability) ?
	    (input) :
	    (0.f)
	    );
    
  }
  
};

//---------------------------------------------------------------------------
//Calculate Delta

struct DropoutCalcDelta {

  DropoutCalcDelta () {}

  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {

    //t<0> is the dropout node's output value
    //t<1> is the dropout node's own delta
    //t<2> is the child node's delta
    thrust::get<2>(t) += ((thrust::get<0>(t) == 0.f) ? (0.f) : (thrust::get<1>(t)));
    
  }
  
};

//---------------------------------------------------------------------------
//Calculate output when activation probability depends on input


struct NodeSampler {

  const float *random_numbers;

  NodeSampler (
	       const float *_random_numbers
	       ) : random_numbers(_random_numbers)
  {}

  __device__
  float operator()(
		   const std::int32_t i,
		   const float input
		   ) {

    return (
	    (random_numbers[i] < input) ?
	    (1.f) :
	    (0.f)
	    );
    
  }
  
};

//---------------------------------------------------------------------------
//Calculate Delta

struct NodeSamplerCalcDelta {

  NodeSamplerCalcDelta () {}

  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {

    thrust::get<1>(t) += thrust::get<0>(t);
    
  }
  
};


}
