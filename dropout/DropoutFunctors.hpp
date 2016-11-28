namespace DropoutFunctors {

//---------------------------------------------------------------------------
//Calculate Delta

struct CalcDelta {

  CalcDelta () {}

  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {

    //t<0> is the output node
    //t<1> is the output node the node's own delta
    //t<2> is the output node parent node's delta
    thrust::get<2>(t) += thrust::get<0>(t)*thrust::get<1>(t);
    
  }
  
};


//---------------------------------------------------------------------------
//Calculate output for standard dropout

struct StandardDropout {

  const float dropout_probability;

  StandardDropout (
		   const float _dropout_probability
		   ) : dropout_probability(_dropout_probability)
  {}

  __device__
  float operator()(
		   const std::int32_t i,
		   const float input
		   ) {

    thrust::default_random_engine rng;
    
    thrust::uniform_real_distribution<float> dist(0.f, 1.f);
    
    rng.discard(i);

    return (
	    (dist(rng) > dropout_probability) ?
	    (input) :
	    (0.f)
	    );
    
  }
  
};

}
