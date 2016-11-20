namespace LogicalGateFunctors {

//---------------------------------------------------------------------------
//Case 1: Two input nodes

struct LogicalGateForwardPropagation2 {

  const float a,b,c,d;

  LogicalGateForwardPropagation2 (
				  const float _a,
				  const float _b,
				  const float _c,
				  const float _d
				  ) : a(_a), 
				      b(_b), 
				      c(_c), 
				      d(_d) {}
  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {
    
    //t<2> is the output node
    thrust::get<2>(t) = a + b
      *(c*thrust::get<0>(t) + d)
      *(c*thrust::get<1>(t) + d)
      ;
    
  }
  
};

struct LogicalGateBackpropagation2 {

  const float a,b,c,d;

  LogicalGateBackpropagation2 (
			       const float _a,
			       const float _b,
			       const float _c,
			       const float _d
			       ) : a(_a), 
				   b(_b), 
				   c(_c), 
				   d(_d) {}

  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {

    //t<0> delta
    //t<1>, t<2> : input0, input1
    //t<3>, t<4> : delta0, delta1
    
    float delta = thrust::get<0>(t);
    
    thrust::get<3>(t) = delta*b*c
      *(c*thrust::get<2>(t) + d)
      ;
    
    thrust::get<4>(t) = delta*b*c
      *(c*thrust::get<1>(t) + d)
      ;
      
  }

};

//---------------------------------------------------------------------------
//Case 2: Three input nodes

struct LogicalGateForwardPropagation3 {

  const float a,b,c,d;

  LogicalGateForwardPropagation3 (
				  const float _a,
				  const float _b,
				  const float _c,
				  const float _d
				  ) : a(_a), 
				      b(_b), 
				      c(_c), 
				      d(_d) {}
  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {
    
    //t<3> is the output node
    thrust::get<3>(t) = a + b
      *(c*thrust::get<0>(t) + d)
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<2>(t) + d)      
      ;
    
  }
  
};

struct LogicalGateBackpropagation3 {

  const float a,b,c,d;

  LogicalGateBackpropagation3 (
			       const float _a,
			       const float _b,
			       const float _c,
			       const float _d
			       ) : a(_a), 
				   b(_b), 
				   c(_c), 
				   d(_d) {}

  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {

    //t<0> delta
    //t<1>, t<2>, t<3>: input0, input1, input2
    //t<4>, t<5>, t<6>: delta0, delta1, delta2
    
    float delta = thrust::get<0>(t);
    
    thrust::get<4>(t) = delta*b*c
      *(c*thrust::get<2>(t) + d)
      *(c*thrust::get<3>(t) + d)
      ;
    
    thrust::get<5>(t) = delta*b*c
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<3>(t) + d)
      ;

    thrust::get<6>(t) = delta*b*c
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<2>(t) + d)
      ;
      
  }

};

//---------------------------------------------------------------------------
//Case 3: Four input nodes

struct LogicalGateForwardPropagation4 {

  const float a,b,c,d;

  LogicalGateForwardPropagation4 (
				  const float _a,
				  const float _b,
				  const float _c,
				  const float _d
				  ) : a(_a), 
				      b(_b), 
				      c(_c), 
				      d(_d) {}
  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {
    
    //t<4> is the output node
    thrust::get<4>(t) = a + b
      *(c*thrust::get<0>(t) + d)
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<2>(t) + d)  
      *(c*thrust::get<3>(t) + d)      
      ;
    
  }
  
};

struct LogicalGateBackpropagation4 {

  const float a,b,c,d;

  LogicalGateBackpropagation4 (
			       const float _a,
			       const float _b,
			       const float _c,
			       const float _d
			       ) : a(_a), 
				   b(_b), 
				   c(_c), 
				   d(_d) {}

  template <typename Tuple> 
  __device__
  void operator()(Tuple t) {

    //t<0> delta
    //t<1>, t<2>, t<3>, t<4>: input0, input1, input2, input3
    //t<5>, t<6>, t<7>, t<8>: delta0, delta1, delta2, delta3
    
    float delta = thrust::get<0>(t);
    
    thrust::get<5>(t) = delta*b*c
      *(c*thrust::get<2>(t) + d)
      *(c*thrust::get<3>(t) + d)
      *(c*thrust::get<4>(t) + d)
      ;
    
    thrust::get<6>(t) = delta*b*c
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<3>(t) + d)
      *(c*thrust::get<4>(t) + d)
      ;

    thrust::get<7>(t) = delta*b*c
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<2>(t) + d)
      *(c*thrust::get<4>(t) + d)      
      ;

    thrust::get<8>(t) = delta*b*c
      *(c*thrust::get<1>(t) + d)
      *(c*thrust::get<2>(t) + d)
      *(c*thrust::get<3>(t) + d)
      ;
      
  }

};
     
}

