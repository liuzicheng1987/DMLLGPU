class NANDGateCpp: public LogicalGateCpp {

public:

  NANDGateCpp (
	       std::int32_t  _node_number, 
	       std::int32_t  _dim,
	       std::int32_t *_hidden_nodes_fed_into_me, 
	       std::int32_t  _hidden_nodes_fed_into_me_length	
	       ): LogicalGateCpp(
				 _node_number, 
				 _dim,
				 _hidden_nodes_fed_into_me, 
				 _hidden_nodes_fed_into_me_length	
				 ) {
    
    this->a_ = 0.f;
    this->b_ = 1.f;
    this->c_ = -1.f;
    this->d_ = 1.f;

  };
	
  ~NANDGateCpp() {};
  
};
