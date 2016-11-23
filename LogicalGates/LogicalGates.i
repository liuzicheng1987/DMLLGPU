//----------------------------------------------------------------------------------------------
//class LogicalGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class LogicalGateCpp: public NeuralNetworkNodeGPUCpp {

public:

  LogicalGateCpp (
		  std::int32_t  _node_number, 
		  std::int32_t  _dim,
		  std::int32_t *_hidden_nodes_fed_into_me, 
		  std::int32_t  _hidden_nodes_fed_into_me_length
		  );
};

//----------------------------------------------------------------------------------------------
//class ANDGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class ANDGateCpp: public LogicalGateCpp {

public:

  ANDGateCpp (
	      std::int32_t  _node_number, 
	      std::int32_t  _dim,
	      std::int32_t *_hidden_nodes_fed_into_me, 
	      std::int32_t  _hidden_nodes_fed_into_me_length	
	      );
	  
};

//----------------------------------------------------------------------------------------------
//class ORGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class ORGateCpp: public LogicalGateCpp {

public:

  ORGateCpp (
	     std::int32_t  _node_number, 
	     std::int32_t  _dim,
	     std::int32_t *_hidden_nodes_fed_into_me, 
	     std::int32_t  _hidden_nodes_fed_into_me_length	
	     );
	  
};

//----------------------------------------------------------------------------------------------
//class XORGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class XORGateCpp: public LogicalGateCpp {

public:

  XORGateCpp (
	      std::int32_t  _node_number, 
	      std::int32_t  _dim,
	      std::int32_t *_hidden_nodes_fed_into_me, 
	      std::int32_t  _hidden_nodes_fed_into_me_length	
	      );
	  
};

//----------------------------------------------------------------------------------------------
//class XNORGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class XNORGateCpp: public LogicalGateCpp {

public:

  XNORGateCpp (
	       std::int32_t  _node_number, 
	       std::int32_t  _dim,
	       std::int32_t *_hidden_nodes_fed_into_me, 
	       std::int32_t  _hidden_nodes_fed_into_me_length	
	       );
	  
};

//----------------------------------------------------------------------------------------------
//class NORGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class NORGateCpp: public LogicalGateCpp {

public:

  NORGateCpp (
	      std::int32_t  _node_number, 
	      std::int32_t  _dim,
	      std::int32_t *_hidden_nodes_fed_into_me, 
	      std::int32_t  _hidden_nodes_fed_into_me_length	
	      );
	  
};

//----------------------------------------------------------------------------------------------
//class NANDGateCpp

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_hidden_nodes_fed_into_me, std::int32_t _hidden_nodes_fed_into_me_length)};

class NANDGateCpp: public LogicalGateCpp {

public:

  NANDGateCpp (
	       std::int32_t  _node_number, 
	       std::int32_t  _dim,
	       std::int32_t *_hidden_nodes_fed_into_me, 
	       std::int32_t  _hidden_nodes_fed_into_me_length	
	       );
	  
};

//----------------------------------------------------------------------------------------------
