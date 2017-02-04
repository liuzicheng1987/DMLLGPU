//----------------------------------------------------------------------------------------------
//class AggregationCpp

class AggregationCpp : public NeuralNetworkNodeCpp
{

public:
  AggregationCpp(
      std::int32_t _node_number,
      std::int32_t _dim,
      std::int32_t _input_network,
      bool _use_timestamps,
      std::int32_t _i_share_weights_with,
      bool _no_weight_updates);

  ~AggregationCpp();

};


//----------------------------------------------------------------------------------------------
//class SumCpp

class SumCpp : public AggregationCpp
{

public:
  SumCpp(
      std::int32_t _node_number,
      std::int32_t _dim,
      std::int32_t _input_network,
      bool _use_timestamps);

  ~SumCpp();

};

//----------------------------------------------------------------------------------------------
//class AvgCpp

class AvgCpp : public AggregationCpp
{

public:
  AvgCpp(
      std::int32_t _node_number,
      std::int32_t _dim,
      std::int32_t _input_network,
      bool _use_timestamps);

  ~AvgCpp();

};

//----------------------------------------------------------------------------------------------
//class CountCpp

class CountCpp : public AggregationCpp
{

public:
  CountCpp(
      std::int32_t _node_number,
      std::int32_t _input_network,
      bool _use_timestamps);

  ~CountCpp();

};

//----------------------------------------------------------------------------------------------
//class FirstCpp

class FirstCpp : public AggregationCpp
{

public:
  FirstCpp(
      std::int32_t _node_number,
      std::int32_t _dim,      
      std::int32_t _input_network,
      bool _use_timestamps);

  ~FirstCpp();

};

//----------------------------------------------------------------------------------------------
//class LastCpp

class LastCpp : public AggregationCpp
{

public:
  LastCpp(
      std::int32_t _node_number,
      std::int32_t _dim,      
      std::int32_t _input_network,
      bool _use_timestamps);

  ~LastCpp();
  
};

