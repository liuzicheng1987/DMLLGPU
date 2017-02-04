class FirstCpp : public AggregationCpp
{

  public:
    FirstCpp(
        std::int32_t _node_number,
        std::int32_t _dim,
        std::int32_t _input_network,
        bool _use_timestamps);

    ~FirstCpp();

    //Calculate the output of the node
    void calc_output(
        const std::int32_t _batch_num,
        const std::int32_t _batch_size);

    //Calculate the delta of the node (which is used for backpropagation)
    void calc_delta(std::int32_t _batch_size);

    //dLdw is not needed - sum aggregation is weightless!
};