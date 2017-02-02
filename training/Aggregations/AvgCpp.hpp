#ifndef AVGCPP_HPP_
#define AVGCPP_HPP_

class AvgCpp : public AggregationCpp
{

  public:
    AvgCpp(
        std::int32_t _node_number,
        std::int32_t _dim,
        std::int32_t _input_network,
        bool _use_timestamps);

    ~AvgCpp();

    //Calculate the output of the node
    void calc_output(
        const std::int32_t _batch_num,
        const std::int32_t _batch_size);

    //Calculate the delta of the node (which is used for backpropagation)
    void calc_delta(std::int32_t _batch_size);

    //dLdw is not needed - sum aggregation is weightless!
};

#endif /* AVGCPP_HPP_ */
