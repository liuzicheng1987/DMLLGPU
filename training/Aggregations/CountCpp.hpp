#ifndef COUNTCPP_HPP_
#define COUNTCPP_HPP_

class CountCpp : public AggregationCpp
{

  public:
    CountCpp(
        std::int32_t _node_number,
        std::int32_t _input_network,
        bool _use_timestamps);

    ~CountCpp();

    //Calculate the output of the node
    void calc_output(
        const std::int32_t _batch_num,
        const std::int32_t _batch_size);

    //calc_delta is not needed

    //dLdw is not needed - sum aggregation is weightless!
};

#endif /* COUNTCPP_HPP_ */
