#ifndef SUMCPP_HPP_
#define SUMCPP_HPP_

class SumCpp : public AggregationCpp
{

  public:
    SumCpp(
        std::int32_t _node_number,
        std::int32_t _dim,
        std::int32_t _input_network,
        bool _use_timestamps);

    ~SumCpp();

    //One, if element is included in aggregation,
    //zero otherwise
    thrust::device_vector<float> included_in_aggregation_;

    //Calculate the output of the node
    void calc_output(
        const std::int32_t _batch_num,
        const std::int32_t _batch_size);

    //Calculate the delta of the node (which is used for backpropagation)
    void calc_delta(std::int32_t _batch_size);

    //dLdw is not needed - sum aggregation is weightless!
};

#endif /* SUMCPP_HPP_ */