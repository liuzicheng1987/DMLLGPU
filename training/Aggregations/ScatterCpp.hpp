class ScatterCpp : public NeuralNetworkNodeCpp
{

    //Pointer to the relational network
    RelationalNetworkCpp *relational_net_;

    std::int32_t target_node_;

    //A vector of ones - necessary for matrix operation
    thrust::device_vector<float> ones_;

    //Pointer to included_in_aggregation_
    float *ones_ptr_;

  public:
    ScatterCpp(
        std::int32_t _node_number,
        std::int32_t _dim,
        std::int32_t _target_node);

    ~ScatterCpp();

    //Calculate the output of the node
    void calc_output(
        const std::int32_t _batch_num,
        const std::int32_t _batch_size);

    //Calculate the delta of the node (which is used for backpropagation)
    void calc_delta(std::int32_t _batch_size);

    //dLdw is not needed - sum aggregation is weightless!

    //Set pointer to relational net containing this nodes
    void set_relational_net(RelationalNetworkCpp *relational_net)
    {

        this->relational_net_ = relational_net;
    }
};