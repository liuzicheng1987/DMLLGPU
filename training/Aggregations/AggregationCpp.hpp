class AggregationCpp : public NeuralNetworkNodeCpp
{

  protected:
    //Index signifying input network
    std::int32_t input_network_;

    //Pointer to input network
    NeuralNetworkCpp *input_network_ptr_;

    //Pointer to ones_
    float *ones_ptr_;

    //One, if element is included in aggregation,
    //zero otherwise
    thrust::device_vector<float> included_in_aggregation_;

    //Pointer to included_in_aggregation_
    float *included_in_aggregation_ptr_;

    //Batch size of the aggregations - since timestamps can reduce
    //the effective batch size!
    std::vector<std::int32_t> batch_size_aggregation_considering_timestamps_;
    
    //Pointer to relational net containing this node
    RelationalNetworkCpp *relational_net_;

    //Boolean signifying whether timestamps are relevant for this aggregation
    bool use_timestamps_;

    //Forward propagation in the input network needs to be applied by all aggregations
    //- therefore it makes sense to modularise it
    void apply_forward_propagation_in_input_network(
        std::int32_t _batch_num_input,
        std::int32_t _batch_size_input);

    //Applies backpropagation in the input network
    void backpropagate_and_calculate_dldw_in_input_network(
        std::int32_t _batch_num_input,
        std::int32_t _batch_size_input);

    //Calculates the batch size of the aggregation
    void calc_batch_size_aggregation_considering_timestamps(
        std::int32_t *_join_keys_left,
        std::int32_t _batch_num,
        std::int32_t _batch_size);

    //Undertakes some initialisation steps at the beginning of forward propagation
    void initialise(std::int32_t _batch_size);

    //Sets all deltas in input_network to zero
    void init_delta_in_input_network();

  public:
    AggregationCpp(
        std::int32_t _node_number,
        std::int32_t _dim,
        std::int32_t _input_network,
        bool _use_timestamps,
        std::int32_t _i_share_weights_with,
        bool _no_weight_updates);

    ~AggregationCpp();

    //Used to get the number of weights required
    //(very important to finalise the neural network)
    virtual std::int32_t get_num_weights_required() { return 0; };

    //Calculate the output of the node
    virtual void calc_output(
        const std::int32_t _batch_num,
        const std::int32_t _batch_size){};

    //Calculate the delta of the node (which is used for backpropagation)
    virtual void calc_delta(std::int32_t _batch_size){};

    //Calculate the derivatives for the individual weights
    virtual void calc_dLdw(
        float *_dLdw,
        const std::int32_t _batch_num,
        const std::int32_t _batch_size){};

    //Getters and setters

    //Get input_network_
    std::int32_t get_input_network()
    {
        return this->input_network_;
    }

    //Set pointer to input network
    void set_input_network_ptr(NeuralNetworkCpp *input_network_ptr)
    {

        this->input_network_ptr_ = input_network_ptr;
    }

    //Set pointer to relational net containing this nodes
    void set_relational_net(RelationalNetworkCpp *relational_net)
    {

        this->relational_net_ = relational_net;
    }
};
