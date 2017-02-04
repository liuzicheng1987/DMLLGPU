class RelationalNetworkCpp : public NumericallyOptimisedAlgorithmCpp
{

  private:
    //Output of the relational network
    //- contains aggregation nodes that
    //link to the input networks
    NeuralNetworkCpp *output_network_;

    //Input to to the aggregations contained in
    //the output network
    std::vector<NeuralNetworkCpp *> input_networks_;

    //Keys by which the input data is merged
    //with the output data, must be in form of
    //integers, from 0 to number of unique keys
    //minus one
    std::vector<std::vector<std::int32_t>> join_keys_left_;

    //Signifies which of the join_keys_left
    //each input_network_ uses
    std::vector<std::int32_t> join_key_used_;

    //Time stamps associated with the output network
    //If timestamps are not applicable, its length is 0
    std::vector<std::vector<float>> time_stamps_output_;

    //Time stamps associated with input network
    //Every input network must have one timestamp
    //If timestamps are not applicable, its length is 0
    std::vector<std::vector<std::vector<float>>> time_stamps_input_;

    //Vector containing weights of ALL neural networks
    thrust::device_vector<float> W_;

    //Pointer to W_
    float *w_ptr_;

    //Batch sizes of the aggregations of each
    //of the input networks
    std::vector<std::vector<std::int32_t>> batch_size_aggregation_;

    //Beginning of the current batch
    std::int32_t batch_begin_;

    //This is the equivalent to cumulative_num_weights_required_ in the
    //NeuralNetworkCpp class. Whereas cumulative_num_weights_required_ in
    //the NeuralNetworkCpp class records individual nodes,
    //this vector records the entries for the input and output network
    std::vector<std::int32_t> cumulative_num_weights_required_;

    //This vector keeps the pointers for the derivatives for the input networks
    std::vector<float *> dldw_ptr_for_input_networks_;

    //Signifies whether the relational network has been finalised
    bool finalised_;

    //Used to optimise the relational network
    OptimiserCpp *optimiser_;

    //Sum of squared gradients (the gradient norm) for each epoch
    std::vector<float> sum_gradients_;

    //Set num_samples and sample (boolean) for all input networks
    //and output network
    void set_num_samples_and_sample(bool _sample,
                                    std::vector<std::int32_t> &_batch_size);

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    //for all input networks and output network
    void init_cublas_cusparse_handles();

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    //for all input networks and output network
    void destroy_cublas_cusparse_handles();

    //This function needs to be called at the beginning of fit and transform
    void ensure_all_networks_are_finalised_and_join_keys_match();

  public:
    RelationalNetworkCpp() : NumericallyOptimisedAlgorithmCpp()
    {
        this->finalised_ = false;
    };

    ~RelationalNetworkCpp(){};

    //Finalises the relational network
    void finalise(float _weight_init_range);

    //The purpose of this function is to calculate the batch size
    //and make sure that the batch sizes in all samples are coherent
    std::vector<std::int32_t> calculate_batch_size_and_ensure_coherence();

    //The purpose of this function is to fit the relational network through
    //backpropagation
    void fit(/*MPI_Comm comm,*/
             OptimiserCpp *_optimiser, std::int32_t _global_batch_size,
             const float _tol, const std::int32_t _max_num_epochs,
             const bool _sample);

    //The purpose of this function is to calculate the gradient of the weights
    void dfdw(/*MPI_Comm comm,*/
              float *_dLdw, const float *_W, const std::int32_t _batch_begin,
              const std::int32_t _batch_end, const std::int32_t _batch_size,
              const std::int32_t _batch_num, const std::int32_t _epoch_num);

    //The purpose of this functions is to load the dense input data into the GPU
    void load_dense_data(std::int32_t _num_input_network,
                         std::int32_t _num_input_node, float *_X, std::int32_t _num_samples,
                         std::int32_t _dim, std::int32_t _global_batch_size,
                         std::int32_t *_indptr, std::int32_t _indptr_length);

    //The purpose of this function is to load the targets for the output network into the GPU,
    //which is necessary for training
    //Since only the output network receives targets, this is actually just a simple
    //wrapper function.
    void load_dense_targets(std::int32_t _num_output_node,
                            float *_Y, std::int32_t _num_samples,
                            std::int32_t _dim, std::int32_t _global_batch_size);

    //The purpose of this function is to load the time stamps for the output network into the GPU
    void load_time_stamps_output(float *_time_stamps_output,
                                 std::int32_t _time_stamps_output_length,
                                 std::int32_t _global_batch_size);

    //The purpose of this function is to load the time stamps for the input networks into the GPU
    void load_time_stamps_input(float *_time_stamps_input,
                                std::int32_t _time_stamps_input_length,
                                std::int32_t *_indptr,
                                std::int32_t _indptr_length);

    //The purpose of this function is to generate a prediction through the fitted network
    void transform(float *_Yhat, std::int32_t _Y2_num_samples,
                   std::int32_t _Y2_dim, bool _sample, std::int32_t _sample_size,
                   bool _get_hidden_nodes);

    //This function needs to be called at the end of fit and transform
    void clean_up();

    //Input networks are the networks that occur under the aggregation
    void add_input_network(NeuralNetworkCpp *_input_network,
                           std::int32_t _join_key_used)
    {
        if (this->finalised_)
            throw std::invalid_argument("Relational network has already been finalised!");

        this->input_networks_.push_back(_input_network);
        this->join_key_used_.push_back(_join_key_used);
    }

    //Join keys left are indices that connect the input networks with
    //the output networks
    void add_join_keys_left(std::int32_t *_join_keys_left,
                            std::int32_t _join_keys_left_length)
    {

        if (!this->finalised_)
            throw std::invalid_argument(
                "Relational network has not been finalised!");

        this->join_keys_left_.push_back(
            std::vector<std::int32_t>(_join_keys_left,
                                      _join_keys_left + _join_keys_left_length));
    }

    //Initialises the weights for the input networks and the output network
    void set_node_weights(const float *_w_ptr);

    //A bunch of getters and setters

    void set_output_network(NeuralNetworkCpp *_output_network)
    {
        if (this->finalised_)
            throw std::invalid_argument("Relational network has already been finalised!");

        this->output_network_ = _output_network;
    }

    std::vector<std::int32_t> &get_join_keys_left(std::int32_t _i)
    {

        return this->join_keys_left_[_i];
    }

    std::int32_t *get_join_keys_left_ptr(
        std::int32_t _input_network)
    {

        return this->join_keys_left_[this->join_key_used_[_input_network]].data() + this->batch_begin_;
    }

    std::int32_t get_batch_size_aggregation(std::int32_t _input_network,
                                            std::int32_t _i)
    {

        return this->batch_size_aggregation_[_input_network][this->batch_begin_ + _i];
    }

    std::int32_t get_sum_output_dim()
    {

        return this->output_network_->get_sum_output_dim();
    }

    float get_time_stamps_output(std::int32_t _batch_num, std::int32_t _i)
    {

        return this->time_stamps_output_[_batch_num][_i];
    }

    std::vector<float> &get_time_stamps_input(std::int32_t _num_input_network,
                                              std::int32_t _batch_num)
    {

        return this->time_stamps_input_[_num_input_network][_batch_num];
    }

    std::vector<std::int32_t> &get_cumulative_num_weights_required()
    {

        return this->cumulative_num_weights_required_;
    }

    float *get_dldw_ptr_for_input_networks(std::int32_t _num_input_network)
    {

        return this->dldw_ptr_for_input_networks_[_num_input_network];
    }

    //This functions returns the length of sum of the gradients during each training epoch
    //Identical to the number of epochs
    std::int32_t get_sum_gradients_length()
    {
        return static_cast<std::int32_t>(this->sum_gradients_.size());
    }

    //This functions returns the sum of the gradients during each training epoch
    void get_sum_gradients(float *_sum_gradients,
                           std::int32_t _sum_gradients_size)
    {
        std::copy(this->sum_gradients_.begin(), this->sum_gradients_.end(),
                  _sum_gradients);
    }
};
