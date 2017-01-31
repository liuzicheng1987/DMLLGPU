//protected methods:

void AggregationCpp::apply_forward_propagation_in_input_network(
    std::int32_t _batch_num_input,
    std::int32_t _batch_size_input)
{

    //Apply feedforward propagation in input network
    for (NeuralNetworkNodeCpp *node : this->input_network_ptr_->get_nodes())
    {
        node->calc_output(
            _batch_num_input,
            _batch_size_input);
    }
}

void AggregationCpp::backpropagate_and_calculate_dldw_in_input_network(
    std::int32_t _batch_num_input,
    std::int32_t _batch_size_input)
{

    //Backpropagation
    for (std::size_t n = 1; n <= this->input_network_ptr_->get_nodes().size(); ++n)
    {
        this->input_network_ptr_->get_nodes()[this->input_network_ptr_->get_nodes().size() - n]->calc_delta(_batch_size_input);
    }

    //For convenience and readability
    float *dldw_input_network_ptr = this->relational_net_->get_dldw_ptr_for_input_networks(this->input_network_);

    //Calculate derivative
    for (std::size_t n = 0; n < this->input_network_ptr_->get_nodes().size(); ++n)
    {
        if (this->input_network_ptr_->get_nodes()[n]->get_no_weight_updates() == false)
        {
            this->input_network_ptr_->get_nodes()[n]->calc_dLdw(
                dldw_input_network_ptr + this->input_network_ptr_->get_cumulative_num_weights_required()[n],
                _batch_num_input, _batch_size_input);
        }
    }

}

void AggregationCpp::calc_batch_size_aggregation_considering_timestamps(
    std::int32_t *_join_keys_left,
    std::int32_t _batch_num,
    std::int32_t _batch_size)
{

    std::int32_t batch_size_aggregation, batch_size_aggregation_considering_timestamps;

    //tso = time stamp of the output sample
    float tso;

    this->batch_size_aggregation_considering_timestamps_ = std::vector<std::int32_t>(_batch_size);

    for (std::int32_t i = 0; i < _batch_size; ++i)
    {

        //Sometimes the join key refers to an entry that does not exist - in this case,
        //we do not need to calculate this
        if (_join_keys_left[i] >= 0 && _join_keys_left[i] < this->input_network_ptr_->get_num_batches())
        {
            batch_size_aggregation = this->relational_net_->get_batch_size_aggregation(this->input_network_, i);

            if (this->use_timestamps_)
            {
                batch_size_aggregation_considering_timestamps = 0;

                tso = this->relational_net_->get_time_stamps_output(_batch_num, i);

                for (float tsi : this->relational_net_->get_time_stamps_input(this->input_network_, _join_keys_left[i]))
                {

                    //If the time stamp of the input sample is greater
                    //than the time stamp of the output sample,
                    //break.
                    //Since the input samples are ordered by the time stamp,
                    //all subsequent samples are irrelevant for the aggregation
                    if (tsi > tso)
                    {
                        break;
                    }

                    ++batch_size_aggregation_considering_timestamps;
                }

                this->batch_size_aggregation_considering_timestamps_[i] = batch_size_aggregation_considering_timestamps;
            }
            else //if (this->use_timestamps_) is not true
            {
                this->batch_size_aggregation_considering_timestamps_[i] = batch_size_aggregation;
            }
        }
    }
}

void AggregationCpp::initialise(std::int32_t _batch_size)
{

    //Resize output and delta, if necessary
    //Both output and delta are stored in NeuralNetworkNodeCpp
    //base class!
    if (static_cast<std::int32_t>(this->output_.size()) != this->dim_ * _batch_size)
    {

        //Resize output
        this->output_.resize(this->dim_ * _batch_size);
        this->output_ptr_ = thrust::raw_pointer_cast(this->output_.data());

        //Resize delta
        this->delta_.resize(this->dim_ * _batch_size);
        this->delta_ptr_ = thrust::raw_pointer_cast(this->delta_.data());
    }

    //Init output to zero
    thrust::fill(this->output_.begin(),
                 this->output_.begin() + this->dim_ * _batch_size,
                 0.f);
}

void AggregationCpp::init_delta_in_input_network()
{

    for (NeuralNetworkNodeCpp *node : this->input_network_ptr_->get_nodes())
    {
        thrust::fill(node->get_delta().begin(),
                     node->get_delta().end(),
                     0.f);

        //Because thrust::fill sometimes reallocates the entire vector,
        //it is necessary to reset delta_ptr_
        node->reset_delta_ptr();

    }
}

//public methods:

AggregationCpp::AggregationCpp(
    std::int32_t _node_number,
    std::int32_t _dim,
    std::int32_t _input_network,
    bool _use_timestamps,
    std::int32_t _i_share_weights_with,
    bool _no_weight_updates) : NeuralNetworkNodeCpp(_node_number,
                                                    _dim,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    0,
                                                    _i_share_weights_with,
                                                    _no_weight_updates,
                                                    nullptr)
{

    this->input_network_ = _input_network;
    this->use_timestamps_ = _use_timestamps;
};

AggregationCpp::~AggregationCpp(){};