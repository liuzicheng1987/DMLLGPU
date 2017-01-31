#ifndef RELATIONALNETWORKCPPFUNCTIONS_HPP_
#define RELATIONALNETWORKCPPFUNCTIONS_HPP_

//private methods:

void RelationalNetworkCpp::ensure_all_networks_are_finalised_and_join_keys_match()
{
    //Make sure that relational network has been finalised!
    if (!this->finalised_)
        throw std::invalid_argument(
            "Relational network has not been finalised!");

    //Size of join_keys_left_ cannot be greater than
    //size of input_networks_
    //(but can be smaller)
    if (this->join_keys_left_.size() > this->input_networks_.size())
        throw std::invalid_argument(
            "Length of join_keys_left cannot be greater than size of input_networks!");

    //Elements contained in join_key_used cannot be negative
    //and cannot exceed size of join_keys_left_
    auto test_join_key_used = [this](std::int32_t i) {
        return (
            i < 0 || i >= static_cast<std::int32_t>(this->join_keys_left_.size()));
    };

    if (std::any_of(this->join_key_used_.begin(), this->join_key_used_.end(),
                    test_join_key_used))
    {
        throw std::invalid_argument("Element of join_key_used out of range!");
    }

    //Length of join_keys_left_ must have same length as tables in output
    //network
    //We can compare with ANY input table, because the function
    //calculate_batch_size_and_ensure_coherence() makes sure that these are
    //consistent.
    std::int32_t length;

    if (this->output_network_->get_dense_input_data_size() > 0)
    {

        length = 0;
        for (auto &data : this->output_network_->get_dense_input_data(0))
            length += data.batch_size;
    }
    else if (this->output_network_->get_sparse_input_data_size() > 0)
    {

        length = 0;
        for (auto &data : this->output_network_->get_sparse_input_data(0))
            length += data.batch_size;
    }

    for (auto &join_key : this->join_keys_left_)
    {
        if (static_cast<std::int32_t>(join_key.size()) != length)
            ;
    }
}

void RelationalNetworkCpp::set_num_samples_and_sample(bool _sample,
                                                      std::vector<std::int32_t> &_batch_size)
{

    //For readability
    std::vector<std::int32_t> *batch_size_aggregation;

    for (NeuralNetworkCpp *input_network : this->input_networks_)
    {

        this->batch_size_aggregation_.push_back(
            input_network->calculate_batch_size_and_ensure_coherence());

        batch_size_aggregation = &(
            this->batch_size_aggregation_[this->batch_size_aggregation_.size() - 1]);

        input_network->set_num_samples(
            std::accumulate(batch_size_aggregation[0].begin(),
                            batch_size_aggregation[0].end(), 0));

        input_network->set_sample(_sample);
    };

    this->output_network_->set_num_samples(
        std::accumulate(_batch_size.begin(), _batch_size.end(),
                        0));

    this->output_network_->set_sample(_sample);
}

void RelationalNetworkCpp::init_cublas_cusparse_handles()
{

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    for (NeuralNetworkCpp *input_network : this->input_networks_)
        input_network->init_cublas_cusparse_handles();

    this->output_network_->init_cublas_cusparse_handles();
}

void RelationalNetworkCpp::destroy_cublas_cusparse_handles()
{

    //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
    for (NeuralNetworkCpp *input_network : this->input_networks_)
        input_network->destroy_cublas_cusparse_handles();

    this->output_network_->destroy_cublas_cusparse_handles();
}

//public methods:

//Finalises the relational network
void RelationalNetworkCpp::finalise(float _weight_init_range)
{

    if (this->finalised_)
        throw std::invalid_argument("Relational network has already been finalised!");

    //Make sure that all input networks have been finalised!
    auto test_input_network = [](NeuralNetworkCpp *neural_net) {
        return neural_net->is_finalised() == false;
    };

    if (std::any_of(this->input_networks_.begin(), this->input_networks_.end(),
                    test_input_network))
        throw std::invalid_argument(
            "Not all input networks have been finalised!");

    //Make sure output network has been finalised!
    if (this->output_network_->is_finalised() == false)
        throw std::invalid_argument("Output network has not been finalised!");

    //Set input network and relational net in all aggregations
    //contained in the output network

    for (NeuralNetworkNodeCpp *node : this->output_network_->get_nodes())
    {

        if (node->get_input_network() < 0 || node->get_input_network() >= static_cast<std::int32_t>(this->input_networks_.size()))
        {
            throw std::invalid_argument("Integer for input network out of range!");
        }

        node->set_input_network_ptr(
            this->input_networks_[node->get_input_network()]);

        node->set_relational_net(this);
    }

    //Initialise weights and calculate cumulative_num_weights_required_
    std::int32_t length_W = 0;
    this->cumulative_num_weights_required_.clear();

    for (NeuralNetworkCpp *input_network : this->input_networks_)
    {

        this->cumulative_num_weights_required_.push_back(length_W);
        length_W += input_network->get_length_params();
    }

    this->cumulative_num_weights_required_.push_back(length_W);

    length_W += this->output_network_->get_length_params();
    this->cumulative_num_weights_required_.push_back(length_W);

    //Init Whost
    std::vector<float> Whost(length_W);

    //Note that we deliberately choose a constant seed to get the same output every time we call the function
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dist(_weight_init_range * (-1.0f),
                                               _weight_init_range);

    //Initialise weight vector
    for (std::int32_t i = 0;
         i < length_W;
         ++i)
    {
        Whost[i] = dist(gen);
    }

    //Transfor to device vector
    this->W_ = thrust::device_vector<float>(Whost.data(),
                                            Whost.data() + Whost.size());

    this->w_ptr_ = thrust::raw_pointer_cast(this->W_.data());

    //Set finalised to true so we know we can now fit the neural network
    this->finalised_ = true;
}

//The purpose of this functions is to load the dense input data into the GPU
void RelationalNetworkCpp::load_dense_data(std::int32_t _num_input_network,
                                           std::int32_t _num_input_node, float *_X, std::int32_t _num_samples,
                                           std::int32_t _dim, std::int32_t _global_batch_size,
                                           std::int32_t *_indptr, std::int32_t _indptr_length)
{

    if (!this->finalised_)
        throw std::invalid_argument(
            "Relational network has not been finalised!");

    std::int32_t num_batches;

    //-1 means we are loading the output network
    if (_num_input_network == -1)
    {

        num_batches = calc_num_batches(
            /*MPI_Comm comm,*/
            _num_samples, _global_batch_size);

        std::vector<std::int32_t> indptr;

        std::int32_t batch_begin, batch_end, batch_size;

        indptr.push_back(0);

        for (std::int32_t batch_num = 0; batch_num < num_batches; ++batch_num)
        {

            this->calc_batch_begin_end(batch_begin, batch_end, batch_size,
                                       batch_num, _num_samples, num_batches);

            indptr.push_back(batch_end);
        }

        this->output_network_->load_dense(
            this->output_network_->get_dense_input_data(
                _num_input_node),
            _X, _num_samples, _dim,
            num_batches, false, //false means do not transpose
            indptr.data(), static_cast<std::int32_t>(indptr.size()));
    }
    else if (_num_input_network >= 0)
    {

        num_batches = _indptr_length - 1;

        this->input_networks_[_num_input_network]->load_dense(
            this->input_networks_[_num_input_network]->get_dense_input_data(
                _num_input_node),
            _X, _num_samples, _dim, num_batches,
            false, //do not transpose
            _indptr, _indptr_length);
    }
    else
    {

        throw std::invalid_argument("_num_input_network out of range!");
    }
}

//Load target data into GPU
void RelationalNetworkCpp::load_dense_targets(std::int32_t _num_output_node,
                                              float *_Y, std::int32_t _num_samples,
                                              std::int32_t _dim, std::int32_t _global_batch_size)
{

    this->output_network_->load_dense_targets(_num_output_node, _Y, _num_samples,
                                              _dim, _global_batch_size);
}

//Load time stamps for input networks into GPU
void RelationalNetworkCpp::load_time_stamps_input(
    float *_time_stamps_input,
    std::int32_t _time_stamps_input_length,
    std::int32_t *_indptr,
    std::int32_t _indptr_length)
{

    this->time_stamps_input_.push_back(
        std::vector<std::vector<float>>(0));

    if (_time_stamps_input_length > 0)
        for (std::int32_t i = 0; i < _indptr_length - 1; ++i)
        {

            this->time_stamps_input_[this->time_stamps_input_.size() - 1].push_back(
                std::vector<float>(
                    _time_stamps_input + _indptr[i],
                    _time_stamps_input + _indptr[i + 1]));
        }
}

//Load time stamps for output network into GPU
void RelationalNetworkCpp::load_time_stamps_output(
    float *_time_stamps_output,
    std::int32_t _time_stamps_output_length,
    std::int32_t _global_batch_size)
{

    if (_time_stamps_output_length != 0)
    {
        std::int32_t num_batches = calc_num_batches(
            /*MPI_Comm comm,*/
            _time_stamps_output_length, _global_batch_size);

        std::int32_t batch_begin, batch_end, batch_size;

        for (std::int32_t batch_num = 0; batch_num < num_batches; ++batch_num)
        {

            this->calc_batch_begin_end(batch_begin, batch_end, batch_size,
                                       batch_num, _time_stamps_output_length, num_batches);

            this->time_stamps_output_.push_back(
                std::vector<float>(
                    _time_stamps_output + batch_begin,
                    _time_stamps_output + batch_end));
        }
    }
}

void RelationalNetworkCpp::set_node_weights(const float *_w_ptr)
{

    const float *w = _w_ptr;

    for (NeuralNetworkCpp *input_network : this->input_networks_)
    {

        input_network->set_node_weights(w);

        w += input_network->get_length_params();
    }

    this->output_network_->set_node_weights(w);
}

void RelationalNetworkCpp::clean_up()
{

    //Clear join_keys and batches
    this->join_keys_left_.clear();
    this->join_key_used_.clear();
    this->batch_size_aggregation_.clear();

    //Clear timestamps
    this->time_stamps_output_.clear();
    this->time_stamps_input_.clear();

    //Clear dldw_ptr_for_input_networks_ and dldw_input_networks_
    this->dldw_ptr_for_input_networks_.clear();

    //Clear data, so it does not unnecessarily take up space on the GPU
    for (NeuralNetworkCpp *input_network : this->input_networks_)
        input_network->delete_data();

    this->output_network_->delete_data();
}

void RelationalNetworkCpp::fit(/*MPI_Comm comm,*/
                               OptimiserCpp *_optimiser, std::int32_t _global_batch_size,
                               const float _tol, const std::int32_t _max_num_epochs,
                               const bool _sample)
{

    //Make sure that relational network has been finalised!
    if (!this->finalised_)
        throw std::invalid_argument("Relational network has not been finalised!");

    this->ensure_all_networks_are_finalised_and_join_keys_match();

    //Calculate the batch size and ensure number of samples is coherent
    std::vector<std::int32_t> batch_size =
        this->output_network_->calculate_batch_size_and_ensure_coherence();

    this->output_network_->ensure_coherence_for_target_values(batch_size);

    //Get num_batches
    std::int32_t num_batches = static_cast<std::int32_t>(batch_size.size());

    //Get batch_size_aggregation
    this->batch_size_aggregation_.clear();

    this->set_num_samples_and_sample(_sample, batch_size);

    this->optimiser_ = _optimiser;

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    this->init_cublas_cusparse_handles();

    //Do the actual optimisation
    this->optimiser_->minimise(this, this->output_network_->get_num_samples(),
                               this->W_, _global_batch_size, _tol,
                               _max_num_epochs, this->sum_gradients_);

    //Clear data. so it does not unnecessarily take up space on the
    this->clean_up();

    //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
    this->destroy_cublas_cusparse_handles();
}

//The purpose of this function is to calculate the gradient of the weights
void RelationalNetworkCpp::dfdw(/*MPI_Comm comm,*/
                                float *_dLdw, const float *_W, const std::int32_t _batch_begin,
                                const std::int32_t _batch_end, const std::int32_t _batch_size,
                                const std::int32_t _batch_num, const std::int32_t _epoch_num)
{

    //Set pointers contained in the NeuralNetworkNodes class
    this->set_node_weights(_W);

    //Record _batch_begin. This is necessary for communication
    //between the RelationalNetworkCpp class and the AggregationCpp
    //class
    this->batch_begin_ = _batch_begin;

    //Set this->dldw_ptr_for_input_networks_
    std::int32_t length_dldw = 0;
    this->dldw_ptr_for_input_networks_.clear();

    for (NeuralNetworkCpp *input_network : this->input_networks_)
    {

        this->dldw_ptr_for_input_networks_.push_back(_dLdw + length_dldw);
        length_dldw += input_network->get_length_params();
    }

    //Forward propagation
    for (NeuralNetworkNodeCpp *node : this->output_network_->get_nodes())
    {
        node->calc_output(_batch_num, _batch_size);
    }

    //Initialise delta
    //Needs to be after forward propagation,
    //because forward propagation might resize delta
    for (NeuralNetworkNodeCpp *node : this->output_network_->get_nodes())
    {
        thrust::fill(node->get_delta().begin(), node->get_delta().end(), 0.f);

        //Because thrust::fill sometimes reallocates the entire vector,
        //it is necessary to reset delta_ptr_
        node->reset_delta_ptr();
    }

    //Calculate loss for dense targets
    for (std::int32_t n = 0; n < this->output_network_->get_num_output_nodes_dense(); ++n)
    {
        this->output_network_->get_loss()->dloss_dyhat_dense(
            /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/
            this->output_network_->get_dense_targets(n, _batch_num),
            this->output_network_->get_output_nodes_dense()[n]->get_output(),
            this->output_network_->get_output_nodes_dense()[n]->get_output_ptr(),
            this->output_network_->get_output_nodes_dense()[n]->get_delta());
    }

    //Calculate loss for sparse targets
    for (std::int32_t n = 0; n < this->output_network_->get_num_output_nodes_sparse(); ++n)
    {
        this->output_network_->get_loss()->dloss_dyhat_sparse(
            /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/
            this->output_network_->get_sparse_targets(n, _batch_num),
            this->output_network_->get_output_nodes_sparse()[n]->get_output(),
            this->output_network_->get_output_nodes_sparse()[n]->get_output_ptr(),
            this->output_network_->get_output_nodes_sparse()[n]->get_delta());
    }

    //Backpropagation
    for (std::size_t n = 1; n <= this->output_network_->get_nodes().size(); ++n)
    {
        this->output_network_->get_nodes()[this->output_network_->get_nodes().size() - n]->calc_delta(_batch_size);
    }

    //For convenience
    float *dldw_output_network = _dLdw + this->cumulative_num_weights_required_[this->input_networks_.size()];

    //Calculate derivative
    for (std::size_t n = 0; n < this->output_network_->get_nodes().size(); ++n)
    {
        if (this->output_network_->get_nodes()[n]->get_no_weight_updates() == false)
        {
            this->output_network_->get_nodes()[n]->calc_dLdw(
                dldw_output_network + this->output_network_->get_cumulative_num_weights_required()[n],
                _batch_num, _batch_size);
        }
    }

    //Add all localdZdW and store the result in dZdW
    //MPI_Allreduce(localdZdW, this->optimiser->dZdW, this->lengthW, MPI_DOUBLE, MPI_SUM, comm);
    //Barrier: Wait until all processes have reached this point
    //MPI_Barrier(comm);
}

//The purpose of this function is to generate a prediction through the fitted network
void RelationalNetworkCpp::transform(float *_Yhat, std::int32_t _Y2_num_samples,
                                     std::int32_t _Y2_dim, bool _sample, std::int32_t _sample_size,
                                     bool _get_hidden_nodes)
{

    //Make sure that relational network has been finalised!
    if (!this->finalised_)
        throw std::invalid_argument("Relational network has not been finalised!");

    this->ensure_all_networks_are_finalised_and_join_keys_match();

    //Calculate the batch size and ensure number of samples is coherent
    std::vector<std::int32_t> batch_size =
        this->output_network_->calculate_batch_size_and_ensure_coherence();

    //Get num_batches
    std::int32_t num_batches = static_cast<std::int32_t>(batch_size.size());

    //Get batch_size_aggregation
    this->batch_size_aggregation_.clear();

    this->set_num_samples_and_sample(_sample, batch_size);

    if (!_sample)
    {
        _sample_size = 1;
    }

    const double SampleAvg = 1.0 / ((double)_sample_size);

    //Set pointers contained in the NeuralNetworkNodes class
    this->set_node_weights(this->w_ptr_);

    //Init YhatTemp
    thrust::device_vector<float> Yhat_temp(_Yhat,
                                           _Yhat + _Y2_num_samples * _Y2_dim);

    this->init_cublas_cusparse_handles();

    //For readability
    NeuralNetworkNodeCpp *output_network_node;

    //Calculate output
    this->batch_begin_ = 0;
    for (std::int32_t batch_num = 0; batch_num < num_batches;
         ++batch_num, this->batch_begin_ += batch_size[batch_num])
    {
        for (std::int32_t iteration = 0; iteration < _sample_size;
             ++iteration)
        {

            //Calculate nodes
            for (NeuralNetworkNodeCpp *node : this->output_network_->get_nodes())
                node->calc_output(batch_num, batch_size[batch_num]);

            //Add to YhatTemp
            std::int32_t col_num = 0;
            for (std::int32_t node_num = 0;
                 node_num < this->output_network_->get_num_output_nodes();
                 ++node_num)
            {
                for (std::int32_t dim = 0;
                     dim < this->output_network_->get_output_nodes()[node_num]->get_dim();
                     ++dim)
                {

                    output_network_node = this->output_network_->get_output_nodes()[node_num];

                    thrust::transform(
                        output_network_node->get_output().begin() + dim * batch_size[batch_num],

                        output_network_node->get_output().begin() + (dim + 1) * batch_size[batch_num],

                        Yhat_temp.begin() + _Y2_num_samples * col_num + this->batch_begin_,

                        Yhat_temp.begin() + _Y2_num_samples * col_num + this->batch_begin_,

                        thrust::plus<float>());

                    ++col_num;
                }
            }
        }
    }

    //Get data from YhatTemp and transpose
    for (std::int32_t i = 0; i < _Y2_num_samples; ++i)
        for (std::int32_t j = 0; j < _Y2_dim; ++j)
            _Yhat[i * _Y2_dim + j] = Yhat_temp[j * _Y2_num_samples + i];

    this->clean_up();

    this->destroy_cublas_cusparse_handles();
}

#endif /* RELATIONALNETWORKCPPFUNCTIONS_HPP_ */
