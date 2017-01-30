void NeuralNetworkCpp::dfdw(/*MPI_Comm comm,*/
                            float *_dLdw, const float *_W, const std::int32_t _batch_begin,
                            const std::int32_t _batch_end, const std::int32_t _batch_size,
                            const std::int32_t _batch_num, const std::int32_t _epoch_num)
{

    //Set pointers contained in the NeuralNetworkNodes class
    this->set_node_weights(_W);

    //Forward propagation
    for (auto node : this->nodes_)
        node->calc_output(_batch_num, _batch_size);

    //Initialise delta
    //Needs to be after forward propagation,
    //because forward propagation might resize delta
    for (auto node : this->nodes_)
    {
        thrust::fill(node->delta_.begin(), node->delta_.end(), 0.f);
    }

    //Calculate loss for dense targets
    for (std::int32_t n = 0; n < this->num_output_nodes_dense_; ++n)
        this->loss_->dloss_dyhat_dense(
            /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/
            this->dense_targets_[n][_batch_num],
            this->output_nodes_dense_[n]->output_,
            this->output_nodes_dense_[n]->output_ptr_,
            this->output_nodes_dense_[n]->delta_);

    //Calculate loss for sparse targets
    for (std::int32_t n = 0; n < this->num_output_nodes_sparse_; ++n)
        this->loss_->dloss_dyhat_sparse(
            /*MPI_Comm comm, const std::int32_t rank, const std::int32_t size,*/
            this->sparse_targets_[n][_batch_num],
            this->output_nodes_sparse_[n]->output_,
            this->output_nodes_sparse_[n]->output_ptr_,
            this->output_nodes_sparse_[n]->delta_);

    //Backpropagation
    for (std::size_t n = 1; n <= this->nodes_.size(); ++n)
        this->nodes_[this->nodes_.size() - n]->calc_delta(_batch_size);

    //Calculate derivative
    for (std::size_t n = 0; n < this->nodes_.size(); ++n)
        if (this->nodes_[n]->no_weight_updates_ == false)
            this->nodes_[n]->calc_dLdw(
                _dLdw + this->cumulative_num_weights_required_[n],
                _batch_num, _batch_size);

    //Add all localdZdW and store the result in dZdW
    //MPI_Allreduce(localdZdW, this->optimiser->dZdW, this->lengthW, MPI_DOUBLE, MPI_SUM, comm);
    //Barrier: Wait until all processes have reached this point
    //MPI_Barrier(comm);
}

void NeuralNetworkCpp::fit(/*MPI_Comm comm,*/
                           OptimiserCpp *_optimiser, std::int32_t _global_batch_size, const float _tol,
                           const std::int32_t _max_num_epochs,
                           const std::int32_t _MinibatchSizeStandard, const bool _sample)
{

    //Make sure that neural network has been finalised!
    if (!this->finalised_)
        throw std::invalid_argument("Neural network has not been finalised!");

    std::vector<std::int32_t> batch_size =
        this->calculate_batch_size_and_ensure_coherence();

    //Calculate this->num_samples
    this->num_samples_ = std::accumulate(batch_size.begin(), batch_size.end(),
                                         0);

    //Get num_batches
    std::int32_t num_batches = (std::int32_t)(batch_size.size());

    //Make sure that the batch_sizes are identical for all matrices provided!
    this->ensure_coherence_for_target_values(batch_size);

    this->optimiser_ = _optimiser;
    this->sample_ = _sample;

    //Init cuBLAS handle, cuSPARSE handle and matrix descriptor
    this->init_cublas_cusparse_handles();

    //Do the actual optimisation
    this->optimiser_->minimise(this, this->num_samples_, this->W_,
                               _global_batch_size, _tol, _max_num_epochs, this->sum_gradients_);

    //Destroy cuBLAS handle, cuSPARSE handle and matrix descriptor
    this->destroy_cublas_cusparse_handles();

    //Clear data. so it does not unnecessarily take up space on the
    this->delete_data();
}
