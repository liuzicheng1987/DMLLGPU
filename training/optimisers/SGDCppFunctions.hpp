void SGDCpp::min(/*MPI_Comm comm,*/
		 NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
		 thrust::device_vector<float> &_W,
		 const float _tol,
		 const std::int32_t _max_num_epochs,
		 std::vector<float> &_sum_gradients)
{

    float current_learning_rate;

    //Initialise this->update_
    this->update_ = thrust::device_vector<float>(_W.size());

    thrust::fill(this->update_.begin(),
		 this->update_.end(),
		 0.f);

    for (;
	 this->epoch_num_ < _max_num_epochs;
	 ++(this->epoch_num_))
    { //epoch_num layer

	//Init this->sum_dldw_
	thrust::fill(
	    this->sum_dldw_.begin(),
	    this->sum_dldw_.end(),
	    0.f);

	//this->num_batches_ is inherited from the Optimiser class
	for (std::int32_t batch_num = 0;
	     batch_num < this->num_batches_;
	     ++batch_num)
	{ //batch_num layer

	    this->calculate_and_record_dldw(batch_num, _numerically_optimised_algorithm);

	    //Update updates_
	    thrust::transform(dldw_.begin(),
			      dldw_.end(),
			      this->update_.begin(),
			      this->update_.begin(),
			      utils::axpy<float>(1.f,
						 this->momentum_));

	    //Calculate current learning rate
	    //Learning rates are always divided by the sample size
	    current_learning_rate =
		(this->learning_rate_ / pow(
					    static_cast<float>(this->epoch_num_ + 1),
					    this->learning_rate_power_));

	    //Update W
	    thrust::transform(this->update_.begin(),
			      this->update_.end(),
			      _W.begin(),
			      _W.begin(),
			      utils::axpy<float>((-1.f) * current_learning_rate,
						 1.f));

	    //Post-update manipulations are used to impose restrictions on the weights,
	    //such as that all weights must be greater or equal to 0.

	} //batch_num layer

	//Check whether convergence condition is met. If yes, break
	if (this->record_sum_dldw_and_check_convergence(_sum_gradients, _tol))
	    break;

    } //epoch_num layer
}
