void AdamCpp::min(/*MPI_Comm comm,*/
		  NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
		  thrust::device_vector<float> &_W,
		  const float _tol,
		  const std::int32_t _max_num_epochs,
		  std::vector<float> &_sum_gradients)
{

    //Initialise first moment estimate
    this->first_moment_ = thrust::device_vector<float>(_W.size());

    thrust::fill(this->first_moment_.begin(),
		 this->first_moment_.end(),
		 0.f);

    //Initialise sum_dldw_squared_
    this->sum_dldw_squared_ = thrust::device_vector<float>(_W.size());

    thrust::fill(this->sum_dldw_squared_.begin(),
		 this->sum_dldw_squared_.end(),
		 0.f);

    for (;
	 this->epoch_num_ < _max_num_epochs;
	 ++(this->epoch_num_))
    { //epoch_num layer

	//Init this->sum_dldw_
	thrust::fill(this->sum_dldw_.begin(),
		     this->sum_dldw_.end(),
		     0.f);

	//this->num_batches_ is inherited from the Optimiser class
	for (std::int32_t batch_num = 0;
	     batch_num < this->num_batches_;
	     ++batch_num)
	{ //batch_num layer

	    this->calculate_and_record_dldw(batch_num, _numerically_optimised_algorithm);

	    //Update _W
	    thrust::for_each(thrust::make_zip_iterator(
				 thrust::make_tuple(this->dldw_.begin(),
						    this->sum_dldw_squared_.begin(),
						    _W.begin(),
						    this->first_moment_.begin())),
			     thrust::make_zip_iterator(
				 thrust::make_tuple(this->dldw_.end(),
						    this->sum_dldw_squared_.end(),
						    _W.end(),
						    this->first_moment_.end())),
			     OptimiserFunctors::AdamFunctor(this->epoch_num_ + 1, this->learning_rate_, this->decay_mom1_, this->decay_mom2_, this->offset_));

	    //Post-update manipulations are used to impose restrictions on the weights,
	    //such as that all weights must be greater or equal to 0.

	} //batch_num layer

	//Check whether convergence condition is met. If yes, break
	if (this->record_sum_dldw_and_check_convergence(_sum_gradients, _tol))
	    break;

    } //epoch_num layer
}
