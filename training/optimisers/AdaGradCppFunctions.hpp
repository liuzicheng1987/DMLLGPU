void AdaGradCpp::min(/*MPI_Comm comm,*/
		     NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
		     thrust::device_vector<float> &_W,
		     const float _tol,
		     const std::int32_t _max_num_epochs,
		     std::vector<float> &_sum_gradients)
{

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
						    _W.begin())),
			     thrust::make_zip_iterator(
				 thrust::make_tuple(this->dldw_.end(),
						    this->sum_dldw_squared_.end(),
						    _W.end())),
			     OptimiserFunctors::AdaGradFunctor(this->learning_rate_, this->offset_));

	    //Post-update manipulations are used to impose restrictions on the weights,
	    //such as that all weights must be greater or equal to 0.

	} //batch_num layer

	//Check whether convergence condition is met. If yes, break
	if (this->record_sum_dldw_and_check_convergence(_sum_gradients, _tol))
	    break;

    } //epoch_num layer
}
