void NadamCpp::min(/*MPI_Comm comm,*/
		   NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
		   thrust::device_vector<float> &_W,
		   const float _tol,
		   const std::int32_t _max_num_epochs,
		   std::vector<float> &_sum_gradients)
{

    // iterations
    std::int32_t t;

    //Initialise biased first moment estimate
    this->est_mom1_b_ = thrust::device_vector<float>(_W.size());

    thrust::fill(this->est_mom1_b_.begin(),
		 this->est_mom1_b_.end(),
		 0.f);

    //Initialise est_mom2_b
    this->est_mom2_b_ = thrust::device_vector<float>(_W.size());

    thrust::fill(this->est_mom2_b_.begin(),
		 this->est_mom2_b_.end(),
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

	    //Initialising and updating momentum schedule before passing to functor

	    // Due to the recommendations in [http://www.cs.toronto.edu/~fritz/absps/momentum.pdf], i.e. warming momentum schedule

	    // Moment decay schedule
	    float m_schedule = 1.f;

	    t = epoch_num_ + 1;
	    this->momentum_cache_t_ = this->beta_1_ * (1.f - 0.5 * pow(t * schedule_decay_, 0.96));
	    this->momentum_cache_t_1_ = this->beta_1_ * (1.f - 0.5 * pow((t + 1) * schedule_decay_, 0.96));
	    this->m_schedule_new_ = m_schedule * this->momentum_cache_t_;
	    this->m_schedule_next_ = m_schedule * this->momentum_cache_t_ * this->momentum_cache_t_1_;
	    m_schedule = this->m_schedule_new_;

	    //Update _W
	    thrust::for_each(thrust::make_zip_iterator(
				 thrust::make_tuple(this->dldw_.begin(),
						    this->est_mom1_b_.begin(),
						    this->est_mom2_b_.begin(),
						    _W.begin())),
			     thrust::make_zip_iterator(
				 thrust::make_tuple(this->dldw_.end(),
						    this->est_mom1_b_.end(),
						    this->est_mom2_b_.end(),
						    _W.end())),
			     OptimiserFunctors::NadamFunctor(this->epoch_num_ + 1,
							     this->learning_rate_,
							     this->beta_1_,
							     this->beta_2_,
							     this->momentum_cache_t_,
							     this->momentum_cache_t_1_,
							     this->m_schedule_new_,
							     this->m_schedule_next_,
							     this->offset_));

	    //Post-update manipulations are used to impose restrictions on the weights,
	    //such as that all weights must be greater or equal to 0.

	} //batch_num layer

	//Check whether convergence condition is met. If yes, break
	if (this->record_sum_dldw_and_check_convergence(_sum_gradients, _tol))
	    break;

    } //epoch_num layer
}
