void NadamCpp::min(/*MPI_Comm comm,*/
		   NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
		   thrust::device_vector<float> &_W,
		   const float _tol,
		   const std::int32_t _max_num_epochs,
		   std::vector<float> &_sum_gradients)
{

    std::int32_t batch_begin, batch_end, batch_size, global_batch_size;

    //The sum of all sum of gradients - will be recorded in _sum_gradients
    float sum_gradients;

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

	    //We must find out our current values for batch_begin and batch_end.
	    //We do so by calling this->calc_batch_begin_end,
	    //which is inherited from the optimiser class.
	    _numerically_optimised_algorithm->calc_batch_begin_end(batch_begin,
								   batch_end,
								   batch_size,
								   batch_num,
								   this->num_samples_,
								   this->num_batches_);

	    //Calculate global_batch_size
	    global_batch_size = batch_size;

	    //Init this->dldw_
	    //VERY IMPORTANT CONVENTION: Optimisers must set dldw to zero before
	    //passing it to the neural network!
	    thrust::fill(this->dldw_.begin(),
			 this->dldw_.end(),
			 0.f);

	    //You must also recast, the pointer, because thrust::fill
	    //sometimes completely reallocates the vector
	    this->dldw_ptr_ = thrust::raw_pointer_cast(this->dldw_.data());

	    //Barrier: Wait until all processes have reached this point
	    //MPI_Barrier(comm);
	    //Call dfdw()
	    //Note that it is the responsibility of whoever writes the underlying algorithm
	    //to make sure that this->dLdW and this->SumdLdW are passed to ALL processes
	    //It is, however, your responsibility to place a barrier after that, if required
	    _numerically_optimised_algorithm->dfdw(/*comm,*/
						   this->dldw_ptr_,
						   w_ptr_,
						   batch_begin,
						   batch_end,
						   batch_size,
						   batch_num,
						   epoch_num_);

	    //Add all batch_size and store the result in global_batch_size
	    //MPI_Allreduce(&batch_size, &global_batch_size, 1, MPI_INT, MPI_SUM, comm);
	    //global_batch_sizeFloat = (float)global_batch_size;

	    //Barrier: Wait until all processes have reached this point
	    //MPI_Barrier(comm);

	    //Record sum_dldw for sum_gradients
	    thrust::transform(dldw_.begin(),
			      dldw_.end(),
			      this->sum_dldw_.begin(),
			      this->sum_dldw_.begin(),
			      thrust::plus<float>());

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
							     this->decay_mom1_,
							     this->decay_mom2_,
							     this->schedule_decay_,
							     this->offset_));

	    //Post-update manipulations are used to impose restrictions on the weights,
	    //such as that all weights must be greater or equal to 0.

	} //batch_num layer

	//Record sum_dldw
	sum_gradients = thrust::transform_reduce(
	    this->sum_dldw_.begin(),
	    this->sum_dldw_.end(),
	    utils::square<float>(),
	    0.f,
	    thrust::plus<float>());

	//sum_gradients != sum_gradients means sum_gradients is nan
	if (sum_gradients != sum_gradients)
	    throw std::invalid_argument(
		"The gradients seem to have spun out of control! You might want to reduce the learning rate!");

	_sum_gradients.push_back(sum_gradients);

	//Check whether convergence condition is met. num_samplesf yes, break
	if (sum_gradients / (static_cast<float>(_W.size())) < _tol)
	    break;

    } //epoch_num layer
}