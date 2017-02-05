OptimiserCpp::OptimiserCpp(/*const std::int32_t _size, const std::int32_t _rank*/)
{

  this->epoch_num_ = 0;

  /*this->size = _size;
      this->rank = _rank;*/
}

OptimiserCpp::~OptimiserCpp() {}

//dev_function_type is defined in file OptimiserCpp.hpp
void OptimiserCpp::minimise(/*MPI_Comm _comm,*/
                            NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm,
                            std::int32_t _num_samples,
                            thrust::device_vector<float> &_W,
                            std::int32_t _global_batch_size,
                            const float _tol,
                            const std::int32_t _max_num_epochs,
                            std::vector<float> &_sum_gradients)
{

  //Store all of the input values
  this->num_samples_ = _num_samples;
  this->global_batch_size_ = _global_batch_size;

  //Set this->w_ptr_
  this->w_ptr_ = thrust::raw_pointer_cast(_W.data());

  //Initialise dLdw and dLdwPtr
  this->dldw_ = thrust::device_vector<float>(_W.size());
  this->dldw_ptr_ = thrust::raw_pointer_cast(this->dldw_.data());

  //Initialise sum_dldw
  this->sum_dldw_ = thrust::device_vector<float>(_W.size());

  //Calculate the number of batches needed
  this->num_batches_ = _numerically_optimised_algorithm->calc_num_batches(
      _num_samples,
      _global_batch_size);

  //Create the threads and pass the values they need
  this->min(/*comm,*/
            _numerically_optimised_algorithm,
            _W,
            _tol,
            _max_num_epochs,
            _sum_gradients);

  //Clear this->dLdw and this->SumdLdw, so they don't take up space on the
  this->dldw_.clear();
  this->sum_dldw_.clear();
}

void OptimiserCpp::calculate_and_record_dldw(std::int32_t _batch_num,
                                             NumericallyOptimisedAlgorithmCpp *_numerically_optimised_algorithm)
{

  std::int32_t batch_begin, batch_end, batch_size;

  //We must find out our current values for batch_begin and batch_end.
  //We do so by calling this->calc_batch_begin_end,
  //which is inherited from the optimiser class.
  _numerically_optimised_algorithm->calc_batch_begin_end(batch_begin,
                                                         batch_end,
                                                         batch_size,
                                                         _batch_num,
                                                         this->num_samples_,
                                                         this->num_batches_);

  //Calculate global_batch_size
  //global_batch_size = batch_size;

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
                                         _batch_num,
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
}

bool OptimiserCpp::record_sum_dldw_and_check_convergence(std::vector<float> &_sum_gradients,
                                                         float _tol)
{

  //Record sum_dldw
  float sum_gradients = thrust::transform_reduce(
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

  //Check whether convergence condition is met.
  return (sum_gradients / (static_cast<float>(this->sum_dldw_.size())) < _tol);
}
