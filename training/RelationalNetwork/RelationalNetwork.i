//----------------------------------------------------------------------------------------------
//class RelationalNetworkCpp

%apply (float* IN_ARRAY1, int DIM1) {(float *_W, std::int32_t _length_W)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_X, std::int32_t _num_samples, std::int32_t _dim)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_X_data, std::int32_t _X_data_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_X_indices, std::int32_t _X_indices_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_X_indptr, std::int32_t _X_indptr_length)};

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_indptr, std::int32_t _indptr_length)};

%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_join_keys_left, std::int32_t _join_keys_left_length)};

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *_Y, std::int32_t _num_samples, std::int32_t _dim)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_Y_data, std::int32_t _Y_data_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_Y_indices, std::int32_t _Y_indices_length)};
%apply (int* IN_ARRAY1, int DIM1) {(std::int32_t *_Y_indptr, std::int32_t _Y_indptr_length)};

%apply (float* IN_ARRAY1, int DIM1) {(float *_join_keys_left, std::int32_t _join_keys_left_length)};

%apply (float* IN_ARRAY1, int DIM1) {(float *_time_stamps_output, std::int32_t _time_stamps_output_length)};
%apply (float* IN_ARRAY1, int DIM1) {(float *_time_stamps_input, std::int32_t _time_stamps_input_length)};

%apply (float* IN_ARRAY1, int DIM1) {(float *_sum_gradients, std::int32_t _sum_gradients_size)};

class RelationalNetworkCpp: public NumericallyOptimisedAlgorithmCpp {
		
public:
	
  RelationalNetworkCpp ();
	
  ~RelationalNetworkCpp();

  	//Adds an input network to the relational network
	//Cannot be removed, once added
    void add_input_network(NeuralNetworkCpp *_input_network,
                           std::int32_t _join_key_used);

	//Sets the output network
    void set_output_network(NeuralNetworkCpp *_output_network);
  
    //Finalises the relational network
    void finalise(float _weight_init_range);

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

	//User needs to provide the keys (in form of integers) to connect input networks to
	//output network
    void add_join_keys_left(std::int32_t *_join_keys_left,
                            std::int32_t _join_keys_left_length);

    //This function needs to be called at the end of fit and transform
    void clean_up();
                            
    //The purpose of this function is to fit the relational network through
    //backpropagation
    void fit(/*MPI_Comm comm,*/
             OptimiserCpp *_optimiser, std::int32_t _global_batch_size,
             const float _tol, const std::int32_t _max_num_epochs,
             const bool _sample);                            

    //The purpose of this function is to generate a prediction through the fitted network
    void transform(float *_Yhat, std::int32_t _Y2_num_samples,
                   std::int32_t _Y2_dim, bool _sample, std::int32_t _sample_size,
                   bool _get_hidden_nodes);		

    //Getters and setters
    std::int32_t get_sum_output_dim();

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

//----------------------------------------------------------------------------------------------
