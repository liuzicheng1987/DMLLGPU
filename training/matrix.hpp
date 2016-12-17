namespace matrix {

//Because the neural network can consist of a number of input matrices, both dense and sparse, we keep them in a vector of structs
struct DenseMatrix {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions

  thrust::device_vector<float> X;//Vector containing data

  float *X_ptr;//Pointer to data contained in X, for convenience and readability
  
};

struct COOVector {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions
  std::int32_t num_non_zero;//Number of non-zero elements in matrix

  thrust::device_vector<float> X_data;//Data vector
  thrust::device_vector<std::int32_t> X_indices;//column indices

  float *X_data_ptr;//Pointer to data contained in X_data
  std::int32_t *X_indices_ptr;//Pointer to data contained in X_indices
 
};

struct CSRMatrix {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions
  std::int32_t num_non_zero;//Number of non-zero elements in matrix

  thrust::device_vector<float> X_data;//Data vector 
  thrust::device_vector<std::int32_t> X_indices;//indices for data 
  thrust::device_vector<std::int32_t> X_indptr;//indptr for data 

  float *X_data_ptr;//Pointer to data contained in X_data, for convenience and readability
  std::int32_t *X_indices_ptr;//Pointer to data contained in X_indices, for convenience and readability
  std::int32_t *X_indptr_ptr;//Pointer to data contained in X_indptr, for convenience and readability

};

}
