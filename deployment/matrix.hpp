namespace matrix {

//Because the neural network can consist of a number of input matrices, both dense and sparse, we keep them in a vector of structs
struct DenseMatrix {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions

  float *X_ptr;//Pointer to data contained in matrix
  
};

struct CSRMatrix {

  std::int32_t batch_size;//Number of samples
  std::int32_t dim;//Number of dimensions
  std::int32_t num_non_zero;//Number of non-zero elements in matrix

  std::vector<std::int32_t> X_indptr;//Index pointer for sparse matrices

  float *X_data_ptr;//Pointer to data contained in matrix
  std::int32_t *X_indices_ptr;//Pointer to indices contained in matrix
  std::int32_t *X_indptr_ptr;//Pointer to index pointer contained in matrix

};

  void matrix_multiplication(
			     const bool         transa,
			     const std::int32_t dim0,
			     const std::int32_t dim1,
			     const std::int32_t dim2,
			     const float        alpha,
			     const float       *A,
			     const float       *B,
			     const float        beta,
			     float             *C
			     ) {

    //Initialise C
    for (std::int32_t i; i < dim0*dim1; ++i)
      C[i] *= beta; 
   
    if (transa) {

      //AT: (dim0 X dim2)-matrix 
      //B: (dim2 X dim1)-matrix 
      //C: (dim0 X dim1)-matrix 

      //Matrix multiplication
      for (std::int32_t j = 0; j < dim1; ++j)
	for (std::int32_t i = 0; i < dim0; ++i)
	  for (std::int32_t k = 0; k < dim2; ++k)
	    C[dim0*j + i] += alpha*A[dim2*i + k]*B[dim2*j + k];
      

    } else {

      //A: (dim0 X dim2)-matrix 
      //B: (dim2 X dim1)-matrix 
      //C: (dim0 X dim1)-matrix 

      //Matrix multiplication
      for (std::int32_t k = 0; k < dim2; ++k)
	for (std::int32_t j = 0; j < dim1; ++j)
	  for (std::int32_t i = 0; i < dim0; ++i)
	    C[dim0*j + i] += alpha*A[dim0*k + i]*B[dim2*j + k];
      
    }
    
  }

  void matrix_multiplication_sparse(
				    const std::int32_t  dim0,
				    const std::int32_t  dim1,
				    const std::int32_t  dim2,
				    const std::int32_t  num_non_zero,
				    const float         alpha,
				    const float        *A_data,
				    const std::int32_t *A_indptr,
				    const std::int32_t *A_indices,
				    const float        *B,
				    const float         beta,
				    float              *C
				    ) {

    //Initialise C
    for (std::int32_t i; i < dim0*dim1; ++i)
      C[i] *= beta; 
    
    for (std::int32_t i = 0; i < dim0; ++i)
      for (std::int32_t j = 0; j < dim1; ++j)
	for (
	     std::int32_t k = A_indptr[i]; 
	     k < A_indptr[i+1]; 
	     ++k
	     ) 
	  C[dim0*j + i] += alpha*A_data[k]*B[dim2*j + A_indices[k]];
	
  }
}
