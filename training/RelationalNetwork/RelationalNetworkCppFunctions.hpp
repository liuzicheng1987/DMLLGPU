//Finalises the relational network
void RelationalNetworkCpp::finalise() {};
	  
//The purpose of this function is to calculate the gradient of the weights
void RelationalNetworkCpp::dfdw(/*MPI_Comm comm,*/
				float             *_dLdw, 
				const float       *_W, 
				const std::int32_t _batch_begin, 
				const std::int32_t _batch_end, 
				const std::int32_t _batch_size, 
				const std::int32_t _batch_num, 
				const std::int32_t _epoch_num
				) {};

};
