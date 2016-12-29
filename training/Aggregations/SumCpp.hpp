#ifndef SUMCPP_HPP_
#define SUMCPP_HPP_

class SumCpp {

private:

	//One, if element is included in aggregation,
	//zero otherwise
	thrust::device_vector<float> included_in_aggregation_;

	//Pointer to included_in_aggregation_
	float *included_in_aggregation_ptr_;

public:

	SumCpp(
			std::int32_t _node_number,
			std::int32_t _dim,
			std::int32_t _input_network
			);

	~SumCpp();

	//One, if element is included in aggregation,
	//zero otherwise
	thrust::device_vector<float> included_in_aggregation_;


	//Calculate the output of the node
	void calc_output(
			const std::int32_t _batch_num,
			const std::int32_t _batch_size
			);

	//Calculate the delta of the node (which is used for backpropagation)
	void calc_delta(std::int32_t _batch_size);

	//dLdw is not needed - sum aggregation is weightless!


};

#endif /* SUMCPP_HPP_ */
