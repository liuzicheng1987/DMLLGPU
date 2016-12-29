#ifndef RELATIONALNETWORKCPPFUNCTIONS_HPP_
#define RELATIONALNETWORKCPPFUNCTIONS_HPP_

void RelationalNetworkCpp::set_join_keys_left(
		std::int32_t *_join_keys_left,
		std::int32_t  _join_keys_left_length
	  ) {

	  this->join_keys_left_.push_back(
		std::vector<std::int32_t>(
			_join_keys_left,
			_join_keys_left + _join_keys_left_length
		);
	  );

	  //Size of join_keys_left_ cannot be greater than
	  //size of input_networks_
	  //(but can be smaller)
	  if (
			  this->join_keys_left_.size()
			  > this->input_networks_.size()
	  )
		  throw std::invalid_argument(
				  "Length of join_keys_left cannot be greater than size of input_networks!"
				  );

}

void RelationalNetworkCpp::set_join_keys_used(
		std::int32_t *_join_keys_used,
		std::int32_t  _join_keys_used_length
	  ) {

	  this->join_key_used_ = std::vector<std::int32_t>(
			  _join_keys_used,
			  _join_keys_used_length
	  );

	  //Size of join_key_used_ must equal size of input_networks_
	  if (
		this->join_key_used_.size()
		!= this->input_networks_.size()
	  )
		  throw std::invalid_argument(
				  "Length of join_key_used must be equal to size of input_networks!"
				  );

	  //Elements contained in join_key_used cannot be negative
	  //and cannot exceed size of join_keys_left_
	  if(
		std::any_of(
				this->join_keys_used_.begin(),
				this->join_keys_used_.end(),
				[] (std::int32_t i) {
		  	  	  return (
		  	  			  i < 0
		  	  			  || i >= static_cast<std::int32_t>(
		  	  					  this_join_keys_left_size()
		  	  					  )
		  	  			  );
	            }
	    )
	  ) {
		  throw std::invalid_argument(
			"Element of join_key_used out of range!"
		  );
	  }

  }


#endif /* RELATIONALNETWORKCPPFUNCTIONS_HPP_ */
