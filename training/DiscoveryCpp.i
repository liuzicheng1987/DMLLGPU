%module DiscoveryCpp
%{
#define SWIG_FILE_WITH_INIT
#include "DiscoveryCpp.hpp"
%}

%include numpy.i
%include exception.i
%include stdint.i

%init %{
import_array();
%}


%exception { 
    try {
        $action
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_UnknownError, e.what());
    } 
 }

%include LossFunctions/LossFunctions.i
%include regularisers/regularisers.i
%include optimisers/optimisers.i
%include NeuralNetwork.i
%include RelationalNetwork/RelationalNetwork.i
%include LogicalGates/LogicalGates.i
%include dropout/dropout.i
%include Aggregations/Aggregations.i
