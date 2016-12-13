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

%include NeuralNetwork.i
 //%include LogicalGates/LogicalGates.i
