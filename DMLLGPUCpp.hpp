#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <iostream>
#include <functional>
#include <math.h>
#include <random>
#include <vector>
#include <memory>
#include <cstdint>

#include "utils.hpp"

#include "LossFunctions/LossFunctionCpp.hpp"
#include "LossFunctions/SquareLossCpp.hpp"

#include "regularisers/RegulariserCpp.hpp"
#include "regularisers/L2RegulariserCpp.hpp"

#include "NeuralNetworkNodeGPUCpp.hpp"
#include "NeuralNetworkGPUCpp.hpp"

#include "optimisers/OptimiserCpp.hpp"
#include "optimisers/OptimiserCppFunctions.hpp"
#include "optimisers/SGDCpp.hpp"
#include "optimisers/SGDCppFunctions.hpp"

#include "NeuralNetworkNodeGPUCppFunctions.hpp"
#include "NeuralNetworkGPUCppNonParallelFunctions.hpp"
#include "NeuralNetworkGPUCppParallelFunctions.hpp"

#include "ActivationFunctions/ActivationFunctions.hpp"

#include "ActivationFunctions/ActivationFunctionGPUCpp.hpp"
#include "ActivationFunctions/ActivationFunctionGPUCppFunctions.hpp"

#include "ActivationFunctions/LogisticActivationFunctionGPUCpp.hpp"
#include "ActivationFunctions/LinearActivationFunctionGPUCpp.hpp"
#include "ActivationFunctions/SoftmaxActivationFunctionGPUCpp.hpp"
