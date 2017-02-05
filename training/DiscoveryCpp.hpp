#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
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
#include <time.h>

#include "utils.hpp"
#include "matrix.hpp"

#include "regularisers/RegulariserCpp.hpp"
#include "regularisers/L2RegulariserCpp.hpp"

#include "optimisers/NumericallyOptimisedAlgorithmCpp.hpp"
#include "optimisers/OptimiserFunctors.hpp"
#include "optimisers/OptimiserCpp.hpp"
#include "optimisers/OptimiserCppFunctions.hpp"
#include "optimisers/SGDCpp.hpp"
#include "optimisers/SGDCppFunctions.hpp"
#include "optimisers/AdaGradCpp.hpp"
#include "optimisers/AdaGradCppFunctions.hpp"
#include "optimisers/AdaDeltaCpp.hpp"
#include "optimisers/AdaDeltaCppFunctions.hpp"
#include "optimisers/RMSPropCpp.hpp"
#include "optimisers/RMSPropCppFunctions.hpp"
#include "optimisers/AdamCpp.hpp"
#include "optimisers/AdamCppFunctions.hpp"
#include "optimisers/NadamCpp.hpp"
#include "optimisers/NadamCppFunctions.hpp"

#include "NeuralNetworkNodeCpp.hpp"
#include "NeuralNetworkCpp.hpp"

#include "LossFunctions/LossFunctionCpp.hpp"
#include "LossFunctions/SquareLossCpp.hpp"

#include "NeuralNetworkNodeCppFunctions.hpp"
#include "NeuralNetworkCppNonParallelFunctions.hpp"
#include "NeuralNetworkCppParallelFunctions.hpp"

#include "RelationalNetwork/RelationalNetworkCpp.hpp"
#include "RelationalNetwork/RelationalNetworkCppFunctions.hpp"

#include "ActivationFunctions/ActivationFunctions.hpp"

#include "ActivationFunctions/ActivationFunctionCpp.hpp"
#include "ActivationFunctions/ActivationFunctionCppFunctions.hpp"

#include "ActivationFunctions/LogisticActivationFunctionCpp.hpp"
#include "ActivationFunctions/LinearActivationFunctionCpp.hpp"
#include "ActivationFunctions/SoftmaxActivationFunctionCpp.hpp"

#include "Aggregations/ScatterCpp.hpp"
#include "Aggregations/ScatterCppFunctions.hpp"

#include "Aggregations/AggregationFunctors.hpp"
#include "Aggregations/AggregationCpp.hpp"
#include "Aggregations/AggregationCppFunctions.hpp"
#include "Aggregations/SumCpp.hpp"
#include "Aggregations/SumCppFunctions.hpp"
#include "Aggregations/CountCpp.hpp"
#include "Aggregations/CountCppFunctions.hpp"
#include "Aggregations/AvgCpp.hpp"
#include "Aggregations/AvgCppFunctions.hpp"
#include "Aggregations/FirstCpp.hpp"
#include "Aggregations/FirstCppFunctions.hpp"
#include "Aggregations/LastCpp.hpp"
#include "Aggregations/LastCppFunctions.hpp"

#include "dropout/DropoutFunctors.hpp"
#include "dropout/DropoutCpp.hpp"
#include "dropout/DropoutCppFunctions.hpp"
#include "dropout/NodeSamplerCpp.hpp"
#include "dropout/NodeSamplerCppFunctions.hpp"

#include "LogicalGates/LogicalGateFunctors.hpp"
#include "LogicalGates/LogicalGateCpp.hpp"
#include "LogicalGates/LogicalGateCppFunctions.hpp"

#include "LogicalGates/ANDGateCpp.hpp"
#include "LogicalGates/ORGateCpp.hpp"
#include "LogicalGates/XORGateCpp.hpp"
#include "LogicalGates/XNORGateCpp.hpp"
#include "LogicalGates/NORGateCpp.hpp"
#include "LogicalGates/NANDGateCpp.hpp"