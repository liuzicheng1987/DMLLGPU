"""
discovery: Neural network module that has been specifically
developed for business applications.
"""

import numpy as np
import warnings
from datetime import datetime
import DiscoveryCpp

# Optimisers


class SGD(object):
    """
    Stochastic gradient descent optimiser.
    """

    def __init__(
            self,
            learning_rate,
            learning_rate_power,
            momentum=0.0):
        """
        learning_rate: Learning rate of the optimisation problem
        learning_rate_power: Speed by which learning rate decreases
        momentum: Momentum of the optimiser
        """
        self.thisptr = DiscoveryCpp.SGDCpp(
            learning_rate,
            learning_rate_power,
            momentum)


class AdaGrad(object):
    """
    Adaptive gradient optimiser.
    """

    def __init__(self, learning_rate):
        """
        learning_rate: Learning rate of the optimisation problem
        """
        self.thisptr = DiscoveryCpp.AdaGradCpp(learning_rate)


class RMSProp(object):
    """
    RMSProp optimiser.
    """

    def __init__(self, learning_rate, gamma=0.9):
        """
        learning_rate: Learning rate of the optimisation problem
        gamma: Decay rate of the previous squared gradients
        """
        self.thisptr = DiscoveryCpp.RMSPropCpp(learning_rate, gamma)


class Adam(object):
    """
    Adam optimiser.
    """

    def __init__(self, learning_rate=0.0002, decay_mom1=0.9, 
        decay_mom2=0.999, offset=1e-08):
        """
        learning_rate: Learning rate parameter
        decay_mom1: Exponential decay parameter for first moment estimate
        decay_mom2: Exponential decay parameter for second moment estimate
        offset: Safety offset for division by estimate of second moment
        """
        self.thisptr = DiscoveryCpp.AdamCpp(learning_rate, decay_mom1, decay_mom2, offset)


class Nadam(object):
    """
    Adam optimiser with Nesterov momentum
    //Default params Keras: lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004
    """

    def __init__(self, learning_rate=0.002, decay_mom1=0.9, 
        decay_mom2=0.999, schedule_decay=0.004, offset=1e-08):
        """
        learning_rate: Learning rate parameter
        decay_mom1: Exponential decay parameter for first moment estimate
        decay_mom2: Exponential decay parameter for second moment estimate
        offset: Safety offset for division by estimate of second moment
        """
        self.thisptr = DiscoveryCpp.AdamCpp(learning_rate, decay_mom1, decay_mom2, offset)

# Loss functions


class SquareLoss(object):
    """
    Squared loss function.
    """

    def __init__(self):
        self.thisptr = DiscoveryCpp.SquareLossCpp()

# Regularisers


class Regulariser(object):
    """
    Regulariser base class.
    When this class is used, then no regularisation
    is applied.
    """

    def __init__(self):
        """
        alpha: Weight associated with regulariser
        """
        self.thisptr = DiscoveryCpp.RegulariserCpp()


class L2Regulariser(object):
    """
    L2 norm regulariser.
    """

    def __init__(self, alpha=1.0):
        """
        alpha: Weight associated with regulariser
        """
        self.thisptr = DiscoveryCpp.L2RegulariserCpp(alpha)

# Activation functions


class ActivationFunction(object):
    """
    Activation function node.
    """

    def __init__(
            self,
            node_number,
            dim=1,
            activation="logistic",
            input_dense=np.asarray([]).astype(np.int32),
            input_sparse=np.asarray([]).astype(np.int32),
            hidden=np.asarray([]).astype(np.int32),
            i_share_weights_with=-1,
            no_weight_updates=False,
            regulariser=Regulariser()
    ):
        """
        node_number: Number of node in the neural network. Every node
        number must be assigned before neural network is finalised.
        dim: Dimension of the output
        activation: Activation function for the node.
        Expects one of the following:
        "logistic", "linear", "rectified linear", "probit", "laplace",
        "cauchy", "exponential", "tanh", "pareto", "weibull".
        hidden: List of hidden nodes (in form of their respective
        node_numbers) that are fed into this node. All of these nodes
        must have a node number that is smaller than the node number of
        this node. Defaults to zero-length array (meaning that no hidden
        nodes are fed into this node).
        input: List of input nodes (in form of their respective
        node_numbers) that are fed into this node. Defaults to zero-length
        array (meaning that no input nodes are fed into this node).
        i_share_weights_with: Node number of a node that this node share
        weights with. -1 if it shares weights with no node. Defaults to -1.
        no_weight_updates: True if this node is to receive no updates during
        the training process.
        regulariser: Regulariser object
        """

        self.node_number = node_number

        # Store regulariser
        self.regulariser = regulariser

        if activation == "logistic":
            self.thisptr = DiscoveryCpp.LogisticActivationFunctionCpp(
                self.node_number,
                dim,
                np.asarray(input_dense).astype(np.int32),
                np.asarray(input_sparse).astype(np.int32),
                np.asarray(hidden).astype(np.int32),
                i_share_weights_with,
                no_weight_updates,
                self.regulariser.thisptr
            )

        elif activation == "linear":
            self.thisptr = DiscoveryCpp.LinearActivationFunctionCpp(
                self.node_number,
                dim,
                np.asarray(input_dense).astype(np.int32),
                np.asarray(input_sparse).astype(np.int32),
                np.asarray(hidden).astype(np.int32),
                i_share_weights_with,
                no_weight_updates,
                self.regulariser.thisptr
            )


class SoftmaxActivationFunction(object):
    """
    Softmax activation function node.
    Has its own class, because it requires more parameters than
    other activation functions.
    """

    def __init__(
            self,
            node_number,
            num_vars=1,
            num_states_per_var=5,
            input_dense=np.asarray([]).astype(np.int32),
            input_sparse=np.asarray([]).astype(np.int32),
            hidden=np.asarray([]).astype(np.int32),
            i_share_weights_with=-1,
            no_weight_updates=False,
            regulariser=Regulariser()
    ):
        """
        node_number: Number of node in the neural network.
        Every node number must be assigned before neural
        network is finalised.
        num_vars: Number of distinct discrete variables
        (output dimension is num_vars*num_states_per_var)
        num_states_per_var: Number of states each variable
        can assume (output dimension is num_vars*num_states_per_var)
        hidden: List of hidden nodes (in form of their respective
        node_numbers) that are fed into this node. All of these
        nodes must have a node number that is smaller than the node
        number of this node. Defaults to zero-length array
        (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective
        node_numbers) that are fed into this node. Defaults to
        zero-length array (meaning that no input nodes are fed
        into this node).
        i_share_weights_with: Node number of a node that this
        node share weights with. -1 if it shares weights with
        no node. Defaults to -1.
        no_weight_updates: True if this node is to receive no
        updates during the training process.
        regulariser: Regulariser object
        """

        self.node_number = node_number

        # Store regulariser
        self.regulariser = regulariser

        self.thisptr = DiscoveryCpp.SoftmaxActivationFunctionCpp(
            self.node_number,
            num_vars,
            num_states_per_var,
            np.asarray(input_dense).astype(np.int32),
            np.asarray(input_sparse).astype(np.int32),
            np.asarray(hidden).astype(np.int32),
            i_share_weights_with,
            no_weight_updates,
            self.regulariser.thisptr
        )

# Dropout


class Dropout(object):
    """
    Dropout node.
    Randomly removes nodes from network.
    """

    def __init__(
            self,
            node_number,
            dropout_probability=0.5,
            numbers_per_kernel=10000,
            num_kernels=1000,
            hidden=np.asarray([]).astype(np.int32)
    ):
        """
        node_number: Number of node in the neural network.
        Every node number must be assigned before neural network is finalised.
        dropout_probability: Probability of node being dropped out.
        numbers_per_kernel: Number of random numbers generated by one kernel
        (only important for performance)
        num_kernels: Number of kernels used when generating random numbers
        (only important for performance)
        hidden: List of hidden nodes (in form of their respective node_numbers)
        that are fed into this node. All of these nodes must have a node number
        that is smaller than the node number of this node. Defaults
        to zero-length array
        (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective node_numbers)
        that are fed into this node. Defaults to zero-length array
        (meaning that no input nodes are fed into this node).
        """

        self.node_number = node_number

        self.thisptr = DiscoveryCpp.DropoutCpp(
            self.node_number,
            dropout_probability,
            numbers_per_kernel,
            num_kernels,
            np.asarray(hidden).astype(np.int32)
        )


class NodeSampler(object):
    """
    NodeSampler node.
    Generate samples based on the nodes fed into NodeSampler.
    """

    def __init__(
            self,
            node_number,
            numbers_per_kernel=10000,
            num_kernels=1000,
            hidden=np.asarray([]).astype(np.int32)
    ):
        """
        node_number: Number of node in the neural network.
        Every node number must be assigned before neural network is finalised.
        numbers_per_kernel: Number of random numbers generated by one kernel
        (only important for performance)
        num_kernels: Number of kernels used when generating random number
        (only important for performance)
        hidden: List of hidden nodes (in form of their respective node_numbers)
        that are fed into this node. All of these nodes must have a node number
        that is smaller than the node number of this node. Defaults to
        zero-length array (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective node_numbers)
        that are fed into this node. Defaults to zero-length array (meaning
        that no input nodes are fed into this node).
        """

        self.node_number = node_number

        self.thisptr = DiscoveryCpp.NodeSamplerCpp(
            self.node_number,
            numbers_per_kernel,
            num_kernels,
            np.asarray(hidden).astype(np.int32)
        )

# Logical gates


class LogicalGate(object):
    """
    LogicalGate node.
    Logical gates are a weightless transformation that make it easier to
    interpret the neural network.
    The logical gate is applied dimension-wise.
    """

    def __init__(
            self,
            node_number,
            dim,
            hidden,
            activation="AND"
    ):
        """
        The logical gate is applied dimension-wise.
        node_number: Number of node in the neural network. Every node number
        must be assigned before neural network is finalised.
        hidden: List of hidden nodes (in form of their respective node_numbers)
        that are fed into this node. All of these nodes must have a node number
        that is smaller than the node number of this node. Number of hidden
        nodes fed into LogicalGate must be between 2 and 4.
        dim: Dimension of the output. Must be identical to dimensions of all
        input nodes.
        activation: Activation function for the node.
        Expects one of the following:
        "AND", "OR", "XOR", "XNOR", "NAND", "NOR".
        """

        self.node_number = node_number

        if activation == "AND":
            self.thisptr = DiscoveryCpp.ANDGateCpp(
                self.node_number,
                dim,
                hidden
            )

        elif activation == "OR":
            self.thisptr = DiscoveryCpp.ORGateCpp(
                self.node_number,
                dim,
                hidden
            )

        elif activation == "XOR":
            self.thisptr = DiscoveryCpp.XORGateCpp(
                self.node_number,
                dim,
                hidden
            )

        elif activation == "XNOR":
            self.thisptr = DiscoveryCpp.XNORGateCpp(
                self.node_number,
                dim,
                hidden
            )

        elif activation == "NOR":
            self.thisptr = DiscoveryCpp.NORGateCpp(
                self.node_number,
                dim,
                hidden
            )

        elif activation == "NAND":
            self.thisptr = DiscoveryCpp.NANDGateCpp(
                self.node_number,
                dim,
                hidden
            )

        else:
            warnings.warn(
                "Logical gate of type '"
                + activation
                + "' not known!"
            )

# Aggregations


class StandardAggregation(object):
    """
    StandardAggregation node.
    Standard aggregations replicate basic SQL aggregation in relational
    nets. They are weightless.
    The standard aggregation is applied dimension-wise.
    """

    def __init__(
            self,
            node_number,
            dim,
            input_network,
            use_timestamps,
            aggregation="SUM"
    ):
        """
        node_number: Number of node in the neural network. Every node number
        must be assigned before neural network is finalised.
        dim: Dimension of the output. Must be identical to output dimension of
        input network, unless aggregation is "COUNT", in which case dim will
        be ignored, since the dimension of count aggregations always equals one.
        input_network: Integer signifying which input network is fed into
        aggregation.
        use_timestamps: Boolean signifying whether we the aggregation should
        consider timestamps
        aggregation: Signifies which aggregation to use.
        Expects one of the following:
        "SUM", "COUNT", "AVG", "MIN", "MAX", "FIRST", "LAST"
        """

        self.node_number = node_number

        if aggregation == "SUM":
            self.thisptr = DiscoveryCpp.SumCpp(
                self.node_number,
                dim,
                input_network,
                use_timestamps
            )

        elif aggregation == "COUNT":
            self.thisptr = DiscoveryCpp.CountCpp(
                self.node_number,
                input_network,
                use_timestamps
            )

        elif aggregation == "AVG":
            self.thisptr = DiscoveryCpp.AvgCpp(
                self.node_number,
                dim,
                input_network,
                use_timestamps
            )

        elif aggregation == "FIRST":
            self.thisptr = DiscoveryCpp.FirstCpp(
                self.node_number,
                dim,
                input_network,
                use_timestamps
            )

        elif aggregation == "LAST":
            self.thisptr = DiscoveryCpp.LastCpp(
                self.node_number,
                dim,
                input_network,
                use_timestamps
            )

        else:
            warnings.warn(
                "Standard aggregation of type '"
                + aggregation
                + "' not known!"
            )

# Neural network


class NeuralNetwork(object):
    """
    NeuralNetwork class.
    Basic feed-forward neural network.
    """

    def __init__(
            self,
            num_input_nodes_dense=[],
            num_input_nodes_sparse=[],
            num_output_nodes_dense=0,
            num_output_nodes_sparse=0,
            names_input=np.zeros(0).astype(str),
            loss=SquareLoss()
    ):
        """
        Initialise neural network.
        num_input_nodes_dense: Number of dimensions for dense inputs.
        For instance, if your inputs consist of two dense matrices with
        300 and 400 nodes respectively, then num_input_nodes_dense=[300,400].
        num_input_nodes_sparse: Number of dimensions for sparse inputs. For
        instance, if your inputs consist of two sparse matrices with 3000
        and 4000 nodes respectively, then num_input_nodes_dense=[3000,4000].
        (It is possible to have dense and sparse inputs at the same time.)
        LossFunction: Loss function object
        num_output_nodes: Number of output nodes
        Names
        """
        self.num_input_nodes_dense = num_input_nodes_dense
        self.num_input_nodes_sparse = num_input_nodes_sparse
        self.num_input_nodes_dense_length = len(num_input_nodes_dense)
        self.num_input_nodes_sparse_length = len(num_input_nodes_sparse)
        self.num_hidden_nodes = 0
        self.num_output_nodes_dense = num_output_nodes_dense
        self.num_output_nodes_sparse = num_output_nodes_sparse
        self.loss = loss

        # Initialise list of nodes
        self.nodes = []
        for i in range(
                self.num_output_nodes_dense
                + self.num_output_nodes_sparse
                + self.num_hidden_nodes
        ):
            self.nodes += [None]

        # Initialise NeuralNetworkCpp
        self.thisptr = DiscoveryCpp.NeuralNetworkCpp(
            num_input_nodes_dense,
            num_input_nodes_sparse,
            num_output_nodes_dense,
            num_output_nodes_sparse,
            self.loss.thisptr
        )

    def init_hidden_node(self, hidden_node, name=""):
        """
        Initialise hidden node in the neural network.
        hidden_node: Discovery.NeuralNetworkNode object
        name: Name of the node (optional)
        """

        # If node number exceeds number of hidden nodes so far, then extend
        # hidden nodes and names accordingly
        if hidden_node.node_number >= self.num_hidden_nodes:
            NumAdditionalNodes = hidden_node.node_number + 1
            - self.num_hidden_nodes
            for i in range(NumAdditionalNodes):
                self.nodes.insert(self.num_hidden_nodes, None)

            # Increase num_hidden_nodes
            self.num_hidden_nodes += NumAdditionalNodes

            # Increase node_number of output_nodes
            for i in range(self.num_output_nodes_dense
                           + self.num_output_nodes_sparse):
                if type(self.nodes[self.num_hidden_nodes + i]) != type(None):
                    self.nodes[self.num_hidden_nodes +
                               i].node_number += NumAdditionalNodes

        # Store node, both in Python and C++
        self.nodes[hidden_node.node_number] = hidden_node
        self.thisptr.init_hidden_node(
            self.nodes[hidden_node.node_number].thisptr)

    def init_output_node(self, output_node, name=""):
        """
        Initialise output node in the neural network.
        output_node: Discovery.NeuralNetworkNode object
        name: Name of the node (optional)
        """

        # Store node, both in Python and C++
        self.nodes[output_node.node_number] = output_node
        self.thisptr.init_output_node(
            self.nodes[output_node.node_number].thisptr)

    def finalise(self, weight_init_range=0.7):
        """
        Prepares neural network for fitting.
        weight_init_range: Range within which the
        weights are randomly initialised
        """
        self.thisptr.finalise(weight_init_range)

    def get_input_dense(self, node_number):
        """
        Get the dense input nodes which are fed into node.
        node_number: Node number of the node we are interested in
        """
        input_dense = np.zeros(
            self.thisptr.get_input_nodes_fed_into_me_dense_length(node_number)
        ).astype(np.int32)

        self.thisptr.get_input_nodes_fed_into_me_dense(
            node_number,
            input_dense
        )
        return input_dense

    def get_input_sparse(self, node_number):
        """
        Get the sparse input nodes which are fed into node.
        node_number: Node number of the node we are interested in
        """
        input_sparse = np.zeros(
            self.thisptr.get_input_nodes_fed_into_me_dense_length(node_number)
        ).astype(np.int32)
        self.thisptr.get_input_nodes_fed_into_me_dense(
            node_number,
            input_sparse
        )
        return input_sparse

    def get_hidden(self, node_number):
        """
        Get the hidden nodes which are fed into node.
        node_number: Node number of the node we are interested in
        """
        hidden = np.zeros(self.thisptr.get_hidden_nodes_fed_into_me_length(
            node_number)).astype(np.int32)
        self.thisptr.get_hidden_nodes_fed_into_me(node_number, hidden)
        return hidden

    def get_params(self):
        """
        Get the weights that have been trained
        """
        params = np.zeros(self.thisptr.get_length_params()).astype(np.float32)
        self.thisptr.get_params(params)
        return params

    def set_params(self, params):
        """
        Set the weights of the neural network
        """
        self.thisptr.set_params(params)

    def fit(
            self,
            Xdense=[],
            Xsparse=[],
            Ydense=[],
            Ysparse=[],
            optimiser=SGD(1.0, 0.0),
            global_batch_size=200,
            tol=1e-08,
            max_num_epochs=200,
            MinibatchSizeStandard=20,
            sample=True,
            root=0
    ):
        """
        Fit the neural network to training data.
        Xdense: Array of numpy.arrays. Number of dimensions must match the
        num_input_nodes_dense set when declaring the class and number of
        samples must be identical.
        Xsparse: Array of scipy.sparse.csr_matrices. Number of dimensions
        must match the num_input_nodes_sparse set when declaring the class
        and number of samples must be identical.
        Ydense: Array of numpy.arrays. Number of dimensions must match the
        dimensions of respective output nodes.
        Ysparse: Array of scipy.sparse.csr_matrices. Number of dimensions
        must match the dimensions of respective output nodes.
        optimiser: Optimiser class
        global_batch_size: Approximate size of batch used in each iteration.
        tol: num_samplesf sum of squared gradients is below tol, stop training
        prematurely.
        max_num_epochs: Maximum number of epochs for training
        MinibatchSizeStandard: Number of samples calculated at once
        sample: Whether you want to sample
        root: Number of root process
        """

        # Make sure length of data vectors provided matches lengths that were
        # previously defined
        if len(Xdense) != self.num_input_nodes_dense_length:
            warnings.warn(
                "Number of matrices in Xdense must be identical to "
                + "length of num_input_nodes_dense. Expected "
                + str(self.num_input_nodes_dense_length)
                + ", got "
                + str(len(Xdense))
                + "."
            )

        if len(Xsparse) != self.num_input_nodes_sparse_length:
            warnings.warn(
                "Number of matrices in Xsparse must be "
                + "identical to length of num_input_nodes_sparse. "
                + "Expected "
                + str(self.num_input_nodes_sparse_length)
                + ", got "
                + str(len(Xsparse))
                + "."
            )

        if len(Ydense) != self.num_output_nodes_dense:
            warnings.warn(
                "Number of matrices in Ydense must be identical to "
                + "length of num_output_nodes_dense. Expected "
                + str(self.num_output_nodes_dense)
                + ", got "
                + str(len(Ydense))
                + "."
            )

        if len(Ysparse) != self.num_output_nodes_sparse:
            warnings.warn(
                "Number of matrices in Ysparse must be "
                + "identical to length of num_output_nodes_sparse. "
                + "Expected "
                + str(self.num_output_nodes_sparse)
                + ", got "
                + str(len(Ysparse))
                + "."
            )

        start_timing = datetime.now()

        # Load Xdense into
        for i, X in enumerate(Xdense):
            self.thisptr.load_dense_data(i, X, global_batch_size)

        # Load Xsparse into
        for i, X in enumerate(Xsparse):

            # cuSPARSE expects sorted indices!
            Xsparse_sorted = X.sorted_indices()

            self.thisptr.load_sparse_data(
                i,
                Xsparse_sorted.data,
                Xsparse_sorted.indices,
                Xsparse_sorted.indptr,
                Xsparse_sorted.shape[0],
                Xsparse_sorted.shape[1],
                global_batch_size
            )
            del Xsparse_sorted

        # Load Ydense into
        for i, Y in enumerate(Ydense):
            self.thisptr.load_dense_targets(i, Y, global_batch_size)

        # Load Ysparse into
        for i, Y in enumerate(Ysparse):

            # cuSPARSE expects sorted indices!
            Ysparse_sorted = Y.sorted_indices()

            self.thisptr.load_sparse_targets(
                i,
                Ysparse_sorted.data,
                Ysparse_sorted.indices,
                Ysparse_sorted.indptr,
                Ysparse_sorted.shape[0],
                Ysparse_sorted.shape[1],
                global_batch_size
            )
            del Ysparse_sorted

        stop_timing = datetime.now()
        time_elapsed = stop_timing - start_timing
        print "Neural network loaded data into GPU."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (
            time_elapsed.seconds // 3600,
            (time_elapsed.seconds // 60) % 60,
            time_elapsed.seconds % 60,
            time_elapsed.microseconds
        )
        print

        start_timing = datetime.now()

        self.thisptr.fit(optimiser.thisptr, global_batch_size,
                         tol, max_num_epochs, MinibatchSizeStandard, sample)

        stop_timing = datetime.now()
        time_elapsed = stop_timing - start_timing
        print "Trained neural network."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (
            time_elapsed.seconds // 3600,
            (time_elapsed.seconds // 60) % 60,
            time_elapsed.seconds % 60,
            time_elapsed.microseconds
        )
        print

    def transform(
            self,
            Xdense=[],
            Xsparse=[],
            global_batch_size=200,
            sample=False,
            sample_size=100,
            get_hidden_nodes=False
    ):
        """
        Transform input values to predictions.
        Xdense: Array of numpy.arrays.
        Number of dimensions must match the num_input_nodes_dense set when
        declaring the class and number of samples must be identical.
        Xsparse: Array of scipy.sparse.csr_matrices. Number of dimensions
        must match the num_input_nodes_sparse set when declaring the class
        and number of samples must be identical.
        global_batch_size: Number of batches calculated at once
        sample: Whether you would like to sample (relevant format
        dropout nodes)
        sample_size: Size of the samples
        get_hidden_nodes: Whether you would like to get the hidden
        nodes as well
        """

        # Make sure length of Xdense matches num_input_nodes_dense_length
        if len(Xdense) != self.num_input_nodes_dense_length:
            warnings.warn(
                "Number of matrices in Xdense must be identical "
                + "to length of num_input_nodes_dense. Expected "
                + str(self.num_input_nodes_dense_length)
                + ", got "
                + str(len(Xdense))
                + "."
            )
            return

        if len(Xsparse) != self.num_input_nodes_sparse_length:
            warnings.warn(
                "Number of matrices in Xsparse must be identical"
                + "to length of num_input_nodes_sparse. Expected "
                + str(self.num_input_nodes_sparse_length)
                + ", got "
                + str(len(Xsparse))
                + "."
            )
            return

        # Load dense data into
        for i, X in enumerate(Xdense):
            self.thisptr.load_dense_data(i, X, global_batch_size)

        # Load sparse data into
        for i, X in enumerate(Xsparse):

            # cuSPARSE expects sorted indices!
            Xsparse_sorted = X.sorted_indices()

            self.thisptr.load_sparse_data(
                i,
                Xsparse_sorted.data,
                Xsparse_sorted.indices,
                Xsparse_sorted.indptr,
                Xsparse_sorted.shape[0],
                Xsparse_sorted.shape[1],
                global_batch_size
            )

            del Xsparse_sorted

        if len(Xdense) > 0:
            Yhat = np.zeros((
                Xdense[0].shape[0],
                self.thisptr.get_sum_output_dim()
            )).astype(np.float32)
        else:
            Yhat = np.zeros((
                Xsparse[0].shape[0],
                self.thisptr.get_sum_output_dim()
            )).astype(np.float32)

        # Check whether the input X is a numpy vector
        self.thisptr.transform(
            Yhat,
            sample,
            sample_size,
            get_hidden_nodes
        )

        return Yhat

    def get_sum_gradients(self):

        sum_gradients = np.zeros(
            self.thisptr.get_sum_gradients_length()).astype(np.float32)

        self.thisptr.get_sum_gradients(sum_gradients)

        return sum_gradients


class RelationalNetwork(object):
    """
    RelationalNetwork class.
    Neural network optimised for relational data.
    """

    def __init__(
            self,
            input_networks,
            output_network
    ):
        """
        input_networks:
        output_network:
        join_keys_left:
        """

        self.thisptr = DiscoveryCpp.RelationalNetworkCpp()

        # Add input and index signifying which join key to use
        for i in input_networks:
            self.thisptr.add_input_network(
                i[0].thisptr,
                i[1])

        # Set output network
        self.thisptr.set_output_network(output_network.thisptr)

    def __reorder_join_keys_and_data(
            self,
            X,
            join_keys_input,
            time_stamps_input):

        # Reorder according to join_keys_input
        X_reordered = X[np.lexsort((time_stamps_input, join_keys_input))]

        join_keys_input_reordered = join_keys_input[
            np.lexsort((time_stamps_input, join_keys_input))
        ]

        # Remove any elements in which join_keys_input_reordered
        # is negative
        #(because that means that there is no entry in the index)
        X_reordered = X_reordered[join_keys_input_reordered >= 0]
        join_keys_input_reordered \
            = join_keys_input_reordered[
                join_keys_input_reordered >= 0
            ]

        return X_reordered, join_keys_input_reordered

    def __create_indptr(self, join_keys_right_reordered):
        indptr = [0] * (join_keys_right_reordered[0] + 1)
        j = 0
        for i in range(len(join_keys_right_reordered)):
            if join_keys_right_reordered[i]\
                    != join_keys_right_reordered[j]:

                indptr += [i]

                # If there are keys, for which there are no entries
                # then keep on adding the last entry to the indptr
                indptr += [indptr[len(indptr) - 1]] * (
                    join_keys_right_reordered[i]
                    - join_keys_right_reordered[j]
                    - 1
                )

                j = i

        indptr += [len(join_keys_right_reordered)]

        # Transform indptr to np.array
        indptr = np.asarray(indptr)

        return indptr

    def __load_dense_data_into_input_networks(
        self,
        X_dense_input,
        join_keys_input,
        time_stamps_input,
        global_batch_size
    ):
        for num_input_network, X_list in enumerate(X_dense_input):
            for num_input_node, X in enumerate(X_list):

                X_reordered, join_keys_input_reordered \
                    = self.__reorder_join_keys_and_data(
                        X,
                        join_keys_input[
                            num_input_network
                        ],
                        time_stamps_input[num_input_network]
                    )

                indptr = self.__create_indptr(join_keys_input_reordered)

                # Load dense data
                self.thisptr.load_dense_data(
                    num_input_network,
                    num_input_node,
                    X_reordered.astype(np.float32),
                    global_batch_size,
                    indptr.astype(np.int32)
                )

                del X_reordered
                del indptr

    def __load_time_stamps_into_input_networks(
            self,
            time_stamps_input,
            join_keys_input):
        for num_input_network, tsi in enumerate(time_stamps_input):
            tsi_reordered, join_keys_input_reordered = \
                self.__reorder_join_keys_and_data(
                    tsi,
                    join_keys_input[
                        num_input_network
                    ],
                    tsi
                )

            indptr = self.__create_indptr(join_keys_input_reordered)

            self.thisptr.load_time_stamps_input(
                tsi_reordered.astype(np.float32),
                indptr.astype(np.int32)
            )

            del tsi_reordered
            del indptr

    def __load_join_keys_output(
            self,
            join_keys_output):
        for join_key in join_keys_output:
            self.thisptr.add_join_keys_left(join_key.astype(np.int32))

    def __load_dense_data_into_output_network(
        self,
        X_dense_output,
        global_batch_size
    ):
        for num_input_node, X in enumerate(X_dense_output):
            self.thisptr.load_dense_data(
                -1,  # -1 implies it is the output network
                num_input_node,
                X.astype(np.float32),
                global_batch_size,
                # No indptr is needed, so we use a pivot
                np.zeros(0).astype(np.int32)
            )

    def __load_dense_targets_into_output_network(
        self,
        Y_dense,
        global_batch_size
    ):
        for num_output_node, Y in enumerate(Y_dense):
            self.thisptr.load_dense_targets(
                num_output_node,
                Y.astype(np.float32),
                global_batch_size
            )

    def __load_time_stamps_into_output_network(
        self,
        time_stamps_output,
        global_batch_size
    ):
        self.thisptr.load_time_stamps_output(
            time_stamps_output.astype(np.float32),
            global_batch_size
        )

    def finalise(self, weight_init_range=0.7):
        self.thisptr.finalise(weight_init_range)

    def fit(
        self,
        X_dense_input=[],
        X_sparse_input=[],
        join_keys_input=[],
        time_stamps_input=[],
        X_dense_output=[],
        X_sparse_output=[],
        join_keys_output=[],
        time_stamps_output=np.zeros(0).astype(np.float32),
        Y_dense=[],
        Y_sparse=[],
        optimiser=SGD(1.0, 0.0),
        global_batch_size=200,
        tol=1e-08,
        max_num_epochs=200,
        MinibatchSizeStandard=20,
        sample=True
    ):

        start_timing = datetime.now()

        self.thisptr.clean_up()

        self.__load_dense_data_into_input_networks(
            X_dense_input,
            join_keys_input,
            time_stamps_input,
            global_batch_size
        )

        self.__load_time_stamps_into_input_networks(
            time_stamps_input,
            join_keys_input
        )

        self.__load_join_keys_output(join_keys_output)

        self.__load_dense_data_into_output_network(
            X_dense_output,
            global_batch_size
        )

        self.__load_time_stamps_into_output_network(
            time_stamps_output,
            global_batch_size
        )

        self.__load_dense_targets_into_output_network(
            Y_dense,
            global_batch_size
        )

        stop_timing = datetime.now()
        time_elapsed = stop_timing - start_timing
        print "Relational network loaded data into GPU."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (
            time_elapsed.seconds // 3600,
            (time_elapsed.seconds // 60) % 60,
            time_elapsed.seconds % 60,
            time_elapsed.microseconds
        )
        print

        start_timing = datetime.now()

        self.thisptr.fit(
            optimiser.thisptr,
            global_batch_size,
            tol,
            max_num_epochs,
            sample
        )

        stop_timing = datetime.now()
        time_elapsed = stop_timing - start_timing
        print "Trained relational network."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (
            time_elapsed.seconds // 3600,
            (time_elapsed.seconds // 60) % 60,
            time_elapsed.seconds % 60,
            time_elapsed.microseconds
        )
        print

    def transform(
        self,
        X_dense_input=[],
        X_sparse_input=[],
        join_keys_input=[],
        time_stamps_input=[],
        X_dense_output=[],
        X_sparse_output=[],
        join_keys_output=[],
        time_stamps_output=np.zeros(0).astype(np.float32),
        global_batch_size=200,
        sample=False,
        sample_size=100,
        get_hidden_nodes=False
    ):

        self.thisptr.clean_up()

        self.__load_dense_data_into_input_networks(
            X_dense_input,
            join_keys_input,
            time_stamps_input,
            global_batch_size
        )

        self.__load_time_stamps_into_input_networks(
            time_stamps_input,
            join_keys_input
        )

        self.__load_join_keys_output(join_keys_output)

        self.__load_dense_data_into_output_network(
            X_dense_output,
            global_batch_size
        )

        self.__load_time_stamps_into_output_network(
            time_stamps_output,
            global_batch_size
        )

        # Prepare Yhat
        Yhat = np.zeros((
            len(join_keys_output[0]),
            self.thisptr.get_sum_output_dim()
        )).astype(np.float32)

        # Apply transformation
        self.thisptr.transform(
            Yhat,
            sample,
            sample_size,
            get_hidden_nodes
        )

        return Yhat

    def get_sum_gradients(self):

        sum_gradients = np.zeros(
            self.thisptr.get_sum_gradients_length()).astype(np.float32)

        self.thisptr.get_sum_gradients(sum_gradients)

        return sum_gradients
