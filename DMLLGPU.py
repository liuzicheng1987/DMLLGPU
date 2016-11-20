import numpy as np
import warnings
from datetime import datetime
import DMLLGPUCpp

#Optimisers

class SGD:
    def __init__(self, learning_rate, learning_rate_power):
        """
        learning_rate: Learning rate of the optimisation problem
        learning_rate_power: Speed by which learning rate decreases
        """
        self.thisptr = DMLLGPUCpp.SGDCpp(learning_rate, learning_rate_power)
    
#Loss functions

class SquareLoss:
    def __init__(self):
        self.thisptr = DMLLGPUCpp.SquareLossCpp()

#Regularisers
class Regulariser:
    def __init__(self):
        """
        alpha: Weight associated with regulariser
        """
        self.thisptr = DMLLGPUCpp.RegulariserCpp()

class L2Regulariser:
    def __init__(self, alpha=1.0):
        """
        alpha: Weight associated with regulariser
        """
        self.thisptr = DMLLGPUCpp.L2RegulariserCpp(alpha)

#Activation functions
class ActivationFunction:
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
        node_number: Number of node in the neural network. Every node number must be assigned before neural network is finalised.
        dim: Dimension of the output
        activation: Activation function for the node. Expects one of the following: "logistic", "linear", "rectified linear", "probit", "laplace", "cauchy", "exponential", "tanh", "pareto", "weibull".
        hidden: List of hidden nodes (in form of their respective node_numbers) that are fed into this node. All of these nodes must have a node number that is smaller than the node number of this node. Defaults to zero-length array (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective node_numbers) that are fed into this node. Defaults to zero-length array (meaning that no input nodes are fed into this node).
        i_share_weights_with: Node number of a node that this node share weights with. -1 if it shares weights with no node. Defaults to -1.
        no_weight_updates: True if this node is to receive no updates during the training process.
        regulariser: Regulariser object
        """       
        
        self.node_number = node_number
        
        #Store regulariser
        self.regulariser = regulariser
        
        if activation == "logistic":
            self.thisptr = DMLLGPUCpp.LogisticActivationFunctionGPUCpp(
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
            self.thisptr = DMLLGPUCpp.LinearActivationFunctionGPUCpp(
                self.node_number,
                dim,
                np.asarray(input_dense).astype(np.int32),
                np.asarray(input_sparse).astype(np.int32),
                np.asarray(hidden).astype(np.int32),
                i_share_weights_with,
                no_weight_updates, 
                self.regulariser.thisptr
            )


class SoftmaxActivationFunction:
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
        node_number: Number of node in the neural network. Every node number must be assigned before neural network is finalised.
        num_vars: Number of distinct discrete variables (output dimension is num_vars*num_states_per_var)
        num_states_per_var: Number of states each variable can assume (output dimension is num_vars*num_states_per_var)
        hidden: List of hidden nodes (in form of their respective node_numbers) that are fed into this node. All of these nodes must have a node number that is smaller than the node number of this node. Defaults to zero-length array (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective node_numbers) that are fed into this node. Defaults to zero-length array (meaning that no input nodes are fed into this node).
        i_share_weights_with: Node number of a node that this node share weights with. -1 if it shares weights with no node. Defaults to -1.
        no_weight_updates: True if this node is to receive no updates during the training process.
        regulariser: Regulariser object
        """       
        
        self.node_number = node_number
        
        #Store regulariser
        self.regulariser = regulariser
        
        self.thisptr = DMLLGPUCpp.SoftmaxActivationFunctionGPUCpp(
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

#Logical gates

class LogicalGate:
    def __init__(
            self, 
            node_number, 
            dim,
            hidden,
            activation="AND"
    ):
        """
        Logical gates are a weightless transformation that make it easier to interpret the neural entwork.
        The logical gate is applied dimension-wise.
        node_number: Number of node in the neural network. Every node number must be assigned before neural network is finalised.
        hidden: List of hidden nodes (in form of their respective node_numbers) that are fed into this node. All of these nodes must have a node number that is smaller than the node number of this node. Number of hidden nodes fed into LogicalGate must be between 2 and 4.
        dim: Dimension of the output. Must be identical to dimensions of all input nodes.
        activation: Activation function for the node. Expects one of the following: "AND", "OR", "XOR", "XNOR", "NAND", "NOR".
        """
        
        self.node_number = node_number
                
        if activation == "AND":
            self.thisptr = DMLLGPUCpp.ANDGateCpp(
                self.node_number, 
                dim, 
                hidden
            )
            
        elif activation == "OR":
            self.thisptr = DMLLGPUCpp.ORGateCpp(
                self.node_number, 
                dim, 
                hidden
            )
            
        elif activation == "XOR":
            self.thisptr = DMLLGPUCpp.XORGateCpp(
                self.node_number, 
                dim, 
                hidden
            )
            
        elif activation == "XNOR":
            self.thisptr = DMLLGPUCpp.XNORGateCpp(
                self.node_number, 
                dim, 
                hidden
            )
            
        elif activation == "NOR":
            self.thisptr = DMLLGPUCpp.NORGateCpp(
                self.node_number, 
                dim, 
                hidden
            )
            
        elif activation == "NAND":
            self.thisptr = DMLLGPUCpp.NANDGateCpp(
                self.node_number, 
                dim, 
                hidden
            )
            

#Neural network

class NeuralNetwork:
    def __init__(
            self, 
            num_input_nodes_dense=[],
            num_input_nodes_sparse=[],
            num_output_nodes_dense=0,
            num_output_nodes_sparse=0,
            NamesInput=np.zeros(0).astype(str),
            loss=SquareLoss()
    ):
        """
        num_samplesnitialise neural network.
        num_input_nodes_dense: Number of dimensions for dense inputs. For instance, if your inputs consist of two dense matrices with 300 and 400 nodes respectively, then num_input_nodes_dense=[300,400].
        num_input_nodes_sparse: Number of dimensions for sparse inputs. For instance, if your inputs consist of two sparse matrices with 3000 and 4000 nodes respectively, then num_input_nodes_dense=[3000,4000]. (It is possible to have dense and sparse inputs at the same time.)
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
        
        #Initialise list of nodes
        self.nodes = []
        for i in range(
                self.num_output_nodes_dense + self.num_output_nodes_sparse + self.num_hidden_nodes
        ):
            self.nodes += [None]
        
        #Initialise NeuralNetworkGPUCpp
        self.thisptr = DMLLGPUCpp.NeuralNetworkGPUCpp(
            num_input_nodes_dense,
            num_input_nodes_sparse,
            num_output_nodes_dense,
            num_output_nodes_sparse,
            self.loss.thisptr
        )
        
    def init_hidden_node(self, hidden_node, name=""):
        """
        num_samplesnitialise hidden node in the neural network.
        hidden_node: DMLL.NeuralNetworkNode object
        name: Name of the node (optional)
        """
        
        #If node number exceeds number of hidden nodes so far, then extend hidden nodes and names accordingly
        if hidden_node.node_number >= self.num_hidden_nodes:
            NumAdditionalNodes = hidden_node.node_number + 1 - self.num_hidden_nodes
            for i in range(NumAdditionalNodes):
                self.nodes.insert(self.num_hidden_nodes, None)
            
            #Increase num_hidden_nodes
            self.num_hidden_nodes += NumAdditionalNodes 
            
            #Increase node_number of output_nodes
            for i in range(self.num_output_nodes_dense + self.num_output_nodes_sparse):
                if type(self.nodes[self.num_hidden_nodes + i]) != type(None):
                    self.nodes[self.num_hidden_nodes + i].node_number += NumAdditionalNodes
        
        #Store node, both in Python and C++
        self.nodes[hidden_node.node_number] = hidden_node
        self.thisptr.init_hidden_node(self.nodes[hidden_node.node_number].thisptr)
                
    def init_output_node(self, output_node, name=""):
        """
        num_samplesnitialise output node in the neural network.
        output_node: DMLL.NeuralNetworkNode object
        name: Name of the node (optional)
        """
                
        #Store node, both in Python and C++
        self.nodes[output_node.node_number] = output_node
        self.thisptr.init_output_node(self.nodes[output_node.node_number].thisptr) 
        
    def finalise(self, weight_init_range=0.7):
        """
        Prepares neural network for fitting.
        weight_init_range: Range within which the weights are randomly initialised
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
        hidden = np.zeros(self.thisptr.get_hidden_nodes_fed_into_me_length(node_number)).astype(np.int32)
        self.thisptr.get_hidden_nodes_fed_into_me(node_number, hidden)
        return hidden
                
    def get_params(self):
        """
        Get the weights that have been trained
        """
        params  = np.zeros(self.thisptr.get_length_params()).astype(np.float32)
        self.thisptr.get_params(params)
        return params

    def fit(
            self,
            Xdense=[],
            Xsparse=[],
            Ydense=[],
            Ysparse=[],
            optimiser=SGD(1.0, 0.0),
            global_batch_size=200,
            tol=1e-08, max_num_epochs=200,
            MinibatchSizeStandard=20,
            sample=True,
            root=0
    ):
        """
        Fit the neural network to training data.
        Xdense: Array of numpy.arrays. Number of dimensions must match the num_input_nodes_dense set when declaring the class and number of samples must be identical.
        Xsparse: Array of scipy.sparse.csr_matrices. Number of dimensions must match the num_input_nodes_sparse set when declaring the class and number of samples must be identical.
        Ydense: Array of numpy.arrays. Number of dimensions must match the dimensions of respective output nodes.
        Ysparse: Array of scipy.sparse.csr_matrices. Number of dimensions must match the dimensions of respective output nodes.
        optimiser: Optimiser class
        global_batch_size: Approximate size of batch used in each iteration.
        tol: num_samplesf sum of squared gradients is below tol, stop training prematurely.
        max_num_epochs: Maximum number of epochs for training
        MinibatchSizeStandard: Number of samples calculated at oncer
        sample: Whether you want to sample
        root: Number of root process
        """
        
        #Make sure length of data vectors provided matches lengths that were previously defined
        if len(Xdense) != self.num_input_nodes_dense_length:
            warnings.warn("Number of matrices in Xdense must be identical to length of num_input_nodes_dense. Expected " + str(self.num_input_nodes_dense_length) + ", got " + str(len(Xdense)) + ".")
        
        if len(Xsparse) != self.num_input_nodes_sparse_length:
            warnings.warn("Number of matrices in Xsparse must be identical to length of num_input_nodes_sparse. Expected " + str(self.num_input_nodes_sparse_length) + ", got " + str(len(Xsparse)) + ".")
       
        if len(Ydense) != self.num_output_nodes_dense:
            warnings.warn("Number of matrices in Ydense must be identical to length of num_output_nodes_dense. Expected " + str(self.num_output_nodes_dense) + ", got " + str(len(Ydense)) + ".")
        
        if len(Ysparse) != self.num_output_nodes_sparse:
            warnings.warn("Number of matrices in Ysparse must be identical to length of num_output_nodes_sparse. Expected " + str(self.num_output_nodes_sparse) + ", got " + str(len(Ysparse)) + ".")
        
        StartTiming = datetime.now()
        
        #Load Xdense into GPU
        for i, X in enumerate(Xdense): 
            self.thisptr.load_dense_data(i, X, global_batch_size)        

        #Load Xsparse into GPU
        for i, X in enumerate(Xsparse): 
            self.thisptr.load_sparse_data(i, X.data, X.indices, X.indptr, X.shape[0], X.shape[1], global_batch_size)                 
        
        #Load Ydense into GPU
        for i, Y in enumerate(Ydense): 
            self.thisptr.load_dense_targets(i, Y, global_batch_size)        
        
        #Load Ysparse into GPU
        for i, Y in enumerate(Ysparse): 
            self.thisptr.load_sparse_targets(i, Y.data, Y.indices, Y.indptr, Y.shape[0], Y.shape[1], global_batch_size)
        
        StopTiming = datetime.now()
        TimeElapsed = StopTiming - StartTiming		
        print "Neural network loaded data into GPU."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, (TimeElapsed.seconds//60)%60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
        print				                   
        
        StartTiming = datetime.now()
        
        self.thisptr.fit(optimiser.thisptr, global_batch_size, tol, max_num_epochs, MinibatchSizeStandard, sample) 
        
        StopTiming = datetime.now()
        TimeElapsed = StopTiming - StartTiming		
        print "Trained neural network."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, (TimeElapsed.seconds//60)%60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
        print				                   
        
    def transform(
            self, 
            Xdense=[],
            Xsparse=[],
            global_batch_size=200,
            sample=False,
            sample_size=1,
            get_hidden_nodes=False
    ):
        """
        Transform input values to predictions.
        Xdense: Array of numpy.arrays. Number of dimensions must match the num_input_nodes_dense set when declaring the class and number of samples must be identical.
        Xsparse: Array of scipy.sparse.csr_matrices. Number of dimensions must match the num_input_nodes_sparse set when declaring the class and number of samples must be identical.
        global_batch_size: Number of batches calculated at once
        sample (bool): Whether you would like to sample (relevant for dropout nodes)
        sample_size (int): Size of the samples
        get_hidden_nodes (bool): Whether you would like to get the hidden nodes as well
        """
        
        #Make sure length of Xdense matches num_input_nodes_dense_length
        if len(Xdense) != self.num_input_nodes_dense_length:
            warnings.warn("Number of matrices in Xdense must be identical to length of num_input_nodes_dense. Expected " + str(self.num_input_nodes_dense_length) + ", got " + str(len(Xdense)) + ".")
        
        if len(Xsparse) != self.num_input_nodes_sparse_length:
            warnings.warn("Number of matrices in Xsparse must be identical to length of num_input_nodes_sparse. Expected " + str(self.num_input_nodes_sparse_length) + ", got " + str(len(Xsparse)) + ".")
        
        #Load dense data into GPU
        for i, X in enumerate(Xdense): 
            self.thisptr.load_dense_data(i, X, global_batch_size)        

        #Load sparse data into GPU
        for i, X in enumerate(Xsparse): 
            self.thisptr.load_sparse_data(i, X.data, X.indices, X.indptr, X.shape[0], X.shape[1], global_batch_size)     
            
        if len(Xdense) > 0:    
            Yhat = np.zeros((Xdense[0].shape[0], self.num_output_nodes_dense + self.num_output_nodes_sparse)).astype(np.float32)
        else:
            Yhat = np.zeros((Xsparse[0].shape[0], self.num_output_nodes_dense + self.num_output_nodes_sparse)).astype(np.float32)
        
        #Check whether the input X is a numpy vector
        self.thisptr.transform(Yhat, sample, sample_size, get_hidden_nodes)
        
        return Yhat
        
    def get_sum_gradients(self):
        
        sum_gradients = np.zeros(self.thisptr.get_sum_gradients_length()).astype(np.float32)
        
        self.thisptr.get_sum_gradients(sum_gradients)
        
        return sum_gradients




