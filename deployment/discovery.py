import numpy as np
import warnings
from datetime import datetime
import DiscoveryCpp

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
            i_share_weights_with=-1
            ):
        """
        node_number: Number of node in the neural network. Every node number must be assigned before neural network is finalised.
        dim: Dimension of the output
        activation: Activation function for the node. Expects one of the following: "logistic", "linear", "rectified linear", "probit", "laplace", "cauchy", "exponential", "tanh", "pareto", "weibull".
        hidden: List of hidden nodes (in form of their respective node_numbers) that are fed into this node. All of these nodes must have a node number that is smaller than the node number of this node. Defaults to zero-length array (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective node_numbers) that are fed into this node. Defaults to zero-length array (meaning that no input nodes are fed into this node).
        """       
        
        self.node_number = node_number
                
        if activation == "logistic":
            self.thisptr = DiscoveryCpp.LogisticActivationFunctionCpp(
                self.node_number, 
                dim, 
                np.asarray(input_dense).astype(np.int32),
                np.asarray(input_sparse).astype(np.int32),
                np.asarray(hidden).astype(np.int32),
                i_share_weights_with
            )
            
        elif activation == "linear":
            self.thisptr = DiscoveryCpp.LinearActivationFunctionCpp(
                self.node_number,
                dim,
                np.asarray(input_dense).astype(np.int32),
                np.asarray(input_sparse).astype(np.int32),
                np.asarray(hidden).astype(np.int32),
                i_share_weights_with
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
            

#Neural network

class NeuralNetwork:
    def __init__(
            self, 
            num_input_nodes_dense=[],
            num_input_nodes_sparse=[],
            num_output_nodes_dense=0,
            num_output_nodes_sparse=0,
            NamesInput=np.zeros(0).astype(str)
    ):
        """
        Initialise neural network.
        num_input_nodes_dense: Number of dimensions for dense inputs. For instance, if your inputs consist of two dense matrices with 300 and 400 nodes respectively, then num_input_nodes_dense=[300,400].
        num_input_nodes_sparse: Number of dimensions for sparse inputs. For instance, if your inputs consist of two sparse matrices with 3000 and 4000 nodes respectively, then num_input_nodes_dense=[3000,4000]. (It is possible to have dense and sparse inputs at the same time.)
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
        
        #Initialise list of nodes
        self.nodes = []
        for i in range(
                self.num_output_nodes_dense + self.num_output_nodes_sparse + self.num_hidden_nodes
        ):
            self.nodes += [None]
        
        #Initialise NeuralNetworkCpp
        self.thisptr = DiscoveryCpp.NeuralNetworkCpp(
            num_input_nodes_dense,
            num_input_nodes_sparse,
            num_output_nodes_dense,
            num_output_nodes_sparse
        )
        
    def init_hidden_node(self, hidden_node, name=""):
        """
        num_samplesnitialise hidden node in the neural network.
        hidden_node: Discovery.NeuralNetworkNode object
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
        output_node: Discovery.NeuralNetworkNode object
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
        
    def set_params(self, params):
        """
        Set the weights of the neural network
        """
        self.thisptr.set_params(params)
        
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
        
        #Load dense data into 
        for i, X in enumerate(Xdense): 
            self.thisptr.load_dense_data(i, X, global_batch_size)        

        #Load sparse data into 
        for i, X in enumerate(Xsparse): 
                        
            self.thisptr.load_sparse_data(
                i, 
                X.data, 
                X.indices, 
                X.indptr, 
                X.shape[0],
                X.shape[1],
                global_batch_size
            )
                        
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
        
        self.thisptr.transform(
            Yhat, 
            sample, 
            sample_size, 
            get_hidden_nodes
        )
        
        return Yhat
        
    def get_sum_gradients(self):
        
        sum_gradients = np.zeros(self.thisptr.get_sum_gradients_length()).astype(np.float32)
        
        self.thisptr.get_sum_gradients(sum_gradients)
        
        return sum_gradients




