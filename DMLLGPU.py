import numpy as np
import warnings
from datetime import datetime
import DMLLGPUCpp

#Optimisers

class SGD:
    def __init__(self, LearningRate, LearningRatePower):
        """
        LearningRate: Learning rate of the optimisation problem
        LearningRatePower: Speed by which learning rate decreases
        """
        self.thisptr = DMLLGPUCpp.SGDCpp(LearningRate, LearningRatePower)
    

#Loss functions

class SquareLoss:
    def __init__(self):
        self.thisptr = DMLLGPUCpp.SquareLossCpp()

#Regularisers

#Activation functions
class ActivationFunction:
    def __init__(self, NodeNumber, activation, InputDense=np.asarray([]).astype(np.int32), InputSparse=np.asarray([]).astype(np.int32), hidden=np.asarray([]).astype(np.int32), IShareWeightsWith=-1, NoWeightUpdates=False):
        """
        NodeNumber: Number of node in the neural network. Every node number must be assigned before neural network is finalised.
        activation: Activation function for the node. Expects one of the following: "logistic", "linear", "rectified linear", "probit", "laplace", "cauchy", "exponential", "tanh", "pareto", "weibull".
        hidden: List of hidden nodes (in form of their respective NodeNumbers) that are fed into this node. All of these nodes must have a node number that is smaller than the node number of this node. Defaults to zero-length array (meaning that no hidden nodes are fed into this node).
        input: List of input nodes (in form of their respective NodeNumbers) that are fed into this node. Defaults to zero-length array (meaning that no input nodes are fed into this node).
        IShareWeightsWith: Node number of a node that this node share weights with. -1 if it shares weights with no node. Defaults to -1.
        NoWeightUpdates: True if this node is to receive no updates during the training process.
        """       
        
        #Transform activation function into integer
        self.NodeNumber = NodeNumber
        
        if  activation == "logistic":
            self.thisptr = DMLLGPUCpp.LogisticActivationFunctionGPUCpp(self.NodeNumber, np.asarray(InputDense).astype(np.int32), np.asarray(InputSparse).astype(np.int32), np.asarray(hidden).astype(np.int32), IShareWeightsWith, NoWeightUpdates)
            
        elif activation == "linear":
            self.thisptr = DMLLGPUCpp.LinearActivationFunctionGPUCpp(self.NodeNumber, np.asarray(InputDense).astype(np.int32), np.asarray(InputSparse).astype(np.int32), np.asarray(hidden).astype(np.int32), IShareWeightsWith, NoWeightUpdates)

#Neural network

class NeuralNetwork:
   #def __init__(self, NumInputNodes, LossFunction, NumOutputNodes=1, NumHiddenNodes=0, regulariser=Regulariser(), NamesInput=np.zeros(0).astype(str)):
    def __init__(self, NumInputNodesDense=[], NumInputNodesSparse=[], NumOutputNodesDense=0, NumOutputNodesSparse=0, NamesInput=np.zeros(0).astype(str), loss=SquareLoss()):
        """
        Initialise neural network.
        NumInputNodesDense: Number of dimensions for dense inputs. For instance, if your inputs consist of two dense matrices with 300 and 400 nodes respectively, then NumInputNodesDense=[300,400].
        NumInputNodesSparse: Number of dimensions for sparse inputs. For instance, if your inputs consist of two sparse matrices with 3000 and 4000 nodes respectively, then NumInputNodesDense=[3000,4000]. (It is possible to have dense and sparse inputs at the same time.)
        LossFunction: Loss function object
        NumOutputNodes: Number of output nodes
        regulariser: regulariser object
        Names
        """
        self.NumInputNodesDense = NumInputNodesDense
        self.NumInputNodesSparse = NumInputNodesSparse
        self.NumInputNodesDenseLength = len(NumInputNodesDense)
        self.NumInputNodesSparseLength = len(NumInputNodesSparse)
        self.NumHiddenNodes = 0
        self.NumOutputNodesDense = NumOutputNodesDense	       
        self.NumOutputNodesSparse = NumOutputNodesSparse	       
        self.loss = loss	
        #self.regulariser = regulariser	  
        
        #Initialise list of nodes
        self.nodes = []
        for i in range(self.NumOutputNodesDense + self.NumOutputNodesSparse + self.NumHiddenNodes):
            self.nodes += [None]
        self.thisptr = DMLLGPUCpp.NeuralNetworkGPUCpp(NumInputNodesDense, NumInputNodesSparse, NumOutputNodesDense, NumOutputNodesSparse, self.loss.thisptr)#, self.regulariser.thisptr)
        
    def init_hidden_node(self, HiddenNode, name=""):
        """
        Initialise hidden node in the neural network.
        HiddenNode: DMLL.NeuralNetworkNode object
        name: Name of the node (optional)
        """
        
        #If node number exceeds number of hidden nodes so far, then extend hidden nodes and names accordingly
        if HiddenNode.NodeNumber >= self.NumHiddenNodes:
            NumAdditionalNodes = HiddenNode.NodeNumber + 1 - self.NumHiddenNodes
            for i in range(NumAdditionalNodes):
                self.nodes.insert(self.NumHiddenNodes, None)
            
            #Increase NumHiddenNodes
            self.NumHiddenNodes += NumAdditionalNodes 
            
            #Increase NodeNumber of OutputNodes
            for i in range(self.NumOutputNodesDense + self.NumOutputNodesSparse):
                if type(self.nodes[self.NumHiddenNodes + i]) != type(None):
                    self.nodes[self.NumHiddenNodes + i].NodeNumber += NumAdditionalNodes
        
        #Store node, both in Python and C++
        self.nodes[HiddenNode.NodeNumber] = HiddenNode
        self.thisptr.init_hidden_node(self.nodes[HiddenNode.NodeNumber].thisptr)
                
    def init_output_node(self, OutputNode, name=""):
        """
        Initialise output node in the neural network.
        OutputNode: DMLL.NeuralNetworkNode object
        name: Name of the node (optional)
        """
                
        #Store node, both in Python and C++
        self.nodes[OutputNode.NodeNumber] = OutputNode
        self.thisptr.init_output_node(self.nodes[OutputNode.NodeNumber].thisptr) 
        
    def finalise(self, WeightInitRange=0.7):
        """
        Prepares neural network for fitting.
        WeightInitRange: Range within which the weights are randomly initialised
        """
        self.thisptr.finalise(WeightInitRange)  
        
    def get_input_dense(self, NodeNumber):
        """
        Get the dense input nodes which are fed into node.
        NodeNumber: Node number of the node we are interested in
        """
        InputDense = np.zeros(self.thisptr.get_input_nodes_fed_into_me_dense_length(NodeNumber)).astype(np.int32)
        self.thisptr.get_input_nodes_fed_into_me_dense(NodeNumber, InputDense)
        return InputDense
        
    def get_input_sparse(self, NodeNumber):
        """
        Get the sparse input nodes which are fed into node.
        NodeNumber: Node number of the node we are interested in
        """
        InputSparse = np.zeros(self.thisptr.get_input_nodes_fed_into_me_dense_length(NodeNumber)).astype(np.int32)
        self.thisptr.get_input_nodes_fed_into_me_dense(NodeNumber, InputSparse)
        return InputSparse        
        
    def get_hidden(self, NodeNumber):
        """
        Get the hidden nodes which are fed into node.
        NodeNumber: Node number of the node we are interested in
        """
        hidden = np.zeros(self.thisptr.get_hidden_nodes_fed_into_me_length(NodeNumber)).astype(np.int32)
        self.thisptr.get_hidden_nodes_fed_into_me(NodeNumber, hidden)
        return hidden
                
    def get_params(self):
        """
        Get the weights that have been trained
        """
        params  = np.zeros(self.thisptr.get_length_params()).astype(np.float32)
        self.thisptr.get_params(params)
        return params

    def fit(self, Xdense=[], Xsparse=[], Ydense=[], Ysparse=[], optimiser=SGD(1.0, 0.0), GlobalBatchSize=200, tol=1e-08, MaxNumEpochs=200, MinibatchSizeStandard=20, sample=True, root=0):
        """
        Fit the neural network to training data.
        Xdense: Array of numpy.arrays. Number of dimensions must match the NumInputNodesDense set when declaring the class and number of samples must be identical.
        Xsparse: Array of scipy.sparse.csr_matrices. Number of dimensions must match the NumInputNodesSparse set when declaring the class and number of samples must be identical.
        Ydense: Array of numpy.arrays. Number of dimensions must match the dimensions of respective output nodes.
        Ysparse: Array of scipy.sparse.csr_matrices. Number of dimensions must match the dimensions of respective output nodes.
        optimiser: Optimiser class
        GlobalBatchSize: Approximate size of batch used in each iteration.
        tol: If sum of squared gradients is below tol, stop training prematurely.
        MaxNumEpochs: Maximum number of epochs for training
        MinibatchSizeStandard: Number of samples calculated at oncer
        sample: Whether you want to sample
        root: Number of root process
        """
        
        #Make sure length of data vectors provided matches lengths that were previously defined
        if len(Xdense) != self.NumInputNodesDenseLength:
            warnings.warn("Number of matrices in Xdense must be identical to length of NumInputNodesDense. Expected " + str(self.NumInputNodesDenseLength) + ", got " + str(len(Xdense)) + ".")
        
        if len(Xsparse) != self.NumInputNodesSparseLength:
            warnings.warn("Number of matrices in Xsparse must be identical to length of NumInputNodesSparse. Expected " + str(self.NumInputNodesSparseLength) + ", got " + str(len(Xsparse)) + ".")
       
        if len(Ydense) != self.NumOutputNodesDense:
            warnings.warn("Number of matrices in Ydense must be identical to length of NumOutputNodesDense. Expected " + str(self.NumOutputNodesDense) + ", got " + str(len(Ydense)) + ".")
        
        if len(Ysparse) != self.NumOutputNodesSparse:
            warnings.warn("Number of matrices in Ysparse must be identical to length of NumOutputNodesSparse. Expected " + str(self.NumOutputNodesSparse) + ", got " + str(len(Ysparse)) + ".")
        
        StartTiming = datetime.now()
        
        #Load Xdense into GPU
        for i, X in enumerate(Xdense): 
            self.thisptr.load_dense_data(i, X, GlobalBatchSize)        

        #Load Xsparse into GPU
        for i, X in enumerate(Xsparse): 
            self.thisptr.load_sparse_data(i, X.data, X.indices, X.indptr, X.shape[0], X.shape[1], GlobalBatchSize)                 
        
        #Load Ydense into GPU
        for i, Y in enumerate(Ydense): 
            self.thisptr.load_dense_targets(i, Y, GlobalBatchSize)        
        
        #Load Ysparse into GPU
        for i, Y in enumerate(Ysparse): 
            self.thisptr.load_sparse_targets(i, Y.data, Y.indices, Y.indptr, Y.shape[0], Y.shape[1], GlobalBatchSize)
        
        StopTiming = datetime.now()
        TimeElapsed = StopTiming - StartTiming		
        print "Neural network loaded data into GPU."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, (TimeElapsed.seconds//60)%60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
        print				                   
        
        StartTiming = datetime.now()
        
        self.thisptr.fit(optimiser.thisptr, GlobalBatchSize, tol, MaxNumEpochs, MinibatchSizeStandard, sample) 
        
        StopTiming = datetime.now()
        TimeElapsed = StopTiming - StartTiming		
        print "Trained neural network."
        print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, (TimeElapsed.seconds//60)%60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
        print				                   
        
    def transform(self, Xdense=[], Xsparse=[], GlobalBatchSize=200, sample=False, SampleSize=1, GetHiddenNodes=False):
        """
        Transform input values to predictions.
        Xdense: Array of numpy.arrays. Number of dimensions must match the NumInputNodesDense set when declaring the class and number of samples must be identical.
        Xsparse: Array of scipy.sparse.csr_matrices. Number of dimensions must match the NumInputNodesSparse set when declaring the class and number of samples must be identical.
        GlobalBatchSize: Number of batches calculated at once
        sample (bool): Whether you would like to sample (relevant for dropout nodes)
        SampleSize (int): Size of the samples
        GetHiddenNodes (bool): Whether you would like to get the hidden nodes as well
        """
        
        #Make sure length of Xdense matches NumInputNodesDenseLength
        if len(Xdense) != self.NumInputNodesDenseLength:
            warnings.warn("Number of matrices in Xdense must be identical to length of NumInputNodesDense. Expected " + str(self.NumInputNodesDenseLength) + ", got " + str(len(Xdense)) + ".")
        
        if len(Xsparse) != self.NumInputNodesSparseLength:
            warnings.warn("Number of matrices in Xsparse must be identical to length of NumInputNodesSparse. Expected " + str(self.NumInputNodesSparseLength) + ", got " + str(len(Xsparse)) + ".")
        
        #Load dense data into GPU
        for i, X in enumerate(Xdense): 
            self.thisptr.load_dense_data(i, X, GlobalBatchSize)        

        #Load sparse data into GPU
        for i, X in enumerate(Xsparse): 
            self.thisptr.load_dense_data(i, X.data, X.indices, X.indptr, X.shape[0], X.shape[1], GlobalBatchSize)     
        
        Yhat = np.zeros((Xdense[0].shape[0], self.NumOutputNodesDense + self.NumOutputNodesSparse)).astype(np.float32)
            
        #Check whether the input X is a numpy vector
        self.thisptr.transform(Yhat, sample, SampleSize, GetHiddenNodes)
        
        return Yhat
        
    def get_sum_gradients(self):
        
        sum_gradients = np.zeros(self.thisptr.get_sum_gradients_length()).astype(np.float32)
        
        self.thisptr.get_sum_gradients(sum_gradients)
        
        return sum_gradients




