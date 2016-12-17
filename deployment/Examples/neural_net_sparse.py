import discovery
import numpy as np
import scipy.stats
import scipy.sparse
import matplotlib.pyplot as plt
import sklearn.datasets

#-------------------------------------------
#Set up neural network

nn = discovery.NeuralNetwork(
    num_input_nodes_sparse=[2],
    num_output_nodes_dense=1
)

nn.init_hidden_node(
    discovery.ActivationFunction(
        node_number=0, 
        dim=50, 
        activation="logistic", 
        input_sparse=[0]
    )
)

nn.init_output_node(
    discovery.ActivationFunction(
        node_number=1, 
        dim=1, 
        activation="logistic", 
        hidden=[0]
    )
)

nn.finalise()

#-------------------------------------------
#Load saved parameters

params = np.load(
    open("Examples/params_dense.np", "rb")
    )

nn.set_params(params)

#-------------------------------------------
#Display decision function

#Create grid
h = .02
x_min, x_max = -4.0, 4.0
y_min, y_max = -4.0, 4.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = nn.transform(
    Xsparse = [scipy.sparse.csr_matrix(
        np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]
    )]
)[:,0]
Z = Z.reshape(xx.shape)

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#plot
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

