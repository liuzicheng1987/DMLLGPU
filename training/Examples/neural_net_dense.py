import discovery
import numpy as np
import scipy.stats
import scipy.sparse
import matplotlib.pyplot as plt
import sklearn.datasets

#-------------------------------------------
#Prepare data set

#Set sample_size
#You can vary this number to see what happens
sample_size = 40000

X, Y = sklearn.datasets.make_classification(
    n_samples=sample_size, 
    n_features=2, 
    n_informative=2, 
    n_redundant=0
)
X = X.astype(np.float32)
Y = Y.reshape(len(Y), 1).astype(np.float32)

Xsparse = scipy.sparse.csr_matrix(X)
Ysparse = scipy.sparse.csr_matrix(Y)

plt.grid(True)
plt.plot(X[Y[:,0]==0.0, 0], X[Y[:,0]==0, 1], 'co')
plt.plot(X[Y[:,0]==1.0, 0], X[Y[:,0]==1, 1], 'ro')
plt.show()

#-------------------------------------------
#Randomly split in training and testing set

ix = np.random.rand(Xsparse.shape[0]) > 0.5

X_train = X[ix]
Y_train = Y[ix]

X_test = X[ix == False]
Y_test = Y[ix == False]

Xsparse_train = Xsparse[ix]
Ysparse_train = Ysparse[ix]

Xsparse_test = Xsparse[ix == False]
Ysparse_test = Ysparse[ix == False]

#-------------------------------------------
#Set up neural network

nn = discovery.NeuralNetwork(
    num_input_nodes_dense=[2],
    num_output_nodes_dense=1
)

nn.init_hidden_node(
    discovery.ActivationFunction(
        node_number=0, 
        dim=50, 
        activation="logistic", 
        input_dense=[0], 
        regulariser=discovery.L2Regulariser(0.00001)
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
#Fit neural network

nn.fit(
    Xdense=[X_train], 
    Ydense=[Y_train], 
    optimiser=discovery.SGD(), 
    tol=0.0, 
    global_batch_size=2000, 
    max_num_epochs=2000
)

plt.grid(True)
plt.plot(nn.get_sum_gradients())
plt.show()

#-------------------------------------------
#Evaluate results on testing set

Y_hat = nn.transform(Xdense = [X_test])

print scipy.stats.pearsonr(Y_hat.ravel(), Y_test.ravel())

#-------------------------------------------
#Display decision function

#Create grid
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = nn.transform(
    Xdense = [
        np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]
    ]
)
Z = Z.reshape(xx.shape)

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#plot
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

#-------------------------------------------
#Display decision function with data

#Create grid
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = nn.transform(
    Xdense = [
        np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]
    ]
)
Z = Z.reshape(xx.shape)

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#plot
plt.plot(X_test[Y_test[:,0]==0.0, 0], X_test[Y_test[:,0]==0, 1], 'co')
plt.plot(X_test[Y_test[:,0]==1.0, 0], X_test[Y_test[:,0]==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
