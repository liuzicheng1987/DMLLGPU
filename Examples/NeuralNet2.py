import DMLLGPU
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import sklearn.datasets

nn = DMLLGPU.NeuralNetwork(NumInputNodesDense=[2], NumOutputNodesDense=1)

for i in range(8):
    nn.init_hidden_node(DMLLGPU.ActivationFunction(i, "logistic", InputDense=[0]))

nn.init_output_node(DMLLGPU.ActivationFunction(8, "logistic", hidden=np.arange(5)))
#nn.init_output_node(DMLLGPU.ActivationFunction(5, "logistic", InputDense=[0]))

nn.finalise()

#Set SampleSize
#You can vary this number to see what happens
SampleSize = 2001

X, Y = sklearn.datasets.make_classification(n_samples=SampleSize, n_features=2, n_informative=2, n_redundant=0)
X = X.astype(np.float32)
Y = Y.reshape(len(Y), 1).astype(np.float32)

plt.grid(True)
plt.plot(X[Y[:,0]==0.0, 0], X[Y[:,0]==0, 1], 'co')
plt.plot(X[Y[:,0]==1.0, 0], X[Y[:,0]==1, 1], 'ro')
plt.show()

nn.fit(Xdense=[X], Ydense=[Y], optimiser=DMLLGPU.SGD(0.1, 0.1), tol=0.0, GlobalBatchSize=200, MaxNumEpochs=2000)

plt.grid(True)
plt.plot(nn.get_sum_gradients())
plt.show()

Yhat = nn.transform(Xdense=[X])

print scipy.stats.pearsonr(Yhat.ravel(), Y.ravel())

#Create grid
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#----------------------------------training set-----------------------------------------
Z = nn.transform(Xdense=[np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]])
Z = Z.reshape(xx.shape)

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(X[Y[:,0]==0.0, 0], X[Y[:,0]==0, 1], 'co')
plt.plot(X[Y[:,0]==1.0, 0], X[Y[:,0]==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
