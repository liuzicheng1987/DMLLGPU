import DMLLGPU
import numpy as np

nn = DMLLGPU.NeuralNetwork(num_input_nodes_dense=[5], num_output_nodes_dense=1)

nn.init_hidden_node(DMLLGPU.ActivationFunction(0, "linear", InputDense=[0]))
nn.init_hidden_node(DMLLGPU.ActivationFunction(1, "linear", hidden=[0]))
nn.init_hidden_node(DMLLGPU.ActivationFunction(2, "linear", hidden=[1]))
nn.init_hidden_node(DMLLGPU.ActivationFunction(3, "linear", hidden=[2]))
nn.init_hidden_node(DMLLGPU.ActivationFunction(4, "linear", hidden=[3]))

nn.init_output_node(DMLLGPU.ActivationFunction(5, "linear", hidden=[4]))

nn.finalise()

print nn.get_input_dense(0)
print nn.get_input_dense(2)
print nn.get_input_dense(3)
print nn.get_input_dense(4)
print nn.get_input_dense(5)

print nn.get_hidden(0)
print nn.get_hidden(1)
print nn.get_hidden(2)
print nn.get_hidden(3)
print nn.get_hidden(4)
print nn.get_hidden(5)

print nn.get_params()

X = np.zeros((50,5)).astype(np.float32)

for i in range(X.shape[0]):
    for j in range(i % X.shape[1] + 1):
        X[i, j] = 1.0

Yhat = nn.transform(Xdense=[X])

W = nn.get_params() 
Y = W[:6].sum()
Y = Y*W[6] + W[7]
Y = Y*W[8] + W[9]
Y = Y*W[10] + W[11]
Y = Y*W[12] + W[13]
Y = Y*W[14] + W[15]
