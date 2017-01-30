import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.sparse

import discovery

# pylint: disable-msg=C0103

#-------------------------------------------
# Set up input network

input_network = discovery.NeuralNetwork(
    num_input_nodes_dense=[2],
    num_output_nodes_dense=1
)

input_network.init_hidden_node(
    discovery.ActivationFunction(
        node_number=0,
        dim=50,
        activation="logistic",
        input_dense=[0],
        regulariser=discovery.L2Regulariser(0.00001)
    )
)

input_network.init_output_node(
    discovery.ActivationFunction(
        node_number=1,
        dim=10,
        activation="logistic",
        hidden=[0]
    )
)

input_network.finalise()

#-------------------------------------------
# Set up output network

output_network = discovery.NeuralNetwork(
    num_input_nodes_dense=[2],
    num_output_nodes_dense=1
)

output_network.init_hidden_node(
    discovery.StandardAggregation(
        node_number=0,
        dim=10,
        input_network=0,
        use_timestamps=False,
        aggregation="SUM"
    )
)

output_network.init_output_node(
    discovery.ActivationFunction(
        node_number=1,
        dim=1,
        activation="linear",
        hidden=[0]
    )
)

output_network.finalise()

#-------------------------------------------
# Set up relational network

relational_network = discovery.RelationalNetwork(
    input_networks=[
        [input_network, 0]
    ],
    output_network=output_network
)

relational_network.finalise()

#----------------
# Generate output

join_keys_input = []
targets = []
for i in range(500):
    j = int(10.0*np.random.rand(1)[0])
    join_keys_input += [i] * j
    targets += [float(j)]

join_keys_input = np.asarray(join_keys_input)
targets = np.asarray(targets).reshape(len(targets), 1)

right_table = np.random.rand(len(join_keys_input), 2).astype(np.float32)

left_table = np.random.rand(500, 2).astype(np.float32)

join_keys_output = np.arange(500)

time_stamps_input = np.random.rand(len(join_keys_input)).astype(np.float32)

time_stamps_output = np.random.rand(500).astype(np.float32)

prediction = relational_network.transform(
    X_dense_input=[
        [right_table]
    ],
    join_keys_input=[join_keys_input],
    time_stamps_input=[time_stamps_input],
    X_dense_output=[left_table],
    join_keys_output=[join_keys_output],
    time_stamps_output=time_stamps_output
)

print scipy.stats.pearsonr(prediction.ravel(), targets.ravel())

relational_network.fit(
    X_dense_input=[
        [right_table]
        ],
    join_keys_input=[join_keys_input],
    time_stamps_input=[time_stamps_input],
    X_dense_output=[left_table],
    join_keys_output=[join_keys_output],
    time_stamps_output=time_stamps_output,
    Y_dense=[targets],
    optimiser=discovery.RMSProp(100.0, 0.1),
    tol=0.0,
    max_num_epochs=50,
    sample=False
)

plt.plot(
    relational_network.get_sum_gradients()
)
plt.grid(True)
plt.show()

prediction = relational_network.transform(
    X_dense_input=[
        [right_table]
    ],
    join_keys_input=[join_keys_input],
    time_stamps_input=[time_stamps_input],
    X_dense_output=[left_table],
    join_keys_output=[join_keys_output],
    time_stamps_output=time_stamps_output
)

print scipy.stats.pearsonr(prediction.ravel(), targets.ravel())
