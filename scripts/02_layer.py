import numpy as np

input = np.array([1.0, 2.0, 3.0])

weights1 = np.array([
    [0.1, 0.2, 0.3],
    [0.3, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.1, 0.3, 0.5]
])

biases1 = np.array([0.1,0.1,0.1,0.1])

weights2 = np.array([
    [0.1, 0.2, 0.3, 0.4],   # output neuron 1
    [0.5, 0.6, 0.7, 0.8],   # output neuron 2
])

def relu(x):
    return np.maximum(0,x)

biases2 = np.array([0.1, 0.1])

#forward pass
layer1_output = relu(np.dot(weights1, input) + biases1)

print (f"layer1 output is : {layer1_output}")

layer2_output = relu(np.dot(weights2, layer1_output) + biases2)

print(f"layer 2 output is {layer2_output}")