"""
This is a test program that apply XOR logic
with two level of perceptron.
"""
from math import exp
from random import seed
from random import random
from time import time


# Initialize the network. 
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    
    # Build the hidden layer. 
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)

    # Build the output layer. 
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)

    return network

# Caculate the sum of neuron. 
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]

    return activation

# Define the activate function. 
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Network forward propagate.
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs

    return inputs

# Define the derivation of transfer function. 
def transfer_derivative(output):
    return output * (1 - output)

# Network backward propagate. 
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['responsibility'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['responsibility'] = errors[j] * transfer_derivative(neuron['output'])

# Update weights based on the errors. 
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]

        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['responsibility'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['responsibility']

# Train the network.
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] -  outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        # print('>周期=%d, 误差=%.3f' % (epoch, sum_error))
        print('>Epoch = %d, Error = %.3f' % (epoch, sum_error))

# Predict.
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

if __name__ == '__main__':

    seed(int(time()))

    dataset = [[1,1,0],[1,0,1],[0,1,1],[0,0,0]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.5, 2000, n_outputs)

    for layer in network:
        print(layer)

    for row in dataset:
        prediction = predict(network, row)
        # print('预期值=%d, 实际输出=%d' % (row[-1], prediction))
        print('Expect = %d, Prediction = %d' % (row[-1], prediction))



"""
# Define the Perceptron.
class Perceptron:
    def __init__(self, input_para_num, acti_func):
        # Initialize the activate function.
        self.activator = acti_func

        # Set the initial weight to 0.
        self.weights = [0.0 for _ in range(input_para_num)]

    def __str__(self):
        # Print all the weights
        return repr(self.weights)

    def predict(self, row_vec):
        # Input the vector and predict result.
        act_values = 0.0

        for i in range(len(self.weights)):
            act_values += self.weights[i] * row_vec[i]

        return self.activator(act_values)

    def train(self, dataset, iteration, rate):
        # Input the data: vectors, label,
        # training iteration and learning rate.
        for i in range(iteration):
            for input_vec_label in dataset:
                prediction = self.predict(input_vec_label)
                self._update_weights(input_vec_label, prediction, rate)

    def _update_weights(self, input_vec_label, prediction, rate):
        # Update weights
        delta = input_vec_label[-1] - prediction
        for i in range(len(self.weights)):
            self.weights[i] += rate * delta * input_vec_label[i]

# Define the activator function.
def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0

# Define the func to get the training dataset.
def get_training_dataset():
    dataset = [[-1, 1, 1, 0], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]]
    return dataset
"""


