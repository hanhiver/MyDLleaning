from random import seed
from random import random

def initialize_network(n_input, n_hidden, n_output):
	network = list()
	
	hidden_layer = [{'weights':[random() for i in range(n_input + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)

	output_layer = [{'weights' : [random() for i in range(n_output + 1)]} for i in range(n_output)]
	network.append(output_layer)

	return network

def activate(weights, inputs):
	activation = weights[-1]
	
	for i in range(len(weights) - 1):
		activation += weights[i] * input[i]

	return activation

def transfer_derivative(output):
	output = 1/(1 + exp(-1 * input))
	return output

if __name__ == '__main__':
	
	seed(1)

	network = initialize_network(2, 2, 2)
	for layer in network:
		print(layer)
