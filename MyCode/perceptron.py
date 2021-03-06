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

def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0

def get_training_dataset():
    dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
    return dataset

def train_and_perceptron():
    p = Perceptron(3, func_activator)

    dataset = get_training_dataset()
    p.train(dataset, 1000, 0.01)

    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)

    print ('1 and 1 = %d' % and_perception.predict([-1, 1, 1]))
    print ('0 and 0 = %d' % and_perception.predict([-1, 0, 0]))
    print ('1 and 0 = %d' % and_perception.predict([-1, 1, 0]))
    print ('0 and 1 = %d' % and_perception.predict([-1, 0, 1]))
        
