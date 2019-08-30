# This class are based on post: https://enlight.nyc/projects/neural-network/
# Special thanks the creator. <3

import numpy


# PS: The structure of this neural network is:
#   O  O
#      O  O
#   O  O
#
# This neural network is not dynamic layers. For works with more layers is necessary adaptations

class NeuralNetwork:
    # Constants
    INPUT_SIZE = 2
    HIDDEN_SIZE = 3
    OUTPUT_SIZE = 1


    def __init__(self):
        # Creates self variables
        self.input_weights = None
        self.hidden_weights = None
        self.hidden_propagation = None

        # Creates matrix (this can easily identify where weights are of first or second input).
        self.input_weights = numpy.random.randn(self.INPUT_SIZE, self.HIDDEN_SIZE)
        self.hidden_weights = numpy.random.randn(self.HIDDEN_SIZE, self.OUTPUT_SIZE)

    def get_output(self, inputs):
        input_calc = numpy.dot(inputs, self.input_weights)
        self.hidden_propagation = self.sigmoid(input_calc)
        output_calc = numpy.dot(self.hidden_propagation, self.hidden_weights)
        return self.sigmoid(output_calc)

    def train(self, inputs, ai_output, right_output):
        output_error = right_output - ai_output
        delta_output = output_error * self.sigmoid_derivative(ai_output)

        hidden_error = delta_output.dot(self.hidden_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_propagation)

        self.hidden_weights += self.hidden_propagation.T.dot(delta_output)
        self.input_weights += inputs.T.dot(hidden_delta)

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def sigmoid_derivative(self, s):
        return s * (1 - s)