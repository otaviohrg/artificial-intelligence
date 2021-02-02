import numpy as np
from utils import sigmoid, sigmoid_derivative
from math import log


class NeuralNetwork:
    """
    Represents a two-layers Neural Network (NN) for multi-class classification.
    The sigmoid activation function is used for all neurons.
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs, alpha):
        """
        Constructs a two-layers Neural Network.

        :param num_inputs: number of inputs of the NN.
        :type num_inputs: int.
        :param num_hiddens: number of neurons in the hidden layer.
        :type num_hiddens: int.
        :param num_outputs: number of outputs of the NN.
        :type num_outputs: int.
        :param alpha: learning rate.
        :type alpha: float.
        """
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.alpha = alpha
        self.weights = [None] * 3
        self.biases = [None] * 3
        self.weights[1] = 0.001 * np.matrix(np.random.randn(num_hiddens, num_inputs))
        self.weights[2] = 0.001 * np.matrix(np.random.randn(num_outputs, num_hiddens))
        self.biases[1] = np.zeros((num_hiddens, 1))
        self.biases[2] = np.zeros((num_outputs, 1))

    def forward_propagation(self, input):
        """
        Executes forward propagation.
        Notice that the z and a of the first layer (l = 0) are equal to the NN's input.

        :param input: input to the network.
        :type input: (num_inputs, 1) numpy matrix.
        :return z: values computed by applying weights and biases at each layer of the NN.
        :rtype z: 3-dimensional list of (num_neurons[l], 1) numpy matrices.
        :return a: activations computed by applying the activation function to z at each layer.
        :rtype a: 3-dimensional list of (num_neurons[l], 1) numpy matrices.
        """
        z = [None] * 3
        a = [None] * 3
        z[0] = input
        a[0] = input
        w = self.weights
        b = self.biases
        z[1] = np.dot(w[1], a[0])+b[1]
        a[1] = sigmoid(z[1])
        z[2] = np.dot(w[2], a[1])+b[2]
        a[2] = sigmoid(z[2])
        return z, a

    def compute_cost(self, inputs, expected_outputs):
        """
        Computes the logistic regression cost of this network.

        :param inputs: inputs to the network.
        :type inputs: list of numpy matrices.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        :return: logistic regression cost.
        :rtype: float.
        """
        cost = 0.0
        num_cases = len(inputs)
        outputs = [None] * num_cases
        for i in range(num_cases):
            z, a = self.forward_propagation(inputs[i])
            outputs[i] = a[-1]
        y = expected_outputs
        yhat = outputs
        for i in range(num_cases):
            for c in range(self.num_outputs):
                cost += -(y[i][c] * log(yhat[i][c]) + (1.0 - y[i][c]) * log(1.0 - yhat[i][c]))
        cost /= num_cases
        return cost

    def compute_gradient_back_propagation(self, inputs, expected_outputs):
        """
        Computes the gradient with respect to the NN's parameters using back propagation.

        :param inputs: inputs to the network.
        :type inputs: list of numpy matrices.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        :return weights_gradient: gradients of the weights at each layer.
        :rtype weights_gradient: L-dimensional list of numpy matrices.
        :return biases_gradient: gradients of the biases at each layer.
        :rtype biases_gradient: L-dimensional list of numpy matrices.
        """
        weights_gradient = [None] * 3
        biases_gradient = [None] * 3
        weights_gradient[1] = np.zeros((self.num_hiddens, self.num_inputs))
        weights_gradient[2] = np.zeros((self.num_outputs, self.num_hiddens))
        biases_gradient[1] = np.zeros((self.num_hiddens, 1))
        biases_gradient[2] = np.zeros((self.num_outputs, 1))

        X = np.array(inputs)[:, :, 0].T
        Y = np.array(expected_outputs)[:, :, 0].T
        Z, A = self.forward_propagation(X)
        num_inputs = X.shape[1]
        for i in range(num_inputs):
            a = [None] * 3
            z = [None] * 3
            a[0] = A[0][:, i, np.newaxis]
            a[1] = A[1][:, i, np.newaxis]
            a[2] = A[2][:, i, np.newaxis]
            z[0] = Z[0][:, i, np.newaxis]
            z[1] = Z[1][:, i, np.newaxis]
            z[2] = Z[2][:, i, np.newaxis]
            y = Y[:, i, np.newaxis]
            db2 = a[2]-y
            dw2 = np.dot(a[1], db2.T).T
            db1 = np.dot(self.weights[2].T, db2)
            db1 = np.multiply(db1, sigmoid_derivative(z[1]))
            dw1 = np.dot(db1, a[0].T)
            weights_gradient[1] += dw1
            weights_gradient[2] += dw2
            biases_gradient[1] += db1
            biases_gradient[2] += db2
        weights_gradient[1] /= num_inputs
        weights_gradient[2] /= num_inputs
        biases_gradient[1] /= num_inputs
        biases_gradient[2] /= num_inputs
        return weights_gradient, biases_gradient

    def back_propagation(self, inputs, expected_outputs):
        """
        Executes the back propagation algorithm to update the NN's parameters.

        :param inputs: inputs to the network.
        :type inputs: list of numpy matrices.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        """
        weights_gradient, biases_gradient = self.compute_gradient_back_propagation(inputs, expected_outputs)
        self.weights[1] -= self.alpha * weights_gradient[1]
        self.weights[2] -= self.alpha * weights_gradient[2]
        self.biases[1] -= self.alpha * biases_gradient[1]
        self.biases[2] -= self.alpha * biases_gradient[2]
