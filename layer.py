import numpy as np

from activation import Linear, Sigmoid, SoftMax

class Layer:
    """
    A class for the hidden layers in the neural network
    """

    def __init__(self, num_neurons, num_previous_neurons, act, lrate=0.01, wr=[-0.1, 0.1], br=[-0.1, 0.1]):
        self.input = None #(x,1)
        self.lrate = lrate
        self.act = act[0]
        self.derivative_act = act[1]
        self.W_T = np.random.uniform(wr[0], wr[1], (num_neurons, num_previous_neurons))
        self.B =  np.random.uniform(br[0], br[1], (num_neurons,1)) #(x,1)
        self.sum = None
        self.output = None

        self.J_Z_sum = None
        self.J_Z_Y = None

    
    def calculate_output(self, X):
        self.input = X #output of past layer
        self.sum = np.einsum('ij,jk->ik', self.W_T, self.input) + self.B #dot product
        self.output = self.act(self.sum)
        return self.output


class SoftMaxLayer(Layer):

    def __init__(self):
        self.act = SoftMax.apply
        self.derivative_act = SoftMax.derivative
        self.W_T = None
        self.B = None


    def calculate_output(self, X):
        if self.act == SoftMax.apply:
            self.input = X
            self.sum = self.input
            self.output = self.act(self.sum)
            return self.output
        else:
            return super().calculate_output(X)


