import numpy as np

from activation import Sigmoid

class Layer:
    """
    A class for the hidden layers in the neural network
    """

    def __init__(self, num_neurons, num_previous_neurons, act, lrate=0.01, wr=[-0.1, 0.1], br=[-0.1, 0.1]):
        self.input = None #(x,1)
        self.act = act[0]
        self.derivative_act = act[1]
        self.W_T = np.random.uniform(wr[0], wr[1], (num_neurons, num_previous_neurons))
        self.B =  np.random.uniform(br[0], br[1], (num_neurons,1)) #(x,1)
        self.sum = None
        self.output = None

        self.J_Z_sum = None
        self.J_Z_Y = None

    
    def calculate_output(self, X):
        self.input = X #(x,1)
        self.sum = np.einsum('ij,jk->ik', self.W_T, self.input) + self.B #dot product
        self.output = self.act(self.sum)
        #BACKWARDS Flytt denne til forward pass
        ###self.J_Z_sum = np.diag(self.derivative_act(self.sum).reshape(-1,))
        ###self.J_Z_Y = np.einsum('ij,jk->ik', self.J_Z_sum, self.W_T) # dot product
        #self.J_Z_W = np.outer(self.input, np.diag(self.J_Z_sum)) #np.einsum('ij,kj->', self.input, self.J_Z_sum)
        ###self.J_Z_W = np.einsum('i,j->ij', self.input.ravel(), np.diag(self.J_Z_sum).ravel()) #np.einsum('ij,kj->', self.input, self.J_Z_sum)

        #self.J_Z_Wb = np.outer([1], np.diag(self.J_Z_sum))
        ###self.J_Z_Wb = np.einsum('i,j->ij', [1], np.diag(self.J_Z_sum).ravel())
        #BACKWARDS
        return self.output



    

if __name__ == "__main__":
    l1 = Layer(2,2, (Sigmoid.apply, Sigmoid.derivative), 1, [-0.5,0.5], [-0.5,0.5])
    l2 = Layer(1,2, (Sigmoid.apply, Sigmoid.derivative), 1, [-0.5,0.5], [-0.5,0.5])

    x = np.zeros((2,4))
    y = np.zeros((1,4))
    
    x[0,0] = 0
    x[1,0] = 0
    y[0,0] = 0

    x[0,1] = 1
    x[1,1] = 0
    y[0,1] = 1

    x[0,2] = 0
    x[1,2] = 1
    y[0,2] = 1

    x[0,3] = 1
    x[1,3] = 1
    y[0,3] = 0

