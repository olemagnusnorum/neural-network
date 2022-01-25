
from unittest import case
from activation import MSE, Linear, Relu, Sigmoid
from layer import Layer
from config_reader import ModelParser
import numpy as np
import random as r

class NeuralNetwork:
    """
    A class for the neural network containg the differen network layers
    """

    def __init__(self, input_zise=None):
        self.input_zise = input_zise
        self.hidden_layers = []
        self.output_layer = None
        self.loss_function = MSE.apply
        self.derivate_loss_function = MSE.derivative

    

    def config_data(self):
        pass


    def add_hidden_layer(self, layer): # agrumenter for å lage layer
        self.hidden_layers.append(layer)

    def set_output_layer(self, layer): # argumenter for å lage output layer
        self.output_layer = layer

    def forward_pass(self, input):
        """
        Function for the forward pass 

        Args:
            Input vector with correct dimentions

        Returns:
            The output of the output layer as a vector
        """
        for i in range(len(self.hidden_layers)):
            input = self.hidden_layers[i].calculate_output(input)
        return input

    def backward_pass(self, pred, target):
        gradients_w = []
        gradients_b = []
        J_l_z = self.jacobian_loss_z(pred, target)
        for i in range(len(self.hidden_layers)-1,-1,-1):
            ###from forward pass
            J_z_sum = np.diag(self.hidden_layers[i].derivative_act(self.hidden_layers[i].sum).reshape(-1,))
            J_z_y = np.einsum('ij,jk->ik', J_z_sum, self.hidden_layers[i].W_T) # dot product
            J_z_w = np.einsum('i,j->ij', self.hidden_layers[i].input.ravel(), np.diag(J_z_sum).ravel())
            J_z_wb = np.einsum('i,j->ij', [1], np.diag(J_z_sum).ravel())
            ###
            J_l_w = J_l_z * J_z_w
            J_l_wb = J_l_z * J_z_wb
            J_l_y = np.einsum('ij,jk->ik', J_l_z, J_z_y)
            J_l_z = J_l_y
            gradients_w.append(J_l_w)
            gradients_b.append(J_l_wb)
        return gradients_w, gradients_b


    def jacobian_loss_z(self, z, t):
        J_L_output = self.derivate_loss_function(z, t) # J_L_Z loss = z-t
        return J_L_output

    def jacobian_for_prev_layer(self, layer):
        return layer.J_Z_Y

    def jacobian_layer_weight(self):
        pass


    def train(self, input, target, epochs):
        for e in range(epochs):
            batch_delta_w = []
            batch_delta_b = []
            for x, y in zip(input.T,target.T):
                pred = self.forward_pass(x.reshape(-1,1))
                deltas_w, deltas_b = self.backward_pass(pred, y.reshape(-1,1))
                batch_delta_w.append(deltas_w)
                batch_delta_b.append(deltas_b)
            
            summed_batch_delta_w = batch_delta_w[0]
            for i in range(1, len(batch_delta_w)):
                for j in range(len(batch_delta_w[i])):
                    summed_batch_delta_w[j] += batch_delta_w[i][j]
            
            summed_batch_delta_b = batch_delta_b[0]
            for i in range(1, len(batch_delta_b)):
                for j in range(len(batch_delta_b[i])):
                    summed_batch_delta_b[j] += batch_delta_b[i][j]
            
            for i in range(len(summed_batch_delta_w)):
                summed_batch_delta_w[i] = summed_batch_delta_w[i] / len(batch_delta_w)
                summed_batch_delta_b[i] = summed_batch_delta_b[i] / len(batch_delta_b)

    
            self.update_weights(summed_batch_delta_w)
            self.update_biases(summed_batch_delta_b)

    
    def predict(self, input):
        pred = self.forward_pass(input)
        return pred
    

    def update_weights(self, deltas_w):
        for i in range(len(deltas_w)):
            self.hidden_layers[-i-1].W_T += -0.75 * deltas_w[i].T
            
    def update_biases(self, deltas_b):
        for i in range(len(deltas_b)):
            self.hidden_layers[-i-1].B += -0.75 * deltas_b[i].T
        




if __name__ == "__main__":

    nn = NeuralNetwork()

    nn.add_hidden_layer(Layer(2,2, (Sigmoid.apply, Sigmoid.derivative), 1, [-0.5,0.5], [-0.5,0.5]))
    nn.add_hidden_layer(Layer(1,2, (Sigmoid.apply, Sigmoid.derivative), 1, [-0.5,0.5], [-0.5,0.5]))
    """
    x = np.zeros((2,50))
    y = np.zeros((1,50))
    for i in range(50):
        rnd = r.randint(0,3)
        if rnd == 0:
            x[0,i] = 0
            x[1,i] = 0
            y[0,i] = 0
        if rnd == 1:
            x[0,i] = 1
            x[1,i] = 0
            y[0,i] = 1
        if rnd == 2:
            x[0,i] = 0
            x[1,i] = 1
            y[0,i] = 1
        if rnd == 3:
            x[0,i] = 1
            x[1,i] = 1
            y[0,i] = 0
    

    nn.train(x, y, 10000)

    print(nn.forward_pass(np.array([[1],[1]])))
    print(nn.forward_pass(np.array([[1],[0]])))
    print(nn.forward_pass(np.array([[0],[1]])))
    print(nn.forward_pass(np.array([[0],[0]])))
    """

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

    nn.train(x,y, 10000)
    print(nn.forward_pass(np.array([[1],[1]])))
    print(nn.forward_pass(np.array([[1],[0]])))
    print(nn.forward_pass(np.array([[0],[1]])))
    print(nn.forward_pass(np.array([[0],[0]])))

    print(nn.forward_pass(x))
    
    
