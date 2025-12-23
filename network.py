import math
from activation import MSE, CrossEntropy, Linear, Relu, Sigmoid, SoftMax
from layer import Layer, SoftMaxLayer
from config_reader import ModelParser
import numpy as np
import random as r

class NeuralNetwork:
    """
    A class for the neural network containg the differen network layers
    """

    def __init__(self, loss_function=(MSE.apply, MSE.derivative), wreg=0, wrt="", verbose=False):
        self.hidden_layers = []
        self.softmax_layer = None
        self.loss_function = loss_function[0] 
        self.derivate_loss_function = loss_function[1]
        
        self.verbose = verbose

        self.wreg = wreg
        self.wrt = wrt
        
        self.loss_graph = []
        self.batch_graph = []

        self.valid_loss_graph = []
        self.valid_batch_graph = []

    

    def set_wreg(self, value):
        self.wreg = value

    def set_wrt(self, value):
        self.wrt = value

    def set_verbose(self, value):
        self.verbose = value

    def set_loss_function(self, function):
        self.loss_function = function[0]
        self.derivate_loss_function = function[1]

    def set_softmax_layer(self, layer):
        self.softmax_layer = layer

    def add_hidden_layer(self, layer): 
        """
        Function for adding a hidden layer
        """
        self.hidden_layers.append(layer)


    def set_softmax_layer(self, layer):
        """
        Function for adding the softmax layer
        """
        self.softmax_layer = layer


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
        
        if self.softmax_layer != None:
            input = self.softmax_layer.calculate_output(input)
            
        return input


    def backward_pass(self, pred, target):
        """
        Function for one bacward pass of the network, can have multiple targets and cases

        Args:
            pred (2d-array): predictions from the forward pass
            target (2d-array): target for the predictions

        Returns:
            two lists containing 3d-array of gradients for weights and bias, for each case and layer 
        """
        gradients_w = []
        gradients_b = []
        # computing the first jacobian with regard to loss function
        J_l_z = self.jacobian_loss_z(pred, target).T[:,np.newaxis,:]
        if self.softmax_layer != None:
            # moving from softmax layer to hidden layer
            J_l_z = np.einsum('hij,hjk->hik', J_l_z, self.softmax_layer.derivative_act(self.softmax_layer.input))
        for i in range(len(self.hidden_layers)-1,-1,-1):
            # every array here is a 3D-array
            # getting J_z_sum that is a 3d array with diagonal matrix with sum for each array
            J_z_sum = self.hidden_layers[i].derivative_act(self.hidden_layers[i].sum).T[:,:,np.newaxis]*np.eye(self.hidden_layers[i].sum.shape[0])
            # taking the dot-product of J_z_sum and the transposed weights to get J_z_y
            J_z_y = np.einsum('hij,hjk->hik', J_z_sum, self.hidden_layers[i].W_T[np.newaxis,:,:]) # dot product
            # taking the outer-product of the input to the layer and the J_z_sum diagonal to get J_z_w
            J_z_w = np.einsum('hki,hkj->hij', self.hidden_layers[i].input.T[:,np.newaxis,:], np.diagonal(J_z_sum, axis1=1, axis2=2)[:,np.newaxis,:])
            # taking the outer-product of 1(the input for bias) and J_z_sum to get J_z_wb 
            J_z_wb = np.einsum('i,hkj->hij', [1], np.diagonal(J_z_sum, axis1=1, axis2=2)[:,np.newaxis,:])
            
            # computing the gradient to the weights J_l_w
            # where we add regularization J_z_w + c * self.hidden_layers[i].W_T.T
            J_l_w = J_l_z * J_z_w
            if self.wrt == "L1":
                J_l_w = J_l_w + self.wreg * np.sign(self.add_hidden_layer[i].W_T.T)
            if self.wrt == "L2":
                J_l_w = J_l_w + self.wreg * self.hidden_layers[i].W_T.T
            # computing the gradient to the bias J_l_wb
            J_l_wb = J_l_z * J_z_wb
            # propagating up a layer by computing J_l_y as dot product of J_l_z and J_z_y
            J_l_y = np.einsum('hij,hjk->hik', J_l_z, J_z_y)
            J_l_z = J_l_y
            # adding the gradients of the layer to the list
            gradients_w.append(J_l_w)
            gradients_b.append(J_l_wb)
        # returns lists of the gradients for each layer and each case
        return gradients_w, gradients_b


    def jacobian_loss_z(self, z, t):
        """
        Function for calculating the loss gradient for the output layer
        
        Args:
            z (2d-array): array of the predicted value
            t (2d-array): array of the target value
        """
        J_l_output = self.derivate_loss_function(z, t)
        return J_l_output


    def train(self, input, target, epochs, batch=1, valid_input=[], valid_target=[], test_input=[], test_target=[]):
        """
        Function for training the neural net

        Args:
            input (2d-array): array of the arguments for the case/cases
            target(2d-array): array of the target value for the case/cases
            epochs (int): number of times traning throug the hole dataset
            batch_size (int): size of the batch used to calculate gradients
        """
        # for graphing
        if self.verbose:
            print("Network inputs:")
            print(input)
            print("Network outputs:")
            pred = self.forward_pass(input)
            print(pred)
            print("Network targets:")
            print(target)
            print("Network loss:")
            print(self.loss_function(pred, target).mean())
        step = 0
        # number of batches
        number = math.ceil(input.shape[1]/batch)
        input_batch = np.array_split(input, number, axis=1)
        target_batch = np.array_split(target, number, axis=1)
        for e in range(epochs):
            if e%10 == 0:
                print(f"epoch: {e}")
            # going through all batches
            for i in range(number):
                step += 1
                i_input = input_batch[i]
                i_target = target_batch[i]
                # prediction for the case
                pred = self.forward_pass(i_input)
                # getting gradients for the weights and biases from each case and layer
                deltas_w, deltas_b = self.backward_pass(pred, i_target)
                
                # getting the average gradient of weights for each layer
                mean_batch_delta_w = []
                for i in range(len(deltas_w)):
                    mean_batch_delta_w.append(deltas_w[i].mean(axis=0))

                # getting the average gradient of biases for each bias
                mean_batch_delta_b = []
                for i in range(len(deltas_b)):
                    mean_batch_delta_b.append(deltas_b[i].mean(axis=0))

                # updating the weight and biases
                self.update_weights(mean_batch_delta_w)
                self.update_biases(mean_batch_delta_b)
                # adding to the loss graph 
                self.loss_graph.append(self.loss_function(pred, i_target).mean())
                self.batch_graph.append(step)
                

            # run forward pass on valid set and save the result
            if len(valid_input) > 0:
                valid_pred = self.forward_pass(valid_input)
                self.valid_loss_graph.append(self.loss_function(valid_pred, valid_target).mean())
                self.valid_batch_graph.append(step)
        # run forward pass on test set and save the result  
        if len(test_input) > 0:
            test_pred = self.forward_pass(test_input)
            print(test_pred[0:5].T)
            print(test_target[0:5].T)
            print("Test set loss:")
            print(self.loss_function(test_pred, test_target).mean())
        # print verbose after train
        if self.verbose:
            print("Network inputs:")
            print(input)
            print("Network outputs:")
            pred = self.forward_pass(input)
            print(pred)
            print("Network targets:")
            print(target)
            print("Network loss:")
            print(self.loss_function(pred, target).mean())
    
    def predict(self, input):
        """
        Function for predicting case

        Args:
            input (2d-array): array of the arguments for the case/cases

        Returns:
            predicted values 
        """
        pred = self.forward_pass(input)
        return pred
    

    def update_weights(self, deltas_w):
        """
        Function for updating layer weights
        
        Args:
            deltas_w (list): list of gradients, the last layer first
        """
        for i in range(len(deltas_w)):
            self.hidden_layers[-i-1].W_T = self.hidden_layers[-i-1].W_T -self.hidden_layers[-i-1].lrate * deltas_w[i].T

            
    def update_biases(self, deltas_b):
        """
        Function for updating layer biases

        Args:
            deltas_b (list): list of gradients, the last layer first
        """
        for i in range(len(deltas_b)):
            self.hidden_layers[-i-1].B = self.hidden_layers[-i-1].B -self.hidden_layers[-i-1].lrate * deltas_b[i].T #+ (0.001 * self.hidden_layers[-i-1].B) # regualarization L2
        




if __name__ == "__main__":
    #testing softmax by training the NN to learn the XOR opperation
    nn = NeuralNetwork(loss_function=(CrossEntropy.apply, CrossEntropy.derivative))

    nn.add_hidden_layer(Layer(2,2, (Sigmoid.apply, Sigmoid.derivative), 0.1, [-0.5,0.5], [-0.5,0.5]))
    nn.add_hidden_layer(Layer(2,2, (Sigmoid.apply, Sigmoid.derivative), 0.1, [-0.5,0.5], [-0.5,0.5]))
    nn.set_softmax_layer(SoftMaxLayer())



    x = np.zeros((2,4))
    y = np.zeros((2,4))
    
    x[0,0] = 0
    x[1,0] = 0
    y[0,0] = 0 
    y[1,0] = 1 

    

    x[0,1] = 1
    x[1,1] = 0
    y[0,1] = 1
    y[1,1] = 0

    x[0,2] = 0
    x[1,2] = 1
    y[0,2] = 1
    y[1,2] = 0

    x[0,3] = 1
    x[1,3] = 1
    y[0,3] = 0
    y[1,3] = 1

    nn.train(x,y, 20000)

    print(nn.forward_pass(x))
    