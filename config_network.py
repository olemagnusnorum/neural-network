from activation import MSE, CrossEntropy, Linear, Relu, Sigmoid, Tanh
from layer import Layer, SoftMaxLayer
from config_reader import ModelParser
from network import NeuralNetwork
import numpy as np

class ConfigNetwork:
    
    def __init__(self, file_path):
        self.modul_parser = ModelParser(file_path)
        self.neural_network = NeuralNetwork()
        self.input_size = None
        # activation functions and their derivatives
        self.functions = {"sigmoid": (Sigmoid.apply, Sigmoid.derivative), 
                        "relu": (Relu.apply, Relu.derivative), 
                        "tanh": (Tanh.apply, Tanh.derivative), 
                        "linear": (Linear.apply, Linear.derivative),
                        "mse": (MSE.apply, MSE.derivative),
                        "cross_entropy": (CrossEntropy.apply, CrossEntropy.derivative)}

    def _get_size(self, layer):
            return layer.get("size")

    def _get_lrate(self, globals ,layer):
        if "lrate" not in layer.keys():
            lrate = globals.get("lrate")
        else:
            lrate = layer.get("lrate")
        return lrate

    def _get_wr(self, globals, layer):
        if "wr" not in layer.keys():
            wr = globals.get("wr")
        else:
            wr = layer.get("wr")
        return wr
    
    def _get_act(self, globals, layer):
        if "act" not in layer.keys():
            act = globals.get("act")
        else:
            act = layer.get("act")
        return self.functions[act]

    def _get_loss_function(self, globals):
        loss_function = globals.get("loss")
        return self.functions[loss_function]
    
    def _set_verbose(self, globals):
        if "verbose" in globals.keys():
            if globals.get("verbose") == "true":
                self.neural_network.set_verbose(True)
            else:
                self.neural_network.set_verbose(False)

    def _set_wreg(self, globals):
        if "wreg" in globals.keys():
            wreg = globals.get("wreg")
            self.neural_network.set_wreg(wreg)
    
    def _set_wrt(self, globals):
        if "wrt" in globals.keys():
            wrt = globals.get("wrt")
            self.neural_network.set_wrt(wrt)

    
    def _config_network(self):
        global_values = self.modul_parser.get_globals()
        self._set_verbose(global_values)
        loss_function = self._get_loss_function(global_values)
        self.neural_network.set_loss_function(loss_function)
        self._set_wreg(global_values)
        self._set_wrt(global_values)


    def _config_layers(self):
        global_values = self.modul_parser.get_globals()
        input = self.modul_parser.get_input_layer()
        self.input_zise = self._get_size(input)
        hidden_layers_list = self.modul_parser.get_hidden_layers()
        num_previous_neurons = self.input_zise
        for i in hidden_layers_list:
            num_neurons = self._get_size(i)
            lrate = self._get_lrate(global_values, i)
            wr = self._get_wr(global_values, i)
            act = self._get_act(global_values, i)
            self.neural_network.add_hidden_layer(Layer(num_neurons, num_previous_neurons, act, lrate, wr))
            num_previous_neurons = num_neurons
        
        output_layer = self.modul_parser.get_output_layer()
        if len(output_layer) != 0:
            if output_layer.get('type') == 'softmax':
                self.neural_network.set_softmax_layer(SoftMaxLayer())

    def _config_data(self):
        pass

    def config_neural_network(self):
        self._config_network()
        self._config_layers()
        return self.neural_network


if __name__ == "__main__":
    cn = ConfigNetwork()
    nn = cn.config_neural_network()

    x = np.zeros((2,4))
    y = np.zeros((1,4))
    
    x[0,0] = 0
    x[1,0] = 0
    y[0,0] = 0 # sansynlighet for 1er

    

    x[0,1] = 1
    x[1,1] = 0
    y[0,1] = 1

    x[0,2] = 0
    x[1,2] = 1
    y[0,2] = 1

    x[0,3] = 1
    x[1,3] = 1
    y[0,3] = 0

    nn.train(x,y, 20000)

    print(nn.forward_pass(x))
    
    

