from activation import Linear, Relu, Sigmoid, Tanh
from layer import Layer
from config_reader import ModelParser
from network import NeuralNetwork
import numpy as np

class ConfigNetwork:
    
    def __init__(self):
        self.modul_parser = ModelParser()
        self.neural_network = NeuralNetwork()
        self.input_size = None
        # activation functions and their derivatives
        self.functions = {"sigmoid": (Sigmoid.apply, Sigmoid.derivative), 
                        "relu": (Relu.apply, Relu.derivative), 
                        "tanh": (Tanh.apply, Tanh.derivative), 
                        "linear": (Linear.apply, Linear.derivative)}

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
        
    def _dummy_act(self, x):
        return x

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

    def _config_data(self):
        pass

    def config_neural_network(self):
        self._config_layers()
        return self.neural_network


if __name__ == "__main__":
    cn = ConfigNetwork()
    nn = cn.config_neural_network()
    nn.forward_pass(np.ones(20))
    

