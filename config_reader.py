from configparser import ConfigParser
import json


class ModelParser:

    def __init__(self):
        self.parser = ConfigParser()
        self.parser.read("model_config.txt")

    def get_globals(self):
        globals_values = self.parser.options("globals")
        for i in globals_values:
            globals_string = self.parser.get("globals", i)
            globals_json = globals_string.replace("\'", "\"")
            globals_dict = json.loads(globals_json)
            return globals_dict

    def get_input_layer(self):
        input_string = self.parser.get("layers", "input")
        input_json = input_string.replace("\'", "\"")
        input_dict = json.loads(input_json)
        return input_dict

    def get_output_layer(self):
        output_string = self.parser.get("layers", "output")
        output_json = output_string.replace("\'", "\"")
        output_dict = json.loads(output_json)
        return output_dict

    def get_hidden_layers(self):
        layers = self.parser.options("layers")
        hidden_layers = []
        for i in layers:
            #ignoring output and input
            if i == "input" or i == "output":
                continue
            hidden_layer = self.parser.get("layers", i)
            hidden_layer_json = hidden_layer.replace("\'", "\"")
            hidden_layer_dict = json.loads(hidden_layer_json)
            hidden_layers.append(hidden_layer_dict)
        
        return hidden_layers

  


            



if __name__ == "__main__":
    m = ModelParser()
    print(m.get_globals())
    print(m.get_input_layer())
    print(m.get_hidden_layers())
    m.get_output_layer()
