from configparser import ConfigParser
import json


class ModelParser:

    def __init__(self, file_path):
        self.parser = ConfigParser()
        self.parser.read(file_path)

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
        if self.parser.options("layers")[-1] != 'output':
            return {}
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

    def get_data_generator(self):
        generator_string = self.parser.get("data", "data")
        generator_json = generator_string.replace("\'", "\"")
        generator_dict = json.loads(generator_json)
        return generator_dict

    def get_image_verbose(self):
        if "images" in self.parser.options("data"):
            images_string = self.parser.get("data", "images")
            images_json = images_string.replace("\'", "\"")
            images_dict = json.loads(images_json)
            return images_dict
        return {}

  


            



if __name__ == "__main__":
    m = ModelParser()
    print(m.get_globals())
    print(m.get_input_layer())
    print(m.get_hidden_layers())
    print(m.get_output_layer())
    print(m.get_data_generator())
