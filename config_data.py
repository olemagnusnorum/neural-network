from re import T
from config_reader import ModelParser
from generator import Generator


class ConfigData:

    def __init__(self) -> None:
        self.data = None
        self.model_parser = ModelParser()
        self.generator = Generator()
    
    def generate_data(self):
        data_dict = self.model_parser.get_data_generator()
        print(data_dict)
        number = data_dict.get('number')
        size = data_dict.get('size')
        split = data_dict.get('split')
        if len(split) == 0:
            split = (None, None, None)
        else:
            split = (split[0], split[1], split[2])
        noise = data_dict.get('noise')
        object_height_range = data_dict.get('object_height_range')
        if len(object_height_range) == 0:
            object_height_range = (10,10)
        else:
            object_height_range = (object_height_range[0], object_height_range[1])
        object_width_range = data_dict.get('object_width_range')
        if len(object_width_range) == 0:
            object_width_range = (10,10)
        else:
            object_width_range = (object_width_range[0], object_width_range[1])
        centerd = data_dict.get('centerd')
        if centerd == 'false':
            centerd = False
        else: 
            centerd = True
        flattend = data_dict.get('flattend')
        if flattend == 'false':
            flattend = False
        else:
            flattend = True
        images = self.generator.generate(number, size, split, noise, object_height_range, object_width_range, centerd, flattend)
        return images



if __name__ == "__main__":
    cd = ConfigData()
    print(cd.generate_data())