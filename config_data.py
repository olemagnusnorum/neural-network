from re import T

from matplotlib import pyplot as plt
import numpy as np
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

        self.show_images(images, size, flattend)

        return images


    def show_images(self, images, size, flattend):
        image_verbose = self.model_parser.get_image_verbose()
        if len(image_verbose) > 0:
            if image_verbose.get("verbose") == "true":
                if flattend:
                    
                    x_train = np.squeeze(images[0][0]).T

                    x_valid = np.squeeze(images[1][0]).T

                    x_test = np.squeeze(images[2][0]).T

                    fig = plt.figure(figsize=(8, 8))
                    for i in range(10):
                        train_img = x_train[:,i].reshape(size, size)
                        fig.add_subplot(10,10,i+1)
                        plt.axis('off')
                        plt.imshow(train_img, "gray")
                    for i in range(10):
                        valid_img = x_valid[:,i].reshape(size, size)
                        fig.add_subplot(10,10,10+i+1)
                        plt.axis('off')
                        plt.imshow(valid_img, "summer")
                    for i in range(10):
                        test_img = x_test[:,i].reshape(size, size)
                        fig.add_subplot(10,10,20+i+1)
                        plt.axis('off')
                        plt.imshow(test_img, "spring")
                    plt.show()


if __name__ == "__main__":
    cd = ConfigData()
    print(cd.generate_data())