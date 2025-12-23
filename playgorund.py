"""
THE MAIN FILE OF THE PROJECT
"""
import matplotlib.pyplot as plt
import numpy as np


from config_data import ConfigData
from config_network import ConfigNetwork
from generator import Generator

def show_images():
    """
    generates and shows all pictures in the training set
    """
    g = Generator()
    i_set = g.generate(number=100, size=40, split=(0.7, 0.2, 0.1), noise=0.01, object_height_range=(20,39), object_width_range=(20,39), centerd=True, flattend=False)
    traning_set = i_set[0][0]
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(traning_set)):
        img = traning_set[i]
        fig.add_subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()



def run_network(file_path):
    cd = ConfigData(file_path)
    cn = ConfigNetwork(file_path)

    nn = cn.config_neural_network()
    data = cd.generate_data()
    print("generated")

    x_train = np.squeeze(data[0][0]).T
    y_train = data[0][1]

    x_valid = np.squeeze(data[1][0]).T
    y_valid = data[1][1]

    x_test = np.squeeze(data[2][0]).T
    y_test = data[2][1]

    nn.train(input=x_train, target=y_train, epochs=100, batch=20, valid_input=x_valid, valid_target=y_valid, test_input=x_test, test_target=y_test)

    # plotting graph
    plt.plot(nn.batch_graph, nn.loss_graph)
    plt.plot(nn.valid_batch_graph, nn.valid_loss_graph)
    plt.savefig("training.png")
    plt.show()

if __name__ == "__main__":
    #show_images()
    run_network("model_config2.ini")